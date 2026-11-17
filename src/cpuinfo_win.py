"""Logical processors of performance cores on Windows (hybrid CPUs).

Uses GetSystemCpuSetInformation (Windows 10+).  Returns one logical
processor per physical core of the highest efficiency class, so that pinned
benchmark processes neither share a core (SMT) nor run on efficiency cores.
"""
from __future__ import annotations

import ctypes
from ctypes import wintypes


class _CpuSet(ctypes.Structure):
    _fields_ = [("Id", wintypes.DWORD), ("Group", wintypes.WORD),
                ("LogicalProcessorIndex", ctypes.c_ubyte), ("CoreIndex", ctypes.c_ubyte),
                ("LastLevelCacheIndex", ctypes.c_ubyte), ("NumaNodeIndex", ctypes.c_ubyte),
                ("EfficiencyClass", ctypes.c_ubyte), ("AllFlags", ctypes.c_ubyte),
                ("SchedulingClass", wintypes.DWORD), ("AllocationTag", ctypes.c_uint64)]


class _Info(ctypes.Structure):
    _fields_ = [("Size", wintypes.DWORD), ("Type", wintypes.DWORD), ("CpuSet", _CpuSet)]


def cpu_sets():
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    need = wintypes.ULONG(0)
    k32.GetSystemCpuSetInformation(None, 0, ctypes.byref(need), None, 0)
    buf = (ctypes.c_byte * need.value)()
    if not k32.GetSystemCpuSetInformation(buf, need, ctypes.byref(need), None, 0):
        raise OSError(ctypes.get_last_error())
    out, off = [], 0
    while off < need.value:
        info = _Info.from_buffer(buf, off)
        cs = info.CpuSet
        out.append(dict(logical=cs.LogicalProcessorIndex, core=cs.CoreIndex,
                        eff=cs.EfficiencyClass, group=cs.Group))
        off += info.Size
    return out


def performance_cores():
    """(list of logical processor indices, one per P-core; summary dict)."""
    sets = cpu_sets()
    top = max(s["eff"] for s in sets)
    first = {}
    for s in sorted(sets, key=lambda s: s["logical"]):
        if s["eff"] == top and s["core"] not in first:
            first[s["core"]] = s["logical"]
    classes = {}
    for s in sets:
        classes[s["eff"]] = classes.get(s["eff"], 0) + 1
    return sorted(first.values()), dict(n_logical=len(sets), logical_per_class=classes,
                                        n_pcores=len(first), efficiency_class_used=top)


if __name__ == "__main__":
    lps, info = performance_cores()
    print(info)
    print("one logical processor per P-core:", lps)
