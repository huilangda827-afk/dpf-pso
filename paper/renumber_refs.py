"""Renumber bracketed citations by order of first appearance (journal style)
and reorder the reference list accordingly.

The templates cite references as [12], [13], [14] or as ranges [7]–[10];
the reference list is a sequence of paragraphs starting with "[n] ".  When
new references are inserted, numbers in the text may no longer follow the
order of first citation.  renumber(md) expands ranges, maps every old number
to its rank of first appearance in the body, rewrites the citations
(re-collapsing runs of three or more consecutive numbers into ranges) and
rewrites the reference list in the new order.  References that are never
cited are dropped with a warning; citations without a list entry raise.
"""
from __future__ import annotations

import re
import sys

CITE = re.compile(r"\[(\d+)\](?:(?:, \[(\d+)\])|(?:–\[(\d+)\]))*")
RANGE = re.compile(r"\[(\d+)\]–\[(\d+)\]")
SINGLE = re.compile(r"\[(\d+)\]")


def _expand(text):
    """[7]–[10] -> [7], [8], [9], [10] (inside the body only)."""
    return RANGE.sub(lambda m: ", ".join(f"[{k}]" for k in range(int(m.group(1)), int(m.group(2)) + 1)), text)


def _collapse(nums):
    """[1, 2, 3, 5] -> '[1]–[3], [5]' (ranges only for runs of three or more)."""
    out, i = [], 0
    while i < len(nums):
        j = i
        while j + 1 < len(nums) and nums[j + 1] == nums[j] + 1:
            j += 1
        if j - i >= 2:
            out.append(f"[{nums[i]}]–[{nums[j]}]")
        else:
            out.extend(f"[{k}]" for k in nums[i:j + 1])
        i = j + 1
    return ", ".join(out)


def first_appearance_map(md):
    """{old number: new number} by order of first citation in the body."""
    head = md.partition("\n# References")[0]
    order = []
    for m in SINGLE.finditer(_expand(head)):
        k = int(m.group(1))
        if k not in order:
            order.append(k)
    return {old: i + 1 for i, old in enumerate(order)}


def apply_map(md, new):
    """Rewrite the citations of a document (e.g. the supplement) with a
    mapping computed from another document (the main text)."""
    body = _expand(md)

    def group_sub(m):
        nums = sorted({new[int(x)] for x in SINGLE.findall(m.group(0))})
        return _collapse(nums)
    return re.sub(r"\[\d+\](?:, \[\d+\])*", group_sub, body)


def renumber(md):
    head, sep, refs = md.partition("\n# References")
    if not sep:
        raise ValueError("no '# References' section")
    body = _expand(head)
    order = []
    for m in SINGLE.finditer(body):
        k = int(m.group(1))
        if k not in order:
            order.append(k)
    new = {old: i + 1 for i, old in enumerate(order)}
    # reference list entries
    entries = {}
    for m in re.finditer(r"^\[(\d+)\] (.*?)(?=^\[\d+\] |\Z)", refs, flags=re.M | re.S):
        entries[int(m.group(1))] = re.sub(r"\n:::\s*$", "", m.group(2)).rstrip("\n")   # last entry: drop the fence
    missing = [k for k in order if k not in entries]
    if missing:
        raise ValueError(f"cited but not in the reference list: {missing}")
    unused = sorted(set(entries) - set(order))
    if unused:
        print(f"renumber_refs: dropping uncited references {unused}", file=sys.stderr)

    # rewrite citation groups in the body: a group is a maximal run of "[k]" separated by ", "
    def group_sub(m):
        nums = sorted({new[int(x)] for x in SINGLE.findall(m.group(0))})
        return _collapse(nums)
    body = re.sub(r"\[\d+\](?:, \[\d+\])*", group_sub, body)
    # rewrite the list
    pre = refs[:refs.find("[")] if "[" in refs else refs
    post_m = re.search(r"\n:::\s*$", refs)
    post = refs[post_m.start():] if post_m else ""
    items = "\n\n".join(f"[{new[k]}] {entries[k].strip()}" for k in order)
    return body + "\n# References" + pre.rstrip("\n") + "\n" + items + "\n" + post.strip("\n") + "\n"


if __name__ == "__main__":
    src = open(sys.argv[1], encoding="utf-8").read()
    open(sys.argv[2], "w", encoding="utf-8").write(renumber(src))
