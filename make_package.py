"""Build an anonymous code-and-data package for double-blind review.

    python make_package.py

Creates dist/DPF-PSO_code_data.zip with the source code, experiment scripts,
paper sources, figures and the raw results of the second and third revision.
Logs and first-revision archives are excluded.  The public DEM is not redistributed
(data/get_dem.py downloads it).  Every text file is scanned for strings that
could identify the authors or their machine; the build fails if one is found.
"""
from __future__ import annotations

import getpass
import os
import re
import sys
import zipfile

ROOT = os.path.dirname(os.path.abspath(__file__))
DIST = os.path.join(ROOT, "dist")
TAGS = ["main", "heldout", "public", "eqtime", "n30", "budget5", "b2", "timing", "tuning", "analysis",
        "heldout_band", "heldout_eqtime", "check_cmaest_identity"]
TEXT_EXT = {".py", ".md", ".json", ".jsonl", ".ps1", ".txt", ".csv"}


def forbidden_patterns():
    pats = {getpass.getuser(), os.path.basename(os.path.dirname(ROOT)), "multisim"}
    home = os.path.expanduser("~")
    pats.add(os.path.basename(home))
    pats.add(ROOT.replace("\\", "/"))
    pats.add(ROOT)
    return [p for p in pats if p and len(p) >= 3]


def collect():
    files = []
    for sub in ("src", "notes"):
        for dp, _, fs in os.walk(os.path.join(ROOT, sub)):
            if "__pycache__" in dp:
                continue
            files += [os.path.join(dp, f) for f in fs if os.path.splitext(f)[1] in TEXT_EXT]
    for f in ("README.md", "run_all.ps1", "run_rev3.ps1", "make_package.py"):
        files.append(os.path.join(ROOT, f))
    for f in ("build2.py", "tables2.py", "supplement_tables.py", "docx_tools.py", "make_reference.py",
              "paper_template2.md", "supplement_template.md", "reference.docx", "supplement.pdf"):
        p = os.path.join(ROOT, "paper", f)
        if os.path.exists(p):
            files.append(p)
    files += [os.path.join(ROOT, "data", f) for f in ("get_dem.py", "SPSO_license.txt")]
    files += [os.path.join(ROOT, "figures", f) for f in os.listdir(os.path.join(ROOT, "figures"))
              if f.endswith(".png")]
    for t in TAGS:
        d = os.path.join(ROOT, "results", t)
        if os.path.isdir(d):
            for dp, _, fs in os.walk(d):
                files += [os.path.join(dp, f) for f in fs if f.endswith((".npz", ".json", ".jsonl", ".md"))]
    return [f for f in files if os.path.exists(f)]


def main():
    files = collect()
    pats = forbidden_patterns()
    bad = []
    for f in files:
        ext = os.path.splitext(f)[1]
        if ext in TEXT_EXT:
            txt = open(f, encoding="utf-8", errors="ignore").read()
        elif ext == ".pdf":                       # supplementary material: text and metadata
            import fitz
            with fitz.open(f) as doc:
                txt = "".join(page.get_text() for page in doc) + repr(doc.metadata)
        else:
            continue
        for p in pats:
            if p in txt and os.path.basename(f) != "make_package.py":
                bad.append((os.path.relpath(f, ROOT), p))
    if bad:
        for f, p in bad:
            print("identifying string found:", f, repr(p))
        sys.exit(1)
    os.makedirs(DIST, exist_ok=True)
    out = os.path.join(DIST, "DPF-PSO_code_data.zip")
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for f in files:
            z.write(f, os.path.join("DPF-PSO", os.path.relpath(f, ROOT)))
    size = os.path.getsize(out) / 2 ** 20
    print(f"{out}: {len(files)} files, {size:.1f} MB; no identifying strings found "
          f"(checked {len(pats)} patterns)")


if __name__ == "__main__":
    main()
