"""Helpers for the MDPI Word template.

    python paper/mdpi_tools.py convert <template.dot> <out.docx>   # Word, macros disabled
    python paper/mdpi_tools.py styles <file.docx>                  # list paragraph/character styles

The template is a downloaded file: it is only ever opened read-only with
macros disabled and saved as a plain .docx for inspection.
"""
from __future__ import annotations

import os
import sys


def dot_to_docx(src, dst):
    import pythoncom
    import win32com.client
    pythoncom.CoInitialize()
    word = win32com.client.DispatchEx("Word.Application")
    word.Visible = False
    word.DisplayAlerts = 0
    word.AutomationSecurity = 3          # msoAutomationSecurityForceDisable: no macros
    try:
        doc = word.Documents.Open(os.path.abspath(src), ReadOnly=True, AddToRecentFiles=False)
        doc.SaveAs2(os.path.abspath(dst), FileFormat=12)      # wdFormatXMLDocument (.docx, no macros)
        doc.Close(False)
    finally:
        word.Quit()


def list_styles(path):
    import docx
    from docx.enum.style import WD_STYLE_TYPE
    d = docx.Document(path)
    used = {}
    for p in d.paragraphs:
        used.setdefault(p.style.name, []).append(p.text.strip()[:60])
    print("== paragraph styles used in the template body (first text sample):")
    for name, texts in used.items():
        print(f"  {name!r}: {len(texts)} paragraphs | {next((t for t in texts if t), '')!r}")
    print("== all paragraph styles defined:")
    print("  " + ", ".join(sorted(s.name for s in d.styles if s.type == WD_STYLE_TYPE.PARAGRAPH)))
    print("== character styles defined:")
    print("  " + ", ".join(sorted(s.name for s in d.styles if s.type == WD_STYLE_TYPE.CHARACTER)))
    print("== table styles defined:")
    print("  " + ", ".join(sorted(s.name for s in d.styles if s.type == WD_STYLE_TYPE.TABLE)))
    s = d.sections[0]
    print(f"== page: {s.page_width.mm:.0f} x {s.page_height.mm:.0f} mm, margins L{s.left_margin.mm:.0f} R{s.right_margin.mm:.0f} "
          f"T{s.top_margin.mm:.0f} B{s.bottom_margin.mm:.0f} mm; tables in template: {len(d.tables)}")


if __name__ == "__main__":
    if sys.argv[1] == "convert":
        dot_to_docx(sys.argv[2], sys.argv[3])
        print("written", sys.argv[3])
    elif sys.argv[1] == "styles":
        list_styles(sys.argv[2])
