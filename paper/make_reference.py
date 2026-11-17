"""Create reference.docx (IEEE-like conference styles, A4) from pandoc's default."""
import os

import docx
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Mm, Pt, RGBColor

HERE = os.path.dirname(os.path.abspath(__file__))
TNR = "Times New Roman"


def set_font(style, size, bold=None, italic=None, small_caps=None):
    f = style.font
    f.name = TNR
    f.size = Pt(size)
    f.color.rgb = RGBColor(0, 0, 0)
    if bold is not None:
        f.bold = bold
    if italic is not None:
        f.italic = italic
    if small_caps is not None:
        f.small_caps = small_caps
    rpr = style.element.get_or_add_rPr()
    rf = rpr.find(qn("w:rFonts"))
    if rf is None:
        rf = OxmlElement("w:rFonts")
        rpr.insert(0, rf)
    for k in ("w:ascii", "w:hAnsi", "w:cs", "w:eastAsia"):
        rf.set(qn(k), TNR)
    for k in ("w:asciiTheme", "w:hAnsiTheme", "w:cstheme", "w:eastAsiaTheme"):
        if rf.get(qn(k)) is not None:
            del rf.attrib[qn(k)]


def set_par(style, before=0, after=0, align=None, first=None, left=None, hanging=None, keep_next=None):
    pf = style.paragraph_format
    pf.space_before = Pt(before)
    pf.space_after = Pt(after)
    pf.line_spacing = 1.0
    if align is not None:
        pf.alignment = align
    if first is not None:
        pf.first_line_indent = first
    if left is not None:
        pf.left_indent = left
    if hanging is not None:
        pf.left_indent = hanging
        pf.first_line_indent = -hanging
    if keep_next is not None:
        pf.keep_with_next = keep_next


def style(d, name, base="Normal"):
    try:
        return d.styles[name]
    except KeyError:
        s = d.styles.add_style(name, 1)
        s.base_style = d.styles[base]
        return s


def main():
    d = docx.Document(os.path.join(HERE, "ref_default.docx"))
    J = WD_ALIGN_PARAGRAPH.JUSTIFY
    C = WD_ALIGN_PARAGRAPH.CENTER
    for name in ("Normal", "Body Text", "First Paragraph", "Compact", "Block Text"):
        s = style(d, name)
        set_font(s, 10)
        set_par(s, align=J, first=Mm(3.6) if name in ("Body Text", "First Paragraph") else Mm(0))
    set_font(style(d, "Title"), 24)
    set_par(style(d, "Title"), after=6, align=C)
    for name in ("Author", "Subtitle", "Date"):
        set_font(style(d, name), 10)
        set_par(style(d, name), after=2, align=C)
    for name in ("Abstract", "Keywords"):
        s = style(d, name)
        set_font(s, 9, bold=True)
        set_par(s, after=6 if name == "Keywords" else 4, align=J, first=Mm(3.6))
    h1 = style(d, "Heading 1")
    set_font(h1, 10, bold=False, italic=False, small_caps=True)
    set_par(h1, before=8, after=4, align=C, keep_next=True)
    h2 = style(d, "Heading 2")
    set_font(h2, 10, bold=False, italic=True)
    set_par(h2, before=4, after=2, align=WD_ALIGN_PARAGRAPH.LEFT, keep_next=True)
    h3 = style(d, "Heading 3")
    set_font(h3, 10, bold=False, italic=True)
    set_par(h3, before=2, after=1, align=WD_ALIGN_PARAGRAPH.LEFT, keep_next=True)
    for name in ("Image Caption", "Caption"):
        set_font(style(d, name), 8, italic=False)
        set_par(style(d, name), before=2, after=8, align=J)
    tc = style(d, "Table Caption")
    set_font(tc, 8, italic=False, small_caps=True)
    set_par(tc, before=6, after=3, align=C, keep_next=True)
    for name in ("Figure", "Captioned Figure"):
        set_par(style(d, name), before=4, after=0, align=C, keep_next=True)
    ref = style(d, "Reference")
    set_font(ref, 8)
    set_par(ref, after=1, align=J, hanging=Mm(5.5))
    note = style(d, "TableNote")
    set_font(note, 7)
    set_par(note, before=1, after=6, align=J)
    for name in ("Heading 4", "Heading 5", "Heading 6"):
        set_font(style(d, name), 10, bold=False, italic=True)
    # page setup: A4, IEEE-like margins; columns are set by build.py
    sec = d.sections[0]
    sec.page_width, sec.page_height = Mm(210), Mm(297)
    sec.top_margin, sec.bottom_margin = Mm(19), Mm(25)
    sec.left_margin = sec.right_margin = Mm(16)
    d.save(os.path.join(HERE, "reference.docx"))
    print("reference.docx written")


if __name__ == "__main__":
    main()
