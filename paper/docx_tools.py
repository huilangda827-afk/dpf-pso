"""Post-processing of the pandoc output: IEEE-like two-column layout,
numbered display equations, three-line tables, and PDF export via Word."""
from __future__ import annotations

import copy
import os

import docx
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Mm, Pt

M_NS = "http://schemas.openxmlformats.org/officeDocument/2006/math"
PAGE_W, PAGE_H = Mm(210), Mm(297)
MARG_T, MARG_B, MARG_LR = Mm(19), Mm(25), Mm(16)
COL_GAP = Mm(5)
TEXT_W = PAGE_W - 2 * MARG_LR
COL_W = (TEXT_W - COL_GAP) // 2


def _twips(emu):
    return int(round(emu / 635))


def _sectpr(cols, continuous=True):
    s = OxmlElement("w:sectPr")
    if continuous:
        t = OxmlElement("w:type")
        t.set(qn("w:val"), "continuous")
        s.append(t)
    sz = OxmlElement("w:pgSz")
    sz.set(qn("w:w"), str(_twips(PAGE_W)))
    sz.set(qn("w:h"), str(_twips(PAGE_H)))
    s.append(sz)
    mg = OxmlElement("w:pgMar")
    for k, v in (("top", MARG_T), ("bottom", MARG_B), ("left", MARG_LR), ("right", MARG_LR),
                 ("header", Mm(10)), ("footer", Mm(10)), ("gutter", 0)):
        mg.set(qn(f"w:{k}"), str(_twips(v)))
    s.append(mg)
    c = OxmlElement("w:cols")
    c.set(qn("w:num"), str(cols))
    c.set(qn("w:space"), str(_twips(COL_GAP)))
    s.append(c)
    return s


def _shrink(p):
    for r in p.runs:
        r.text = ""
    pf = p.paragraph_format
    pf.space_before = pf.space_after = Pt(0)
    pf.line_spacing = Pt(1)
    pf.first_line_indent = Mm(0)
    rpr = p._p.get_or_add_pPr()
    mark = OxmlElement("w:rPr")
    sz = OxmlElement("w:sz")
    sz.set(qn("w:val"), "2")
    mark.append(sz)
    rpr.append(mark)


def _ptext(e):
    return "".join(t.text or "" for t in e.iter(qn("w:t"))).strip()


def _float_box(elements, width):
    """Full-width floating container (1x1 table anchored at the top margin of
    the page where it falls) holding the given body elements."""
    tbl = OxmlElement("w:tbl")
    tblPr = OxmlElement("w:tblPr")
    pp = OxmlElement("w:tblpPr")
    # absolute position at the top margin of the anchor's page: the box never
    # drags its anchor paragraph to the next page
    for k, val in (("leftFromText", "0"), ("rightFromText", "0"), ("topFromText", "0"),
                   ("bottomFromText", "200"), ("vertAnchor", "page"), ("horzAnchor", "margin"),
                   ("tblpXSpec", "center"), ("tblpY", str(_twips(MARG_T)))):
        pp.set(qn(f"w:{k}"), val)
    tblPr.append(pp)
    ov = OxmlElement("w:tblOverlap")
    ov.set(qn("w:val"), "never")
    tblPr.append(ov)
    w = OxmlElement("w:tblW")
    w.set(qn("w:w"), str(_twips(width)))
    w.set(qn("w:type"), "dxa")
    tblPr.append(w)
    jc = OxmlElement("w:jc")
    jc.set(qn("w:val"), "center")
    tblPr.append(jc)
    b = OxmlElement("w:tblBorders")
    for tag in ("top", "left", "bottom", "right", "insideH", "insideV"):
        b.append(_border(tag, 0))
    tblPr.append(b)
    lay = OxmlElement("w:tblLayout")
    lay.set(qn("w:type"), "fixed")
    tblPr.append(lay)
    cm = OxmlElement("w:tblCellMar")
    for side in ("top", "left", "bottom", "right"):
        e = OxmlElement(f"w:{side}")
        e.set(qn("w:w"), "0")
        e.set(qn("w:type"), "dxa")
        cm.append(e)
    tblPr.append(cm)
    look = OxmlElement("w:tblLook")        # no header-row formatting from the default style
    for flag in ("firstRow", "lastRow", "firstColumn", "lastColumn", "noHBand", "noVBand"):
        look.set(qn(f"w:{flag}"), "1" if flag in ("noHBand", "noVBand") else "0")
    tblPr.append(look)
    tbl.append(tblPr)
    grid = OxmlElement("w:tblGrid")
    gc = OxmlElement("w:gridCol")
    gc.set(qn("w:w"), str(_twips(width)))
    grid.append(gc)
    tbl.append(grid)
    tr = OxmlElement("w:tr")
    tc = OxmlElement("w:tc")
    tcPr = OxmlElement("w:tcPr")
    tw = OxmlElement("w:tcW")
    tw.set(qn("w:w"), str(_twips(width)))
    tw.set(qn("w:type"), "dxa")
    tcPr.append(tw)
    cb = OxmlElement("w:tcBorders")
    for tag in ("top", "left", "bottom", "right"):
        cb.append(_border(tag, 0))
    tcPr.append(cb)
    tc.append(tcPr)
    for e in elements:
        tc.append(e)
    if elements and elements[-1].tag != qn("w:p"):
        tc.append(OxmlElement("w:p"))
    tr.append(tc)
    tbl.append(tr)
    return tbl


def layout_sections(d, first_cols=1, wide_mode="float", columns=2):
    """Markers (own paragraphs): [[COLS2]] ends the one-column title block;
    [[WIDE]] ... [[/WIDE]] encloses full-width material, either as a floating
    box at the top of the page (wide_mode='float') or as a one-column section."""
    body = d.element.body
    if wide_mode == "float":
        kids = list(body)
        i = 0
        while i < len(kids):
            e = kids[i]
            if e.tag == qn("w:p") and _ptext(e) == "[[WIDE]]":
                j = i + 1
                while _ptext(kids[j]) != "[[/WIDE]]":
                    j += 1
                inner = kids[i + 1:j]
                for x in inner:
                    body.remove(x)
                box = _float_box(inner, TEXT_W)
                e.addprevious(box)
                body.remove(e)
                body.remove(kids[j])
                kids = list(body)
                i = kids.index(box) + 1
                continue
            i += 1
    for p in list(d.paragraphs):
        t = p.text.strip()
        if t == "[[COLS2]]":
            _shrink(p)
            p._p.get_or_add_pPr().append(_sectpr(1, continuous=False))
        elif t == "[[WIDE]]":
            _shrink(p)
            p._p.get_or_add_pPr().append(_sectpr(2))
        elif t == "[[/WIDE]]":
            _shrink(p)
            p._p.get_or_add_pPr().append(_sectpr(1))
    body = d.element.body
    last = body.find(qn("w:sectPr"))
    new = _sectpr(columns)
    if last is not None:
        body.replace(last, new)
    else:
        body.append(new)
    first = None
    for p in d.paragraphs:
        sp = p._p.find(qn("w:pPr"))
        if sp is not None and sp.find(qn("w:sectPr")) is not None:
            first = sp.find(qn("w:sectPr"))
            break
    if first is not None:   # the very first section is a new page section
        t = first.find(qn("w:type"))
        if t is not None:
            first.remove(t)


def number_equations(d, num_w=Mm(7), width=COL_W):
    """Each display equation becomes a borderless 1x2 table: the equation
    (kept as display math, so Word can break long lines) in the left cell and
    a right-aligned label (k) in the right cell."""
    k = 0
    for p in list(d.paragraphs):
        omp = p._p.find(f"{{{M_NS}}}oMathPara")
        if omp is None:
            continue
        k += 1
        tb = d.add_table(rows=1, cols=2)
        tbl = tb._tbl
        p._p.addnext(tbl)
        tblPr = tbl.tblPr
        for tag in ("w:tblStyle", "w:tblW", "w:tblBorders", "w:tblLayout", "w:tblCellMar"):
            for old in tblPr.findall(qn(tag)):
                tblPr.remove(old)
        b = OxmlElement("w:tblBorders")
        for tag in ("top", "left", "bottom", "right", "insideH", "insideV"):
            b.append(_border(tag, 0))
        tblPr.append(b)
        w = OxmlElement("w:tblW")
        w.set(qn("w:w"), str(_twips(width)))
        w.set(qn("w:type"), "dxa")
        tblPr.append(w)
        lay = OxmlElement("w:tblLayout")
        lay.set(qn("w:type"), "fixed")
        tblPr.append(lay)
        cm = OxmlElement("w:tblCellMar")
        for side in ("left", "right", "top", "bottom"):
            e = OxmlElement(f"w:{side}")
            e.set(qn("w:w"), "0")
            e.set(qn("w:type"), "dxa")
            cm.append(e)
        tblPr.append(cm)
        widths = (width - num_w, num_w)
        grid = tbl.find(qn("w:tblGrid"))
        for gc, wd in zip(grid.findall(qn("w:gridCol")), widths):
            gc.set(qn("w:w"), str(_twips(wd)))
        for old in tblPr.findall(qn("w:tblLook")):
            tblPr.remove(old)
        look = OxmlElement("w:tblLook")
        for flag in ("firstRow", "lastRow", "firstColumn", "lastColumn", "noHBand", "noVBand"):
            look.set(qn(f"w:{flag}"), "1" if flag in ("noHBand", "noVBand") else "0")
        tblPr.append(look)
        _sort_tblpr(tblPr)
        c0, c1 = tb.rows[0].cells
        for c, wd in zip((c0, c1), widths):
            tcPr = c._tc.get_or_add_tcPr()
            tw = OxmlElement("w:tcW")
            tw.set(qn("w:w"), str(_twips(wd)))
            tw.set(qn("w:type"), "dxa")
            tcPr.append(tw)
            cb = OxmlElement("w:tcBorders")
            for tag in ("top", "left", "bottom", "right"):
                cb.append(_border(tag, 0))
            tcPr.append(cb)
            va = OxmlElement("w:vAlign")
            va.set(qn("w:val"), "center")
            tcPr.append(va)
        p0 = c0.paragraphs[0]
        p0._p.append(omp)
        p0.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p1 = c1.paragraphs[0]
        p1.add_run(f"({k})").font.size = Pt(10)
        p1.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.RIGHT
        for pp in (p0, p1):
            pf = pp.paragraph_format
            pf.first_line_indent = Mm(0)
            pf.space_before = Pt(2)
            pf.space_after = Pt(2)
        if not p.text.strip() and not p._p.findall(".//" + qn("w:drawing")):
            p._p.getparent().remove(p._p)
    return k


_TBLPR_ORDER = ["tblStyle", "tblpPr", "tblOverlap", "bidiVisual", "tblStyleRowBandSize",
                "tblStyleColBandSize", "tblW", "jc", "tblCellSpacing", "tblInd", "tblBorders",
                "shd", "tblLayout", "tblCellMar", "tblLook", "tblCaption", "tblDescription"]


def _sort_tblpr(tblPr):
    """Reorder tblPr children into the order required by the OOXML schema."""
    kids = list(tblPr)
    rank = {f"{{http://schemas.openxmlformats.org/wordprocessingml/2006/main}}{k}": i
            for i, k in enumerate(_TBLPR_ORDER)}
    kids.sort(key=lambda e: rank.get(e.tag, len(rank)))
    for e in list(tblPr):
        tblPr.remove(e)
    for e in kids:
        tblPr.append(e)


def _border(tag, sz):
    e = OxmlElement(f"w:{tag}")
    e.set(qn("w:val"), "single" if sz else "nil")
    if sz:
        e.set(qn("w:sz"), str(sz))
        e.set(qn("w:space"), "0")
        e.set(qn("w:color"), "000000")
    return e


def _all_tables(d):
    """All content tables, including tables nested in floating containers."""
    from docx.table import Table
    out = []
    for tbl in d.element.body.iter(qn("w:tbl")):
        tblPr = tbl.find(qn("w:tblPr"))
        if tblPr is not None and tblPr.find(qn("w:tblpPr")) is not None:
            continue                       # floating container
        out.append(Table(tbl, d._body))
    return out


def style_tables(d, font_pt=7.5):
    for tb in _all_tables(d):
        tb.alignment = WD_TABLE_ALIGNMENT.CENTER
        tblPr = tb._tbl.tblPr
        for tag in ("w:tblBorders", "w:tblW", "w:tblLayout", "w:tblCellMar"):
            for old in tblPr.findall(qn(tag)):
                tblPr.remove(old)
        b = OxmlElement("w:tblBorders")
        for tag, sz in (("top", 8), ("left", 0), ("bottom", 8), ("right", 0),
                        ("insideH", 0), ("insideV", 0)):
            b.append(_border(tag, sz))
        tblPr.append(b)
        w = OxmlElement("w:tblW")
        w.set(qn("w:w"), "5000")
        w.set(qn("w:type"), "pct")
        tblPr.append(w)
        lay = OxmlElement("w:tblLayout")
        lay.set(qn("w:type"), "autofit")
        tblPr.append(lay)
        cm = OxmlElement("w:tblCellMar")
        for side in ("left", "right"):
            e = OxmlElement(f"w:{side}")
            e.set(qn("w:w"), "40")
            e.set(qn("w:type"), "dxa")
            cm.append(e)
        tblPr.append(cm)
        for old in tblPr.findall(qn("w:tblLayout"))[:-1]:
            tblPr.remove(old)
        _sort_tblpr(tblPr)
        nrows = len(tb.rows)
        for i, row in enumerate(tb.rows):
            trPr = row._tr.get_or_add_trPr()
            cs = OxmlElement("w:cantSplit")
            trPr.append(cs)
            for j, cell in enumerate(row.cells):
                tcPr = cell._tc.get_or_add_tcPr()
                for old in tcPr.findall(qn("w:tcW")):
                    tcPr.remove(old)
                if i == 0:
                    bb = OxmlElement("w:tcBorders")
                    bb.append(_border("bottom", 4))
                    tcPr.append(bb)
                for p in cell.paragraphs:
                    p.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.LEFT if j == 0 \
                        else WD_ALIGN_PARAGRAPH.CENTER
                    p.paragraph_format.first_line_indent = Mm(0)
                    p.paragraph_format.space_before = Pt(0.5)
                    p.paragraph_format.space_after = Pt(0.5)
                    p.paragraph_format.keep_with_next = i < nrows - 1   # keep table in one piece
                    for r in p.runs:
                        r.font.size = Pt(font_pt)
                        r.font.name = "Times New Roman"
                    for mr in p._p.iter(f"{{{M_NS}}}r"):              # math runs: same size
                        rpr = mr.find(qn("w:rPr"))
                        if rpr is None:
                            rpr = OxmlElement("w:rPr")
                            mr.insert(0, rpr)
                        for old in rpr.findall(qn("w:sz")):
                            rpr.remove(old)
                        sz = OxmlElement("w:sz")
                        sz.set(qn("w:val"), str(int(round(font_pt * 2))))
                        rpr.append(sz)
        grid = tb._tbl.find(qn("w:tblGrid"))
        if grid is not None:
            for gc in grid.findall(qn("w:gridCol")):
                del gc.attrib[qn("w:w")]


def center_figures(d):
    from docx.text.paragraph import Paragraph
    for pe in d.element.body.iter(qn("w:p")):
        p = Paragraph(pe, d._body)
        if p.style.name in ("Captioned Figure", "Figure") or pe.findall(".//" + qn("w:drawing")):
            p.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.first_line_indent = Mm(0)


def anonymize(d):
    """Remove identifying document properties (double-blind review)."""
    cp = d.core_properties
    for k in ("author", "last_modified_by", "comments", "keywords", "subject", "category",
              "content_status", "identifier", "version"):
        setattr(cp, k, "")
    cp.title = ""
    cp.revision = 1


def postprocess(src, dst, wide_mode="float", columns=2):
    d = docx.Document(src)
    layout_sections(d, wide_mode=wide_mode, columns=columns)
    style_tables(d)          # real tables first; equation tables are borderless
    n_eq = number_equations(d, width=COL_W if columns == 2 else TEXT_W)
    center_figures(d)
    anonymize(d)
    d.save(dst)
    return n_eq


def export_pdf(docx_path, pdf_path):
    import pythoncom
    import win32com.client
    pythoncom.CoInitialize()
    word = win32com.client.DispatchEx("Word.Application")
    word.Visible = False
    word.DisplayAlerts = 0
    try:
        doc = word.Documents.Open(os.path.abspath(docx_path), ReadOnly=True)
        doc.Repaginate()
        pages = doc.ComputeStatistics(2)
        doc.ExportAsFixedFormat(os.path.abspath(pdf_path), 17)
        doc.Close(False)
    finally:
        word.Quit()
    return pages


def copy_docx(src, dst):
    d = docx.Document(src)
    d.save(dst)
    return copy.copy(dst)
