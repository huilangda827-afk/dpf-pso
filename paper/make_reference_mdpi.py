"""Build the pandoc reference document for the MDPI layout.

    python paper/make_reference_mdpi.py            # -> paper/reference_mdpi.docx

Starts from the converted journal template (paper/mdpi_template/
drones-template.docx: page set-up, headers, all MDPI_* styles), removes the
example text and defines the style names that pandoc and our templates use
as thin aliases of the corresponding MDPI styles (base style only, no direct
formatting), so that the generated document uses the publisher's styles.
"""
from __future__ import annotations

import copy
import os

import docx
from docx.enum.style import WD_STYLE_TYPE
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "mdpi_template", "drones-template.docx")
DST = os.path.join(HERE, "reference_mdpi.docx")

# pandoc / template style name -> MDPI style it should inherit from
ALIASES = {
    "Normal": "MDPI_3.1_text",            # font only (see below): Normal is also the base of table cells
    "Body Text": "MDPI_3.1_text",
    "First Paragraph": "MDPI_3.1_text",
    "Compact": "MDPI_4.2_table_body",     # pandoc uses Compact in table cells and tight lists
    "Heading 1": "MDPI_2.1_heading1",
    "Heading 2": "MDPI_2.2_heading2",
    "Heading 3": "MDPI_2.3_heading3",
    "Title": "MDPI_1.2_title",
    "ArticleType": "MDPI_1.1_article_type",
    "Author": "MDPI_1.3_authornames",
    "Affiliation": "MDPI_1.6_affiliation",
    "Abstract": "MDPI_1.7_abstract",
    "Highlights": "MDPI_1.7_abstract",
    "HighlightQuestion": "MDPI_3.1_text",
    "HighlightBullet": "MDPI_3.8_bullet",
    "Keywords": "MDPI_1.8_keywords",
    "Image Caption": "MDPI_5.1_figure_caption",
    "Captioned Figure": "MDPI_5.2_figure",
    "Figure": "MDPI_5.2_figure",
    "Table Caption": "MDPI_4.1_table_caption",
    "TableNote": "MDPI_4.3_table_footer",
    "Reference": "MDPI_8.1_references",
    "BackMatter": "MDPI_6.2_back_matter",
    "Bibliography": "MDPI_8.1_references",
    "Block Text": "MDPI_3.2_text_no_indent",
    "Algorithm": "MDPI_4.2_table_body",
}


def strip_direct_formatting(style):
    el = style.element
    for tag in ("w:pPr", "w:rPr"):
        for child in el.findall(qn(tag)):
            el.remove(child)


def main():
    d = docx.Document(SRC)
    body = d.element.body
    # remove the example content but keep the final section properties
    for child in list(body):
        if child.tag != qn("w:sectPr"):
            body.remove(child)
    names = {s.name: s for s in d.styles if s.type == WD_STYLE_TYPE.PARAGRAPH}
    for alias, target in ALIASES.items():
        base = names.get(target)
        if base is None:
            raise KeyError(f"MDPI style {target!r} not found in the template")
        st = names.get(alias)
        if st is None:
            st = d.styles.add_style(alias, WD_STYLE_TYPE.PARAGRAPH)
            names[alias] = st
        if alias == "Normal":
            # Normal cannot be based on another style; copy only the font.  The
            # paragraph indent of MDPI_3.1_text (the 4.6 cm line-number gutter)
            # must not be inherited by table cells and equations.
            strip_direct_formatting(st)
            src = base.element.find(qn("w:rPr"))
            if src is not None:
                st.element.append(copy.deepcopy(src))
            continue
        strip_direct_formatting(st)
        st.base_style = base
        st.hidden = False
        st.quick_style = True
        if alias in ("Reference", "Bibliography"):
            # the MDPI reference style numbers automatically; our entries carry "[n]"
            ppr = OxmlElement("w:pPr")
            numpr = OxmlElement("w:numPr")
            numid = OxmlElement("w:numId")
            numid.set(qn("w:val"), "0")
            numpr.append(numid)
            ppr.append(numpr)
            st.element.append(ppr)
    # pandoc marks the first paragraph after a heading with "First Paragraph";
    # MDPI indents every paragraph, which the alias above already gives.
    d.save(DST)
    print("written", DST)
    d2 = docx.Document(DST)
    have = {s.name for s in d2.styles if s.type == WD_STYLE_TYPE.PARAGRAPH}
    print("aliases present:", all(a in have for a in ALIASES), "| missing:", [a for a in ALIASES if a not in have])


if __name__ == "__main__":
    main()
