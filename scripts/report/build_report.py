# build_report.py - assembles the bite497j project i report (word file) in the university template format
"""Builds docs/report/BITE497J_Project_I_Report.docx.

Format rules taken from the template (Project-1 Report final.docx):
  A4 page, margins left 1.5 in / right, top, bottom 1 in; Times New Roman 12, justified, 1.5 line spacing;
  chapter heading 14 pt bold capitals; sub-heading 12 pt without capitals; first paragraph of a chapter
  has no first-line indent, later paragraphs have a 1-tab indent; tick-mark bullets, numbers only for
  types or classifications; figure caption below ("Fig. 2.4 Title Case", 10 pt), table caption above
  ("Table 6.1 Title Case", 10 pt); equations as Word equations numbered (chapter.number) on the right;
  roman page numbers from the cover to the symbols page (cover number hidden), arabic from chapter 1 to
  the references, no numbers in the appendix; APA references.

Run:  python scripts/report/build_report.py      then   scripts/report/finalize_report.ps1 (updates the contents fields)
"""
from __future__ import annotations

import copy
import re
import sys
from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.style import WD_STYLE_TYPE
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING, WD_TAB_ALIGNMENT, WD_TAB_LEADER
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn
from docx.shared import Cm, Emu, Inches, Pt, RGBColor
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "src"))

OUT = ROOT / "docs" / "report" / "BITE497J_Project_I_Report_raw.docx"
FONT = "Times New Roman"
TEXT_W_CM = 21.0 - 3.81 - 2.54            # usable width of the page in cm
SMALL = {"a", "an", "and", "as", "at", "but", "by", "for", "in", "of", "on", "or", "the", "to", "vs", "with", "per", "from", "into", "over", "via"}


# ---------------------------------------------------------------------------------------------- small helpers
def title_case(s: str) -> str:
    words = s.split(" ")
    out = []
    for i, w in enumerate(words):
        core = re.sub(r"^[\(\"“‘]+|[\)\"”’,;:.]+$", "", w)
        if not core:
            out.append(w); continue
        if any(c.isupper() for c in core[1:]) or any(c.isdigit() for c in core) or core.isupper():
            out.append(w); continue
        if i not in (0, len(words) - 1) and core.lower() in SMALL:
            out.append(w.lower() if w.islower() or w[0].isupper() else w); continue
        if "-" in core and not any(c.isupper() for c in core):
            parts = core.split("-")
            new = "-".join(p if p.lower() in SMALL else p.capitalize() for p in parts)
            out.append(w.replace(core, new)); continue
        out.append(w.replace(core, core[0].upper() + core[1:]))
    return " ".join(out)


def set_cell_shade(cell, fill: str):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd"); shd.set(qn("w:val"), "clear"); shd.set(qn("w:color"), "auto"); shd.set(qn("w:fill"), fill)
    tcPr.append(shd)


def set_cell_margins(table, top=40, bottom=40, left=70, right=70):
    tblPr = table._tbl.tblPr
    m = OxmlElement("w:tblCellMar")
    for k, v in (("top", top), ("left", left), ("bottom", bottom), ("right", right)):
        e = OxmlElement(f"w:{k}"); e.set(qn("w:w"), str(v)); e.set(qn("w:type"), "dxa"); m.append(e)
    tblPr.append(m)


def no_borders(table):
    tblPr = table._tbl.tblPr
    b = OxmlElement("w:tblBorders")
    for k in ("top", "left", "bottom", "right", "insideH", "insideV"):
        e = OxmlElement(f"w:{k}"); e.set(qn("w:val"), "nil"); b.append(e)
    tblPr.append(b)


def fixed_layout(table, widths_cm):
    tblPr = table._tbl.tblPr
    lay = OxmlElement("w:tblLayout"); lay.set(qn("w:type"), "fixed"); tblPr.append(lay)
    grid = table._tbl.tblGrid
    for gc, w in zip(grid.findall(qn("w:gridCol")), widths_cm):
        gc.set(qn("w:w"), str(int(w / 2.54 * 1440)))
    for row in table.rows:
        for c, w in zip(row.cells, widths_cm):
            c.width = Cm(w)


class Report:
    def __init__(self):
        self.d = Document()
        self.fig_no = {}
        self.tab_no = {}
        self.eq_no = {}
        self.chapter = 0
        self.first_para = True
        self._num_id = 100
        self._setup_styles()
        self._setup_numbering()

    # ------------------------------------------------------------------ styles
    def _style(self, name, base="Normal", size=12, bold=False, italic=False, align=None, before=0, after=6, line=1.5, keep_next=False, caps=False, outline=None, color=None):
        st = self.d.styles[name] if name in [s.name for s in self.d.styles] else self.d.styles.add_style(name, WD_STYLE_TYPE.PARAGRAPH)
        if base and name != base:
            st.base_style = self.d.styles[base]
        f = st.font
        f.name = FONT; f.size = Pt(size); f.bold = bold; f.italic = italic; f.all_caps = caps
        f.color.rgb = RGBColor(0, 0, 0) if color is None else color
        rpr = st.element.get_or_add_rPr()
        rf = rpr.find(qn("w:rFonts"))
        if rf is None:
            rf = OxmlElement("w:rFonts"); rpr.append(rf)
        for a in ("w:ascii", "w:hAnsi", "w:cs", "w:eastAsia"):
            rf.set(qn(a), FONT)
        for a in ("w:asciiTheme", "w:hAnsiTheme", "w:eastAsiaTheme", "w:cstheme"):
            rf.attrib.pop(qn(a), None)
        pf = st.paragraph_format
        pf.space_before = Pt(before); pf.space_after = Pt(after)
        pf.line_spacing = line
        pf.keep_with_next = keep_next
        if align is not None:
            pf.alignment = align
        if outline is not None:
            pPr = st.element.get_or_add_pPr()
            ol = pPr.find(qn("w:outlineLvl"))
            if ol is None:
                ol = OxmlElement("w:outlineLvl"); pPr.append(ol)
            ol.set(qn("w:val"), str(outline))
        return st

    def _setup_styles(self):
        d = self.d
        n = d.styles["Normal"]
        n.font.name = FONT; n.font.size = Pt(12)
        rpr = n.element.get_or_add_rPr(); rf = rpr.find(qn("w:rFonts"))
        if rf is None:
            rf = OxmlElement("w:rFonts"); rpr.append(rf)
        for a in ("w:ascii", "w:hAnsi", "w:cs", "w:eastAsia"):
            rf.set(qn(a), FONT)
        n.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        n.paragraph_format.line_spacing = 1.5
        n.paragraph_format.space_after = Pt(6)
        n.paragraph_format.widow_control = True
        self._style("Heading 1", size=14, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER, before=0, after=14, line=1.15, keep_next=True, caps=True, outline=0)
        self._style("Heading 2", size=12, bold=True, align=WD_ALIGN_PARAGRAPH.LEFT, before=12, after=6, line=1.5, keep_next=True, outline=1)
        self._style("Heading 3", size=12, bold=True, italic=True, align=WD_ALIGN_PARAGRAPH.LEFT, before=8, after=4, line=1.5, keep_next=True, outline=2)
        self._style("Front Heading", base="Normal", size=14, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER, before=0, after=14, line=1.15, keep_next=True, caps=True, outline=0)
        self._style("Front Heading Plain", base="Normal", size=14, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER, before=0, after=14, line=1.15, keep_next=True, caps=True)
        self._style("Figure Caption", size=10, bold=False, align=WD_ALIGN_PARAGRAPH.CENTER, before=4, after=12, line=1.0)
        self._style("Table Caption", size=10, bold=False, align=WD_ALIGN_PARAGRAPH.CENTER, before=10, after=4, line=1.0, keep_next=True)
        self._style("Table Text", size=10, align=WD_ALIGN_PARAGRAPH.LEFT, before=0, after=0, line=1.0)
        self._style("Code Block", size=8.5, align=WD_ALIGN_PARAGRAPH.LEFT, before=0, after=0, line=1.0)
        d.styles["Code Block"].font.name = "Courier New"
        rf = d.styles["Code Block"].element.get_or_add_rPr().find(qn("w:rFonts"))
        for a in ("w:ascii", "w:hAnsi", "w:cs", "w:eastAsia"):
            rf.set(qn(a), "Courier New")
        self._style("Reference", size=12, align=WD_ALIGN_PARAGRAPH.LEFT, before=0, after=6, line=1.15)
        d.styles["Reference"].paragraph_format.left_indent = Cm(1.27)
        d.styles["Reference"].paragraph_format.first_line_indent = Cm(-1.27)
        self._style("List Line", size=12, align=WD_ALIGN_PARAGRAPH.LEFT, before=0, after=2, line=1.15)
        # contents styles (names follow word's built-in names so field updates use them)
        for lvl, (b, ind) in {1: (True, 0), 2: (False, 0.6), 3: (False, 0), 4: (False, 0)}.items():
            st = self._style(f"toc {lvl}", size=12, bold=b, align=WD_ALIGN_PARAGRAPH.LEFT, before=0, after=3, line=1.15)
            st.paragraph_format.left_indent = Cm(ind)
            st.paragraph_format.tab_stops.add_tab_stop(Cm(TEXT_W_CM), WD_TAB_ALIGNMENT.RIGHT, WD_TAB_LEADER.DOTS)
        for lvl in (3, 4):
            d.styles[f"toc {lvl}"].paragraph_format.left_indent = Cm(1.8)
            d.styles[f"toc {lvl}"].paragraph_format.first_line_indent = Cm(-1.8)

    def _setup_numbering(self):
        numbering = self.d.part.numbering_part.element
        self._abs_bullet = 90; self._abs_decimal = 91
        xml_bullet = (
            f'<w:abstractNum {nsdecls("w")} w:abstractNumId="{self._abs_bullet}"><w:multiLevelType w:val="hybridMultilevel"/>'
            '<w:lvl w:ilvl="0"><w:start w:val="1"/><w:numFmt w:val="bullet"/><w:lvlText w:val="✔"/><w:lvlJc w:val="left"/>'
            '<w:pPr><w:ind w:left="720" w:hanging="360"/></w:pPr><w:rPr><w:rFonts w:ascii="Segoe UI Symbol" w:hAnsi="Segoe UI Symbol" w:cs="Segoe UI Symbol"/></w:rPr></w:lvl>'
            '<w:lvl w:ilvl="1"><w:start w:val="1"/><w:numFmt w:val="bullet"/><w:lvlText w:val="–"/><w:lvlJc w:val="left"/>'
            '<w:pPr><w:ind w:left="1260" w:hanging="360"/></w:pPr></w:lvl></w:abstractNum>')
        xml_dec = (
            f'<w:abstractNum {nsdecls("w")} w:abstractNumId="{self._abs_decimal}"><w:multiLevelType w:val="hybridMultilevel"/>'
            '<w:lvl w:ilvl="0"><w:start w:val="1"/><w:numFmt w:val="decimal"/><w:lvlText w:val="%1."/><w:lvlJc w:val="left"/>'
            '<w:pPr><w:ind w:left="720" w:hanging="360"/></w:pPr></w:lvl></w:abstractNum>')
        first_num = numbering.find(qn("w:num"))
        for x in (xml_bullet, xml_dec):
            el = parse_xml(x)
            if first_num is not None:
                first_num.addprevious(el)
            else:
                numbering.append(el)
        self._bullet_num = self._new_num(self._abs_bullet)

    def _new_num(self, abs_id, restart=True):
        numbering = self.d.part.numbering_part.element
        self._num_id += 1
        num = parse_xml(f'<w:num {nsdecls("w")} w:numId="{self._num_id}"><w:abstractNumId w:val="{abs_id}"/>'
                        + ('<w:lvlOverride w:ilvl="0"><w:startOverride w:val="1"/></w:lvlOverride>' if restart else "") + "</w:num>")
        numbering.append(num)
        return self._num_id

    # ------------------------------------------------------------------ text with light markup
    TAG = re.compile(r"(</?b>|</?i>|</?sub>|</?sup>|</?u>)")

    def runs(self, p, text, size=None, bold=None, italic=None, font=None):
        text = self.resolve(text)
        state = {"b": False, "i": False, "sub": False, "sup": False, "u": False}
        for tok in self.TAG.split(text):
            if not tok:
                continue
            m = re.fullmatch(r"<(/?)(b|i|sub|sup|u)>", tok)
            if m:
                state[m.group(2)] = not m.group(1)
                continue
            r = p.add_run(tok)
            if state["b"] or bold:
                r.bold = True
            if state["i"] or italic:
                r.italic = True
            if state["sub"]:
                r.font.subscript = True
            if state["sup"]:
                r.font.superscript = True
            if state["u"]:
                r.underline = True
            if size:
                r.font.size = Pt(size)
            if font:
                r.font.name = font
                rpr = r._r.get_or_add_rPr(); rf = rpr.find(qn("w:rFonts"))
                if rf is None:
                    rf = OxmlElement("w:rFonts"); rpr.insert(0, rf)
                for a in ("w:ascii", "w:hAnsi", "w:cs"):
                    rf.set(qn(a), font)
        return p

    def resolve(self, text):
        text = text.replace("&amp;", "&")

        def sub(m):
            kind, key = m.group(1), m.group(2)
            try:
                if kind == "F":
                    return f"Fig. {self.fig_no[key]}"
                if kind == "T":
                    return f"Table {self.tab_no[key]}"
                if kind == "E":
                    return f"Eqn. {self.eq_no[key]}"
            except KeyError:
                return f"[missing {kind}:{key}]"
            return m.group(0)
        return re.sub(r"\{([FTE]):([A-Za-z0-9_]+)\}", sub, text)

    # ------------------------------------------------------------------ block writers
    def para(self, text, style="Normal", indent=True, align=None, keep_next=False):
        p = self.d.add_paragraph(style=style)
        if indent and not self.first_para:
            p.paragraph_format.first_line_indent = Cm(1.27)
        self.first_para = False
        self.runs(p, text)
        if align is not None:
            p.paragraph_format.alignment = align
        if keep_next:
            p.paragraph_format.keep_with_next = True
        return p

    def bullets(self, items, numbered=False):
        num = self._new_num(self._abs_decimal) if numbered else self._bullet_num
        for it in items:
            lvl = 0
            if isinstance(it, tuple):
                lvl, it = it
            p = self.d.add_paragraph()
            p.paragraph_format.space_after = Pt(3)
            pPr = p._p.get_or_add_pPr()
            numPr = parse_xml(f'<w:numPr {nsdecls("w")}><w:ilvl w:val="{lvl}"/><w:numId w:val="{num}"/></w:numPr>')
            pPr.append(numPr)
            self.runs(p, it)
        self.first_para = False

    def heading(self, text, level):
        style = {1: "Heading 1", 2: "Heading 2", 3: "Heading 3"}[level]
        p = self.d.add_paragraph(style=style)
        if level == 1:
            p.paragraph_format.page_break_before = True
            label, title = text
            self.runs(p, label)
            p.add_run().add_break()
            self.runs(p, title)
            self.first_para = True
        else:
            self.runs(p, text)
        return p

    def figure(self, key, path, caption, width_cm=13.5, max_h_cm=20.5):
        p = self.d.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.keep_with_next = True
        p.paragraph_format.space_before = Pt(6); p.paragraph_format.space_after = Pt(0); p.paragraph_format.line_spacing = 1.0
        w, h = Image.open(path).size
        wc = min(width_cm, TEXT_W_CM)
        if wc * h / w > max_h_cm:
            wc = max_h_cm * w / h
        p.add_run().add_picture(str(path), width=Cm(wc))
        c = self.d.add_paragraph(style="Figure Caption")
        c.add_run(f"Fig. {self.fig_no[key]} {title_case(caption)}")
        self.first_para = False

    def table(self, key, caption, header, rows, widths_cm=None, font=9.5, align_cols=None, header_fill="D9D9D9", bold_first_col=False):
        cap = self.d.add_paragraph(style="Table Caption")
        cap.add_run(f"Table {self.tab_no[key]} {title_case(caption)}")
        ncol = len(header)
        if widths_cm is None:
            widths_cm = [TEXT_W_CM / ncol] * ncol
        scale = TEXT_W_CM / sum(widths_cm)
        widths_cm = [w * scale for w in widths_cm]
        t = self.d.add_table(rows=1, cols=ncol)
        t.style = "Table Grid"; t.alignment = WD_TABLE_ALIGNMENT.CENTER
        set_cell_margins(t)
        for i, h in enumerate(header):
            c = t.rows[0].cells[i]
            c.text = ""
            pp = c.paragraphs[0]; pp.style = self.d.styles["Table Text"]; pp.alignment = WD_ALIGN_PARAGRAPH.CENTER
            self.runs(pp, h, size=font, bold=True)
            set_cell_shade(c, header_fill)
        trPr = t.rows[0]._tr.get_or_add_trPr()
        th = OxmlElement("w:tblHeader"); th.set(qn("w:val"), "true"); trPr.append(th)
        for r in rows:
            cells = t.add_row().cells
            for i, v in enumerate(r):
                cells[i].text = ""
                pp = cells[i].paragraphs[0]; pp.style = self.d.styles["Table Text"]
                if align_cols and align_cols[i] == "c":
                    pp.alignment = WD_ALIGN_PARAGRAPH.CENTER
                self.runs(pp, str(v), size=font, bold=(bold_first_col and i == 0))
        for row in t.rows:
            trPr = row._tr.get_or_add_trPr()
            cs = OxmlElement("w:cantSplit"); cs.set(qn("w:val"), "true"); trPr.append(cs)
        fixed_layout(t, widths_cm)
        sp = self.d.add_paragraph(); sp.paragraph_format.space_after = Pt(6); sp.paragraph_format.line_spacing = 1.0
        sp.paragraph_format.space_before = Pt(0)
        sp.add_run("").font.size = Pt(4)
        self.first_para = False

    def equation(self, key, omml):
        t = self.d.add_table(rows=1, cols=3)
        no_borders(t); set_cell_margins(t, 20, 20, 30, 30)
        fixed_layout(t, [1.2, TEXT_W_CM - 2.4, 1.2])
        mid = t.rows[0].cells[1].paragraphs[0]
        mid.alignment = WD_ALIGN_PARAGRAPH.CENTER; mid.paragraph_format.line_spacing = 1.0; mid.paragraph_format.space_after = Pt(0)
        mid._p.append(parse_xml(omml))
        num = t.rows[0].cells[2].paragraphs[0]
        num.alignment = WD_ALIGN_PARAGRAPH.RIGHT; num.paragraph_format.line_spacing = 1.0; num.paragraph_format.space_after = Pt(0)
        r = num.add_run(f"({self.eq_no[key]})"); r.font.size = Pt(12)
        for c in t.rows[0].cells:
            c.vertical_alignment = 1
        sp = self.d.add_paragraph(); sp.paragraph_format.space_after = Pt(0); sp.paragraph_format.line_spacing = 0.6
        self.first_para = False

    def code(self, text):
        lines = text.rstrip("\n").split("\n")
        t = self.d.add_table(rows=1, cols=1)
        t.style = "Table Grid"; set_cell_margins(t, 60, 60, 100, 100)
        fixed_layout(t, [TEXT_W_CM])
        c = t.rows[0].cells[0]; set_cell_shade(c, "F2F2F2")
        c.text = ""
        first = True
        for ln in lines:
            pp = c.paragraphs[0] if first else c.add_paragraph()
            first = False
            pp.style = self.d.styles["Code Block"]
            r = pp.add_run(ln if ln else " ")
        sp = self.d.add_paragraph(); sp.paragraph_format.space_after = Pt(6); sp.paragraph_format.line_spacing = 1.0
        self.first_para = False

    # ------------------------------------------------------------------ numbering pre-pass
    def number_blocks(self, chapters):
        for ch_no, ch in chapters:
            f = t = e = 0
            for b in ch["blocks"]:
                if b[0] == "fig":
                    f += 1; self.fig_no[b[1]] = f"{ch_no}.{f}"
                elif b[0] == "tbl":
                    t += 1; self.tab_no[b[1]] = f"{ch_no}.{t}"
                elif b[0] == "eq":
                    e += 1; self.eq_no[b[1]] = f"{ch_no}.{e}"

    def render_blocks(self, blocks, fig_paths):
        for b in blocks:
            k = b[0]
            if k == "h2":
                self.heading(b[1], 2)
            elif k == "h3":
                self.heading(b[1], 3)
            elif k == "p":
                self.para(b[1])
            elif k == "bul":
                self.bullets(b[1])
            elif k == "num":
                self.bullets(b[1], numbered=True)
            elif k == "fig":
                width = b[4] if len(b) > 4 else 13.5
                maxh = b[5] if len(b) > 5 else 20.5
                self.figure(b[1], fig_paths[b[2]], b[3], width, maxh)
            elif k == "tbl":
                opts = b[6] if len(b) > 6 else {}
                self.table(b[1], b[2], b[3], b[4], b[5], **opts)
            elif k == "eq":
                self.equation(b[1], b[2])
            elif k == "code":
                self.code(b[1])
            elif k == "codefile":
                lines = (ROOT / b[1]).read_text(encoding="utf-8").splitlines()
                self.code(chr(10).join(lines[: b[2]]))
            elif k == "codeslice":
                lines = (ROOT / b[1]).read_text(encoding="utf-8").splitlines()
                self.code(chr(10).join(lines[b[2] - 1: b[3]]))
            elif k == "pb":
                self.d.add_page_break()
            else:
                raise ValueError(k)

    # ------------------------------------------------------------------ fields and sections
    def field_paragraph(self, instr, placeholder="Right-click and choose Update Field."):
        p = self.d.add_paragraph(style="Normal")
        p.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.LEFT
        for t, extra in (("begin", ' w:dirty="true"'), ("instr", ""), ("separate", ""), ("text", ""), ("end", "")):
            r = OxmlElement("w:r")
            if t == "instr":
                it = OxmlElement("w:instrText"); it.set(qn("xml:space"), "preserve"); it.text = f" {instr} "; r.append(it)
            elif t == "text":
                tt = OxmlElement("w:t"); tt.text = placeholder; r.append(tt)
            else:
                fc = OxmlElement("w:fldChar"); fc.set(qn("w:fldCharType"), t)
                if t == "begin":
                    fc.set(qn("w:dirty"), "true")
                r.append(fc)
            p._p.append(r)
        return p

    def footer_page_number(self, section, visible=True):
        section.footer.is_linked_to_previous = False
        f = section.footer
        for p in list(f.paragraphs):
            for r in list(p.runs):
                r._r.getparent().remove(r._r)
        p = f.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        if visible:
            r = OxmlElement("w:r")
            for t in ("begin", "instr", "end"):
                if t == "instr":
                    it = OxmlElement("w:instrText"); it.set(qn("xml:space"), "preserve"); it.text = " PAGE "; rr = OxmlElement("w:r"); rr.append(it)
                else:
                    fc = OxmlElement("w:fldChar"); fc.set(qn("w:fldCharType"), t); rr = OxmlElement("w:r"); rr.append(fc)
                p._p.append(rr)
            for rr in p._p.findall(qn("w:r")):
                rpr = OxmlElement("w:rPr"); sz = OxmlElement("w:sz"); sz.set(qn("w:val"), "22"); rpr.append(sz)
                rf = OxmlElement("w:rFonts"); rf.set(qn("w:ascii"), FONT); rf.set(qn("w:hAnsi"), FONT); rpr.insert(0, rf)
                rr.insert(0, rpr)

    def set_pgnum(self, section, fmt, start=None):
        sectPr = section._sectPr
        for e in sectPr.findall(qn("w:pgNumType")):
            sectPr.remove(e)
        pg = OxmlElement("w:pgNumType"); pg.set(qn("w:fmt"), fmt)
        if start is not None:
            pg.set(qn("w:start"), str(start))
        sectPr.append(pg)

    def page_setup(self, section):
        section.page_width = Cm(21.0); section.page_height = Cm(29.7)
        section.left_margin = Inches(1.5); section.right_margin = Inches(1)
        section.top_margin = Inches(1); section.bottom_margin = Inches(1)
        section.header_distance = Cm(1.25); section.footer_distance = Cm(1.0)
