# report_equations.py - small helpers that write word-native (office math) equations for the project report
"""Builds Office Math Markup Language (OMML) strings so equations in the report are real
Word equations (editable with the Word equation tools), not pictures.

Usage:  eq = S("e", "s") + R(" = ") + F(R("a"), R("b"))   -> string of <m:...> elements
        wrap(eq) gives a complete <m:oMathPara> element ready to insert in a paragraph.
"""
from __future__ import annotations

from xml.sax.saxutils import escape

M_NS = "http://schemas.openxmlformats.org/officeDocument/2006/math"
W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"


def R(text: str, style: str | None = None) -> str:
    """A run of math text. style: 'p' upright (names, units), 'b' bold, 'bi' bold italic; None = math italic."""
    pr = f'<m:rPr><m:sty m:val="{style}"/></m:rPr>' if style else ""
    return f'<m:r>{pr}<m:t xml:space="preserve">{escape(text)}</m:t></m:r>'


def T(text: str) -> str:
    """Plain running text inside an equation (spaces kept, document font)."""
    return (f'<m:r><m:rPr><m:nor/></m:rPr><w:rPr xmlns:w="{W_NS}"><w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman" w:cs="Times New Roman"/></w:rPr>'
            f'<m:t xml:space="preserve">{escape(text)}</m:t></m:r>')


def N(text: str) -> str:
    """Upright text such as a function or label name."""
    return R(text, "p")


def S(base: str, sub: str) -> str:
    return f"<m:sSub><m:sSubPr/><m:e>{base}</m:e><m:sub>{sub}</m:sub></m:sSub>"


def P(base: str, sup: str) -> str:
    return f"<m:sSup><m:sSupPr/><m:e>{base}</m:e><m:sup>{sup}</m:sup></m:sSup>"


def SP(base: str, sub: str, sup: str) -> str:
    return f"<m:sSubSup><m:sSubSupPr/><m:e>{base}</m:e><m:sub>{sub}</m:sub><m:sup>{sup}</m:sup></m:sSubSup>"


def F(num: str, den: str) -> str:
    return f"<m:f><m:fPr/><m:num>{num}</m:num><m:den>{den}</m:den></m:f>"


def D(body: str, beg: str = "(", end: str = ")") -> str:
    return f'<m:d><m:dPr><m:begChr m:val="{beg}"/><m:endChr m:val="{end}"/></m:dPr><m:e>{body}</m:e></m:d>'


def ABS(body: str) -> str:
    return D(body, "|", "|")


def SUM(lo: str, hi: str, body: str, chr_: str = "∑") -> str:
    hide_hi = ' <m:supHide m:val="1"/>' if not hi else ""
    return (f'<m:nary><m:naryPr><m:chr m:val="{chr_}"/><m:limLoc m:val="undOvr"/>{hide_hi}</m:naryPr>'
            f"<m:sub>{lo}</m:sub><m:sup>{hi}</m:sup><m:e>{body}</m:e></m:nary>")


def RAD(body: str) -> str:
    return f'<m:rad><m:radPr><m:degHide m:val="1"/></m:radPr><m:deg/><m:e>{body}</m:e></m:rad>'


def HAT(x: str) -> str:
    return f'<m:acc><m:accPr><m:chr m:val="̂"/></m:accPr><m:e>{x}</m:e></m:acc>'


def BAR(x: str) -> str:
    return f'<m:bar><m:barPr><m:pos m:val="top"/></m:barPr><m:e>{x}</m:e></m:bar>'


def FN(name: str, body: str) -> str:
    return f"<m:func><m:funcPr/><m:fName>{N(name)}</m:fName><m:e>{body}</m:e></m:func>"


def LIMLOW(base: str, lim: str) -> str:
    return f"<m:limLow><m:limLowPr/><m:e>{base}</m:e><m:lim>{lim}</m:lim></m:limLow>"


def EQARR(*lines: str) -> str:
    return "<m:eqArr><m:eqArrPr/>" + "".join(f"<m:e>{x}</m:e>" for x in lines) + "</m:eqArr>"


def IND(body: str) -> str:
    """Indicator function 1[ ... ]."""
    return R("1", "b") + D(body, "[", "]")


def wrap(body: str) -> str:
    """A complete display-math paragraph element."""
    return f'<m:oMathPara xmlns:m="{M_NS}"><m:oMath>{body}</m:oMath></m:oMathPara>'
