"""Helpers to edit the BPASTS paper while reusing its own paragraph/table/figure formatting."""
from __future__ import annotations

import copy
import re
from pathlib import Path

import docx
from docx.oxml.ns import qn
from docx.shared import Emu
from docx.text.paragraph import Paragraph

W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"


def body_elements(doc):
    return list(doc.element.body.iterchildren())


def text_of(el) -> str:
    return "".join(t.text or "" for t in el.iter(W + "t"))


def find_index(doc, prefix: str, start: int = 0) -> int:
    for i, el in enumerate(body_elements(doc)):
        if i >= start and el.tag == W + "p" and text_of(el).strip().startswith(prefix):
            return i
    raise KeyError(prefix)


def set_paragraph_text(el, text: str, keep_run: int = 0):
    """Replace all text of a paragraph with ``text`` using the formatting of run ``keep_run``."""
    runs = [r for r in el.iter(W + "r") if r.find(W + "t") is not None]
    template = copy.deepcopy(runs[keep_run]) if runs else None
    for child in list(el):
        if child.tag in (W + "r", W + "proofErr", W + "hyperlink", W + "bookmarkStart", W + "bookmarkEnd"):
            el.remove(child)
    if template is None:
        return el
    for t in template.findall(W + "t"):
        template.remove(t)
    t = template.makeelement(W + "t", {})
    t.text = text
    t.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
    template.append(t)
    el.append(template)
    return el


def set_rich_text(el, parts: list[tuple[str, dict]]):
    """parts: [(text, {"italic": bool, "sub": bool, "sup": bool, "bold": bool})] using run 0 formatting."""
    runs = [r for r in el.iter(W + "r") if r.find(W + "t") is not None]
    base = copy.deepcopy(runs[0])
    for child in list(el):
        if child.tag in (W + "r", W + "proofErr", W + "hyperlink", W + "bookmarkStart", W + "bookmarkEnd"):
            el.remove(child)
    for t in base.findall(W + "t"):
        base.remove(t)
    for text, fmt in parts:
        r = copy.deepcopy(base)
        rpr = r.find(W + "rPr")
        if rpr is None:
            rpr = r.makeelement(W + "rPr", {})
            r.insert(0, rpr)
        if fmt.get("italic"):
            rpr.append(rpr.makeelement(W + "i", {}))
        if fmt.get("bold"):
            rpr.append(rpr.makeelement(W + "b", {}))
        if fmt.get("sub") or fmt.get("sup"):
            rpr.append(rpr.makeelement(W + "vertAlign", {W + "val": "subscript" if fmt.get("sub") else "superscript"}))
        t = r.makeelement(W + "t", {})
        t.text = text
        t.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
        r.append(t)
        el.append(r)
    return el


def parse_math(text: str) -> list[tuple[str, dict]]:
    """Tiny markup: _{x} subscript, ^{x} superscript, *x* italic."""
    parts = []
    for tok in re.split(r"(_\{[^}]*\}|\^\{[^}]*\}|\*[^*]+\*)", text):
        if not tok:
            continue
        if tok.startswith("_{"):
            parts.append((tok[2:-1], {"sub": True}))
        elif tok.startswith("^{"):
            parts.append((tok[2:-1], {"sup": True}))
        elif tok.startswith("*") and tok.endswith("*") and len(tok) > 2:
            parts.append((tok[1:-1], {"italic": True}))
        else:
            parts.append((tok, {}))
    return parts


class Templates:
    def __init__(self, doc, idx: dict[str, int]):
        els = body_elements(doc)
        self.t = {k: copy.deepcopy(els[i]) for k, i in idx.items()}

    def para(self, kind: str, text: str, rich: bool = False):
        el = copy.deepcopy(self.t[kind])
        return set_rich_text(el, parse_math(text)) if rich else set_paragraph_text(el, text)

    def caption(self, label: str, text: str):
        el = copy.deepcopy(self.t["figcap"])
        runs = [r for r in el.iter(W + "r") if r.find(W + "t") is not None]
        runs[0].find(W + "t").text = label + " "
        runs[1].find(W + "t").text = text
        for r in runs[2:]:
            el.remove(r)
        return el

    def table(self, rows: list[list[str]], widths: list[int], bold_first_col: bool = False):
        """Build a table from the template table (header row + one body row are reused)."""
        tpl = copy.deepcopy(self.t["table"])
        trs = tpl.findall(W + "tr")
        head_tr, body_tr = trs[0], trs[1]
        for tr in trs:
            tpl.remove(tr)
        grid = tpl.find(W + "tblGrid")
        for g in list(grid):
            grid.remove(g)
        for w in widths:
            grid.append(grid.makeelement(W + "gridCol", {W + "w": str(w)}))
        tpl.find(W + "tblPr").find(W + "tblW").set(W + "w", str(sum(widths)))
        for ri, row in enumerate(rows):
            src = head_tr if ri == 0 else body_tr
            tr = copy.deepcopy(src)
            tcs = tr.findall(W + "tc")
            for tc in tcs:
                tr.remove(tc)
            for ci, (val, w) in enumerate(zip(row, widths)):
                tc = copy.deepcopy(tcs[min(ci, len(tcs) - 1)] if ci == 0 else tcs[-1])
                tc.find(W + "tcPr").find(W + "tcW").set(W + "w", str(w))
                p = tc.find(W + "p")
                for extra in tc.findall(W + "p")[1:]:
                    tc.remove(extra)
                ppr = p.find(W + "pPr")
                jc = ppr.find(W + "jc") if ppr is not None else None
                if ci == 0 and jc is not None:
                    ppr.remove(jc)
                elif ci > 0 and jc is None and ppr is not None:
                    ppr.append(ppr.makeelement(W + "jc", {W + "val": "center"}))
                set_rich_text(p, parse_math(val))
                if bold_first_col and ci == 0 and ri > 0:
                    for r in p.iter(W + "r"):
                        rpr = r.find(W + "rPr")
                        rpr.append(rpr.makeelement(W + "b", {}))
                tr.append(tc)
            tpl.append(tr)
        return tpl

    def figure(self, doc, image: Path, width_cm: float):
        el = copy.deepcopy(self.t["figure"])
        rid, _ = doc.part.get_or_add_image(str(image))
        from PIL import Image
        with Image.open(image) as im:
            w, h = im.size
        cx = int(width_cm * 360000)
        cy = int(cx * h / w)
        for blip in el.iter("{http://schemas.openxmlformats.org/drawingml/2006/main}blip"):
            blip.set(qn("r:embed"), rid)
        for ext in el.iter("{http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing}extent"):
            ext.set("cx", str(cx)); ext.set("cy", str(cy))
        for ext in el.iter("{http://schemas.openxmlformats.org/drawingml/2006/main}ext"):
            if ext.get("cx"):
                ext.set("cx", str(cx)); ext.set("cy", str(cy))
        Templates._pic_id += 1
        for pr in el.iter("{http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing}docPr"):
            pr.set("id", str(Templates._pic_id)); pr.set("name", f"Picture {Templates._pic_id}")
        return el

    _pic_id = 90000


def replace_range(doc, start: int, stop: int, new_elements: list):
    """Replace body elements [start, stop) with ``new_elements``."""
    els = body_elements(doc)
    anchor = els[start]
    for el in new_elements:
        anchor.addprevious(el)
    for el in els[start:stop]:
        el.getparent().remove(el)
