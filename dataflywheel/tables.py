"""Strict, content-preserving OTSL v1 codec.

Token semantics match PaddleX's Apache-2.0 paddleocr_vl/uilts.py. Unlike its
inference repair routine we reject ambiguous grids instead of silently padding
or truncating ground truth. Original HTML is kept separately by the importer.
"""
from dataclasses import dataclass
import html
import re

from lxml import etree
from lxml import html as lh

TOKENS = re.compile(r"(<(?:fcel|ecel|lcel|ucel|xcel|nl)>)")


@dataclass(frozen=True)
class Cell:
    row: int
    col: int
    rowspan: int
    colspan: int
    text: str


def cell_text(node):
    def walk(n):
        out = n.text or ""
        for child in n:
            if child.tag == "br":
                out += "\n"
            else:
                out += walk(child)
            out += child.tail or ""
        return out
    return walk(node).strip()


def parse_html(value):
    if not isinstance(value, str) or "<table" not in value.lower():
        raise ValueError("table HTML requires <table>")
    if re.search(r"<\s*(script|style|iframe|img)\b", value, re.I):
        raise ValueError("unsupported active/non-text HTML inside table")
    root = lh.fragment_fromstring(value, create_parent="div")
    tables = root.xpath(".//table")
    if len(tables) != 1:
        raise ValueError("nested or multiple tables are not representable as one OTSL table")
    table = tables[0]
    cells, occupied = [], {}
    rows = table.xpath("./tr|./thead/tr|./tbody/tr|./tfoot/tr")
    if not rows:
        raise ValueError("table has no rows")
    for r, row in enumerate(rows):
        c = 0
        for node in row.xpath("./td|./th"):
            while (r, c) in occupied:
                c += 1
            rs, cs = int(node.get("rowspan", "1")), int(node.get("colspan", "1"))
            if not 1 <= rs <= 10000 or not 1 <= cs <= 10000 or r + rs > len(rows):
                raise ValueError("invalid rowspan/colspan")
            if len(occupied) + rs * cs > 1000000:
                raise ValueError("table grid exceeds safety limit")
            cell = Cell(r, c, rs, cs, cell_text(node))
            for rr in range(r, r + rs):
                for cc in range(c, c + cs):
                    if (rr, cc) in occupied:
                        raise ValueError("overlapping spans")
                    occupied[rr, cc] = cell
            cells.append(cell)
            c += cs
    if not occupied:
        raise ValueError("table has no cells")
    width = max(c for _, c in occupied) + 1
    if len(occupied) != len(rows) * width:
        raise ValueError("ragged table grid; refusing to invent cells")
    return cells, len(rows), width


def render_html(cells, rows, cols):
    starts = {(c.row, c.col): c for c in cells}
    parts = ["<table><tbody>"]
    for r in range(rows):
        parts.append("<tr>")
        for c in range(cols):
            cell = starts.get((r, c))
            if cell:
                attrs = (f' rowspan="{cell.rowspan}"' if cell.rowspan > 1 else "")
                attrs += f' colspan="{cell.colspan}"' if cell.colspan > 1 else ""
                text = html.escape(cell.text, quote=False).replace("\n", "<br/>")
                parts.append(f"<td{attrs}>{text}</td>")
        parts.append("</tr>")
    return "".join(parts) + "</tbody></table>"


def html_to_otsl(value):
    cells, rows, cols = parse_html(value)
    grid = {}
    for cell in cells:
        for r in range(cell.row, cell.row + cell.rowspan):
            for c in range(cell.col, cell.col + cell.colspan):
                if (r, c) == (cell.row, cell.col):
                    # PaddleX escapes cell text at HTML export, not in the OTSL
                    # stream. Encoding & here would double-escape official output.
                    text = cell.text
                    if TOKENS.search(text):
                        raise ValueError("literal OTSL delimiter in cell text is ambiguous")
                    grid[r, c] = "<fcel>" + text if text else "<ecel>"
                elif r == cell.row:
                    grid[r, c] = "<lcel>"
                elif c == cell.col:
                    grid[r, c] = "<ucel>"
                else:
                    grid[r, c] = "<xcel>"
    return "".join("".join(grid[r, c] for c in range(cols)) + "<nl>" for r in range(rows))


def parse_otsl(value):
    parts = TOKENS.split(value.strip())
    if parts[0].strip() or len(parts) < 3:
        raise ValueError("OTSL must start with a cell token")
    rows, row = [], []
    for i in range(1, len(parts), 2):
        token, text = parts[i], parts[i + 1]
        if token == "<nl>":
            if text.strip() or not row:
                raise ValueError("invalid row delimiter")
            rows.append(row)
            row = []
        else:
            if token != "<fcel>" and text.strip():
                raise ValueError("only fcel can carry text")
            text = text.strip()
            row.append((token, text))
    if row:
        rows.append(row)  # Last newline is optional; no grid repair is performed.
    if not rows or any(len(r) != len(rows[0]) for r in rows):
        raise ValueError("ragged/empty OTSL")
    owners, texts = {}, {}
    for r, row in enumerate(rows):
        for c, (token, text) in enumerate(row):
            if token in ("<fcel>", "<ecel>"):
                owner = (r, c)
                texts[owner] = text
            elif token == "<lcel>":
                owner = owners.get((r, c - 1))
                if owner is None or owner[0] != r:
                    raise ValueError("orphan lcel")
            elif token == "<ucel>":
                owner = owners.get((r - 1, c))
                if owner is None or owner[1] != c:
                    raise ValueError("orphan ucel")
            else:
                owner = owners.get((r - 1, c))
                if owner is None or owner != owners.get((r, c - 1)) or owner[0] == r or owner[1] == c:
                    raise ValueError("orphan xcel")
            owners[r, c] = owner
    groups = {}
    for pos, owner in owners.items():
        groups.setdefault(owner, []).append(pos)
    cells = []
    for (r, c), positions in groups.items():
        rs = max(p[0] for p in positions) - r + 1
        cs = max(p[1] for p in positions) - c + 1
        if len(positions) != rs * cs:
            raise ValueError("nonrectangular merged cell")
        cells.append(Cell(r, c, rs, cs, texts[r, c]))
    return cells, len(rows), len(rows[0])


def otsl_to_html(value):
    return render_html(*parse_otsl(value))


def canonical_html(value):
    return render_html(*(parse_otsl(value) if TOKENS.match(value.strip()) else parse_html(value)))


def table_features(value):
    cells, rows, cols = parse_html(canonical_html(value))
    return {"rows": rows, "columns": cols, "cells": len(cells),
            "merged_cells": sum(c.rowspan > 1 or c.colspan > 1 for c in cells)}
