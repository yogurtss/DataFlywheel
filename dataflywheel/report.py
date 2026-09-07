"""Self-contained static reports, escaped content, local thumbnails and MathML."""
from collections import Counter
import html
import json
import math
from pathlib import Path

from PIL import Image

from .io import atomic_text, digest, write_json
from .tables import canonical_html

CSS = """body{font:15px system-ui,sans-serif;margin:30px;color:#243247;background:#f5f7fa}h1,h2{color:#14263e}a{color:#195ca4}section,article{background:white;padding:20px;margin:18px 0;border-radius:10px;border:1px solid #dde3eb}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:16px}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px;max-height:360px;overflow:auto}table{border-collapse:collapse;max-width:100%;font-size:13px}td,th{border:1px solid #bcc8d6;padding:5px}img{max-width:100%;max-height:450px}svg{max-width:100%;height:auto}small{color:#64748b}.preview{overflow:auto;max-height:400px}math{font-size:20px}nav a{padding:6px}details{margin-top:12px}"""


def esc(x):
    return html.escape(str(x))


def document(title, body):
    return f'<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>{esc(title)}</title><style>{CSS}</style><body><h1>{esc(title)}</h1>{body}</body></html>'


def bars(title, counts):
    items = list(counts.items())[:30]
    height = max(60, len(items) * 27 + 10)
    maximum = max(counts.values(), default=1) or 1
    svg = f'<svg viewBox="0 0 600 {height}" role="img" aria-label="{esc(title)}">'
    for i, (label, count) in enumerate(items):
        y = i * 27 + 5
        svg += f'<text x="0" y="{y+16}" font-size="12">{esc(label)[:100]}</text><rect x="205" y="{y}" width="{count/maximum*320:.2f}" height="20" fill="#387bb8"/><text x="{210+count/maximum*320:.2f}" y="{y+16}" font-size="12">{count}</text>'
    return f'<section><h2>{esc(title)}</h2>{svg}</svg></section>'


def histogram(title, values):
    values = [v for v in values if v is not None and math.isfinite(v)]
    if not values:
        return bars(title, {})
    hi = max(values) or 1.0
    counts = Counter({f"{hi*i/10:.2g}–{hi*(i+1)/10:.2g}": 0 for i in range(10)})
    labels = list(counts)
    for v in values:
        counts[labels[min(9, max(0, int(v / hi * 10)))]] += 1
    return bars(title, counts)


def render_content(task, text):
    text = text or ""
    rendered = ""
    if task == "table" and text:
        try:
            rendered = canonical_html(text)
        except ValueError as e:
            rendered = f"<small>结构无法渲染：{esc(e)}</small>"
    elif task == "formula" and text:
        try:
            from latex2mathml.converter import convert
            from .metrics import normalize_formula
            from lxml import etree
            root = etree.fromstring(convert(normalize_formula(text)).encode())
            # Only MathML nodes/attributes are permitted; no HTML annotation or links.
            for node in root.iter():
                if etree.QName(node).namespace != "http://www.w3.org/1998/Math/MathML":
                    raise ValueError("unexpected non-MathML node")
                for key in list(node.attrib):
                    if key not in {"display", "mathvariant", "stretchy", "fence", "separator", "accent", "accentunder", "columnalign", "rowspacing", "columnspacing", "linethickness", "displaystyle", "scriptlevel", "width", "height", "depth", "lspace", "rspace", "encoding"}:
                        del node.attrib[key]
            rendered = etree.tostring(root, encoding="unicode")
        except Exception as e:
            rendered = f"<small>公式预览不可用（原文保留）：{esc(e)}</small>"
    return f'<div class="preview">{rendered}</div><details open><summary>原文</summary><pre>{esc(text)}</pre></details>'


def create_report(rows, output, config, before=None, pending=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "assets").mkdir(exist_ok=True)
    before = rows if before is None else before
    pending = pending or []
    pagesize = config["report"]["page_size"]
    if pagesize <= 0:
        raise ValueError("report.page_size must be positive")
    count = max(1, math.ceil(len(rows) / pagesize))
    links = "<nav>" + " ".join(f'<a href="samples-{p+1}.html">{p+1}</a>' for p in range(count)) + "</nav>"
    stats = {"before": len(before), "after": len(rows), "pending": len(pending),
             "exclusions": dict(Counter(r.get("reason", "unknown") for r in pending))}
    write_json(output / "summary.json", stats)
    body = f'<p>输入 {len(before)} · 当前 {len(rows)} · 排除/待处理 {len(pending)}。以下为区域挖掘指标，不是完整 OmniDocBench 分数。</p>{links}<div class="grid">'
    for label, key in [("任务", "task"), ("来源", "source"), ("域", "domain"), ("语言", "language")]:
        body += bars(label + "（筛选前）", Counter(r.get(key, "unknown") for r in before))
        body += bars(label + "（筛选后）", Counter(r.get(key, "unknown") for r in rows))
    for key, label in [("base_score", "原模型质量"), ("sft_score", "SFT 质量"), ("regression", "退化程度"), ("candidate_gap", "候选分差"), ("disagreement", "双模型分歧")]:
        body += histogram(label, [r.get("signals", {}).get(key) for r in before])
    body += histogram("GT 字符长度", [len(r.get("target") or "") for r in before])
    for key, label in [("cells", "表格单元格数"), ("merged_cells", "合并单元格数"), ("rows", "表格行数"), ("columns", "表格列数")]:
        body += histogram(label, [r.get("table_features", {}).get(key) for r in before])
    body += bars("排除原因", stats["exclusions"])
    dots = '<svg viewBox="0 0 340 340"><path d="M30 10V310H330M30 310L330 10" fill="none" stroke="#94a3b8"/>'
    for r in before:
        s = r.get("signals", {})
        if s.get("base_score") is not None and s.get("sft_score") is not None:
            dots += f'<circle cx="{30+300*s["base_score"]:.2f}" cy="{310-300*s["sft_score"]:.2f}" r="2" opacity="0.4" fill="#387bb8"><title>{esc(r["id"])}</title></circle>'
    body += f'<section><h2>原模型 x / SFT y（0–1）</h2>{dots}</svg></section></div>'
    body += '<section><h2>待处理样本</h2><p>完整清单见 pending.jsonl。</p><pre>' + esc(json.dumps([{k: r.get(k) for k in ("id", "reason")} for r in pending[:100]], ensure_ascii=False, indent=2)) + '</pre></section>'
    atomic_text(output / "index.html", document("DataFlywheel 数据报告", body))
    from .io import write_rows
    write_rows(output / "pending.jsonl", pending)
    for p in range(count):
        body = '<a href="index.html">返回分布报告</a>' + links
        for row in rows[p*pagesize:(p+1)*pagesize]:
            name = digest([row["id"], row.get("image_hash")])[:24] + ".png"
            try:
                with Image.open(row["image"]) as im:
                    im = im.convert("RGB")
                    im.thumbnail((1000, 1000))
                    im.save(output / "assets" / name)
                thumb = f'<a href="assets/{name}"><img loading="lazy" src="assets/{name}"></a>'
            except OSError as e:
                thumb = esc(e)
            body += f'<article><h2>{esc(row["id"])} · {esc(row["task"])}</h2><small>{esc(row.get("source"))} / {esc(row.get("domain"))} / {esc(row.get("mining_bucket", ""))}</small><div class="grid"><div>{thumb}</div><div><h3>GT</h3>{render_content(row["task"], row.get("target"))}</div></div><div class="grid">'
            audit = row.get("pair_audit", {})
            for c in row.get("candidates", []):
                badge = "chosen" if c["id"] == audit.get("chosen_id") else "rejected" if c["id"] == audit.get("rejected_id") else ""
                body += f'<section><h3>{esc(c["role"])} / {esc(c["kind"])} {badge}</h3><small>{esc(c.get("quality", {}))}</small>{render_content(row["task"], c.get("text"))}</section>'
            body += '</div><pre>' + esc(json.dumps(audit or row.get("signals", {}), ensure_ascii=False, indent=2)) + '</pre></article>'
        atomic_text(output / f"samples-{p+1}.html", document(f"样本 {p+1}/{count}", body))
    return stats
