"""GT-driven table synthesis: one cell grid produces HTML, pixels and OTSL.

This renderer is deterministic and does not call an LLM; synthesis_agent supplies
vision-generated templates. Templates are declarative data, never executable scripts.
"""
from dataclasses import asdict
import base64
import html
import io
import json
import math
from pathlib import Path
import random
import re
import shutil

from PIL import Image, ImageFilter, ImageOps, ImageDraw

from .io import atomic_text, digest, file_hash, read_rows, write_json, write_rows
from .tables import Cell, canonical_html, html_to_otsl, otsl_to_html, parse_html, render_html, table_features


def builtin_templates():
    """Explicitly synthetic examples, not mined from user data or a benchmark."""
    finance = '<table><tr><td rowspan="2">业务类别 / Segment</td><td colspan="2">本期 / Current period</td><td colspan="2">上期 / Prior period</td></tr><tr><td>收入（万元）</td><td>占比 / Share</td><td>收入（万元）</td><td>占比 / Share</td></tr>'
    for name, a, b in [("企业软件 / Enterprise", 18234.57, 15236.18), ("云服务 / Cloud", 12345.68, 10325.61), ("技术咨询 / Consulting", 5738.92, 4986.12), ("设备维护 / Maintenance", 2168.35, 1958.71)]:
        finance += f'<tr><td>{name}</td><td>{a:,.2f}</td><td>28.35%</td><td>{b:,.2f}</td><td>25.17%</td></tr>'
    finance += '</table>'
    dense = '<table><tr>' + ''.join(f'<td>{x}</td>' for x in ['Item', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Change']) + '</tr>'
    for i in range(16):
        dense += '<tr><td>SKU-' + str(7100+i) + '</td>' + ''.join(f'<td>{(i+1)*173.15+j*91.7:,.2f}</td>' for j in range(6)) + f'<td>{i-7:+.2f}%</td></tr>'
    dense += '</table>'
    wrap = '<table><tr><td>编号</td><td>检测项目 / Test item</td><td>技术要求 / Specification</td><td>结果 / Result</td></tr>'
    for i, (label, desc) in enumerate([
        ('环境适应性测试 Environmental adaptability', '温度范围 −20 °C 至 +60 °C；持续运行 48 h 后，设备应保持正常工作。'),
        ('数据接口兼容性 Interface compatibility', '支持 UTF-8 编码与中英文混合字段；允许保留小数点、括号及百分号。'),
        ('长期运行稳定性 Long-term stability', '连续运行期间不应出现非预期重启；日志保留时间不少于 30 天。'),
        ('异常恢复与告警 Recovery and notification', '网络中断后自动重连；恢复后继续传输尚未确认的数据记录。')], 1):
        wrap += f'<tr><td>{i}</td><td>{label}</td><td>{desc}</td><td>符合 / Pass</td></tr>'
    wrap += '</table>'
    spans = '<table><tr><td rowspan="2" colspan="2">试验分组 / Experimental groups</td><td colspan="3">Response (mean ± SD)</td></tr><tr><td>Day 1</td><td>Day 7</td><td>Day 14</td></tr>'
    for group in ('Control', 'Treatment A', 'Treatment B'):
        for j in range(3):
            spans += '<tr>' + (f'<td rowspan="3">{group}</td>' if j == 0 else '') + f'<td>Replicate {j+1}</td>' + ''.join(f'<td>{12+j+k:.2f} ± 0.{k+2}</td>' for k in range(3)) + '</tr>'
    spans += '</table>'
    templates = []
    for name, title, value, width, font, header in [
        ('merged_header', '业务收入明细 / Segment revenue', finance, 1000, 16, 2),
        ('dense_numeric', 'Inventory movement — monthly detail', dense, 1040, 13, 1),
        ('narrow_wrap', '产品验收记录 / Product acceptance record', wrap, 700, 15, 1),
        ('joint_spans', '重复测量结果 / Repeated measurements', spans, 880, 15, 2),
    ]:
        cells, rows, cols = parse_html(value)
        templates.append({'version': 1, 'template_family_id': name, 'seed_sample_id': 'builtin:' + name,
                          'document_id': 'synthetic-builtin:' + name, 'source': 'synthetic_demo',
                          'domain': 'private', 'split': 'train', 'language': 'en' if name == 'dense_numeric' else 'zh-en',
                          'title': title, 'seed_html': value, 'rows': rows, 'columns': cols,
                          'cells': [asdict(c) for c in cells], 'header_rows': header,
                          'style': {'width': width, 'font_size': font}, 'seed_origin': 'builtin_demo'})
    return templates


def make_templates(input_path, output):
    rejected, templates = [], []
    if input_path is None:
        templates = builtin_templates()
    else:
        from .data import isolate
        candidates, blocked = isolate(read_rows(input_path))
        rejected += [{'id': r.get('id'), 'reason': r.get('reason')} for r in blocked]
        for row in candidates:
            if row.get('task') != 'table':
                rejected.append({'id': row.get('id'), 'reason': 'v1_table_only'})
                continue
            if row.get('split', 'train') != 'train' or not row.get('gt_trusted', True):
                rejected.append({'id': row.get('id'), 'reason': 'held_out_or_untrusted_seed'})
                continue
            try:
                value = next((row.get(k) for k in ('score_gt', 'html', 'otsl', 'gt', 'target') if row.get(k)), None)
                if not isinstance(value, str):
                    raise ValueError('missing table GT')
                canon = canonical_html(value)
                # Verify both provided labels when available before accepting a seed.
                for key in ('html', 'otsl'):
                    if row.get(key) and canonical_html(row[key]) != canon:
                        raise ValueError('conflicting seed annotations')
                cells, rows, cols = parse_html(canon)
                sid = str(row.get('id') or digest(canon)[:16])
                templates.append({'version': 1, 'template_family_id': 'seed-' + digest([sid, canon])[:16],
                                  'seed_sample_id': sid, 'document_id': str(row.get('document_id', sid)),
                                  'source': row.get('source', 'user'), 'domain': row.get('domain', 'private'),
                                  'split': row.get('split', 'train'), 'language': row.get('language', 'unknown'),
                                  'title': '合成表格 / Synthetic table', 'seed_html': canon,
                                  'rows': rows, 'columns': cols, 'cells': [asdict(c) for c in cells],
                                  'header_rows': int(row.get('header_rows', 1)),
                                  'style': {'width': min(1400, max(700, cols * 140)), 'font_size': 15},
                                  'seed_origin': 'gt_structure', 'parent_document_id': row.get('parent_document_id'),
                                  'parent_image_hash': row.get('image_hash'),
                                  'seed_image': row.get('image')})
            except (ValueError, TypeError, KeyError) as e:
                rejected.append({'id': row.get('id'), 'reason': str(e)})
    write_rows(output, templates)
    write_rows(Path(output).with_suffix('.rejected.jsonl'), rejected)
    return {'templates': len(templates), 'rejected': len(rejected)}


def validate_template(t):
    if t.get('version') != 1 or t.get('split', 'train') != 'train':
        raise ValueError('only v1 training templates may be synthesized')
    cells = [Cell(**c) for c in t['cells']]
    if not 1 <= t['rows'] <= 300 or not 1 <= t['columns'] <= 40:
        raise ValueError('template grid outside supported bounds')
    occupied = set()
    for c in cells:
        if not isinstance(c.text, str) or not 0 <= c.row < t['rows'] or not 0 <= c.col < t['columns'] or c.rowspan < 1 or c.colspan < 1:
            raise ValueError('invalid cell')
        for r in range(c.row, c.row + c.rowspan):
            for col in range(c.col, c.col + c.colspan):
                if r >= t['rows'] or col >= t['columns'] or (r, col) in occupied:
                    raise ValueError('overlapping/out-of-bounds span')
                occupied.add((r, col))
    if len(occupied) != t['rows'] * t['columns']:
        raise ValueError('incomplete grid')
    if not 0 <= t['header_rows'] <= t['rows']:
        raise ValueError('invalid header_rows')
    if not 400 <= t['style']['width'] <= 1800 or not 10 <= t['style']['font_size'] <= 30:
        raise ValueError('style outside supported bounds')
    h = render_html(cells, t['rows'], t['columns'])
    if canonical_html(t['seed_html']) != h:
        raise ValueError('template cells disagree with seed_html')
    html_to_otsl(h)
    return cells


def variant(t, index, seed):
    cells = validate_template(t)
    rng = random.Random(digest([t['template_family_id'], index, seed]))
    changed = []
    for cell in cells:
        text = cell.text
        # Preserve labels/identifiers and merged header topology. Numeric body
        # values vary, never infer accounting/experimental semantics from them.
        if index and cell.row >= t['header_rows'] and cell.col > 0:
            match = re.fullmatch(r'([+-]?)(\d[\d,]*)(\.\d+)?(%)?', text)
            if match:
                value = float(text.rstrip('%').replace(',', '')) * rng.uniform(.75, 1.25)
                decimals = len(match[3]) - 1 if match[3] else 0
                text = format(value, f'{"," if "," in text else ""}.{decimals}f') + ('%' if match[4] else '')
        changed.append(Cell(cell.row, cell.col, cell.rowspan, cell.colspan, text))
    style = dict(t['style'])
    style.update(font_size=style['font_size'] - (1 if index % 2 else 0),
                 padding=7 if index % 2 else 10, rule='light' if index % 2 else 'full',
                 stripe=bool(index % 2 == 0), ink='#222222', paper='#ffffff')
    return changed, style


def page_html(table, title, style, header_rows, font_data):
    # No user CSS/scripts are evaluated. Dynamic values are validated or escaped.
    head_selectors = ','.join(f'tr:nth-child({i+1}) td' for i in range(header_rows))
    border = '#bdc3c7' if style['rule'] == 'light' else '#606974'
    bg = 'tr:nth-child(even) td{background:#f4f5f6}' if style['stripe'] else ''
    header_css = f'{head_selectors}' + '{background:#e8edf2;font-weight:600;text-align:center}' if head_selectors else ''
    return f'''<!doctype html><html><meta charset="utf-8"><style>
@font-face{{font-family:DocumentFont;src:url(data:font/otf;base64,{font_data})}}
*{{box-sizing:border-box}}body{{margin:0;background:white;color:{style['ink']};font-family:DocumentFont,sans-serif}}
main{{padding:36px;width:{style['width']+72}px}}h1{{font-size:22px;margin:0 0 9px;font-weight:600}}
.meta{{font-size:11px;color:#626872;margin-bottom:22px}}.note{{font-size:11px;margin-top:16px;color:#626872}}
table{{border-collapse:collapse;width:{style['width']}px;table-layout:fixed;font-size:{style['font_size']}px;line-height:1.55}}
td{{border:1px solid {border};padding:{style['padding']}px;white-space:pre-wrap;overflow-wrap:anywhere;vertical-align:middle}}
{bg}{header_css}</style><main><h1>{html.escape(title)}</h1>
<div class="meta">SYNTHETIC DOCUMENT · generated data · not a real business record</div>
{table}<div class="note">Generated from a single cell grid. Figure and annotation share the same source.</div></main></html>'''


def font_coverage(path, text):
    from fontTools.ttLib import TTFont
    with TTFont(path) as font:
        cmap = font.getBestCmap()
        missing = sorted({c for c in text if not c.isspace() and ord(c) not in cmap})
    if missing:
        raise ValueError('font lacks required glyphs: ' + ''.join(missing[:30]))


def check_records(path, output):
    rows, valid, errors, seen = read_rows(path), [], [], set()
    base = Path(path).resolve().parent
    for row in rows:
        try:
            image = (base / row['image']).resolve()
            with Image.open(image) as im:
                im.verify()
            if file_hash(image) != row['image_hash']:
                raise ValueError('image hash mismatch')
            h = canonical_html(row['html'])
            if h != otsl_to_html(row['otsl']):
                raise ValueError('HTML/OTSL mismatch')
            if row['split'] != 'train' or row.get('qa', {}).get('status') != 'passed':
                raise ValueError('held-out or failed render QA')
            if row['image_hash'] in seen:
                raise ValueError('duplicate image')
            seen.add(row['image_hash'])
            valid.append(row)
        except (ValueError, OSError, KeyError) as e:
            errors.append({'id': row.get('id'), 'reason': str(e)})
    # Emit absolute images so relocating the checked JSONL is safe.
    valid = [{**r, 'image': str((base / r['image']).resolve())} for r in valid]
    write_rows(output, valid)
    write_rows(Path(output).with_suffix('.errors.jsonl'), errors)
    return {'accepted': len(valid), 'rejected': len(errors)}


def render_templates(input_path, output, font_path, variants=2, seed=42):
    from playwright.sync_api import sync_playwright
    if not 1 <= variants <= 100:
        raise ValueError('variants must be between 1 and 100 per family')
    out = Path(output).resolve()
    if out.exists():
        raise ValueError('use a new synthesis output directory')
    templates = read_rows(input_path)
    if not templates:
        raise ValueError('empty template package')
    families = [t['template_family_id'] for t in templates]
    if len(set(families)) != len(families):
        raise ValueError('duplicate template families')
    for t in templates:
        validate_template(t)
    font_path = Path(font_path).resolve()
    font_coverage(font_path, ''.join(c['text'] for t in templates for c in t['cells']) + ''.join(t['title'] for t in templates))
    font_data = base64.b64encode(font_path.read_bytes()).decode()
    out.mkdir(parents=True)
    for folder in ('images', 'pages', 'html', 'specs', 'assets'):
        (out / folder).mkdir()
    shutil.copy2(font_path, out / 'assets' / 'document.otf')
    if (font_path.parent / 'LICENSE').exists():
        shutil.copy2(font_path.parent / 'LICENSE', out / 'assets' / 'FONT-LICENSE.txt')
    write_rows(out / 'templates.jsonl', templates)
    records, failures = [], []
    with sync_playwright() as pw:
        try:
            browser = pw.chromium.launch(headless=True, timeout=30000)
        except Exception as e:
            write_json(out / 'manifest.json', {'status': 'failed', 'reason': str(e)})
            raise RuntimeError('Chromium launch failed; install Playwright browser dependencies. Details in manifest.json') from e
        context = browser.new_context(viewport={'width': 1900, 'height': 1200}, device_scale_factor=1)
        # Everything is inlined. Rendering must not depend on remote assets.
        context.route('**/*', lambda route: route.abort())
        page = context.new_page()
        for t in templates:
            for index in range(variants):
                sid = digest([t, index, seed])[:20]
                try:
                    cells, style = variant(t, index, seed)
                    gt = render_html(cells, t['rows'], t['columns'])
                    otsl = html_to_otsl(gt)
                    if otsl_to_html(otsl) != gt:
                        raise ValueError('roundtrip mismatch')
                    document = page_html(gt, t['title'], style, t['header_rows'], font_data)
                    atomic_text(out / 'html' / f'{sid}.html', document.replace('data:font/otf;base64,' + font_data, '../assets/document.otf'))
                    page.set_content(document, wait_until='load')
                    page.evaluate('document.fonts.ready')
                    page.evaluate('''async () => {await document.fonts.load('16px DocumentFont'); await new Promise(r=>requestAnimationFrame(()=>requestAnimationFrame(r)));}''')
                    geometry = page.locator('table').evaluate('''el => {
                        const rect=o=>({x:o.x,y:o.y,width:o.width,height:o.height});
                        return {table:rect(el.getBoundingClientRect()),cells:[...el.querySelectorAll('td')].map(c=>{
                            const range=document.createRange();range.selectNodeContents(c);
                            const b=c.getBoundingClientRect();const r=range.getBoundingClientRect();
                            return {text:c.textContent,box:rect(b),overflow:c.scrollWidth>c.clientWidth+1 || c.scrollHeight>c.clientHeight+1 || r.right>b.right+1 || r.bottom>b.bottom+1};
                        })};}''')
                    if len(geometry['cells']) != len(cells) or any(c['overflow'] for c in geometry['cells']):
                        raise ValueError('cell count mismatch or text overflow')
                    if [c['text'] for c in geometry['cells']] != [c.text for c in cells]:
                        # textContent omits <br> newlines; inner HTML source remains
                        # authoritative. Compare explicit br using DOM serialization.
                        texts = page.locator('td').evaluate_all("els => els.map(el => {const c=el.cloneNode(true);c.querySelectorAll('br').forEach(b=>b.replaceWith('\\n'));return c.textContent;})")
                        if texts != [c.text for c in cells]:
                            raise ValueError('rendered cell text differs from source')
                    main = page.locator('main')
                    if geometry['table']['height'] > 12000:
                        raise ValueError('table too tall for bounded generation')
                    main.screenshot(path=str(out / 'pages' / f'{sid}.png'))
                    raw = page.locator('table').screenshot()
                    clean = Image.open(io.BytesIO(raw)).convert('RGB')
                    image = ImageOps.expand(clean, border=10, fill='white')
                    augmentation = {'kind': 'clean', 'padding': 10}
                    if index % 2:
                        angle = .35 if index % 4 == 1 else -.35
                        image = ImageOps.grayscale(image).convert('RGB').filter(ImageFilter.GaussianBlur(.25))
                        image = image.rotate(angle, resample=Image.Resampling.BICUBIC, expand=True, fillcolor='white')
                        buf = io.BytesIO()
                        image.save(buf, format='JPEG', quality=78)
                        image = Image.open(io.BytesIO(buf.getvalue())).convert('RGB')
                        augmentation.update(kind='scan_like', rotation_degrees=angle, blur_radius=.25, jpeg_quality=78)
                    image_path = out / 'images' / f'{sid}.png'
                    image.save(image_path)
                    write_json(out / 'specs' / f'{sid}.json', {'cells': [asdict(c) for c in cells], 'style': style,
                               'clean_page_geometry': geometry, 'augmentation': augmentation,
                               'geometry_coordinate_space': 'unaugmented page CSS pixels; not final augmented crop'})
                    record = {'id': sid, 'task': 'table', 'image': str(image_path), 'image_hash': file_hash(image_path),
                              'gt': gt, 'gt_format': 'html', 'html': gt, 'otsl': otsl, 'gt_trusted': True,
                              'source': t['source'], 'domain': t['domain'], 'split': t['split'],
                              'document_id': t['document_id'], 'parent_document_id': t.get('parent_document_id'),
                              'parent_image_hash': t.get('parent_image_hash'),
                              'language': t.get('language', 'unknown'), 'origin': 'synthetic',
                              'seed_sample_id': t['seed_sample_id'], 'template_family_id': t['template_family_id'],
                              'variant_seed': seed, 'variant_index': index, 'seed_origin': t['seed_origin'],
                              'generation_params': {'style': style, 'augmentation': augmentation},
                              'table_features': table_features(gt), 'font_sha256': file_hash(font_path),
                              'qa': {'status': 'passed', 'roundtrip': True, 'glyph_coverage': True,
                                     'dom_text_match': True, 'overflow': False}, 'model_difficulty': 'not_evaluated'}
                    records.append(record)
                except Exception as e:
                    failures.append({'id': sid, 'family': t['template_family_id'], 'reason': str(e)})
        browser.close()
    write_rows(out / 'samples.jsonl', records)
    write_rows(out / 'failures.jsonl', failures)
    manifest = {'generated': len(records), 'failed': len(failures), 'families': len(templates),
                'status': 'complete' if not failures else 'partial',
                'variants_per_family': variants, 'seed': seed, 'method': 'GT-driven parametric table synthesis',
                'vlm_template_generation': False, 'model_inference_performed': False,
                'template_sha256': file_hash(input_path), 'font_sha256': file_hash(font_path)}
    write_json(out / 'manifest.json', manifest)
    gallery(records, out)
    return {**manifest, 'output': str(out), 'preview': str(out / 'index.html')}


def gallery(records, out):
    from .report import document
    parts = ['<p>GT 驱动的参数化合成；尚未运行 OCR 模型，难度标签仅描述设计特征。偶数编号为清晰版，奇数编号为轻度扫描版。图片与 GT 来自同一个单元格网格。</p>']
    thumbs = []
    for row in records:
        sid = row['id']
        label = row['template_family_id'] + ' / ' + row['generation_params']['augmentation']['kind']
        parts.append(f'<article><h2>{html.escape(label)}</h2><a href="images/{sid}.png"><img loading="lazy" src="images/{sid}.png" style="max-height:none"></a><p><a href="pages/{sid}.png">整页预览</a> · <a href="html/{sid}.html">渲染 HTML</a> · <a href="specs/{sid}.json">结构与坐标</a></p><details><summary>GT 表格</summary>{row["html"]}</details><details><summary>OTSL</summary><pre>{html.escape(row["otsl"])}</pre></details></article>')
        with Image.open(row['image']) as im:
            thumb = im.convert('RGB')
            thumb.thumbnail((650, 440))
            card = Image.new('RGB', (690, thumb.height + 65), '#f3f5f7')
            ImageDraw.Draw(card).text((18, 12), label, fill='#1c3048', font_size=16)
            card.paste(thumb, ((690-thumb.width)//2, 45))
            thumbs.append(card)
    atomic_text(out / 'index.html', document('Synthetic table preview', ''.join(parts)))
    if thumbs:
        heights = [max(im.height for im in thumbs[i:i+2]) for i in range(0, len(thumbs), 2)]
        sheet = Image.new('RGB', (1380, sum(heights)), '#f3f5f7')
        y = 0
        for index, height in enumerate(heights):
            for col, im in enumerate(thumbs[index*2:index*2+2]):
                sheet.paste(im, (col*690, y))
            y += height
        sheet.save(out / 'contact-sheet.jpg', quality=90)
