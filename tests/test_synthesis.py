from copy import deepcopy
import json

import pytest

from dataflywheel.synthesis import builtin_templates, make_templates, validate_template, variant, page_html, check_records
from dataflywheel.tables import render_html, html_to_otsl, otsl_to_html
from dataflywheel.io import write_rows, file_hash


def test_all_builtin_topologies_and_variants():
    for t in builtin_templates():
        validate_template(t)
        for i in range(4):
            cells, style = variant(t, i, 42)
            h = render_html(cells, t['rows'], t['columns'])
            assert otsl_to_html(html_to_otsl(h)) == h
            assert variant(t, i, 42) == (cells, style)


def test_invalid_grid_and_eval_seeds():
    t = builtin_templates()[0]
    invalid = deepcopy(t)
    invalid['cells'].append(invalid['cells'][0])
    with pytest.raises(ValueError, match='overlapping'):
        validate_template(invalid)
    t['split'] = 'test'
    with pytest.raises(ValueError, match='training'):
        validate_template(t)


def test_import_seed_preserves_parent_and_domain(tmp_path, image_path):
    src = tmp_path / 'seeds.jsonl'
    t = builtin_templates()[0]
    write_rows(src, [{'id': 'a', 'image': str(image_path), 'task': 'table', 'html': t['seed_html'],
                     'document_id': 'real-document', 'source': 'user', 'domain': 'general', 'split': 'train'}])
    out = tmp_path / 'templates.jsonl'
    assert make_templates(src, out)['templates'] == 1
    obj = json.loads(out.read_text())
    assert obj['document_id'] == 'real-document' and obj['domain'] == 'general'
    assert obj['seed_sample_id'] == 'a'


def test_seed_eval_sibling_excluded(tmp_path, image_path):
    src = tmp_path / 'seeds.jsonl'
    write_rows(src, [{'id': str(i), 'image': str(image_path), 'task': 'table', 'html': builtin_templates()[0]['seed_html'],
                     'document_id': 'doc', 'split': split} for i, split in enumerate(('train','test'))])
    assert make_templates(src, tmp_path / 't.jsonl')['templates'] == 0


def test_document_has_only_validated_html():
    t = builtin_templates()[0]
    cells, style = variant(t, 0, 42)
    result = page_html(render_html(cells,t['rows'],t['columns']), '<script>alert(1)</script>', style, t['header_rows'], '')
    assert '<script>' not in result
    assert '&lt;script&gt;' in result and 'table-layout:fixed' in result


def test_check_detects_tampered_gt_and_duplicate_image(tmp_path, image_path):
    h = '<table><tbody><tr><td>A</td></tr></tbody></table>'
    row = {'id': 'a', 'image': str(image_path), 'image_hash': file_hash(image_path), 'html': h,
           'otsl': '<fcel>A<nl>', 'split': 'train', 'qa': {'status':'passed'}}
    inp = tmp_path / 'samples.jsonl'
    write_rows(inp, [row, {**row, 'id':'b'}, {**row,'id':'c','otsl':'<fcel>B<nl>'}])
    result = check_records(inp, tmp_path / 'valid.jsonl')
    assert result == {'accepted':1,'rejected':2}
