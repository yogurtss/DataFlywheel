import argparse
import io
import json

import httpx
from PIL import Image
import pytest

from dataflywheel.cli import execute
from dataflywheel.data import import_data
from dataflywheel.io import load_config, read_rows, write_rows
from dataflywheel.mining import stratify
from dataflywheel.sft import adapt_sft, resolve_image


def record(image, prompt='OCR:', answer='hello', **extra):
    return {'image_info': [{'image_url': str(image), 'matched_text_index': 0}],
            'text_info': [{'text': prompt, 'tag': 'mask'}, {'text': answer, 'tag': 'no_mask'}], **extra}


@pytest.mark.parametrize('prompt,answer,task,fmt', [
    ('OCR:', 'x^2 is ordinary OCR text', 'text', 'text'),
    ('Table Recognition:', '<fcel>A<nl>', 'table', 'otsl'),
    ('Table Recognition:', '<table><tr><td>A</td></tr></table>', 'table', 'html'),
    ('Formula Recognition:', r'\[x^2\]', 'formula', 'latex')])
def test_official_import(tmp_path, image_path, config, prompt, answer, task, fmt):
    path = tmp_path / 'input.jsonl'
    original = record(image_path.name, prompt, answer)
    write_rows(path, [original])
    rows, errors = import_data(path, config)
    assert not errors and len(rows) == 1
    assert rows[0]['task'] == task and rows[0]['gt_format'] == fmt
    assert rows[0]['text_info'] == original['text_info'] and rows[0]['gt'] == answer
    assert rows[0]['image'] == str(image_path)


@pytest.mark.parametrize('change', [
    {'text_info': [{'text': 'Chart Recognition:', 'tag': 'mask'}, {'text': 'A', 'tag': 'no_mask'}]},
    {'image_info': [{'image_url': 'a.png', 'matched_text_index': 1}]},
    {'image_info': [{'image_url': 'a.png'}, {'image_url': 'b.png'}]},
    {'is_system': 1}, {'task': 'formula'},
    {'text_info': [{'text': 'OCR:', 'tag': 'no_mask'}, {'text': 'A', 'tag': 'mask'}]}])
def test_ambiguous_records_rejected(change):
    with pytest.raises(ValueError):
        adapt_sft({**record('a.png'), **change})


def test_invalid_eval_annotation_still_blocks_image(tmp_path, image_path, config):
    path = tmp_path / 'input.jsonl'
    write_rows(path, [record(image_path, id='train'), record(image_path, 'Unknown:', split='test', id='eval')])
    rows, errors = import_data(path, config)
    assert not rows
    assert any(e.get('reason') == 'evaluation_overlap' for e in errors)


def test_no_task_quotas_and_explicit_opt_in(config):
    rows = [{'id': str(i), 'task': 'table' if i < 10 else 'text', 'domain': 'private',
             'signals': {'priority': 1 if i < 10 else 0}} for i in range(20)]
    selected, stats = stratify(rows, 10, config)
    assert all(r['task'] == 'table' for r in selected)
    assert not stats['task_quotas_enabled'] and not stats['domain_quotas_enabled']
    config['mixture']['task'] = {'table': .5, 'text': .5}
    selected, stats = stratify(rows, 10, config)
    assert stats['task_counts'] == {'table': 5, 'text': 5}
    assert load_config()['mixture'] == {'task': None, 'domain': None}


def test_remote_image_cache(tmp_path, monkeypatch):
    buf = io.BytesIO()
    Image.new('RGB', (10, 10), 'red').save(buf, format='PNG')
    calls = []
    from contextlib import contextmanager
    @contextmanager
    def stream(method, url, **kwargs):
        calls.append(url)
        yield httpx.Response(200, content=buf.getvalue(), request=httpx.Request(method, url))
    monkeypatch.setattr(httpx, 'stream', stream)
    opts = {'image_cache': str(tmp_path / 'cache')}
    first = resolve_image('https://example.test/a.png', tmp_path, opts)
    assert resolve_image('https://example.test/a.png', tmp_path, opts) == first
    assert len(calls) == 1
    opts['max_image_bytes'] = 1
    with pytest.raises(ValueError, match='exceeds'):
        resolve_image('https://example.test/b.png', tmp_path, opts)
    assert not list((tmp_path / 'cache').glob('*.part'))


def test_official_sft_to_dpo_all_tasks(tmp_path, config, monkeypatch):
    prompts = [('OCR:', 'hello world', 'text'), ('Table Recognition:', '<fcel>A<nl>', 'table'),
               ('Formula Recognition:', r'x^{2}+1', 'formula')]
    seeds = []
    answers = {}
    for i, (prompt, gt, task) in enumerate(prompts):
        image = tmp_path / f'{task}.png'
        Image.new('RGB', (20+i, 20), (i*60, 20, 80)).save(image)
        seeds.append(record(image, prompt, gt))
        answers[prompt] = gt
    src = tmp_path / 'official.jsonl'
    write_rows(src, seeds)
    def handler(request):
        body = json.loads(request.content)
        prompt = body['messages'][0]['content'][1]['text']
        answer = answers[prompt] if body['model'] == 'base' else ('<fcel>wrong<nl>' if prompt == 'Table Recognition:' else 'wrong')
        return httpx.Response(200, json={'choices': [{'finish_reason': 'stop', 'message': {'content': answer}}]})
    original = httpx.AsyncClient
    monkeypatch.setattr(httpx, 'AsyncClient', lambda **kw: original(transport=httpx.MockTransport(handler), **kw))
    out = tmp_path / 'result'
    result = execute(argparse.Namespace(command='run', input=str(src), output=str(out)), config)
    assert result['pairs'] == 3
    audit = read_rows(out / 'dpo.audit.jsonl')
    assert {r['task'] for r in audit} == {'text', 'table', 'formula'}
    scored = read_rows(out / 'scored.jsonl')
    assert all(r['candidates'][0]['quality']['score'] == 1 for r in scored)
    assert all('rejected_response' in r for r in read_rows(out / 'dpo.jsonl'))
