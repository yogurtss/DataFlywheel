import json
from pathlib import Path

import httpx
import pytest

from dataflywheel.io import load_config, read_rows, write_rows, redact_config
from dataflywheel.synthesis_agent import VisionClient, run_agent, template_from_reply


def proposal(value='A'):
    return {'html': f'<table><tr><td>{value}</td></tr></table>', 'title': 'Synthetic',
            'header_rows': 0, 'style': {'width': 500, 'font_size': 15}}


def test_openai_payload_and_custom_auth(image_path, monkeypatch):
    monkeypatch.setenv('TEST_VLM_TOKEN', 'env-key')
    def handler(request):
        assert str(request.url) == 'https://example.test/v1/chat/completions'
        assert request.headers['Authorization'] == 'Bearer env-key'
        body = json.loads(request.content)
        assert body['model'] == 'custom-vision' and body['stream'] is False
        parts = body['messages'][0]['content']
        assert parts[0] == {'type': 'text', 'text': 'Inspect'}
        assert parts[1]['image_url']['url'].startswith('data:image/png;base64,')
        return httpx.Response(200, json={'choices': [{'finish_reason': 'stop', 'message': {'content': '```json\n{"ok":true}\n```'}}]})
    client = VisionClient({'url': 'https://example.test/v1', 'model': 'custom-vision',
                           'token_env': 'TEST_VLM_TOKEN'}, httpx.MockTransport(handler))
    try:
        assert client.ask('Inspect', [image_path]) == {'ok': True}
    finally:
        client.close()


@pytest.mark.parametrize('finish', ['length', 'content_filter', None])
def test_truncated_is_not_accepted(finish):
    client = VisionClient({'url': 'https://example.test/v1/chat/completions', 'model': 'm', 'token': 'literal'},
                          httpx.MockTransport(lambda req: httpx.Response(200, json={'choices': [
                              {'finish_reason': finish, 'message': {'content': '{}'}}]})))
    try:
        with pytest.raises(ValueError, match='incomplete'):
            client.ask('x')
    finally:
        client.close()


def test_http_error_does_not_leak_token():
    client = VisionClient({'url': 'https://example.test/v1', 'model': 'm', 'token': 'SECRET'},
                          httpx.MockTransport(lambda req: httpx.Response(401, text='SECRET')))
    try:
        with pytest.raises(RuntimeError, match='401') as err:
            client.ask('x')
        assert 'SECRET' not in str(err.value)
        assert redact_config({'synthesis': {'vlm': {'token': 'SECRET', 'model': 'm'}}})['synthesis']['vlm']['token'] == '[REDACTED]'
    finally:
        client.close()


def test_visual_repair_and_expansion(tmp_path, image_path):
    seeds = tmp_path / 'seeds.jsonl'
    write_rows(seeds, [{'id': 'seed', 'image': str(image_path), 'task': 'table', 'document_id': 'doc',
                        'html': proposal()['html'], 'split': 'train'}])
    config = load_config()
    config['synthesis'] = {'templates_per_seed': 2, 'max_repairs': 1}
    class Client:
        model, token, calls = 'mock-vlm', '', 0
        def ask(self, prompt, images):
            self.calls += 1
            if self.calls in (2, 4, 6):
                assert len(images) == 2
                return {'approved': self.calls != 2, 'issues': ['font too small'] if self.calls == 2 else []}
            if self.calls == 3:
                assert 'font too small' in prompt and len(images) == 2
            if self.calls == 5:
                assert 'Diversify both structure' in prompt
            return proposal(str(self.calls))
    def renderer(path, out, font, **kwargs):
        t = read_rows(path)[0]
        write_rows(out / 'samples.jsonl', [{'id': t['template_family_id'], 'image': str(image_path),
                                          'html': t['seed_html'], 'qa': {'status': 'passed'}}])
    result = run_agent(seeds, tmp_path / 'out', config, image_path, client=Client(), renderer=renderer)
    assert result['generated'] == 2 and result['api_calls'] == 6
    audit = read_rows(tmp_path / 'out/audit.jsonl')
    assert [a['status'] for a in audit] == ['rejected', 'accepted', 'accepted']
    assert read_rows(tmp_path / 'out/templates.jsonl')[0]['parent_document_id'] == 'doc'


def test_eval_siblings_never_sent(tmp_path, image_path):
    seeds = tmp_path / 'seeds.jsonl'
    write_rows(seeds, [{'id': str(i), 'image': str(image_path), 'task': 'table', 'document_id': 'doc',
                        'html': proposal()['html'], 'split': split} for i, split in enumerate(('train', 'test'))])
    with pytest.raises(ValueError, match='no eligible seeds'):
        run_agent(seeds, tmp_path / 'out', load_config(), image_path)


def test_invalid_style_cannot_enter_renderer():
    seed = {'id': 'a', 'image': 'a.png', 'image_hash': 'h', 'document_id': 'doc', 'source': 's', 'domain': 'private'}
    reply = proposal()
    reply['style']['width'] = '500; background:url(http://example.test)'
    with pytest.raises(ValueError, match='types'):
        template_from_reply(reply, seed, 'family')
