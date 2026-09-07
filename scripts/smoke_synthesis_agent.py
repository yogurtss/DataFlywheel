"""Real Chromium + MOCK VLM protocol smoke; no claim of model generation quality."""
import argparse
import json
from pathlib import Path

import httpx

from dataflywheel.io import load_config, write_rows, write_json
from dataflywheel.synthesis import builtin_templates
from dataflywheel.synthesis_agent import VisionClient, run_agent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--font', required=True)
    parser.add_argument('--image', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    root = Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=False)
    fixtures = builtin_templates()
    seed = root / 'seeds.jsonl'
    write_rows(seed, [{'id': 'mock-protocol-seed', 'image': str(Path(args.image).resolve()),
                       'task': 'table', 'html': fixtures[0]['seed_html'],
                       'source': 'mock_agent_smoke', 'document_id': 'mock-smoke-doc', 'split': 'train'}])
    requests = []
    def handler(request):
        body = json.loads(request.content)
        parts = body['messages'][0]['content']
        is_review = parts[0]['text'].startswith('Review the rendered')
        requests.append({'model': body['model'], 'images': len(parts)-1, 'review': is_review})
        if is_review:
            # Mock approves the first template, requests a repair on second.
            needs_repair = len(requests) == 4
            answer = {'approved': not needs_repair, 'issues': ['Increase font size'] if needs_repair else []}
        else:
            t = fixtures[0 if len(requests) == 1 else 3]
            answer = {'html': t['seed_html'], 'title': t['title'], 'header_rows': t['header_rows'],
                      'style': {**t['style'], 'font_size': 18 if len(requests) == 5 else t['style']['font_size']}}
        return httpx.Response(200, json={'choices': [{'finish_reason': 'stop', 'message': {
            'content': json.dumps(answer, ensure_ascii=False)}}]})
    client = VisionClient({'url': 'https://mock.invalid/v1', 'model': 'MOCK-no-real-vlm'}, httpx.MockTransport(handler))
    config = load_config()
    config['synthesis'] = {'templates_per_seed': 2, 'max_repairs': 1}
    try:
        result = run_agent(seed, root / 'result', config, args.font, client=client)
        assert result['generated'] == 2 and result['api_calls'] == 6
        assert [r['images'] for r in requests] == [1, 2, 1, 2, 2, 2]
        write_json(root / 'validation.json', {'mode': 'real Chromium + MOCK VLM', 'requests': requests, 'result': result})
        print(json.dumps(result, indent=2))
    finally:
        client.close()


if __name__ == '__main__':
    main()
