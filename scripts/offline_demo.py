"""Exercise the real pipeline with explicitly simulated model responses.

This is a software smoke test, NOT an OCR/model quality evaluation.
"""
import json
from pathlib import Path
import sys

import httpx

from dataflywheel.cli import main
from make_demo import main as make_data


def run():
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "runs/offline-demo").resolve()
    old = sys.argv
    sys.argv = ["make_demo", str(root / "input")]
    make_data()
    sys.argv = old
    cfg = root / "config.yaml"
    cfg.write_text('''metrics: {formula: fast}
models:
  base: {base_url: "http://simulation/v1", model: base, revision: simulated-base-v1}
  sft: {base_url: "http://simulation/v1", model: sft, revision: simulated-sft-v1}
mining: {budget: 3}
training: {model: /path/to/sft-full-checkpoint}
''')
    def handler(request):
        body = json.loads(request.content)
        prompt = body["messages"][0]["content"][1]["text"]
        good = body["model"] == "base" or (body["temperature"] > 0 and body["seed"] % 2 == 0)
        answers = {
            "Table Recognition:": ("<fcel>Year<fcel>Revenue<nl><fcel>2025<fcel>100<nl>", "<fcel>Year<fcel>Revenue<nl><fcel>2024<fcel>999<nl>"),
            "OCR:": ("The quick brown fox jumps over the lazy dog.", "The quick red fox jumps over a lazy cat."),
            "Formula Recognition:": ("x^{2}+y^{2}=z^{2}", "x^{3}-y^{2}=z^{3}"),
        }
        answer = answers[prompt][0 if good else 1]
        return httpx.Response(200, json={"choices": [{"message": {"role": "assistant", "content": answer}, "finish_reason": "stop"}]})
    original = httpx.AsyncClient
    httpx.AsyncClient = lambda **kwargs: original(transport=httpx.MockTransport(handler), **kwargs)
    try:
        main(["run", "-i", str(root / "input/input.json"), "-o", str(root / "result"), "-c", str(cfg)])
    finally:
        httpx.AsyncClient = original


if __name__ == "__main__":
    run()
