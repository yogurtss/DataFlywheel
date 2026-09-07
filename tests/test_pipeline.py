import json
from pathlib import Path

from PIL import Image
import httpx

from dataflywheel.cli import main
from dataflywheel.io import read_rows


def test_complete_run_three_tasks_and_reuse(tmp_path, monkeypatch):
    records = []
    for task, color, gt, fmt in [("text", "red", "hello", "text"), ("table", "green", "<fcel>good<nl>", "otsl"),
                                 ("formula", "blue", "x+1", "latex")]:
        image = tmp_path / f"{task}.png"
        Image.new("RGB", (40,40), color).save(image)
        records.append({"id": task, "image": str(image), "task": task, "gt": gt, "gt_format": fmt})
    inp = tmp_path / "input.json"
    inp.write_text(json.dumps(records))
    config = tmp_path / "config.yaml"
    config.write_text('''metrics: {formula: fast}
mining: {budget: 3}
models:
  base: {base_url: "http://test/v1", model: base, revision: v1}
  sft: {base_url: "http://test/v1", model: sft, revision: v1}
''')
    count = 0
    def handler(request):
        nonlocal count
        count += 1
        body = json.loads(request.content)
        prompt = body["messages"][0]["content"][1]["text"]
        good, bad = {"OCR:": ("hello", "wrong"), "Table Recognition:": ("<fcel>good<nl>", "<fcel>bad<nl>"),
                     "Formula Recognition:": ("x+1", "y-2")}[prompt]
        text = good if body["model"] == "base" or body["temperature"] > 0 and body["seed"] % 2 == 0 else bad
        return httpx.Response(200, json={"choices": [{"message": {"content": text}, "finish_reason": "stop"}]})
    original = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: original(transport=httpx.MockTransport(handler), **kwargs))
    out = tmp_path / "out"
    argv = ["run", "-i", str(inp), "-o", str(out), "-c", str(config)]
    main(argv)
    pairs = read_rows(out / "dpo.jsonl")
    assert len(pairs) == 3 and count == 18
    assert (out / "report/index.html").is_file()
    main(argv)
    assert count == 18
    assert read_rows(out / "dpo.jsonl") == pairs
    for mode in ("gt_pair", "model_pair", "hybrid"):
        main(["build-dpo", "-i", str(out / "sampled.scored.jsonl"), "-o", str(out / (mode+".jsonl")), "--mode", mode, "-c", str(config)])
        assert len(read_rows(out / (mode+".jsonl"))) == 3
