import asyncio
import json
from pathlib import Path

import httpx
import pytest

from dataflywheel.data import normalize
from dataflywheel.inference import infer_rows
from dataflywheel.io import write_rows
from dataflywheel.pages import parse_pages
from dataflywheel.report import create_report
from dataflywheel.training import preflight, train_command


def test_inference_cache_revision_retry_and_sampling(tmp_path, image_path, config, monkeypatch):
    calls = []
    def handler(request):
        body = json.loads(request.content)
        calls.append(body)
        if len(calls) == 1:
            return httpx.Response(503)
        return httpx.Response(200, json={"choices": [{"message": {"content": "hello"}, "finish_reason": "stop"}]})
    original = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: original(transport=httpx.MockTransport(handler), **kwargs))
    row = normalize({"image": str(image_path), "task": "text", "text": "hello"}, tmp_path, config)
    cache = tmp_path / "cache"
    a = asyncio.run(infer_rows([row], config, cache))
    assert len(calls) == 3
    assert len(a[0]["candidates"]) == 2
    asyncio.run(infer_rows([row], config, cache))
    assert len(calls) == 3
    sampled = asyncio.run(infer_rows(a, config, cache, sampling=True))
    assert len(calls) == 7 and len(sampled[0]["candidates"]) == 6
    config["models"]["sft"]["revision"] = "v2"
    with pytest.raises(ValueError, match="rerun greedy"):
        asyncio.run(infer_rows(a, config, cache, sampling=True))
    asyncio.run(infer_rows([row], config, cache))
    assert len(calls) >= 8


def test_preflight_both_branches_and_train_flags(tmp_path, image_path, config):
    row = {"messages": [{"role": "user", "content": "<image>OCR:"}, {"role": "assistant", "content": "good"}],
           "images": [str(image_path)], "rejected_response": "bad"}
    class Template:
        def encode(self, row):
            return {"chosen_input_ids": [1]*4, "chosen_labels": [-100, -100, 1, 2],
                    "rejected_input_ids": [1]*10, "rejected_labels": [-100]*8+[1,2]}
    config["training"]["max_length"] = 5
    with pytest.raises(ValueError, match="failed preflight"):
        preflight([row], config, tmp_path, template=Template())
    assert "overlength" in (tmp_path / "preflight.errors.jsonl").read_text()
    config["training"]["model"] = "/sft"
    cmd, env = train_command(config, tmp_path / "dpo.jsonl", tmp_path / "train")
    assert cmd[cmd.index("--ref_model")+1] == "/sft"
    assert cmd[cmd.index("--eval_strategy")+1] == "no"
    assert cmd[cmd.index("--truncation_strategy")+1] == "delete"
    assert cmd[cmd.index("--strict")+1] == "true"
    assert "--deepspeed" not in cmd and "--rpo_alpha" not in cmd
    assert "--tuner_type" in cmd and env["NPROC_PER_NODE"] == "1"
    config["training"].update(gpus="0,1", nproc_per_node=2)
    assert train_command(config, "data", "out")[1]["NPROC_PER_NODE"] == "2"


def test_page_export_and_resume(tmp_path, image_path, config):
    inputs = tmp_path / "pages.json"
    write_rows(inputs, [{"image": str(image_path)}])
    calls = []
    class Result:
        def save_to_json(self, path):
            Path(path).write_text('{"page": 0}')
        def save_to_markdown(self, path):
            Path(path, "image_res.md").write_text("hello\n")
    class Pipeline:
        def __init__(self, **kwargs):
            assert kwargs["pipeline_version"] == "v1.6"
        def predict(self, path):
            calls.append(path)
            return [Result()]
    out = tmp_path / "pages"
    assert parse_pages(inputs, out, config, pipeline_factory=Pipeline)["pages"] == 1
    assert (out / "markdown/image.md").read_text() == "hello\n"
    parse_pages(inputs, out, config, pipeline_factory=Pipeline)
    assert len(calls) == 1
    config["models"]["sft"]["revision"] = "new"
    parse_pages(inputs, out, config, pipeline_factory=Pipeline)
    assert len(calls) == 2


def test_report_safe_offline(tmp_path, image_path, config):
    row = {"id": "<script>alert(1)</script>", "image": str(image_path), "task": "table",
           "target": "<table><tr><td>&lt;script&gt;</td></tr></table>", "candidates": []}
    create_report([row], tmp_path / "report", config)
    page = (tmp_path / "report/samples-1.html").read_text()
    assert "<script>" not in page and "&lt;script&gt;" in page
    assert "<table>" in page and "loading=\"lazy\"" in page
    assert "cdn" not in page and (tmp_path / "report/index.html").exists()
