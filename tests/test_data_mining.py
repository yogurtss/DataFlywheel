import json

from PIL import Image
import pytest

from dataflywheel.data import import_data, isolate
from dataflywheel.mining import build_pairs, make_pair, mine_rows, stratify
from conftest import candidate, sample


def test_html_import_conflict_and_no_overwrite(tmp_path, image_path, config):
    raw = {"id": "a", "image": str(image_path), "task": "table", "html": "<table><tr><td>a</td></tr></table>", "otsl": "<fcel>b<nl>"}
    path = tmp_path / "data.json"
    path.write_text(json.dumps([raw]))
    rows, errors = import_data(path, config)
    assert not rows and "conflict" in errors[0]["reason"].lower()
    assert json.loads(path.read_text()) == [raw]


def test_document_leak_and_pixel_dedup(tmp_path, config):
    records = []
    for i, color in enumerate(("red", "green", "blue", "blue")):
        image = tmp_path / f"{i}.png"
        Image.new("RGB", (10, 10), color).save(image)
        records.append({"id": str(i), "image": str(image), "task": "text", "text": "word", "document_id": "shared" if i < 2 else str(i), "split": "eval" if i == 1 else "train"})
    path = tmp_path / "input.json"
    path.write_text(json.dumps(records))
    rows, errors = import_data(path, config)
    assert {r["id"] for r in rows} == {"1", "2"}
    assert {e["reason"] for e in errors} == {"evaluation_overlap", "duplicate_image"}


def test_invalid_eval_gt_still_blocks_crops(tmp_path, image_path, config):
    raw = [{"id": "eval", "image": str(image_path), "task": "table", "html": "bad", "document_id": "doc", "split": "eval"},
           {"id": "crop", "image": str(image_path), "task": "text", "text": "good", "document_id": "doc"}]
    path = tmp_path / "input.json"
    path.write_text(json.dumps(raw))
    rows, errors = import_data(path, config)
    assert not rows
    assert any(e["reason"] == "evaluation_overlap" for e in errors)


def test_model_pair_direction_and_hard_negative(config):
    config["pairs"]["mode"] = "model_pair"
    r = sample([candidate("hello", 1, "base"), candidate("hallo", .8), candidate("helpo", .79, index=1)])
    pair, reason = make_pair(r, config)
    assert reason is None and pair["pair_audit"]["chosen_source"] == "base"
    assert pair["pair"]["rejected_response"] == "hallo"
    r["candidates"] = [candidate("hello", 1), candidate("hallo", .8, "base")]
    assert make_pair(r, config)[0]["pair_audit"]["chosen_source"] == "sft"


@pytest.mark.parametrize("mode,expected", [("model_pair", False), ("gt_pair", True), ("hybrid", True)])
def test_pair_modes(config, mode, expected):
    config["pairs"]["mode"] = mode
    r = sample([candidate("wrong", .3), candidate("also wrong", .2, "base")])
    pair, reason = make_pair(r, config)
    assert bool(pair) == expected
    if pair:
        assert pair["pair_audit"]["chosen_source"] == "gt"


def test_pair_filters(config):
    r = sample([candidate("hello", 1), candidate("hello ", .8, "base")])
    assert make_pair(r, config)[0] is None
    r = sample([candidate("hello", 1), candidate("wrong", 0, "base", finish="length")])
    assert make_pair(r, config)[0] is None
    r = sample([candidate("wrong", .2)], score_gt=None, target=None)
    assert make_pair(r, config)[1] == "missing_gt"
    r = sample([candidate("wrong", .2)], gt_trusted=False)
    assert make_pair(r, config)[1] == "untrusted_gt"


def test_invalid_prediction_can_be_negative(config):
    gt = "<fcel>a<nl>"
    r = sample([candidate("broken", 0, qstatus="invalid_prediction")], task="table", gt=gt)
    pair, _ = make_pair(r, config)
    assert pair["pair"]["messages"][1]["content"] == gt


def test_model_table_positive_requires_native_otsl(config):
    config["pairs"]["mode"] = "model_pair"
    row = sample([candidate("<table><tr><td>a</td></tr></table>", 1, "base"), candidate("<fcel>b<nl>", .5)], task="table", gt="<fcel>a<nl>")
    assert make_pair(row, config)[1] == "no_reliable_model_positive"


def test_stratification_and_shortfalls(config):
    rows = [{"id": str(i), "task": task, "domain": domain} for i, (task, domain) in enumerate(
        [(t, d) for t in ("table", "text", "formula") for d in ("private", "general") for _ in range(100)])]
    selected, stats = stratify(rows, 100, config)
    assert len(selected) == 100
    assert stats["actual"] == stats["desired"]
    short, report = stratify(rows[:3], 100, config)
    assert len(short) == 3 and report["unfilled"] == 97


def test_mining_reproducible_excludes_eval(config):
    rows = [sample([candidate("wrong", .5)], id=str(i), image=f"/{i}.png", document_id=str(i),
                   signals={"sft_score": .5, "gt_error": .5, "regression": .2, "priority": .4, "candidate_gap": .2}) for i in range(30)]
    rows[0]["split"] = "eval"
    a, pending, _ = mine_rows(rows, config)
    b, _, _ = mine_rows(rows, config)
    assert [r["id"] for r in a] == [r["id"] for r in b]
    assert all(r["id"] != "0" for r in a)
    assert len(a) == 10
