from copy import deepcopy
from pathlib import Path

from PIL import Image
import pytest

from dataflywheel.io import DEFAULTS


@pytest.fixture
def config():
    cfg = deepcopy(DEFAULTS)
    cfg["metrics"]["formula"] = "fast"
    cfg["models"] = {role: {"base_url": "http://test/v1", "model": role, "revision": role + "-v1"} for role in ("base", "sft")}
    cfg["mining"]["budget"] = 10
    return cfg


@pytest.fixture
def image_path(tmp_path):
    path = tmp_path / "image.png"
    Image.new("RGB", (50, 50), "white").save(path)
    return path


def candidate(text, score, role="sft", index=0, status="ok", finish="stop", qstatus="ok"):
    return {"id": f"{role}-{index}", "text": text, "role": role, "index": index, "kind": "greedy" if index == 0 else "sample",
            "status": status, "finish_reason": finish, "quality": {"score": score, "status": qstatus}}


def sample(candidates, task="text", gt="hello", **kwargs):
    return {"id": "one", "image": "/fake.png", "task": task, "domain": "private", "source": "test", "document_id": "one",
            "split": "train", "target": gt, "score_gt": gt, "gt_trusted": True, "candidates": candidates, **kwargs}
