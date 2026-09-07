import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file

from dataflywheel.average import average_models


def checkpoint(path, value, buffer=2):
    path.mkdir()
    (path / "config.json").write_text(json.dumps({"model_type": "test", "vocab_size": 2, "tie_word_embeddings": False}))
    (path / "tokenizer.json").write_text('{"vocab": ["a", "b"]}')
    (path / "preprocessor_config.json").write_text('{"size": 100}')
    save_file({"weight": torch.tensor([value, value+1]), "buffer": torch.tensor(buffer)}, path / "model.safetensors")
    return path


def read_model(path):
    tensors = {}
    for p in path.glob("*.safetensors"):
        tensors.update(load_file(p))
    return tensors


def test_weighted_shards_and_n1(tmp_path):
    a, b = checkpoint(tmp_path / "a", 1.0), checkpoint(tmp_path / "b", 3.0)
    out = tmp_path / "out"
    average_models([a,b], out, [1,3], "float32", max_shard_bytes=8)
    assert torch.equal(read_model(out)["weight"], torch.tensor([2.5, 3.5]))
    assert (out / "model.safetensors.index.json").exists()
    assert (out / "tokenizer.json").exists()
    n1 = tmp_path / "n1"
    average_models([out], n1, dtype="float32")
    assert torch.equal(read_model(n1)["weight"], read_model(out)["weight"])


def test_buffer_failure_no_partial_output(tmp_path):
    a, b = checkpoint(tmp_path / "a", 1.0), checkpoint(tmp_path / "b", 3.0, buffer=3)
    with pytest.raises(ValueError, match="buffer mismatch"):
        average_models([a,b], tmp_path / "out")
    assert not (tmp_path / "out").exists()
    assert not list(tmp_path.glob(".averaging-*"))


@pytest.mark.parametrize("kind", ["tokenizer", "architecture", "adapter", "shape", "quantized"])
def test_incompatible_checkpoints(tmp_path, kind):
    a, b = checkpoint(tmp_path / "a", 1.0), checkpoint(tmp_path / "b", 3.0)
    if kind == "tokenizer":
        (b / "tokenizer.json").write_text('{}')
    elif kind == "architecture":
        (b / "config.json").write_text('{"model_type":"other"}')
    elif kind == "adapter":
        (b / "adapter_config.json").write_text('{}')
    elif kind == "quantized":
        (b / "config.json").write_text('{"quantization_config":{"bits":4}}')
    else:
        save_file({"weight": torch.ones(3), "buffer": torch.tensor(2)}, b / "model.safetensors")
    with pytest.raises(ValueError):
        average_models([a,b], tmp_path / "out")


def test_invalid_weights_and_no_overwrite(tmp_path):
    a = checkpoint(tmp_path / "a", 1.0)
    for weights in ([0], [-1], [float("nan")], [1,2]):
        with pytest.raises(ValueError):
            average_models([a], tmp_path / "out", weights)
    with pytest.raises(ValueError):
        average_models([a], a)


def test_tied_weights_and_nonfinite(tmp_path):
    a = checkpoint(tmp_path / "a", 1.0)
    (a / "config.json").write_text('{"tie_word_embeddings": true}')
    save_file({"model.embed_tokens.weight": torch.ones(2,2), "lm_head.weight": torch.zeros(2,2)}, a / "model.safetensors")
    with pytest.raises(ValueError, match="tied"):
        average_models([a], tmp_path / "out")
    save_file({"weight": torch.tensor([float("nan")])}, a / "model.safetensors")
    with pytest.raises(ValueError, match="nonfinite"):
        average_models([a], tmp_path / "out")
