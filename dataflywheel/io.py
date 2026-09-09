"""Atomic artifacts and reproducible configuration."""
import hashlib
import json
import os
import math
from pathlib import Path
import tempfile

import yaml


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_rows(path):
    path = Path(path)
    with path.open(encoding="utf-8") as f:
        if path.suffix == ".jsonl":
            rows = [json.loads(line) for line in f if line.strip()]
        else:
            rows = json.load(f)
    if not isinstance(rows, list) or any(not isinstance(r, dict) for r in rows):
        raise ValueError(f"{path}: expected a JSON list or JSONL of objects")
    return rows


def atomic_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".writing-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def write_json(path, value):
    atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def write_rows(path, rows):
    if Path(path).suffix == ".json":
        write_json(path, rows)
    else:
        atomic_text(path, "".join(json.dumps(r, ensure_ascii=False, allow_nan=False) + "\n" for r in rows))


DEFAULTS = {
    "seed": 42,
    "prompts": {"table": "Table Recognition:", "text": "OCR:", "formula": "Formula Recognition:"},
    "fields": {},
    "data": {"eval_fraction": 0.0, "exclude_manifests": [], "image_root": None,
             "image_cache": "runs/image-cache", "image_timeout": 60, "max_image_bytes": 50_000_000},
    "inference": {"concurrency": 8, "timeout": 180, "retries": 3, "max_tokens": 4096,
                  "samples": 4, "temperature": 0.7, "top_p": 0.9,
                  "preprocess": {"mode": "rgb", "max_side": None}},
    "metrics": {"formula": "cdm", "cdm_python": "python", "cdm_repo": None, "timeout": 120},
    "mining": {"budget": 1000, "weights": [0.4, 0.4, 0.2], "regression_threshold": 0.05,
               "middle_quantiles": [0.2, 0.8], "buckets": {"regression": 0.4, "learnable": 0.4, "coverage": 0.2}},
    "pairs": {"mode": "hybrid", "margin": 0.05,
              "min_quality": {"table": 0.85, "text": 0.95, "formula": 0.90}},
    "mixture": {"domain": None, "task": None},
    "report": {"page_size": 40},
    "training": {"model": None, "ref_model": None, "gpus": "0", "nproc_per_node": 1,
                 "torch_dtype": "bfloat16",
                 "paddle_feature_compat": True,
                 "learning_rate": 1e-6, "beta": 0.1, "num_train_epochs": 1,
                 "per_device_train_batch_size": 1, "gradient_accumulation_steps": 16,
                 "max_length": 8192, "max_pixels": 1003520, "rpo_alpha": None,
                 "save_steps": 100, "logging_steps": 5},
    "pages": {"pipeline_version": "v1.6", "backend": "vllm-server", "options": {}},
}


def merge(a, b):
    out = dict(a)
    for k, v in b.items():
        out[k] = merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def redact_config(value):
    """Exclude credentials from persisted configuration snapshots."""
    if isinstance(value, dict):
        return {k: '[REDACTED]' if k.lower() in {'token', 'api_key', 'authorization', 'password'} else redact_config(v)
                for k, v in value.items()}
    if isinstance(value, list):
        return [redact_config(v) for v in value]
    return value


def load_config(path=None):
    config = merge({}, DEFAULTS)
    if path:
        config = merge(config, yaml.safe_load(Path(path).read_text()) or {})
    if config["metrics"]["formula"] not in {"cdm", "fast"}:
        raise ValueError("metrics.formula must be cdm or fast")
    if config["pairs"]["mode"] not in {"model_pair", "gt_pair", "hybrid"}:
        raise ValueError("pairs.mode must be model_pair, gt_pair or hybrid")
    for section in [config["mining"]["buckets"], *(v for v in config["mixture"].values() if v is not None)]:
        if not section or any(not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0 for v in section.values()) or abs(sum(section.values()) - 1) > 1e-6:
            raise ValueError("Quota ratios must be nonnegative and sum to 1")
    weights = config["mining"]["weights"]
    if len(weights) != 3 or any(not math.isfinite(w) or w < 0 for w in weights) or sum(weights) <= 0:
        raise ValueError("mining.weights must contain three nonnegative finite weights")
    for key in ("concurrency", "samples", "max_tokens", "timeout"):
        if config["inference"][key] <= 0:
            raise ValueError(f"inference.{key} must be positive")
    if config["inference"]["retries"] < 0 or config["mining"]["budget"] < 0:
        raise ValueError("retries and budget cannot be negative")
    if not 0 < config["pairs"]["margin"] <= 1 or any(not 0 <= v <= 1 for v in config["pairs"]["min_quality"].values()):
        raise ValueError("invalid pair margin/quality threshold")
    lo, hi = config["mining"]["middle_quantiles"]
    if not 0 <= lo <= hi <= 1:
        raise ValueError("middle_quantiles must be ordered values in [0,1]")
    if config["inference"]["preprocess"]["mode"] != "rgb":
        raise ValueError("only rgb preprocessing is supported")
    return config
