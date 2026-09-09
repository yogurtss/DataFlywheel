"""ms-swift 4.x full DPO launcher; no evaluation or silent truncation."""
import importlib.metadata
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

from PIL import Image

from .io import digest, file_hash, read_rows, write_json, write_rows


def train_command(config, dataset, output):
    t = config["training"]
    if not t.get("model"):
        raise ValueError("training.model must point to the SFT full checkpoint")
    dtype = t.get("torch_dtype", "bfloat16")
    if dtype not in {"bfloat16", "float16", "float32"}:
        raise ValueError("training.torch_dtype must be bfloat16, float16 or float32")
    cmd = ["swift", "rlhf", "--rlhf_type", "dpo", "--model", t["model"],
           "--ref_model", t.get("ref_model") or t["model"], "--model_type", "paddleocr_vl",
           "--ref_model_type", "paddleocr_vl", "--template", "paddle_ocr_1_5",
           "--tuner_type", "full", "--dataset", str(Path(dataset).resolve()), "--output_dir", str(Path(output).resolve()),
           "--split_dataset_ratio", "0", "--eval_strategy", "no", "--load_best_model_at_end", "false",
           "--loss_type", "sigmoid", "--torch_dtype", dtype,
           "--bf16", str(dtype == "bfloat16").lower(), "--fp16", str(dtype == "float16").lower(),
           "--gradient_checkpointing", "true",
           "--freeze_vit", "false", "--freeze_aligner", "false", "--padding_free", "false",
           # CLI 'delete' maps to template 'raise' in TemplateArguments.
           # 'raise' itself is not a valid CLI choice. Preflight rejects long pairs.
           "--truncation_strategy", "delete", "--strict", "true", "--remove_unused_columns", "false",
           "--report_to", "none"]
    if t.get("paddle_feature_compat", True):
        cmd += ["--external_plugins", str(Path(__file__).resolve().parents[1] / "scripts/swift_paddle_compat.py")]
    for key in ("learning_rate", "beta", "num_train_epochs", "per_device_train_batch_size",
                "gradient_accumulation_steps", "max_length", "max_pixels", "save_steps", "logging_steps"):
        cmd += ["--" + key, str(t[key])]
    if t.get("rpo_alpha") is not None:
        cmd += ["--rpo_alpha", str(t["rpo_alpha"])]
    if t.get("resume_from_checkpoint"):
        cmd += ["--resume_from_checkpoint", str(t["resume_from_checkpoint"])]
    env = {"CUDA_VISIBLE_DEVICES": str(t["gpus"]), "NPROC_PER_NODE": str(t["nproc_per_node"])}
    if int(t["nproc_per_node"]) != len(str(t["gpus"]).split(",")):
        raise ValueError("nproc_per_node must match the number of configured GPUs")
    return cmd, env


def validate_pairs(rows):
    if not rows:
        raise ValueError("empty DPO dataset")
    seen = set()
    for i, row in enumerate(rows):
        if len(row.get("messages", [])) != 2 or [m.get("role") for m in row["messages"]] != ["user", "assistant"]:
            raise ValueError(f"pair {i}: expected user/assistant messages")
        if row["messages"][0]["content"].count("<image>") != 1 or len(row.get("images", [])) != 1:
            raise ValueError(f"pair {i}: expected exactly one image")
        if not isinstance(row.get("rejected_response"), str) or not row["messages"][1]["content"].strip():
            raise ValueError(f"pair {i}: invalid chosen/rejected")
        if row["messages"][1]["content"] == row["rejected_response"]:
            raise ValueError(f"pair {i}: identical chosen and rejected")
        with Image.open(row["images"][0]) as image:
            image.verify()
        h = digest(row)
        if h in seen:
            raise ValueError(f"duplicate preference pair: {i}")
        seen.add(h)


def preflight(rows, config, output, template=None):
    """Encode BOTH branches including visual tokens through the actual template."""
    validate_pairs(rows)
    if template is None:
        if config["training"].get("paddle_feature_compat", True):
            from .paddle_compat import install_patch
            install_patch()
        try:
            from swift.model import get_model_processor
            from swift.template import get_template
        except ImportError as e:
            raise RuntimeError("Install ms-swift 4.x with PaddleOCR-VL-1.6 support in the training environment") from e
        _, processor = get_model_processor(config["training"]["model"], model_type="paddleocr_vl", load_model=False)
        template = get_template(processor, template_type="paddle_ocr_1_5", max_length=None,
                                # Internal template API accepts 'raise', unlike the CLI.
                                max_pixels=config["training"]["max_pixels"], truncation_strategy="raise",
                                remove_unused_columns=False)
        template.set_mode("rlhf")
    valid, errors, lengths = [], [], []
    for i, row in enumerate(rows):
        try:
            encoded = template.encode(row)
            lens = {side: len(encoded[side + "_input_ids"]) for side in ("chosen", "rejected")}
            for side in lens:
                labels = encoded.get(side + "_labels")
                if labels is None or not any(v != -100 for v in labels):
                    raise ValueError(f"no trainable {side} response tokens")
            lengths.append({"index": i, **lens})
            if max(lens.values()) > config["training"]["max_length"]:
                raise ValueError(f"overlength: {lens}")
            valid.append(row)
        except Exception as e:
            errors.append({"index": i, "pair_hash": digest(row), "reason": str(e)})
    write_rows(Path(output, "preflight.lengths.jsonl"), lengths)
    write_rows(Path(output, "preflight.errors.jsonl"), errors)
    if errors:
        raise ValueError(f"{len(errors)} pairs failed preflight; inspect preflight.errors.jsonl, fix/filter explicitly and rerun")
    return lengths


def launch_training(dataset, config, output, dry_run=False, preflight_only=False):
    rows = read_rows(dataset)
    validate_pairs(rows)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    cmd, env = train_command(config, dataset, output / "checkpoints")
    from .io import redact_config
    write_json(output / "launch.json", {"argv": cmd, "environment": env, "dataset_sha256": file_hash(dataset), "config": redact_config(config)})
    if dry_run:
        return {"command": shlex.join(cmd), "environment": env, "length_preflight": "not_run"}
    if not preflight_only:
        probe = subprocess.run([sys.executable, str(Path(__file__).resolve().parents[1] / "scripts/check_training_gpu.py"),
                                "--dtype", config["training"].get("torch_dtype", "bfloat16"),
                                "--expected-devices", str(config["training"]["nproc_per_node"])],
                               env={**os.environ, **env}, capture_output=True, text=True)
        (output / "gpu-check.log").write_text(probe.stdout + probe.stderr)
        if probe.returncode:
            raise RuntimeError(f"GPU/precision check failed; see {output / 'gpu-check.log'}:\n{probe.stdout}{probe.stderr}")
    # Pair export audit binds images to the actual inference inputs.
    audit_path = Path(dataset).with_suffix(".audit.jsonl")
    if audit_path.exists():
        audits = read_rows(audit_path)
        if len(audits) != len(rows):
            raise ValueError("DPO audit length mismatch")
        for row, audit in zip(rows, audits):
            if audit["pair_hash"] != digest(row) or (audit.get("image_hash") and audit["image_hash"] != file_hash(row["images"][0])):
                raise ValueError("DPO data/image changed since export")
    preflight(rows, config, output)
    versions = {}
    for pkg in ("ms-swift", "transformers", "torch", "trl", "peft", "safetensors", "Pillow"):
        try:
            versions[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            versions[pkg] = "missing"
    write_json(output / "environment.preflight.json", {"versions": versions, "status": "template_preflight_passed", "gpu_training_verified": False})
    if preflight_only:
        return {"preflight": "passed", "pairs": len(rows)}
    with (output / "train.log").open("a") as log:
        result = subprocess.run(cmd, env={**os.environ, **env}, stdout=log, stderr=subprocess.STDOUT)
    write_json(output / "training.result.json", {"returncode": result.returncode, "versions": versions})
    if result.returncode:
        raise RuntimeError(f"ms-swift exited {result.returncode}; see {output / 'train.log'}")
    (output / "requirements.verified.txt").write_text("\n".join(f"{k}=={v}" for k, v in versions.items() if v != "missing") + "\n")
    return {"training": "completed", "output": str(output)}
