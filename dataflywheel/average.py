"""Memory-bounded weighted averaging of compatible HF safetensors checkpoints."""
import json
import math
import os
from pathlib import Path
import shutil
import tempfile

from .io import file_hash, write_json


def _config(path):
    config = json.loads((path / "config.json").read_text())
    if config.get("quantization_config"):
        raise ValueError(f"quantized checkpoint is not supported: {path}")
    def clean(value):
        if isinstance(value, dict):
            return {k: clean(v) for k, v in value.items() if k not in {"_name_or_path", "_commit_hash", "transformers_version", "torch_dtype", "dtype"}}
        if isinstance(value, list):
            return [clean(v) for v in value]
        return value
    return clean(config)


def _sidecars(path):
    files = {}
    patterns = ("tokenizer*", "*processor*", "special_tokens_map.json", "added_tokens.json", "vocab*", "merges.txt", "*.model", "chat_template*", "*.py")
    for pattern in patterns:
        for p in path.glob(pattern):
            if p.is_file():
                files[p.name] = file_hash(p)
    if not any(k.startswith("tokenizer") for k in files):
        raise ValueError(f"missing tokenizer files: {path}")
    if not any("processor" in k for k in files):
        raise ValueError(f"missing processor files: {path}")
    return files


def _index(path):
    from safetensors import safe_open
    if (path / "adapter_config.json").exists() or (path / "adapter_model.safetensors").exists():
        raise ValueError("merge adapters into their matching base before averaging")
    index = path / "model.safetensors.index.json"
    if index.exists():
        mapping = json.loads(index.read_text())["weight_map"]
    elif (path / "model.safetensors").exists():
        with safe_open(path / "model.safetensors", framework="pt", device="cpu") as f:
            mapping = {k: "model.safetensors" for k in f.keys()}
    else:
        raise ValueError(f"no HF safetensors model found: {path}")
    if not mapping:
        raise ValueError("empty checkpoint")
    metadata, actual = {}, {}
    for shard in sorted(set(mapping.values())):
        file = (path / shard).resolve()
        if file.parent != path.resolve():
            raise ValueError("shard path must be inside the checkpoint directory")
        with safe_open(file, framework="pt", device="cpu") as f:
            for key in f.keys():
                if key in actual:
                    raise ValueError(f"duplicate tensor in shards: {key}")
                view = f.get_slice(key)
                metadata[key] = (view.get_shape(), view.get_dtype())
                actual[key] = shard
    if actual != mapping:
        raise ValueError("checkpoint weight_map disagrees with shard contents")
    return mapping, metadata


def average_models(models, output, weights=None, dtype="bfloat16", max_shard_bytes=2_000_000_000):
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file
    paths = [Path(p).expanduser().resolve() for p in models]
    output = Path(output).expanduser().resolve()
    if not paths or output.exists() or any(p == output or p in output.parents for p in paths):
        raise ValueError("supply N>=1 inputs and a new output directory outside all inputs")
    weights = [1.0] * len(paths) if weights is None else list(weights)
    if len(weights) != len(paths) or any(not math.isfinite(w) or w < 0 for w in weights) or not sum(weights) > 0:
        raise ValueError("weights must be finite, nonnegative, match N and have positive sum")
    weights = [w / sum(weights) for w in weights]
    if dtype not in {"float32", "float16", "bfloat16"} or max_shard_bytes <= 0:
        raise ValueError("invalid dtype or shard size")
    configs = [_config(p) for p in paths]
    sidecars = [_sidecars(p) for p in paths]
    if any(c != configs[0] for c in configs[1:]):
        raise ValueError("model architecture/configuration mismatch")
    if any(s != sidecars[0] for s in sidecars[1:]):
        raise ValueError("tokenizer/processor/custom-code mismatch")
    indices = [_index(p) for p in paths]
    metadata = indices[0][1]
    supported_dtypes = {"F64", "F32", "F16", "BF16", "I64", "I32", "I16", "I8", "U8", "BOOL", "U16", "U32", "U64"}
    if any(dtype not in supported_dtypes for _, meta in indices for _, dtype in meta.values()):
        raise ValueError("quantized/unsupported tensor dtype; dequantize explicitly before averaging")
    for _, meta in indices[1:]:
        if set(meta) != set(metadata):
            raise ValueError("tensor key mismatch (including tied weight storage)")
        for key in meta:
            if meta[key][0] != metadata[key][0]:
                raise ValueError(f"tensor shape mismatch: {key}")
            float_types = {"F64", "F32", "F16", "BF16"}
            if meta[key][1] != metadata[key][1] and not {meta[key][1], metadata[key][1]} <= float_types:
                raise ValueError(f"tensor dtype mismatch: {key}")
    tied = configs[0].get("tie_word_embeddings", configs[0].get("text_config", {}).get("tie_word_embeddings", False))
    if tied:
        embeddings = [k for k in metadata if k.endswith("embed_tokens.weight")]
        heads = [k for k in metadata if k.endswith("lm_head.weight")]
        if len(embeddings) == 1 and len(heads) == 1:
            for p, (mapping, _) in zip(paths, indices):
                values = []
                for key in (embeddings[0], heads[0]):
                    with safe_open(p / mapping[key], framework="pt", device="cpu") as f:
                        values.append(f.get_tensor(key))
                if not torch.equal(*values):
                    raise ValueError("tied embedding/head tensors disagree")
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".averaging-", dir=output.parent))
    try:
        shards, current, size, total = [], {}, 0, 0
        def flush():
            nonlocal current, size
            if current:
                name = f"part-{len(shards) + 1:05d}.safetensors"
                save_file(current, stage / name, metadata={"format": "pt"})
                shards.append((name, list(current)))
                current, size = {}, 0
        for key in sorted(metadata):
            acc, exact = None, None
            for p, (mapping, _), w in zip(paths, indices, weights):
                with safe_open(p / mapping[key], framework="pt", device="cpu") as f:
                    tensor = f.get_tensor(key)
                if tensor.is_floating_point():
                    if not torch.isfinite(tensor).all():
                        raise ValueError(f"nonfinite input tensor: {key}")
                    if acc is None:
                        acc = torch.zeros_like(tensor, dtype=torch.float32)
                    acc.add_(tensor.float(), alpha=w)
                else:
                    if exact is None:
                        exact = tensor.clone()
                    elif not torch.equal(exact, tensor):
                        raise ValueError(f"nonfloating buffer mismatch: {key}")
            value = acc.to(getattr(torch, dtype)) if acc is not None else exact
            if value.is_floating_point() and not torch.isfinite(value).all():
                raise ValueError(f"output overflow: {key}")
            nbytes = value.numel() * value.element_size()
            if size and size + nbytes > max_shard_bytes:
                flush()
            current[key] = value.contiguous()
            size += nbytes
            total += nbytes
        flush()
        mapping = {}
        for i, (name, keys) in enumerate(shards, 1):
            final = "model.safetensors" if len(shards) == 1 else f"model-{i:05d}-of-{len(shards):05d}.safetensors"
            os.rename(stage / name, stage / final)
            mapping.update({key: final for key in keys})
        if len(shards) > 1:
            write_json(stage / "model.safetensors.index.json", {"metadata": {"total_size": total}, "weight_map": mapping})
        for name in [*sidecars[0], "generation_config.json"]:
            if (paths[0] / name).is_file():
                shutil.copy2(paths[0] / name, stage / name)
        config = json.loads((paths[0] / "config.json").read_text())
        config["torch_dtype"] = dtype
        if "dtype" in config:
            config["dtype"] = dtype
        write_json(stage / "config.json", config)
        manifest = {"models": [{"path": str(p), "weight": w, "shards": {s: file_hash(p / s) for s in sorted(set(index[0].values()))}}
                               for p, w, index in zip(paths, weights, indices)],
                    "dtype": dtype, "accumulation": "float32", "tensors": len(mapping),
                    "sidecar_source": str(paths[0]), "optimizer_averaged": False}
        write_json(stage / "fusion.json", manifest)
        # Read back every shard header before publishing atomically.
        _index(stage)
        os.rename(stage, output)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return manifest
