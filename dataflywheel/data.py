"""Import, provenance preservation, document-level exclusion and deduplication."""
from pathlib import Path

from PIL import Image

from .io import digest, file_hash, read_rows
from .tables import canonical_html, html_to_otsl, table_features
from .sft import adapt_sft, resolve_image

TASKS = {"table", "text", "formula"}


def pixel_hash(path):
    with Image.open(path) as img:
        img = img.convert("RGB")
        import hashlib
        return hashlib.sha256(str(img.size).encode() + img.tobytes()).hexdigest(), img.size


def normalize(row, base, config):
    r = adapt_sft(row)
    for target, source in config["fields"].items():
        if source in row:
            r[target] = row[source]
    if r.get("task") not in TASKS:
        raise ValueError("task must be table/text/formula")
    image = resolve_image(r["image"], base, config["data"])
    phash, size = pixel_hash(image)
    r.update(image=str(image), image_hash=file_hash(image), pixel_hash=phash, image_size=list(size))
    r.setdefault("id", digest([phash, r["task"]])[:20])
    r["id"] = str(r["id"])
    r.setdefault("source", "user")
    r.setdefault("domain", "private")
    if config["mixture"].get("domain") and r["domain"] not in config["mixture"]["domain"]:
        raise ValueError("domain must match mixture.domain (default: private/general)")
    r.setdefault("document_id", r.get("parent_document_id", r["id"]))
    r["document_id"] = str(r["document_id"])
    r.setdefault("split", "train")
    if r["split"] not in {"train", "eval", "val", "test", "validation"}:
        raise ValueError("unsupported split")
    if "split" not in row and config["data"]["eval_fraction"]:
        fraction = config["data"]["eval_fraction"]
        if not 0 <= fraction < 1:
            raise ValueError("eval_fraction must be in [0,1)")
        v = int(digest([config["seed"], r["source"], r["document_id"]])[:8], 16) / 2**32
        r["split"] = "eval" if v < fraction else "train"
    gt, fmt = r.get("gt"), r.get("gt_format")
    if gt is None:
        options = {"table": ["otsl", "html"], "text": ["text"], "formula": ["latex"]}
        for key in options[r["task"]]:
            if r.get(key) is not None:
                gt, fmt = r[key], key
                break
    r["raw_annotation"] = {k: r[k] for k in ("gt", "gt_format", "html", "otsl", "text", "latex") if k in r}
    r["gt"], r["gt_format"] = gt, fmt
    r["conversion"] = {"status": "missing_gt" if gt is None else "ok", "repairs": []}
    r["gt_trusted"] = bool(r.get("gt_trusted", True))
    if gt is None:
        r.update(target=None, score_gt=None)
        return r
    if not isinstance(gt, str) or not gt.strip():
        raise ValueError("GT must be a nonempty string (use null for missing GT)")
    if r["task"] == "table":
        if fmt not in {"otsl", "html"}:
            raise ValueError("table gt_format must be html/otsl")
        canon = canonical_html(gt)
        if r.get("html") is not None and canonical_html(r["html"]) != canon:
            raise ValueError("HTML conflicts with table GT")
        if r.get("otsl") is not None and canonical_html(r["otsl"]) != canon:
            raise ValueError("OTSL conflicts with table GT")
        r.update(score_gt=canon, target=html_to_otsl(canon), table_features=table_features(canon))
        r["otsl"] = r["target"]
        r["conversion"]["normalization"] = "cell topology/text; HTML header/style tags are not encoded in OTSL"
    else:
        expected = "latex" if r["task"] == "formula" else "text"
        if fmt not in {None, expected}:
            raise ValueError(f"expected gt_format={expected}")
        r.update(gt_format=expected, target=gt, score_gt=gt)
    return r


def isolate(rows, excluded=()):
    """Evaluation wins globally, irrespective of input ordering.

    document_id must be shared by pages/crops of a document; hashes cannot infer
    an unknown parent for arbitrary crops. Cross-source parent_id matches are
    deliberately conservative to avoid training leakage.
    """
    forbidden = set()
    heldout = [r for r in rows if r.get("split", "train") != "train"] + list(excluded)
    for r in heldout:
        for key in ("document_id", "parent_document_id", "image_hash", "pixel_hash", "parent_image_hash"):
            if r.get(key):
                kind = "document" if "document" in key else "hash"
                forbidden.add((kind, str(r[key])))
    kept, rejected, seen = [], [], {}
    for r in rows:
        if r.get("split", "train") == "train" and any(
            (("document" if "document" in k else "hash"), str(r[k])) in forbidden
            for k in ("document_id", "parent_document_id", "image_hash", "pixel_hash", "parent_image_hash") if r.get(k)
        ):
            rejected.append({**r, "reason": "evaluation_overlap"})
            continue
        key = r.get("pixel_hash", r.get("image_hash", r["image"]))
        if key in seen:
            previous = seen[key]
            conflict = previous.get("target") != r.get("target") or previous["task"] != r["task"]
            reason = "duplicate_conflicting_gt" if conflict else "duplicate_image"
            rejected.append({**r, "reason": reason, "duplicate_of": previous["id"]})
            if conflict and previous in kept:
                kept.remove(previous)
                rejected.append({**previous, "reason": reason})
            continue
        seen[key] = r
        kept.append(r)
    return kept, rejected


def import_data(path, config):
    path = Path(path).resolve()
    rows, errors, ids = [], [], set()
    raw_rows = read_rows(path)
    excluded = []
    # A bad evaluation annotation must not disable its leakage protection.
    for raw in raw_rows:
        record = dict(raw)
        for target, source in config["fields"].items():
            if source in raw:
                record[target] = raw[source]
        if record.get("split", "train") != "train":
            # Preserve image exclusion even when the SFT annotation is invalid.
            if not record.get("image") and isinstance(record.get("image_info"), list):
                for info in record["image_info"]:
                    if isinstance(info, dict) and isinstance(info.get("image_url"), str):
                        try:
                            ph, _ = pixel_hash(resolve_image(info["image_url"], path.parent, config["data"]))
                            excluded.append({"pixel_hash": ph})
                        except (OSError, ValueError):
                            pass
            if record.get("image"):
                try:
                    record["pixel_hash"], _ = pixel_hash(resolve_image(record["image"], path.parent, config["data"]))
                except (OSError, ValueError):
                    pass
            excluded.append(record)
    for i, raw in enumerate(raw_rows):
        try:
            r = normalize(raw, path.parent, config)
            if r["id"] in ids:
                raise ValueError(f"duplicate id: {r['id']}")
            ids.add(r["id"])
            rows.append(r)
        except (ValueError, OSError, KeyError, TypeError) as e:
            errors.append({"input_index": i, "raw": raw, "reason": f"import_error: {e}"})
    for manifest in config["data"]["exclude_manifests"]:
        for raw in read_rows(manifest):
            if not raw.get("image") and raw.get("image_info"):
                for info in raw["image_info"]:
                    p = resolve_image(info["image_url"], Path(manifest).resolve().parent, config["data"])
                    excluded.append({"pixel_hash": pixel_hash(p)[0]})
            # Normalized exclusion manifests need not have accessible images.
            if not raw.get("pixel_hash") and raw.get("image"):
                p = Path(manifest).resolve().parent / raw["image"]
                raw["pixel_hash"], _ = pixel_hash(p)
            excluded.append(raw)
    rows, leaks = isolate(rows, excluded)
    return rows, errors + leaks
