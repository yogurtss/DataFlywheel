"""Official full-page PaddleOCR pipeline export for external OmniDocBench."""
import importlib.metadata
import json
import os
from pathlib import Path
import tempfile

from .io import atomic_text, digest, file_hash, read_rows, write_json


def parse_pages(input_path, output, config, role="sft", pipeline_factory=None):
    if pipeline_factory is None:
        try:
            from paddleocr import PaddleOCRVL
        except ImportError as e:
            raise RuntimeError("Install PaddleOCR-VL-1.6 official page pipeline in its own environment") from e
        pipeline_factory = PaddleOCRVL
    endpoint = config["models"][role]
    for key in ("base_url", "model", "revision"):
        if not endpoint.get(key):
            raise ValueError(f"models.{role}.{key} is required")
    output = Path(output).resolve()
    rows = read_rows(input_path)
    images, names = [], set()
    for row in rows:
        image = (Path(input_path).resolve().parent / row["image"]).resolve()
        # Input consists of already-rendered page images matching benchmark filenames.
        if image.suffix.lower() not in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}:
            raise ValueError("parse-pages expects one page image per row, not PDFs")
        if image.stem in names:
            raise ValueError(f"duplicate page basename: {image.stem}")
        names.add(image.stem)
        images.append(image)
    options = dict(config["pages"].get("options", {}))
    if any(k.startswith("vl_rec_") or k == "pipeline_version" for k in options):
        raise ValueError("pages.options cannot override VLM endpoint or pipeline version")
    constructor = dict(pipeline_version=config["pages"]["pipeline_version"],
                       vl_rec_backend=config["pages"]["backend"], vl_rec_server_url=endpoint["base_url"],
                       vl_rec_api_model_name=endpoint["model"], **options)
    if endpoint.get("api_key_env"):
        constructor["vl_rec_api_key"] = os.environ[endpoint["api_key_env"]]
    pipeline = pipeline_factory(**constructor)
    identity = {"endpoint": {k: endpoint[k] for k in ("base_url", "model", "revision")},
                "pages_config": config["pages"], "role": role}
    for pkg in ("paddleocr", "paddlex"):
        try:
            identity[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            identity[pkg] = "unavailable"
    manifest = {**identity, "status": "running", "expected_pages": len(images), "pages": []}
    output.mkdir(parents=True, exist_ok=True)
    stale = {p.stem for p in (output / "markdown").glob("*.md")} - names
    if stale:
        raise ValueError("output contains pages outside this input; use a new output directory")
    write_json(output / "manifest.json", manifest)
    for image in images:
        key = digest([identity, file_hash(image)])
        stamp = output / "status" / (image.stem + ".json")
        md = output / "markdown" / (image.stem + ".md")
        if stamp.exists() and md.exists():
            old = json.loads(stamp.read_text())
            if old.get("key") == key and old.get("markdown_hash") == file_hash(md):
                manifest["pages"].append(old)
                continue
        results = list(pipeline.predict(str(image)))
        if len(results) != 1:
            raise ValueError(f"expected one result per page: {image}")
        result = results[0]
        md.parent.mkdir(parents=True, exist_ok=True)
        (output / "json").mkdir(exist_ok=True)
        result.save_to_json(str(output / "json" / (image.stem + ".json")))
        # Save via official exporter to preserve image assets and its serialization.
        with tempfile.TemporaryDirectory(prefix="page-", dir=output) as tmp:
            result.save_to_markdown(tmp)
            files = list(Path(tmp).glob("*.md"))
            if len(files) != 1:
                raise ValueError("official Markdown exporter did not emit exactly one page")
            import shutil
            assetdir = output / "markdown" / "assets" / image.stem
            assetdir.mkdir(parents=True, exist_ok=True)
            text = files[0].read_text()
            for asset in Path(tmp).rglob("*"):
                if asset.is_file() and asset != files[0]:
                    rel = asset.relative_to(tmp)
                    dest = assetdir / rel
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(asset, dest)
                    text = text.replace(str(rel), f"assets/{image.stem}/{rel.as_posix()}")
            atomic_text(md, text)
        record = {"image": str(image), "key": key, "markdown": str(md), "markdown_hash": file_hash(md)}
        write_json(stamp, record)
        manifest["pages"].append(record)
        write_json(output / "manifest.json", manifest)
    manifest["status"] = "complete"
    write_json(output / "manifest.json", manifest)
    return {"pages": len(images), "markdown_dir": str(output / "markdown")}
