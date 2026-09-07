"""Bounded asynchronous inference with immutable request-addressed cache."""
import asyncio
import base64
import io
import json
import os
from pathlib import Path
import random

import httpx
from PIL import Image

from .io import digest, file_hash, write_json


def prepare_image(row, preprocess, cache_dir):
    if file_hash(row["image"]) != row["image_hash"]:
        raise ValueError(f"image changed since import: {row['id']}")
    key = digest([row["image_hash"], preprocess])
    path = Path(cache_dir, "images", key + ".png")
    if not path.exists():
        with Image.open(row["image"]) as im:
            im = im.convert("RGB")
            if preprocess.get("max_side"):
                im.thumbnail((preprocess["max_side"], preprocess["max_side"]), Image.Resampling.LANCZOS)
            buf = io.BytesIO()
            im.save(buf, format="PNG")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(buf.getvalue())
    return path.resolve()


async def infer_rows(rows, config, cache_dir, sampling=False):
    opts = config["inference"]
    roles = ["sft"] if sampling else ["base", "sft"]
    for role in roles:
        endpoint = config.get("models", {}).get(role, {})
        for key in ("base_url", "model", "revision"):
            if not endpoint.get(key):
                raise ValueError(f"models.{role}.{key} is required (revision must change when served weights change)")
    sem = asyncio.Semaphore(opts["concurrency"])
    out = []
    async with httpx.AsyncClient(timeout=opts["timeout"], trust_env=False) as client:
        async def request(row, role, index, image):
            endpoint = config["models"][role]
            prompt = config["prompts"][row["task"]]
            params = {"temperature": opts["temperature"] if sampling else 0,
                      "max_tokens": opts["max_tokens"], "top_p": opts["top_p"] if sampling else 1.0,
                      "seed": config["seed"] + index}
            identity = {"image": row["image_hash"], "prepared_image": file_hash(image),
                        "model": endpoint["model"], "revision": endpoint["revision"],
                        "endpoint": endpoint["base_url"], "prompt": prompt, "params": params,
                        "preprocess": opts["preprocess"], "extra_body": endpoint.get("extra_body", {}),
                        "kind": "sample" if sampling else "greedy", "index": index}
            key = digest(identity)
            path = Path(cache_dir, "responses", key + ".json")
            common = {"id": key, "role": role, "kind": identity["kind"], "index": index,
                      "prompt": prompt, "request": identity, "training_image": str(image)}
            if path.exists():
                cached = json.loads(path.read_text())
                if cached.get("status") == "ok":
                    return {**cached, **common}
            extra = endpoint.get("extra_body", {})
            if set(extra) & {"messages", "model", *params.keys()}:
                raise ValueError("extra_body cannot override messages/model/generation parameters")
            body = {"model": endpoint["model"], **params, **extra,
                    "messages": [{"role": "user", "content": [
                        {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(image.read_bytes()).decode()}},
                        {"type": "text", "text": prompt}]}]}
            headers = {}
            if endpoint.get("api_key_env"):
                key_value = os.environ.get(endpoint["api_key_env"])
                if not key_value:
                    raise ValueError(f"missing credential environment variable: {endpoint['api_key_env']}")
                headers["Authorization"] = "Bearer " + key_value
            error = "unknown error"
            for attempt in range(opts["retries"] + 1):
                try:
                    async with sem:
                        response = await client.post(endpoint["base_url"].rstrip("/") + "/chat/completions", json=body, headers=headers)
                    response.raise_for_status()
                    raw = response.json()
                    choice = raw["choices"][0]
                    content = choice["message"]["content"]
                    if not isinstance(content, str):
                        raise ValueError("response content must be a string")
                    result = {**common, "status": "ok", "text": content,
                              "finish_reason": choice.get("finish_reason"), "raw_response": raw}
                    write_json(path, result)
                    return result
                except (httpx.HTTPError, ValueError, KeyError, IndexError) as e:
                    error = f"{type(e).__name__}: {e}"
                    if isinstance(e, httpx.HTTPStatusError) and e.response.status_code < 500 and e.response.status_code not in (408, 429):
                        break
                    if attempt < opts["retries"]:
                        await asyncio.sleep(min(2**attempt, 8))
            result = {**common, "status": "transport_error", "text": "", "finish_reason": None, "error": error}
            write_json(path, result)
            return result

        # Batches bound task count and encoded image memory, not just HTTP sockets.
        for start in range(0, len(rows), opts["concurrency"]):
            batch = []
            for row in rows[start:start + opts["concurrency"]]:
                r = dict(row)
                image = prepare_image(r, opts["preprocess"], cache_dir)
                r["training_image"] = str(image)
                r["training_image_hash"] = file_hash(image)
                n = opts["samples"] if sampling else 1
                batch.append((r, [request(r, role, i, image) for role in roles for i in range(n)]))
            results = await asyncio.gather(*(asyncio.gather(*jobs) for _, jobs in batch))
            for (r, _), candidates in zip(batch, results):
                existing = [c for c in r.get("candidates", []) if c.get("kind") == "greedy"] if sampling else []
                if sampling:
                    for c in existing:
                        endpoint = config["models"].get(c["role"], {})
                        if (c.get("training_image") != r["training_image"] or c.get("prompt") != config["prompts"][r["task"]]
                            or c.get("request", {}).get("revision") != endpoint.get("revision")
                            or c.get("request", {}).get("model") != endpoint.get("model")):
                            raise ValueError("sampling model/preprocessing/prompt changed; rerun greedy inference")
                r["candidates"] = existing + candidates
                out.append(r)
    return out
