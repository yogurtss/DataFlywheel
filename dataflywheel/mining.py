"""Reproducible stratified selection and preference construction."""
from collections import Counter, defaultdict
import math
import random

from .data import isolate
from .io import digest
from .metrics import equivalent
from .tables import parse_otsl


def apportion(total, ratios):
    raw = {k: total * v / sum(ratios.values()) for k, v in ratios.items()}
    result = {k: math.floor(v) for k, v in raw.items()}
    for k in sorted(raw, key=lambda k: (-(raw[k] - result[k]), str(k)))[:total - sum(result.values())]:
        result[k] += 1
    return result


def stratify(rows, budget, config, shuffle=False, bucketed=False):
    """Optional domain/task quotas; deterministic no-replacement deficit backfill."""
    domains, tasks = config["mixture"].get("domain"), config["mixture"].get("task")
    ratios = {f"{d}/{t}": dv * tv for d, dv in (domains or {"*": 1}).items()
              for t, tv in (tasks or {"*": 1}).items()}
    if bucketed:
        ratios = {f"{key}/{b}": v * bv for key, v in ratios.items() for b, bv in config["mining"]["buckets"].items()}
    def category(r):
        key = f"{r['domain'] if domains else '*'}/{r['task'] if tasks else '*'}"
        return key + "/" + r["mining_bucket"] if bucketed else key
    desired = apportion(budget, ratios)
    ordered = list(rows)
    if shuffle:
        random.Random(config["seed"]).shuffle(ordered)
    else:
        def order_key(r):
            if bucketed and r.get("mining_bucket") == "coverage":
                return (-int(digest([config["seed"], r["id"]])[:8], 16) / 2**32, str(r["id"]))
            return (-r.get("signals", {}).get("priority", 0), str(r["id"]))
        ordered.sort(key=order_key)
    selected, rest, counts = [], [], Counter()
    for r in ordered:
        key = category(r)
        if counts[key] < desired.get(key, 0):
            selected.append(r)
            counts[key] += 1
        else:
            rest.append(r)
    gaps = {k: n - counts[k] for k, n in desired.items() if n > counts[k]}
    selected += rest[:max(0, budget - len(selected))]
    actual = Counter(category(r) for r in selected)
    return selected, {"requested": budget, "desired": desired, "actual": dict(actual),
                      "task_counts": dict(Counter(r['task'] for r in selected)),
                      "domain_counts": dict(Counter(r['domain'] for r in selected)),
                      "task_quotas_enabled": bool(tasks), "domain_quotas_enabled": bool(domains),
                      "quota_shortfall_before_backfill": gaps, "unfilled": max(0, budget - len(selected))}


def quantile(values, q):
    values = sorted(values)
    if not values:
        return 0.0
    position = (len(values) - 1) * q
    lo, hi = math.floor(position), math.ceil(position)
    return values[lo] + (values[hi] - values[lo]) * (position - lo)


def mine_rows(rows, config):
    clean, exclusions = isolate(rows)
    eligible = [r for r in clean if r.get("split", "train") == "train" and r.get("score_gt") is not None and r.get("gt_trusted", True)]
    thresholds = {}
    for task in ("table", "text", "formula"):
        errors = [r["signals"]["gt_error"] for r in eligible if r["task"] == task and r.get("signals", {}).get("gt_error") is not None]
        thresholds[task] = [quantile(errors, q) for q in config["mining"]["middle_quantiles"]]
    buckets = defaultdict(list)
    assigned = {}
    for r in eligible:
        s = r.get("signals", {})
        if s.get("sft_score") is None:
            exclusions.append({**r, "reason": "missing_or_failed_sft_score"})
            continue
        lo, hi = thresholds[r["task"]]
        if (s.get("regression") or 0) >= config["mining"]["regression_threshold"]:
            bucket = "regression"
        elif lo <= s["gt_error"] <= hi and s.get("candidate_gap", 0) >= config["pairs"]["margin"]:
            bucket = "learnable"
        else:
            bucket = "coverage"
        r = {**r, "mining_bucket": bucket}
        assigned[r["id"]] = bucket
        buckets[bucket].append(r)
    allocation = apportion(config["mining"]["budget"], config["mining"]["buckets"])
    chosen, reports = [], {}
    for bucket, count in allocation.items():
        selected, report = stratify(buckets[bucket], count, config, shuffle=bucket == "coverage")
        chosen.extend(selected)
        reports[bucket] = report
    used = {r["id"] for r in chosen}
    remainder = [r for group in buckets.values() for r in group if r["id"] not in used]
    backfill, backfill_report = stratify(remainder, max(0, config["mining"]["budget"] - len(chosen)), config)
    chosen += [{**r, "mining_backfill": True} for r in backfill]
    ids = {r["id"] for r in chosen}
    excluded_ids = {r.get("id") for r in exclusions}
    for r in clean:
        if r["id"] in ids or r["id"] in excluded_ids:
            continue
        reason = "not_selected_budget"
        if r.get("split", "train") != "train":
            reason = "held_out"
        elif r.get("score_gt") is None:
            reason = "missing_gt_disagreement_only"
        elif not r.get("gt_trusted", True):
            reason = "untrusted_gt"
        exclusions.append({**r, "reason": reason})
    return chosen, exclusions, {"buckets": reports, "backfill": backfill_report, "middle_thresholds": thresholds,
                               "selected": len(chosen), "input": len(rows)}


def make_pair(row, config):
    if row.get("split", "train") != "train":
        return None, "held_out"
    if row.get("target") is None:
        return None, "missing_gt"
    if not row.get("gt_trusted", True):
        return None, "untrusted_gt"
    candidates = []
    for c in row.get("candidates", []):
        q = c.get("quality", {})
        if c.get("status") == "ok" and c.get("finish_reason") in {"stop", "eos"} and q.get("score") is not None and q.get("status") in {"ok", "invalid_prediction"}:
            candidates.append(c)
    if not candidates:
        return None, "no_scored_candidates"
    prompt = config["prompts"][row["task"]]
    if any(c.get("prompt", prompt) != prompt for c in candidates):
        return None, "prompt_mismatch"
    if any(c.get("training_image", row.get("training_image", row["image"])) != row.get("training_image", row["image"]) for c in candidates):
        return None, "image_mismatch"
    threshold = config["pairs"]["min_quality"][row["task"]]
    def positive_format(c):
        if row["task"] != "table":
            return True
        try:
            parse_otsl(c["text"])
            return True
        except ValueError:
            return False
    valid = [c for c in candidates if c["quality"]["status"] == "ok" and c["quality"]["score"] >= threshold and positive_format(c)]
    valid.sort(key=lambda c: (-c["quality"]["score"], c["role"] != "sft", c["id"]))
    mode = config["pairs"]["mode"]
    positive = valid[0] if mode != "gt_pair" and valid else None
    if positive is None and mode in {"hybrid", "gt_pair"}:
        positive = {"id": "gt", "role": "gt", "text": row["target"], "quality": {"score": 1.0, "status": "ok"}}
    if positive is None:
        return None, "no_reliable_model_positive"
    margin = config["pairs"]["margin"]
    rejected = [c for c in candidates if positive["quality"]["score"] - c["quality"]["score"] >= margin - 1e-10
                and not equivalent(row["task"], positive["text"], c["text"])]
    if not rejected:
        return None, "no_meaningful_preference_gap"
    rejected.sort(key=lambda c: (c["role"] != "sft", -c["quality"]["score"], c["id"]))
    negative = rejected[0]
    pair = {"messages": [{"role": "user", "content": "<image>" + prompt},
                         {"role": "assistant", "content": positive["text"]}],
            "images": [row.get("training_image", row["image"])], "rejected_response": negative["text"]}
    audit = {"sample_id": row["id"], "chosen_id": positive["id"], "rejected_id": negative["id"],
             "chosen_source": positive["role"], "rejected_source": negative["role"],
             "chosen_score": positive["quality"]["score"], "rejected_score": negative["quality"]["score"],
             "gap": positive["quality"]["score"] - negative["quality"]["score"],
             "mode": mode, "pair_hash": digest(pair), "metric_config": row.get("metric_config"),
             "source": row.get("source"), "document_id": row.get("document_id"),
             "task": row["task"], "domain": row["domain"], "signals": row.get("signals"),
             "mining_bucket": row.get("mining_bucket"), "image_hash": row.get("training_image_hash", row.get("image_hash"))}
    return {**row, "pair": pair, "pair_audit": audit}, None


def build_pairs(rows, config):
    rows, pending = isolate(rows)
    available = []
    for row in rows:
        result, reason = make_pair(row, config)
        if result is None:
            pending.append({**row, "reason": reason})
        else:
            available.append(result)
    selected, report = stratify(available, config["mining"]["budget"], config,
                                bucketed=bool(available) and all("mining_bucket" in r for r in available))
    ids = {r["id"] for r in selected}
    pending += [{**r, "reason": "pair_quota_budget"} for r in available if r["id"] not in ids]
    return selected, pending, report
