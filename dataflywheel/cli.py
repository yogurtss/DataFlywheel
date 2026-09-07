"""CLI commands share the same JSON artifacts; no hidden training side effects."""
import argparse
import asyncio
import json
from pathlib import Path
import sys

from .io import load_config, merge, read_rows, write_json, write_rows


def export_pairs(selected, pending, stats, output):
    output = Path(output)
    write_rows(output, [r["pair"] for r in selected])
    write_rows(output.with_suffix(".audit.jsonl"), [r["pair_audit"] for r in selected])
    write_rows(output.with_suffix(".selected.jsonl"), selected)
    write_rows(output.with_suffix(".pending.jsonl"), pending)
    write_json(output.with_suffix(".stats.json"), stats)


def execute(args, config):
    if args.command == "synth-agent":
        from .synthesis_agent import run_agent
        return run_agent(args.input, args.output, config, args.font)
    if args.command == "synth-template":
        from .synthesis import make_templates
        return make_templates(args.input, args.output)
    if args.command == "synth-render":
        from .synthesis import render_templates
        return render_templates(args.input, args.output, args.font, args.variants, args.seed)
    if args.command == "synth-check":
        from .synthesis import check_records
        return check_records(args.input, args.output)
    if args.command == "average":
        from .average import average_models
        return average_models(args.models, args.output, args.weights, args.dtype, int(args.shard_mb * 1_000_000))
    if args.command == "train":
        from .training import launch_training
        return launch_training(args.input, config, args.output, args.dry_run, args.preflight_only)
    if args.command == "parse-pages":
        from .pages import parse_pages
        return parse_pages(args.input, args.output, config, args.role)
    if args.command == "convert":
        from .data import import_data
        if Path(args.input).resolve() == Path(args.output).resolve():
            raise ValueError("convert output must differ from input")
        rows, errors = import_data(args.input, config)
        write_rows(args.output, rows)
        write_rows(Path(args.output).with_suffix(".errors.jsonl"), errors)
        return {"accepted": len(rows), "quarantined": len(errors)}
    if args.command == "run":
        from .data import import_data
        from .inference import infer_rows
        from .metrics import score_rows
        from .mining import build_pairs, mine_rows
        from .report import create_report
        root = Path(args.output)
        root.mkdir(parents=True, exist_ok=True)
        from .io import redact_config
        write_json(root / "config.resolved.json", redact_config(config))
        rows, errors = import_data(args.input, config)
        write_rows(root / "normalized.jsonl", rows)
        write_rows(root / "import.errors.jsonl", errors)
        if not rows:
            raise ValueError("no valid inputs; inspect import.errors.jsonl")
        print("[1/6] 双模型确定性推理", file=sys.stderr)
        # Fail on missing CDM before spending inference budget.
        from .metrics import MetricEngine
        MetricEngine(config).preflight({r["task"] for r in rows})
        inferred = asyncio.run(infer_rows(rows, config, root / "cache"))
        write_rows(root / "inferred.jsonl", inferred)
        for role in ("base", "sft"):
            if not any(c.get("status") == "ok" for r in inferred for c in r.get("candidates", []) if c["role"] == role):
                raise RuntimeError(f"all {role} requests failed; inspect inferred.jsonl and endpoint configuration")
        print("[2/6] 评分与候选筛选", file=sys.stderr)
        scored = score_rows(inferred, config)
        write_rows(root / "scored.jsonl", scored)
        # Extra shortlist capacity allows meaningful pairs to replace flat cases.
        shortlist_config = merge(config, {"mining": {"budget": config["mining"]["budget"] * 3}})
        shortlist, skipped, mining_stats = mine_rows(scored, shortlist_config)
        write_rows(root / "shortlist.jsonl", shortlist)
        write_json(root / "shortlist.stats.json", mining_stats)
        print("[3/6] 入选样本追加 SFT 采样", file=sys.stderr)
        sampled = asyncio.run(infer_rows(shortlist, config, root / "cache", sampling=True))
        write_rows(root / "sampled.jsonl", sampled)
        print("[4/6] 候选评分与偏好构造", file=sys.stderr)
        scored_samples = score_rows(sampled, config)
        write_rows(root / "sampled.scored.jsonl", scored_samples)
        # Reclassify after sampling, retain shortlist capacity until valid pairs exist.
        remined, rem_pending, rem_stats = mine_rows(scored_samples, shortlist_config)
        write_json(root / "mining.stats.json", rem_stats)
        selected, pending, stats = build_pairs(remined, config)
        pending += skipped + rem_pending + errors
        export_pairs(selected, pending, stats, root / "dpo.jsonl")
        print("[5/6] 离线报告", file=sys.stderr)
        create_report(selected, root / "report", config, before=scored, pending=pending)
        print("[6/6] 完成；训练通过 train 独立启动", file=sys.stderr)
        return {"input": len(rows), "pairs": len(selected), "output": str(root.resolve()),
                "report": str((root / "report/index.html").resolve())}
    rows = read_rows(args.input)
    if args.command == "infer":
        from .inference import infer_rows
        result = asyncio.run(infer_rows(rows, config, args.cache, args.sampling))
        write_rows(args.output, result)
        return {"rows": len(result)}
    if args.command == "score":
        from .metrics import score_rows
        result = score_rows(rows, config)
        write_rows(args.output, result)
        return {"rows": len(result)}
    if args.command == "mine":
        from .mining import mine_rows
        selected, pending, stats = mine_rows(rows, config)
        write_rows(args.output, selected)
        write_rows(Path(args.output).with_suffix(".pending.jsonl"), pending)
        write_json(Path(args.output).with_suffix(".stats.json"), stats)
        return stats
    if args.command == "build-dpo":
        from .mining import build_pairs
        if args.mode:
            config = merge(config, {"pairs": {"mode": args.mode}})
        selected, pending, stats = build_pairs(rows, config)
        export_pairs(selected, pending, stats, args.output)
        return stats
    if args.command == "report":
        from .report import create_report
        return create_report(rows, args.output, config,
                             read_rows(args.before) if args.before else None,
                             read_rows(args.pending) if args.pending else None)


def parser():
    p = argparse.ArgumentParser(prog="dataflywheel", description="PaddleOCR-VL-1.6 数据飞轮")
    sub = p.add_subparsers(dest="command", required=True)
    s = sub.add_parser("synth-agent")
    s.add_argument("--input", "-i", required=True)
    s.add_argument("--output", "-o", required=True)
    s.add_argument("--config", "-c", required=True)
    s.add_argument("--font", required=True)
    for name in ("synth-template", "synth-render", "synth-check"):
        s = sub.add_parser(name)
        s.add_argument("--input", "-i", required=name != "synth-template")
        s.add_argument("--output", "-o", required=True)
        if name == "synth-render":
            s.add_argument("--font", required=True, help="Local font covering all generated characters")
            s.add_argument("--variants", type=int, default=2)
            s.add_argument("--seed", type=int, default=42)
    for name in ("convert", "infer", "score", "mine", "build-dpo", "report", "run", "train", "parse-pages"):
        s = sub.add_parser(name)
        s.add_argument("--config", "-c")
        s.add_argument("--input", "-i", required=True)
        s.add_argument("--output", "-o", required=True)
        if name == "infer":
            s.add_argument("--cache", default="runs/cache")
            s.add_argument("--sampling", action="store_true")
        if name == "build-dpo":
            s.add_argument("--mode", choices=["model_pair", "gt_pair", "hybrid"])
        if name == "report":
            s.add_argument("--before")
            s.add_argument("--pending")
        if name == "train":
            flags = s.add_mutually_exclusive_group()
            flags.add_argument("--dry-run", action="store_true")
            flags.add_argument("--preflight-only", action="store_true")
        if name == "parse-pages":
            s.add_argument("--role", choices=["base", "sft"], default="sft")
    s = sub.add_parser("average")
    s.add_argument("--models", nargs="+", required=True)
    s.add_argument("--weights", nargs="+", type=float)
    s.add_argument("--output", "-o", required=True)
    s.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="bfloat16")
    s.add_argument("--shard-mb", type=float, default=2000)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        result = execute(args, load_config(getattr(args, "config", None)))
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except (ValueError, RuntimeError, OSError, KeyError, ImportError) as e:
        print(f"dataflywheel: {e}", file=sys.stderr)
        raise SystemExit(2) from e
