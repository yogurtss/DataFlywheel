"""Task scores, with metric failures distinct from recognition errors."""
from dataclasses import dataclass, field
import json
import math
from pathlib import Path
import re
import subprocess
import tempfile
import unicodedata

from apted import APTED, Config
from lxml import html as lh
from rapidfuzz.distance import Levenshtein

from .io import digest
from .tables import canonical_html, cell_text


def normalize_text(s):
    return unicodedata.normalize("NFC", s).replace("\r\n", "\n").strip()


def normalize_formula(s):
    s = normalize_text(s)
    for left, right in [("$$", "$$"), (r"\[", r"\]"), (r"\(", r"\)"), ("$", "$")]:
        if s.startswith(left) and s.endswith(right):
            s = s[len(left):-len(right)].strip()
            break
    return s


def normalized_distance(a, b):
    return Levenshtein.distance(a, b) / max(len(a), len(b), 1)


@dataclass
class Node:
    tag: str
    colspan: int = 1
    rowspan: int = 1
    content: list = field(default_factory=list)
    children: list = field(default_factory=list)


class TableCost(Config):
    """TEDS edit costs from IBM PubTabNet (Apache-2.0; see THIRD_PARTY.md)."""
    def rename(self, a, b):
        if (a.tag, a.colspan, a.rowspan) != (b.tag, b.colspan, b.rowspan):
            return 1.0
        return normalized_distance(a.content, b.content) if a.tag == "td" else 0.0


def teds(pred, gt, structure_only=False):
    def tokens(n):
        out = [f"<{n.tag}>"] + list(n.text or "")
        for c in n:
            out += tokens(c)
        if n.tag != "unk":
            out.append(f"</{n.tag}>")
        if n.tag != "td":
            out += list(n.tail or "")
        return out

    def tree(n):
        if n.tag == "td":
            return Node(n.tag, int(n.get("colspan", 1)), int(n.get("rowspan", 1)),
                        [] if structure_only else tokens(n)[1:-1])
        return Node(n.tag, children=[tree(c) for c in n])
    p, g = lh.fromstring(pred), lh.fromstring(gt)
    denom = max(len(p.xpath(".//*")), len(g.xpath(".//*")), 1)
    distance = APTED(tree(p), tree(g), TableCost()).compute_edit_distance()
    return max(0.0, min(1.0, 1 - float(distance) / denom))


class MetricEngine:
    def __init__(self, config):
        self.config = config["metrics"]
        self.cache = {}
        self.cdm_ready = False

    def _cdm(self, gt, pred):
        repo = self.config.get("cdm_repo")
        if not repo or not Path(repo, "src/metrics/cdm/cdm.py").is_file():
            raise RuntimeError("CDM requires metrics.cdm_repo pointing to OmniDocBench with src/metrics/cdm/cdm.py; explicitly set formula: fast for proxy scoring")
        worker = Path(__file__).with_name("cdm_worker.py")
        with tempfile.TemporaryDirectory(prefix="flywheel-cdm-") as tmp:
            inp, out = Path(tmp, "in.json"), Path(tmp, "out.json")
            inp.write_text(json.dumps({"gt": gt, "pred": pred}))
            result = subprocess.run([self.config["cdm_python"], str(worker), str(Path(repo).resolve()),
                                     str(inp), str(out)], capture_output=True, text=True,
                                    timeout=self.config["timeout"])
            if result.returncode or not out.exists():
                raise RuntimeError("CDM worker failed: " + result.stderr[-1500:])
            data = json.loads(out.read_text())
        if data.get("error"):
            raise RuntimeError(data["error"])
        score = float(data["F1_score"])
        if not math.isfinite(score) or not 0 <= score <= 1 or data.get("gt_tokens", 0) <= 0:
            raise RuntimeError("CDM invalid GT/render result")
        if data.get("pred_tokens", 0) <= 0 and pred.strip():
            # Upstream may silently return zero after a rendering failure.
            raise RuntimeError("CDM produced no prediction tokens; cannot distinguish render failure from recognition error")
        return data

    def preflight(self, tasks):
        if "formula" in tasks and self.config["formula"] == "cdm" and not self.cdm_ready:
            result = self._cdm("x+1", "x+1")
            if result["F1_score"] < 0.99:
                raise RuntimeError("CDM self-test failed")
            self.cdm_ready = True

    def compare(self, task, pred, gt):
        key = digest([task, pred, gt, self.config])
        if key in self.cache:
            return dict(self.cache[key])
        try:
            result = self._compare(task, pred, gt)
        except (RuntimeError, subprocess.SubprocessError, OSError, KeyError, ValueError, TypeError) as e:
            result = {"score": None, "status": "metric_error", "error": str(e)}
        self.cache[key] = result
        return dict(result)

    def _compare(self, task, pred, gt):
        if task == "table":
            try:
                g = canonical_html(gt)
            except ValueError as e:
                return {"score": None, "status": "invalid_gt", "error": str(e)}
            try:
                p = canonical_html(pred)
            except ValueError as e:
                return {"score": 0.0, "status": "invalid_prediction", "error": str(e), "metric": "teds"}
            ptext = "\n".join(cell_text(n) for n in lh.fromstring(p).xpath(".//td"))
            gtext = "\n".join(cell_text(n) for n in lh.fromstring(g).xpath(".//td"))
            return {"score": teds(p, g), "teds_s": teds(p, g, True), "metric": "teds",
                    "cell_text_ned": normalized_distance(ptext, gtext), "status": "ok", "exact": p == g}
        p, g = (normalize_formula(pred), normalize_formula(gt)) if task == "formula" else (normalize_text(pred), normalize_text(gt))
        if not g:
            return {"score": None, "status": "invalid_gt", "error": "empty GT"}
        ned = normalized_distance(p, g)
        result = {"score": 1 - ned, "ned": ned, "exact": p == g, "status": "ok", "metric": "text_ned"}
        if task == "text":
            result["cer"] = Levenshtein.distance(p, g) / len(g)
            # WER is meaningful only with whitespace-delimited words.
            result["wer"] = Levenshtein.distance(p.split(), g.split()) / len(g.split()) if " " in g else None
        else:
            result["metric"] = "formula_edit_proxy"
            result["render_status"] = "not_checked"
            if self.config["formula"] == "cdm":
                if not p:
                    result.update(score=0.0, metric="cdm", render_status="empty_prediction")
                else:
                    cdm = self._cdm(g, p)
                    result.update(score=cdm["F1_score"], metric="cdm", cdm=cdm, render_status="ok")
        return result


def equivalent(task, a, b):
    if task == "table":
        try:
            return canonical_html(a) == canonical_html(b)
        except ValueError:
            return a == b
    normalize = normalize_formula if task == "formula" else normalize_text
    return normalize(a) == normalize(b)


def score_rows(rows, config):
    engine = MetricEngine(config)
    engine.preflight({r["task"] for r in rows})
    out = []
    for row in rows:
        r = dict(row)
        candidates = []
        for c in r.get("candidates", []):
            c = dict(c)
            if c.get("status") != "ok" or c.get("finish_reason") not in {"stop", "eos"}:
                c["quality"] = {"score": None, "status": "generation_error", "error": c.get("error", c.get("finish_reason"))}
            elif r.get("score_gt") is None:
                c["quality"] = {"score": None, "status": "missing_gt"}
            else:
                c["quality"] = engine.compare(r["task"], c["text"], r["score_gt"])
            candidates.append(c)
        r["candidates"] = candidates
        base = next((c for c in candidates if c["role"] == "base" and c["kind"] == "greedy"), None)
        sft = next((c for c in candidates if c["role"] == "sft" and c["kind"] == "greedy"), None)
        b = base["quality"]["score"] if base else None
        s = sft["quality"]["score"] if sft else None
        signals = {"base_score": b, "sft_score": s, "gt_error": 1 - s if s is not None else None,
                   "regression": max(b - s, 0) if b is not None and s is not None else None,
                   "disagreement": None}
        if base and sft and all(c["status"] == "ok" and c["finish_reason"] in {"stop", "eos"} for c in (base, sft)):
            forward = engine.compare(r["task"], sft["text"], base["text"])
            backward = engine.compare(r["task"], base["text"], sft["text"])
            signals.update(disagreement_forward=forward, disagreement_backward=backward)
            if forward["score"] is not None and backward["score"] is not None:
                signals["disagreement"] = 1 - (forward["score"] + backward["score"]) / 2
        w = config["mining"]["weights"]
        values = [signals[k] for k in ("gt_error", "regression", "disagreement")]
        den = sum(a for a, v in zip(w, values) if v is not None)
        signals["priority"] = sum(a * v for a, v in zip(w, values) if v is not None) / den if den else 0.0
        scores = [c["quality"]["score"] for c in candidates if c["quality"].get("score") is not None]
        signals["candidate_gap"] = max(scores) - min(scores) if scores else 0.0
        r.update(signals=signals, metric_config=dict(config["metrics"]))
        out.append(r)
    return out
