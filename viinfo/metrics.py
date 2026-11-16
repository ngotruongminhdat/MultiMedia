"""Scoring for ViInfographicVQA predictions.

`paper_anls` reproduces the official scorer (src/common/metrics.py and
src/inference/single/calculate_scores.py in duongtruongbinh/ViInfographicVQA)
so our numbers are comparable with the paper's tables. Note that it is not the
textbook ANLS: it uses `Levenshtein.ratio` (an indel-based similarity) instead
of 1 - normalised Levenshtein distance. `standard_anls` is the textbook one,
reported alongside for reference.
"""
from __future__ import annotations

from collections import defaultdict

import Levenshtein

THRESHOLD = 0.5


def clean_prediction(text: str) -> str:
    return text.lower().strip().rstrip(".").replace('"', "").rstrip(">").lstrip("<")


def clean_answer(text: str) -> str:
    return text.lower().strip()


def paper_anls(answer: str, prediction: str) -> float:
    p = clean_prediction(prediction).replace('"', "").rstrip(".").lower()
    score = Levenshtein.ratio(p, clean_answer(answer))
    return score if score >= THRESHOLD else 0.0


def standard_anls(answer: str, prediction: str) -> float:
    p, g = clean_prediction(prediction), clean_answer(answer)
    if not p and not g:
        return 1.0
    nl = Levenshtein.distance(p, g) / max(len(p), len(g))
    return 1.0 - nl if nl < THRESHOLD else 0.0


def exact_match(answer: str, prediction: str) -> float:
    return float(clean_prediction(prediction) == clean_answer(answer))


METRICS = {"anls": paper_anls, "anls_std": standard_anls, "acc": exact_match}

# Fields to break results down by; whichever exist in the split are used.
GROUP_FIELDS = ("answer_source", "element", "image_type")


def _as_labels(value) -> list[str]:
    """`element` is a string in most rows but a list in a few."""
    if value is None:
        return []
    return [str(v) for v in value] if isinstance(value, list) else [str(value)]


def score(rows: list[dict], pred_key: str = "predict") -> dict:
    """Return {"overall": {...}, "<field>": {"<label>": {...}}} with scores in percent."""
    sums: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    for row in rows:
        vals = {name: fn(row["answer"], str(row.get(pred_key, ""))) for name, fn in METRICS.items()}
        groups = [("overall", "overall")]
        groups += [(f, label) for f in GROUP_FIELDS for label in _as_labels(row.get(f))]
        for field, label in groups:
            bucket = sums[field][label]
            bucket["count"] += 1
            for name, v in vals.items():
                bucket[name] += v
    out: dict = {}
    for field, labels in sums.items():
        table = {label: {"count": int(b["count"]),
                         **{m: round(100 * b[m] / b["count"], 2) for m in METRICS}}
                 for label, b in labels.items()}
        out[field] = table["overall"] if field == "overall" else table
    return out


def format_report(result: dict) -> str:
    lines = [f"{'':32s}{'n':>6s}{'ANLS':>8s}{'ANLS-std':>10s}{'Acc':>8s}"]

    def row(name: str, s: dict) -> str:
        return f"{name[:31]:32s}{s['count']:6d}{s['anls']:8.2f}{s['anls_std']:10.2f}{s['acc']:8.2f}"

    lines.append(row("Overall", result["overall"]))
    for field in GROUP_FIELDS:
        if field in result:
            lines.append(f"-- {field}")
            for label, s in sorted(result[field].items(), key=lambda kv: -kv[1]["count"]):
                lines.append(row("  " + label, s))
    return "\n".join(lines)
