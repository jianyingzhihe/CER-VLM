#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = PROJECT_ROOT / "doc" / "experiments" / "stage1" / "results"
PREFIX = "stage1_composition_screen"


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _f(value: Any) -> float:
    try:
        if value in ("", None):
            return math.nan
        return float(value)
    except Exception:
        return math.nan


def _mean(values: list[float]) -> float:
    vals = [value for value in values if not math.isnan(value)]
    return sum(vals) / len(vals) if vals else math.nan


def _same_answer(a: str, b: str) -> bool:
    aa = " ".join((a or "").lower().strip().split())
    bb = " ".join((b or "").lower().strip().split())
    return bool(aa and bb and aa == bb)


def _label(row: dict[str, Any], margin_threshold: float, rank_threshold: float) -> str:
    if row["blocked"]:
        return "blocked"
    union_margin = _f(row["damage_margin_mask_A_union_B"])
    a_margin = _f(row["damage_margin_mask_A"])
    b_margin = _f(row["damage_margin_mask_B"])
    random_union = _f(row["damage_margin_random_union_size"])
    wrong_margin = _f(row["damage_margin_wrong_image"])
    union_rank = _f(row["damage_rank_mask_A_union_B"])
    a_rank = _f(row["damage_rank_mask_A"])
    b_rank = _f(row["damage_rank_mask_B"])
    max_single_margin = max(a_margin, b_margin)
    max_single_rank = max(a_rank, b_rank)
    if union_margin > max_single_margin + margin_threshold and union_rank >= max_single_rank + rank_threshold and union_margin > random_union + margin_threshold:
        return "multi_evidence_supported"
    if union_margin <= margin_threshold and (math.isnan(wrong_margin) or wrong_margin <= margin_threshold):
        return "shortcut_prior_suspected"
    if max_single_margin >= union_margin - margin_threshold:
        return "one_region_dominated"
    if a_margin > margin_threshold and b_margin > margin_threshold and union_margin <= max_single_margin + margin_threshold:
        return "redundant_evidence"
    return "mixed_or_unclear"


def analyze(tag: str, mode: str, margin_threshold: float, rank_threshold: float) -> dict[str, Any]:
    raw_paths = [
        RESULTS_ROOT / f"{PREFIX}_{mode}_{tag}_gemma.csv",
        RESULTS_ROOT / f"{PREFIX}_{mode}_{tag}_qwen.csv",
    ]
    raw = []
    for path in raw_paths:
        raw.extend(_read_csv(path))
    by_key: dict[tuple[str, str], dict[str, dict[str, str]]] = defaultdict(dict)
    for row in raw:
        by_key[(row.get("model_family", ""), row.get("sample_id", ""))][row.get("condition", "")] = row

    detail: list[dict[str, Any]] = []
    for (model_family, sample_id), conds in sorted(by_key.items()):
        clean = conds.get("clean", {})
        blocked = not clean or clean.get("format_ok") == "0" or clean.get("empty_answer") == "1"
        out: dict[str, Any] = {
            "model_family": model_family,
            "sample_id": sample_id,
            "sample_type": clean.get("sample_type", ""),
            "question_text": clean.get("question_text", ""),
            "answer_text": clean.get("answer_text", ""),
            "clean_predicted_answer": clean.get("predicted_answer", ""),
            "blocked": blocked,
            "blocked_reason": "" if not blocked else "missing_or_invalid_clean",
        }
        clean_margin = _f(clean.get("target_minus_wrong_margin"))
        clean_rank = _f(clean.get("target_rank"))
        clean_logit = _f(clean.get("target_logit"))
        for condition in ["mask_A", "mask_B", "mask_A_union_B", "random_A_size", "random_B_size", "random_union_size", "wrong_image"]:
            row = conds.get(condition, {})
            margin = _f(row.get("target_minus_wrong_margin"))
            rank = _f(row.get("target_rank"))
            logit = _f(row.get("target_logit"))
            out[f"damage_margin_{condition}"] = clean_margin - margin if not math.isnan(clean_margin) and not math.isnan(margin) else ""
            out[f"damage_rank_{condition}"] = rank - clean_rank if not math.isnan(rank) and not math.isnan(clean_rank) else ""
            out[f"damage_logit_{condition}"] = clean_logit - logit if not math.isnan(clean_logit) and not math.isnan(logit) else ""
            out[f"predicted_answer_{condition}"] = row.get("predicted_answer", "")
            out[f"answer_flip_{condition}"] = "" if not clean else int(not _same_answer(clean.get("predicted_answer", ""), row.get("predicted_answer", "")))
        out["composition_effect_margin"] = (
            _f(out["damage_margin_mask_A_union_B"]) - max(_f(out["damage_margin_mask_A"]), _f(out["damage_margin_mask_B"]))
        )
        out["composition_effect_rank"] = (
            _f(out["damage_rank_mask_A_union_B"]) - max(_f(out["damage_rank_mask_A"]), _f(out["damage_rank_mask_B"]))
        )
        out["stage1_label"] = _label(out, margin_threshold, rank_threshold)
        detail.append(out)

    fields = sorted({key for row in detail for key in row.keys()})
    _write_csv(RESULTS_ROOT / "composition_effect_by_sample.csv", detail, fields)
    summary_by_type: list[dict[str, Any]] = []
    for (model_family, sample_type), rows in sorted(defaultdict(list, {k: [] for k in []}).items()):
        pass
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in detail:
        grouped[(row["model_family"], row.get("sample_type", ""))].append(row)
    for (model_family, sample_type), rows in grouped.items():
        summary_by_type.append(
            {
                "model_family": model_family,
                "sample_type": sample_type,
                "n": len(rows),
                "composition_effect_margin_mean": _mean([_f(row.get("composition_effect_margin")) for row in rows]),
                "composition_effect_rank_mean": _mean([_f(row.get("composition_effect_rank")) for row in rows]),
                "multi_evidence_supported_count": sum(row.get("stage1_label") == "multi_evidence_supported" for row in rows),
                "blocked_count": sum(row.get("stage1_label") == "blocked" for row in rows),
            }
        )
    _write_csv(
        RESULTS_ROOT / "composition_effect_by_type.csv",
        summary_by_type,
        ["model_family", "sample_type", "n", "composition_effect_margin_mean", "composition_effect_rank_mean", "multi_evidence_supported_count", "blocked_count"],
    )
    model_summary = []
    for model_family, rows in defaultdict(list, {k: [] for k in []}).items():
        pass
    grouped_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in detail:
        grouped_model[row["model_family"]].append(row)
    for model_family, rows in grouped_model.items():
        labels = Counter(row.get("stage1_label", "") for row in rows)
        model_summary.append(
            {
                "model_family": model_family,
                "n": len(rows),
                "supported_count": labels.get("multi_evidence_supported", 0),
                "one_region_dominated_count": labels.get("one_region_dominated", 0),
                "redundant_count": labels.get("redundant_evidence", 0),
                "shortcut_count": labels.get("shortcut_prior_suspected", 0),
                "blocked_count": labels.get("blocked", 0),
                "mixed_or_unclear_count": labels.get("mixed_or_unclear", 0),
            }
        )
    _write_csv(
        RESULTS_ROOT / "model_comparison_summary.csv",
        model_summary,
        ["model_family", "n", "supported_count", "one_region_dominated_count", "redundant_count", "shortcut_count", "blocked_count", "mixed_or_unclear_count"],
    )
    blocked_or_shortcut = [row for row in detail if row.get("stage1_label") in {"blocked", "shortcut_prior_suspected", "one_region_dominated"}]
    _write_csv(RESULTS_ROOT / "blocked_or_shortcut_samples.csv", blocked_or_shortcut, fields)

    sample_supported = {
        row["sample_id"]
        for row in detail
        if row.get("stage1_label") == "multi_evidence_supported"
    }
    types_supported = {row.get("sample_type", "") for row in detail if row.get("sample_id") in sample_supported}
    labels = Counter(row.get("stage1_label", "") for row in detail)
    if labels.get("blocked", 0) > max(4, len(detail) * 0.35):
        status = "blocked_engineering"
    elif len(sample_supported) >= 16 and len(types_supported) >= 2:
        status = "ready_for_stage2_mechanism"
    elif labels.get("one_region_dominated", 0) > len(detail) * 0.5:
        status = "needs_annotation_revision"
    elif len(detail) > 0:
        status = "needs_more_samples"
    else:
        status = "blocked_missing_raw"
    decision = {
        "created_at": _now(),
        "status": status,
        "mode": mode,
        "tag": tag,
        "raw_rows": len(raw),
        "evaluated_model_sample_pairs": len(detail),
        "supported_sample_count": len(sample_supported),
        "supported_sample_types": sorted(types_supported),
        "label_counts": dict(labels),
        "outputs": {
            "composition_effect_by_sample": str(RESULTS_ROOT / "composition_effect_by_sample.csv"),
            "composition_effect_by_type": str(RESULTS_ROOT / "composition_effect_by_type.csv"),
            "model_comparison_summary": str(RESULTS_ROOT / "model_comparison_summary.csv"),
            "blocked_or_shortcut_samples": str(RESULTS_ROOT / "blocked_or_shortcut_samples.csv"),
        },
    }
    _write_json(RESULTS_ROOT / "stage1_decision.json", decision)
    return decision


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze Paper2 Stage1 composition behavior screen.")
    parser.add_argument("--mode", choices=["viability", "smoke", "full"], default="smoke")
    parser.add_argument("--tag", default="v1")
    parser.add_argument("--margin-threshold", type=float, default=0.05)
    parser.add_argument("--rank-threshold", type=float, default=1.0)
    args = parser.parse_args()
    decision = analyze(args.tag, args.mode, args.margin_threshold, args.rank_threshold)
    print(json.dumps(decision, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
