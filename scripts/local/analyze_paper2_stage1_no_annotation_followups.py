#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
STAGE = ROOT / "doc" / "experiments" / "paper2" / "stage1"
TAG = "no_annotation_followups_v1"


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _f(row: dict[str, Any], key: str, default: float = math.nan) -> float:
    value = row.get(key, "")
    if value in ("", None):
        return default
    try:
        return float(value)
    except Exception:
        return default


def _label(row: dict[str, Any], margin_threshold: float, random_gap: float) -> str:
    if str(row.get("blocked", "")).lower() == "true":
        return "blocked"
    union = _f(row, "damage_margin_mask_A_union_B")
    a = _f(row, "damage_margin_mask_A")
    b = _f(row, "damage_margin_mask_B")
    random_union = _f(row, "damage_margin_random_union_size")
    wrong = _f(row, "damage_margin_wrong_image")
    max_single = max(a, b)
    if (
        union > max_single + margin_threshold
        and union > random_union + random_gap
        and wrong > 0
    ):
        return "multi_evidence_supported"
    if union <= margin_threshold and wrong <= margin_threshold:
        return "shortcut_prior_suspected"
    if max_single >= union - margin_threshold:
        return "one_region_dominated"
    if a > margin_threshold and b > margin_threshold and union <= max_single + margin_threshold:
        return "redundant_evidence"
    return "mixed_or_unclear"


def _mean(values: list[float]) -> float:
    vals = [v for v in values if not math.isnan(v)]
    return sum(vals) / len(vals) if vals else math.nan


def main() -> int:
    STAGE.mkdir(parents=True, exist_ok=True)
    comp_path = STAGE / "composition_effect_by_sample_mask_full_v1.csv"
    strong_path = STAGE / "paper2_stage1_behavior_viability_v1_strong_cases.csv"
    paired_path = STAGE / "paper2_stage1_behavior_viability_v1_paired.csv"
    comp = _read_csv(comp_path)
    strong = _read_csv(strong_path)
    paired = _read_csv(paired_path)

    thresholds = [
        {"margin_threshold": 0.05, "random_gap": 0.05},
        {"margin_threshold": 1.0, "random_gap": 1.0},
        {"margin_threshold": 2.0, "random_gap": 1.0},
        {"margin_threshold": 5.0, "random_gap": 2.0},
    ]

    sweep_rows: list[dict[str, Any]] = []
    labels_by_key: dict[tuple[str, str], list[str]] = defaultdict(list)
    for setting in thresholds:
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in comp:
            label = _label(row, setting["margin_threshold"], setting["random_gap"])
            labels_by_key[(row.get("model_family", ""), row.get("sample_id", ""))].append(label)
            grouped[row.get("model_family", "")].append({**row, "sweep_label": label})
        for model, rows in sorted(grouped.items()):
            counts = Counter(row["sweep_label"] for row in rows)
            sweep_rows.append(
                {
                    "model_family": model,
                    "margin_threshold": setting["margin_threshold"],
                    "random_gap": setting["random_gap"],
                    "n": len(rows),
                    "multi_evidence_supported": counts.get("multi_evidence_supported", 0),
                    "one_region_dominated": counts.get("one_region_dominated", 0),
                    "shortcut_prior_suspected": counts.get("shortcut_prior_suspected", 0),
                    "mixed_or_unclear": counts.get("mixed_or_unclear", 0),
                    "blocked": counts.get("blocked", 0),
                }
            )

    stability_rows: list[dict[str, Any]] = []
    comp_by_key = {(row.get("model_family", ""), row.get("sample_id", "")): row for row in comp}
    for key, labels in sorted(labels_by_key.items()):
        row = comp_by_key[key]
        counts = Counter(labels)
        stability_rows.append(
            {
                "model_family": key[0],
                "sample_id": key[1],
                "sample_type": row.get("sample_type", ""),
                "answer_text": row.get("answer_text", ""),
                "question_text": row.get("question_text", ""),
                "original_stage1_label": row.get("stage1_label", ""),
                "multi_evidence_supported_thresholds": counts.get("multi_evidence_supported", 0),
                "dominant_label_across_thresholds": counts.most_common(1)[0][0] if labels else "",
                "composition_effect_margin": row.get("composition_effect_margin", ""),
                "union_damage_margin": row.get("damage_margin_mask_A_union_B", ""),
                "max_single_damage_margin": max(_f(row, "damage_margin_mask_A"), _f(row, "damage_margin_mask_B")),
                "random_union_damage_margin": row.get("damage_margin_random_union_size", ""),
                "wrong_image_damage_margin": row.get("damage_margin_wrong_image", ""),
                "recommended_use": (
                    "mechanism_priority"
                    if counts.get("multi_evidence_supported", 0) >= 2
                    else "annotation_or_diagnostic_only"
                ),
            }
        )

    annotated_sample_ids = {row.get("sample_id", "") for row in comp}
    strong_candidates: dict[tuple[str, str], dict[str, Any]] = {}
    for row in strong:
        strong_candidates[(row.get("model_family", ""), row.get("sample_id", ""))] = row
    for row in paired:
        key = (row.get("model_family", ""), row.get("sample_id", ""))
        if key not in strong_candidates and row.get("format_ok_both") == "1":
            try:
                score = float(row.get("strong_case_score", "0") or 0)
            except Exception:
                score = 0.0
            if score > 50:
                strong_candidates[key] = row

    annotation_rows: list[dict[str, Any]] = []
    for (model, sample_id), row in sorted(
        strong_candidates.items(),
        key=lambda item: float(item[1].get("strong_case_score", "0") or 0),
        reverse=True,
    ):
        has_mask = sample_id in annotated_sample_ids
        annotation_rows.append(
            {
                "model_family": model,
                "sample_id": sample_id,
                "sample_type": row.get("sample_type", ""),
                "answer_text": row.get("answer_text", ""),
                "question_text": row.get("question_text", ""),
                "has_stage1_ab_mask": int(has_mask),
                "clean_exact_answer": row.get("clean_exact_answer", ""),
                "decoded_changed": row.get("decoded_changed", ""),
                "rank_delta_wrong_minus_clean": row.get("rank_delta_wrong_minus_clean", ""),
                "target_logit_drop_clean_minus_wrong": row.get("target_logit_drop_clean_minus_wrong", ""),
                "strong_case_score": row.get("strong_case_score", ""),
                "annotation_priority": (
                    "already_annotated"
                    if has_mask
                    else ("high" if row.get("clean_exact_answer") == "1" else "medium")
                ),
            }
        )

    robustness_fields = [
        "model_family",
        "margin_threshold",
        "random_gap",
        "n",
        "multi_evidence_supported",
        "one_region_dominated",
        "shortcut_prior_suspected",
        "mixed_or_unclear",
        "blocked",
    ]
    stability_fields = [
        "model_family",
        "sample_id",
        "sample_type",
        "answer_text",
        "original_stage1_label",
        "multi_evidence_supported_thresholds",
        "dominant_label_across_thresholds",
        "composition_effect_margin",
        "union_damage_margin",
        "max_single_damage_margin",
        "random_union_damage_margin",
        "wrong_image_damage_margin",
        "recommended_use",
        "question_text",
    ]
    annotation_fields = [
        "model_family",
        "sample_id",
        "sample_type",
        "answer_text",
        "has_stage1_ab_mask",
        "clean_exact_answer",
        "decoded_changed",
        "rank_delta_wrong_minus_clean",
        "target_logit_drop_clean_minus_wrong",
        "strong_case_score",
        "annotation_priority",
        "question_text",
    ]

    _write_csv(STAGE / f"paper2_stage1_{TAG}_threshold_sweep.csv", sweep_rows, robustness_fields)
    _write_csv(STAGE / f"paper2_stage1_{TAG}_sample_stability.csv", stability_rows, stability_fields)
    _write_csv(STAGE / f"paper2_stage1_{TAG}_annotation_candidates.csv", annotation_rows, annotation_fields)

    high_annotation = [row for row in annotation_rows if row["annotation_priority"] == "high"]
    medium_annotation = [row for row in annotation_rows if row["annotation_priority"] == "medium"]
    robust_multi = [row for row in stability_rows if row["multi_evidence_supported_thresholds"] >= 2]
    decision = {
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "status": "paper2_stage1_no_annotation_followups_ready",
        "tag": TAG,
        "inputs": {
            "composition_rows": len(comp),
            "strong_rows": len(strong),
            "paired_rows": len(paired),
        },
        "robust_multi_evidence_rows": len(robust_multi),
        "robust_multi_evidence_by_model": dict(Counter(row["model_family"] for row in robust_multi)),
        "annotation_needed_high": len(high_annotation),
        "annotation_needed_medium": len(medium_annotation),
        "next_action": (
            "annotate_more_high_priority_cases"
            if len(robust_multi) < 16
            else "ready_for_stage2_mechanism_expansion"
        ),
        "outputs": {
            "threshold_sweep": str(STAGE / f"paper2_stage1_{TAG}_threshold_sweep.csv"),
            "sample_stability": str(STAGE / f"paper2_stage1_{TAG}_sample_stability.csv"),
            "annotation_candidates": str(STAGE / f"paper2_stage1_{TAG}_annotation_candidates.csv"),
        },
    }
    (STAGE / f"paper2_stage1_{TAG}_decision.json").write_text(
        json.dumps(decision, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    lines = [
        "# Paper2 Stage1 No-Annotation Follow-ups",
        "",
        f"Generated: {decision['created_at']}",
        "",
        "## What Was Done",
        "",
        "This analysis reuses existing Stage1 behavior and A/B mask-composition outputs. It does not require new annotation or GPU runs.",
        "",
        "It checks threshold robustness of the multi-evidence label, identifies samples whose label is stable across stricter thresholds, and extracts the next annotation candidates from behavior-strong samples that do not yet have A/B masks.",
        "",
        "## Main Readout",
        "",
        f"- Robust multi-evidence rows across threshold settings: `{len(robust_multi)}`.",
        f"- Robust multi-evidence by model: `{dict(Counter(row['model_family'] for row in robust_multi))}`.",
        f"- High-priority unannotated behavior cases: `{len(high_annotation)}`.",
        f"- Medium-priority unannotated behavior cases: `{len(medium_annotation)}`.",
        "",
        "## Interpretation",
        "",
        "The existing Stage1 data can support case selection and sample triage, but it still does not provide enough robust multi-evidence samples for a broad Paper2 mechanism claim. The next bottleneck is annotation expansion, especially for cases with clean exact answers and strong wrong-image sensitivity.",
        "",
        "## Outputs",
        "",
        f"- `paper2_stage1_{TAG}_threshold_sweep.csv`",
        f"- `paper2_stage1_{TAG}_sample_stability.csv`",
        f"- `paper2_stage1_{TAG}_annotation_candidates.csv`",
        f"- `paper2_stage1_{TAG}_decision.json`",
    ]
    (STAGE / f"004_paper2_stage1_{TAG}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(decision, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
