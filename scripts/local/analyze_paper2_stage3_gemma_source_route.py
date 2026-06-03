#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
STAGE3 = ROOT / "doc" / "experiments" / "paper2" / "stage3_mechanism"
PREFIX = "paper2_stage3_gemma_source_route"


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({k for row in rows for k in row}) if rows else ["status"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _f(row: dict[str, str], key: str) -> float:
    try:
        return float(row.get(key, ""))
    except Exception:
        return math.nan


def _mean(vals: list[float]) -> float:
    clean = [v for v in vals if not math.isnan(v)]
    return sum(clean) / len(clean) if clean else math.nan


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze Paper2 Gemma source-route case probe.")
    parser.add_argument("--tag", default="paper2_mechanism_rank50_v1")
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    for condition in ["mask_A", "mask_B", "mask_A_union_B", "random_union_size"]:
        stem = f"{PREFIX}_{args.mode}_{args.tag}_{condition}"
        compare = _read_csv(STAGE3 / f"{stem}_sample_compare_controlled.csv")
        intervention = _read_csv(STAGE3 / f"{stem}_intervention.csv")
        decision_path = STAGE3 / f"{stem}_decision.json"
        decision = {}
        if decision_path.exists():
            decision = json.loads(decision_path.read_text(encoding="utf-8"))
        cmp = compare[0] if compare else {}
        deltas = [_f(r, "delta_target_logit") for r in intervention if r.get("run") == "B"]
        rows.append(
            {
                "condition": condition,
                "status": decision.get("status", "missing"),
                "sample_compare_rows": len(compare),
                "intervention_rows": len(intervention),
                "sample_id": cmp.get("sample_id", "okvqa_val_1295955"),
                "node_overlap_jaccard": cmp.get("node_overlap_jaccard", ""),
                "edge_overlap_jaccard": cmp.get("edge_overlap_jaccard", ""),
                "clean_target_total_in_abs": cmp.get("b_target_total_in_abs", ""),
                "condition_target_total_in_abs": cmp.get("a_target_total_in_abs", ""),
                "route_total_in_abs_delta_condition_minus_clean": cmp.get("delta_target_total_in_abs", ""),
                "clean_traced_total_path_mass": cmp.get("b_traced_total_path_mass", ""),
                "condition_traced_total_path_mass": cmp.get("a_traced_total_path_mass", ""),
                "route_path_mass_delta_condition_minus_clean": cmp.get("delta_traced_total_path_mass", ""),
                "clean_feature_ratio": cmp.get("b_target_feature_ratio", ""),
                "condition_feature_ratio": cmp.get("a_target_feature_ratio", ""),
                "intervention_delta_target_logit_mean_clean_run": _mean(deltas),
            }
        )
    by_cond = {row["condition"]: row for row in rows}
    def val(cond: str, key: str) -> float:
        try:
            return float(by_cond.get(cond, {}).get(key, ""))
        except Exception:
            return math.nan
    union_weakening = -val("mask_A_union_B", "route_path_mass_delta_condition_minus_clean")
    single_weakening = max(
        -val("mask_A", "route_path_mass_delta_condition_minus_clean"),
        -val("mask_B", "route_path_mass_delta_condition_minus_clean"),
    )
    random_weakening = -val("random_union_size", "route_path_mass_delta_condition_minus_clean")
    verdict = (
        "gemma_case_route_union_specific"
        if union_weakening > single_weakening and union_weakening > random_weakening
        else "gemma_case_route_mixed_or_not_union_specific"
    )
    if not any(row["sample_compare_rows"] for row in rows):
        verdict = "blocked_missing_compare"
    decision = {
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "status": verdict,
        "mode": args.mode,
        "tag": args.tag,
        "rows": len(rows),
        "union_weakening": union_weakening,
        "max_single_weakening": single_weakening,
        "random_union_weakening": random_weakening,
        "interpretation": "Single Gemma Paper2 case diagnostic; do not treat as model-level claim.",
    }
    _write_csv(STAGE3 / f"{PREFIX}_{args.mode}_{args.tag}_summary.csv", rows)
    (STAGE3 / f"{PREFIX}_{args.mode}_{args.tag}_decision.json").write_text(
        json.dumps(decision, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(decision, indent=2, ensure_ascii=False))
    return 0 if verdict != "blocked_missing_compare" else 2


if __name__ == "__main__":
    raise SystemExit(main())
