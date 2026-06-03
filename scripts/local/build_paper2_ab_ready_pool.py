#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
ANNOTATION = ROOT / "annotation"
REGISTRY = ANNOTATION / "_registry"
STAGE1 = ROOT / "doc" / "experiments" / "paper2" / "stage1"
DEFAULT_CURRENT = REGISTRY / "current_followup_annotation_status.csv"
DEFAULT_LEGACY_PACK = ANNOTATION / "stage1_legacy_multi_answer_ab_pack_v1"
PAIRED = STAGE1 / "paper2_stage1_behavior_viability_v1_paired.csv"


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
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


def _paired_models() -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for row in _read_csv(PAIRED):
        sample_id = row.get("sample_id", "").strip()
        model = row.get("model_family", "").strip()
        if sample_id and model:
            out.setdefault(sample_id, [])
            if model not in out[sample_id]:
                out[sample_id].append(model)
    return out


def _current_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in _read_csv(path):
        if row.get("paper2_annotation_status") != "multi_source_mask_ready":
            continue
        mask_dir = Path(row["mask_dir"])
        rows.append(
            {
                "pool_source": "current_followup_labelme",
                "image_filename": row.get("image_filename", ""),
                "sample_ids": row.get("sample_ids", ""),
                "model_families": row.get("model_families", ""),
                "question_texts": row.get("question_texts", ""),
                "answer_texts": row.get("answer_texts", ""),
                "mask_dir": str(mask_dir),
                "region_A_mask_path": str(mask_dir / "region_A.png"),
                "region_B_mask_path": str(mask_dir / "region_B.png"),
                "region_A_union_B_mask_path": str(mask_dir / "region_A_union_B.png"),
                "conversion_rule": "manual_labelme_region_A_region_B_or_answer_letters",
            }
        )
    return rows


def _legacy_rows(pack: Path, paired_models: dict[str, list[str]]) -> list[dict[str, Any]]:
    manifest = pack / "legacy_multi_answer_ab_manifest.csv"
    rows: list[dict[str, Any]] = []
    for row in _read_csv(manifest):
        mask_dir = pack / Path(row["mask_dir"])
        models = sorted(
            {
                model
                for sample_id in row.get("sample_ids", "").split(" | ")
                for model in paired_models.get(sample_id.strip(), [])
            }
        )
        rows.append(
            {
                "pool_source": "legacy_raw_answer_auto_converted",
                "image_filename": row.get("image_filename", ""),
                "sample_ids": row.get("sample_ids", ""),
                "model_families": " | ".join(models),
                "question_texts": row.get("question_texts", ""),
                "answer_texts": row.get("answer_texts", ""),
                "mask_dir": str(mask_dir),
                "region_A_mask_path": str(mask_dir / "region_A.png"),
                "region_B_mask_path": str(mask_dir / "region_B.png"),
                "region_A_union_B_mask_path": str(mask_dir / "region_A_union_B.png"),
                "conversion_rule": row.get("conversion_rule", ""),
            }
        )
    return rows


def build_pool(current_csv: Path, legacy_pack: Path, out_dir: Path) -> dict[str, Any]:
    paired = _paired_models()
    image_rows = [*_current_rows(current_csv), *_legacy_rows(legacy_pack, paired)]
    image_fields = [
        "pool_source",
        "image_filename",
        "sample_ids",
        "model_families",
        "question_texts",
        "answer_texts",
        "mask_dir",
        "region_A_mask_path",
        "region_B_mask_path",
        "region_A_union_B_mask_path",
        "conversion_rule",
    ]
    _write_csv(out_dir / "paper2_ab_ready_pool_by_image_v1.csv", image_rows, image_fields)

    sample_rows: list[dict[str, Any]] = []
    for row in image_rows:
        sample_ids = [sample_id.strip() for sample_id in row.get("sample_ids", "").split(" | ") if sample_id.strip()]
        for sample_id in sample_ids:
            models = paired.get(sample_id) or [
                model.strip() for model in row.get("model_families", "").split(" | ") if model.strip()
            ] or [""]
            for model in models:
                sample_rows.append(
                    {
                        "pool_source": row["pool_source"],
                        "sample_id": sample_id,
                        "model_family": model,
                        "image_filename": row["image_filename"],
                        "question_text": row["question_texts"],
                        "answer_text": row["answer_texts"],
                        "mask_dir": row["mask_dir"],
                        "region_A_mask_path": row["region_A_mask_path"],
                        "region_B_mask_path": row["region_B_mask_path"],
                        "region_A_union_B_mask_path": row["region_A_union_B_mask_path"],
                        "conversion_rule": row["conversion_rule"],
                    }
                )
    sample_fields = [
        "pool_source",
        "sample_id",
        "model_family",
        "image_filename",
        "question_text",
        "answer_text",
        "mask_dir",
        "region_A_mask_path",
        "region_B_mask_path",
        "region_A_union_B_mask_path",
        "conversion_rule",
    ]
    _write_csv(out_dir / "paper2_ab_ready_pool_by_sample_v1.csv", sample_rows, sample_fields)

    decision = {
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "status": "paper2_ab_ready_pool_ready" if image_rows else "blocked_empty_pool",
        "output_dir": str(out_dir),
        "image_rows": len(image_rows),
        "unique_images": len({row["image_filename"] for row in image_rows}),
        "sample_rows": len(sample_rows),
        "unique_samples": len({row["sample_id"] for row in sample_rows}),
        "sample_rows_with_model": sum(1 for row in sample_rows if row.get("model_family")),
        "current_followup_rows": sum(1 for row in image_rows if row["pool_source"] == "current_followup_labelme"),
        "legacy_auto_converted_rows": sum(
            1 for row in image_rows if row["pool_source"] == "legacy_raw_answer_auto_converted"
        ),
    }
    (out_dir / "paper2_ab_ready_pool_decision_v1.json").write_text(
        json.dumps(decision, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(decision, indent=2, ensure_ascii=False))
    return decision


def main() -> int:
    parser = argparse.ArgumentParser(description="Build unified Paper2 A/B-ready sample pool.")
    parser.add_argument("--current-csv", default=str(DEFAULT_CURRENT))
    parser.add_argument("--legacy-pack", default=str(DEFAULT_LEGACY_PACK))
    parser.add_argument("--out-dir", default=str(REGISTRY))
    args = parser.parse_args()
    decision = build_pool(
        Path(args.current_csv).expanduser().resolve(),
        Path(args.legacy_pack).expanduser().resolve(),
        Path(args.out_dir).expanduser().resolve(),
    )
    return 0 if decision["image_rows"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
