#!/usr/bin/env python3
from __future__ import annotations

import csv
import argparse
import html
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
ANNOTATION = ROOT / "annotation"
STAGE1 = ROOT / "doc" / "experiments" / "paper2" / "stage1"
DEFAULT_OUT = ANNOTATION / "stage1_followup_ab_evidence_pack_v2"
SOURCE = STAGE1 / "paper2_stage1_no_annotation_followups_v1_annotation_candidates.csv"
IMAGE_INDEX_SOURCES = [
    STAGE1 / "stage1_composition_screen_viability_v1_gemma.csv",
    STAGE1 / "stage1_composition_screen_viability_v1_qwen.csv",
    STAGE1 / "stage1_composition_screen_viability_v1_manifest.csv",
]


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _image_index() -> dict[str, str]:
    index: dict[str, str] = {}
    for source in IMAGE_INDEX_SOURCES:
        if not source.exists():
            continue
        for row in _read_csv(source):
            sample_id = row.get("sample_id", "").strip()
            image_filename = row.get("image_filename", "").strip()
            if sample_id and image_filename and sample_id not in index:
                index[sample_id] = image_filename
    return index


def _find_image(sample_id: str, image_filename: str) -> Path | None:
    if image_filename:
        for root in [ANNOTATION, ROOT]:
            matches = sorted(root.rglob(image_filename))
            if matches:
                return matches[0]
    # Fallback only: older packs may encode the COCO id in the sample id.
    suffix = sample_id.replace("okvqa_val_", "")
    for path in sorted(ANNOTATION.rglob("images/COCO_val2014_*.jpg")):
        digits = path.stem.replace("COCO_val2014_", "").lstrip("0") or "0"
        if digits == suffix.lstrip("0"):
            return path
    return None


def _page(rows: list[dict[str, Any]]) -> str:
    cards = []
    for row in rows:
        rel_img = Path(row["image_path"]).as_posix()
        mask_dir = html.escape(row["mask_dir"])
        cards.append(
            f"""
<section class="card">
  <h2>{html.escape(row['sample_id'])} · {html.escape(row['model_family'])}</h2>
  <img src="{html.escape(rel_img)}" />
  <p><b>Question:</b> {html.escape(row['question_text'])}</p>
  <p><b>Answer:</b> {html.escape(row['answer_text'])}</p>
  <p><b>Type:</b> {html.escape(row['sample_type'])}</p>
  <p class="todo">Draw and save masks under <code>{mask_dir}</code>:</p>
  <ul>
    <li><code>region_A.png</code>: first necessary evidence region</li>
    <li><code>region_B.png</code>: second necessary evidence region</li>
    <li><code>region_A_union_B.png</code>: union of A and B</li>
  </ul>
</section>"""
        )
    return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Paper2 Stage1 Follow-up Annotation Pack v2</title>
  <style>
    body {{ font-family: Georgia, serif; margin: 24px; background: #f6f1e8; color: #221b14; }}
    .card {{ background: white; border: 1px solid #d7c6aa; border-radius: 12px; padding: 18px; margin: 18px 0; box-shadow: 0 2px 10px #0001; }}
    img {{ max-width: 720px; width: 100%; border-radius: 8px; display: block; margin: 12px 0; }}
    code {{ background: #f2eadc; padding: 2px 5px; border-radius: 4px; }}
    .todo {{ color: #6d3e00; }}
  </style>
</head>
<body>
  <h1>Paper2 Stage1 Follow-up Annotation Pack v2</h1>
  <p>Goal: add A/B evidence masks for high-priority behavior-strong cases that currently lack Stage1 A/B masks.</p>
  {''.join(cards)}
</body>
</html>
"""


def main() -> int:
    parser = argparse.ArgumentParser(description="Build Paper2 Stage1 A/B follow-up annotation pack.")
    parser.add_argument("--include-medium", action="store_true", help="Include medium-priority candidates in the UI.")
    parser.add_argument("--out-name", default="stage1_followup_ab_evidence_pack_v2")
    args = parser.parse_args()

    out_dir = ANNOTATION / args.out_name
    rows = _read_csv(SOURCE)
    image_index = _image_index()
    high = [r for r in rows if r.get("annotation_priority") == "high"]
    medium = [r for r in rows if r.get("annotation_priority") == "medium"]
    selected = high + medium if args.include_medium else high
    out_rows: list[dict[str, Any]] = []
    missing: list[str] = []
    image_dir = out_dir / "images"
    mask_root = out_dir / "exported_masks"
    for row in selected:
        image_filename = image_index.get(row["sample_id"], "")
        image = _find_image(row["sample_id"], image_filename)
        if image is None:
            missing.append(f"{row['sample_id']}:missing_image:{image_filename or 'no_index'}")
            continue
        dst_image = image_dir / image.name
        dst_image.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(image, dst_image)
        stem = dst_image.stem
        mask_dir = mask_root / stem
        mask_dir.mkdir(parents=True, exist_ok=True)
        out_rows.append(
            {
                **row,
                "image_filename": dst_image.name,
                "image_path": str(Path("images") / dst_image.name),
                "mask_dir": str(Path("exported_masks") / stem),
                "region_A_mask_path": str(Path("exported_masks") / stem / "region_A.png"),
                "region_B_mask_path": str(Path("exported_masks") / stem / "region_B.png"),
                "region_A_union_B_mask_path": str(Path("exported_masks") / stem / "region_A_union_B.png"),
                "annotation_status": "needs_region_masks",
            }
        )
    fields = [
        "model_family",
        "sample_id",
        "sample_type",
        "answer_text",
        "image_filename",
        "image_path",
        "question_text",
        "clean_exact_answer",
        "strong_case_score",
        "mask_dir",
        "region_A_mask_path",
        "region_B_mask_path",
        "region_A_union_B_mask_path",
        "annotation_status",
    ]
    _write_csv(out_dir / "stage1_followup_annotation_blank.csv", out_rows, fields)
    _write_csv(out_dir / "stage1_followup_medium_priority_candidates.csv", medium, list(medium[0].keys()) if medium else [])
    (out_dir / "stage1_followup_annotation_ui.html").write_text(_page(out_rows), encoding="utf-8")
    decision = {
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "status": "paper2_stage1_followup_annotation_pack_ready" if out_rows else "blocked_missing_images",
        "include_medium": args.include_medium,
        "high_priority_rows": sum(1 for row in out_rows if row.get("annotation_priority") == "high"),
        "selected_medium_rows": sum(1 for row in out_rows if row.get("annotation_priority") == "medium"),
        "selected_rows": len(selected),
        "medium_priority_rows": len(medium),
        "missing": missing,
        "output_dir": str(out_dir),
    }
    (out_dir / "stage1_followup_annotation_pack_decision.json").write_text(
        json.dumps(decision, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(decision, indent=2, ensure_ascii=False))
    return 0 if out_rows else 2


if __name__ == "__main__":
    raise SystemExit(main())
