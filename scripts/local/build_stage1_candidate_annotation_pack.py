#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import html
import json
import shutil
import time
from collections import Counter
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
BRIDGING_ROOT = PROJECT_ROOT.parent
DEFAULT_SOURCE = BRIDGING_ROOT / "annotation" / "okvqa_type_label_round4_400_mobile_package" / "manifest.csv"
FALLBACK_SOURCES = [
    BRIDGING_ROOT / "annotation" / "okvqa_type_label_round3_320" / "manifest.csv",
    BRIDGING_ROOT / "annotation" / "okvqa_type_label_round4_400" / "manifest.csv",
]
PACK_ROOT = PROJECT_ROOT / "annotation" / "stage1_ab_evidence_pack_v1"


TYPE_QUOTAS = {
    "relation_reasoning": 10,
    "comparison_reasoning": 10,
    "text_object_binding": 10,
    "local_context_inference": 10,
}


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _norm(value: str) -> str:
    return " ".join((value or "").lower().strip().split())


def _local_image_path(row: dict[str, str], source_path: Path) -> Path:
    if row.get("local_image_path"):
        return Path(row["local_image_path"])
    image_url = row.get("image_url", "")
    if image_url:
        return source_path.parent / image_url
    return source_path.parent / "images" / row.get("image_filename", "")


def _sample_type(question: str, row: dict[str, str]) -> str:
    q = _norm(question)
    if any(word in q for word in ["which", "larger", "smaller", "more", "most", "brighter", "taller", "older", "younger", "closer"]):
        return "comparison_reasoning"
    if any(word in q for word in ["sign", "word", "letter", "logo", "label", "written", "writing", "language", "brand", "number"]):
        return "text_object_binding"
    if any(word in q for word in ["left", "right", "next to", "beside", "behind", "front", "on top", "under", "holding", "wearing", "sitting", "standing"]):
        return "relation_reasoning"
    visual_type = _norm(row.get("legacy_visual_type_label", ""))
    if visual_type == "multi_region":
        return "relation_reasoning"
    return "local_context_inference"


def _roles(sample_type: str) -> tuple[str, str]:
    if sample_type == "comparison_reasoning":
        return "first compared object / quantity", "second compared object / quantity"
    if sample_type == "text_object_binding":
        return "text/logo/label evidence", "carrier object or surrounding visual context"
    if sample_type == "relation_reasoning":
        return "object or actor participating in relation", "related object/context needed to resolve relation"
    return "local object evidence", "scene/context evidence needed to infer answer"


def _priority(row: dict[str, str], source_index: int) -> tuple[int, int, int, int]:
    visual = _norm(row.get("legacy_visual_type_label", ""))
    knowledge = _norm(row.get("legacy_knowledge_level_label", ""))
    priority = _norm(row.get("priority", ""))
    visual_score = 0 if visual == "multi_region" else 1 if visual == "localized" else 4
    knowledge_score = 0 if knowledge == "high" else 1 if knowledge == "medium" else 2
    priority_score = 0 if priority == "high" else 1 if priority == "medium" else 2
    return (visual_score, knowledge_score, priority_score, source_index)


def _load_candidates(source_paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for source_index, path in enumerate(source_paths):
        for row in _read_csv(path):
            sample_id = (row.get("sample_id") or "").strip()
            image_filename = (row.get("image_filename") or "").strip()
            question = (row.get("question_text") or row.get("question") or "").strip()
            answer = (row.get("answer_text") or row.get("answer") or "").strip()
            if not sample_id or not image_filename or not question or not answer or sample_id in seen:
                continue
            visual = _norm(row.get("legacy_visual_type_label", ""))
            if visual == "diffuse_global":
                continue
            local_image = _local_image_path(row, path)
            if not local_image.exists():
                continue
            sample_type = _sample_type(question, row)
            region_a_role, region_b_role = _roles(sample_type)
            item = dict(row)
            item.update(
                {
                    "sample_id": sample_id,
                    "question_text": question,
                    "answer_text": answer,
                    "image_filename": image_filename,
                    "local_image_path": str(local_image),
                    "sample_type": sample_type,
                    "region_A_role": region_a_role,
                    "region_B_role": region_b_role,
                    "candidate_source": str(path.relative_to(BRIDGING_ROOT)),
                    "_priority": _priority(row, source_index),
                }
            )
            rows.append(item)
            seen.add(sample_id)
    return rows


def _select_rows(candidates: list[dict[str, Any]], target_count: int) -> list[dict[str, Any]]:
    candidates = sorted(candidates, key=lambda row: (row["_priority"], row["sample_id"]))
    selected: list[dict[str, Any]] = []
    used: set[str] = set()
    by_type: dict[str, list[dict[str, Any]]] = {key: [] for key in TYPE_QUOTAS}
    for row in candidates:
        by_type.setdefault(row["sample_type"], []).append(row)
    for sample_type, quota in TYPE_QUOTAS.items():
        for row in by_type.get(sample_type, []):
            if len([r for r in selected if r["sample_type"] == sample_type]) >= quota:
                break
            if row["sample_id"] not in used:
                selected.append(row)
                used.add(row["sample_id"])
    if len(selected) < target_count:
        for row in candidates:
            if len(selected) >= target_count:
                break
            if row["sample_id"] not in used:
                selected.append(row)
                used.add(row["sample_id"])
    return selected[:target_count]


def _html(rows: list[dict[str, Any]]) -> str:
    cards = []
    for idx, row in enumerate(rows, 1):
        image_rel = f"images/{html.escape(row['image_filename'])}"
        cards.append(
            f"""
<section class="card">
  <h2>{idx}. {html.escape(row['sample_id'])} <span>{html.escape(row['sample_type'])}</span></h2>
  <img src="{image_rel}" alt="{html.escape(row['sample_id'])}">
  <p><b>Question:</b> {html.escape(row['question_text'])}</p>
  <p><b>Gold answer:</b> {html.escape(row['answer_text'])}</p>
  <p><b>Region A:</b> {html.escape(row['region_A_role'])}</p>
  <p><b>Region B:</b> {html.escape(row['region_B_role'])}</p>
  <p class="todo">Draw masks named <code>region_A.png</code>, <code>region_B.png</code>, and <code>region_A_union_B.png</code> under <code>exported_masks/{html.escape(Path(row['image_filename']).stem)}/</code>.</p>
</section>
"""
        )
    return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Paper2 Stage1 A/B Evidence Annotation Pack</title>
  <style>
    body {{ font-family: Georgia, serif; margin: 24px; background: #f6f1e8; color: #1f1a14; }}
    h1 {{ font-size: 30px; }}
    .card {{ background: #fffaf0; border: 1px solid #d8c9ad; padding: 16px; margin: 18px 0; border-radius: 14px; }}
    .card img {{ max-width: 560px; width: 100%; display: block; border-radius: 10px; border: 1px solid #d1c2a7; }}
    .card span {{ font-size: 14px; color: #76664e; }}
    code {{ background: #eee0c5; padding: 1px 4px; border-radius: 4px; }}
    .todo {{ color: #67451b; }}
  </style>
</head>
<body>
  <h1>Paper2 Stage1 A/B Evidence Annotation Pack</h1>
  <p>Mark two visual evidence regions for compositional VQA. A and B should be independently meaningful, and their union should represent the intended multi-evidence support.</p>
  {''.join(cards)}
</body>
</html>
"""


def build_pack(target_count: int, tag: str) -> dict[str, Any]:
    source_paths = [DEFAULT_SOURCE, *FALLBACK_SOURCES]
    candidates = _load_candidates(source_paths)
    selected = _select_rows(candidates, target_count)
    if len(selected) < target_count:
        raise RuntimeError(f"only selected {len(selected)} candidates; target={target_count}")

    pack_dir = PACK_ROOT if tag == "v1" else PROJECT_ROOT / "annotation" / f"stage1_ab_evidence_pack_{tag}"
    images_dir = pack_dir / "images"
    masks_dir = pack_dir / "exported_masks"
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for rank, row in enumerate(selected, 1):
        image_path = Path(row["local_image_path"])
        shutil.copy2(image_path, images_dir / row["image_filename"])
        stem = Path(row["image_filename"]).stem
        (masks_dir / stem).mkdir(parents=True, exist_ok=True)
        wrong = selected[(rank % len(selected))]
        out = {
            "rank": rank,
            "sample_id": row["sample_id"],
            "question_text": row["question_text"],
            "answer_text": row["answer_text"],
            "image_filename": row["image_filename"],
            "image_url": row.get("image_url", ""),
            "local_image_path": str(images_dir / row["image_filename"]),
            "original_image_path": row["local_image_path"],
            "sample_type": row["sample_type"],
            "region_A_role": row["region_A_role"],
            "region_B_role": row["region_B_role"],
            "mask_dir": str(masks_dir / stem),
            "region_A_mask_path": str(masks_dir / stem / "region_A.png"),
            "region_B_mask_path": str(masks_dir / stem / "region_B.png"),
            "region_A_union_B_mask_path": str(masks_dir / stem / "region_A_union_B.png"),
            "random_A_size_mask_path": str(masks_dir / stem / "random_A_size.png"),
            "random_B_size_mask_path": str(masks_dir / stem / "random_B_size.png"),
            "random_union_size_mask_path": str(masks_dir / stem / "random_union_size.png"),
            "wrong_image_sample_id": wrong["sample_id"],
            "wrong_image_filename": wrong["image_filename"],
            "wrong_image_path": str(images_dir / wrong["image_filename"]),
            "candidate_source": row["candidate_source"],
            "legacy_visual_type_label": row.get("legacy_visual_type_label", ""),
            "legacy_knowledge_level_label": row.get("legacy_knowledge_level_label", ""),
            "annotation_status": "needs_region_masks",
        }
        rows.append(out)

    fields = [
        "rank",
        "sample_id",
        "question_text",
        "answer_text",
        "image_filename",
        "image_url",
        "local_image_path",
        "original_image_path",
        "sample_type",
        "region_A_role",
        "region_B_role",
        "mask_dir",
        "region_A_mask_path",
        "region_B_mask_path",
        "region_A_union_B_mask_path",
        "random_A_size_mask_path",
        "random_B_size_mask_path",
        "random_union_size_mask_path",
        "wrong_image_sample_id",
        "wrong_image_filename",
        "wrong_image_path",
        "candidate_source",
        "legacy_visual_type_label",
        "legacy_knowledge_level_label",
        "annotation_status",
    ]
    _write_csv(pack_dir / "stage1_candidate_manifest.csv", rows, fields)
    _write_csv(pack_dir / "stage1_ab_annotation_blank.csv", rows, fields)
    (pack_dir / "stage1_ab_annotation_ui.html").write_text(_html(rows), encoding="utf-8")
    quickstart = """# Paper2 Stage1 A/B Evidence Annotation Quickstart

1. Open `stage1_ab_annotation_ui.html`.
2. For each image, draw `region_A.png`, `region_B.png`, and `region_A_union_B.png`.
3. Save masks under `exported_masks/<image_stem>/`.
4. Run `preflight_stage1_ab_masks.py` before starting GPU behavior screening.

Do not use Paper1 `answer/relate/union` labels as a semantic substitute. Paper2 needs two explicit evidence roles.
"""
    (pack_dir / "ANNOTATION_QUICKSTART.md").write_text(quickstart, encoding="utf-8")
    summary = {
        "created_at": _now(),
        "status": "annotation_pack_ready",
        "pack_dir": str(pack_dir),
        "candidate_count": len(rows),
        "source_paths": [str(path) for path in source_paths],
        "sample_types": dict(Counter(row["sample_type"] for row in rows)),
        "legacy_visual_types": dict(Counter(row["legacy_visual_type_label"] for row in rows)),
    }
    _write_json(pack_dir / "stage1_candidate_manifest_summary.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Build Paper2 Stage1 A/B evidence annotation pack.")
    parser.add_argument("--target-count", type=int, default=40)
    parser.add_argument("--tag", default="v1")
    args = parser.parse_args()
    summary = build_pack(args.target_count, args.tag)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
