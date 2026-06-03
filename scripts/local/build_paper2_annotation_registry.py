#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
BRIDGING_ROOT = ROOT.parent
ANNOTATION_ROOTS = [
    BRIDGING_ROOT / "annotation",
    ROOT / "annotation",
]
OUT_DIR = ROOT / "annotation" / "_registry"
PAPER2_STAGE1 = ROOT / "doc" / "experiments" / "paper2" / "stage1"
PAPER2_STAGE2 = ROOT / "doc" / "experiments" / "paper2" / "stage2"
PAPER2_STAGE3 = ROOT / "doc" / "experiments" / "paper2" / "stage3_mechanism"
PAPER1_STAGE3 = BRIDGING_ROOT / "doc" / "experiments" / "stage3" / "paperpack72"
TYPE_LABEL_SOURCES = sorted(
    path
    for root in (BRIDGING_ROOT / "annotation").glob("okvqa_type_label*")
    for path in [
        root / "manifest.csv",
        root / "labels.csv",
        root / "round4_400_five_axis_blank.csv",
        root / "five_axis_annotation_blank.csv",
        root / "three_label_annotation_blank.csv",
    ]
    if path.exists()
)
EVIDENCE_MANIFEST_SOURCES = sorted(
    path
    for root in (BRIDGING_ROOT / "annotation").glob("okvqa_evidence_labelme*")
    for path in [root / "manifest.csv", root / "manifest_runready.csv"]
    if path.exists()
)


METADATA_SOURCES = [
    PAPER2_STAGE1 / "stage1_composition_screen_viability_v1_manifest.csv",
    PAPER2_STAGE1 / "stage1_composition_screen_viability_v1_gemma.csv",
    PAPER2_STAGE1 / "stage1_composition_screen_viability_v1_qwen.csv",
    PAPER2_STAGE1 / "paper2_stage1_behavior_viability_v1_paired.csv",
    PAPER2_STAGE1 / "paper2_stage1_behavior_viability_v1_strong_cases.csv",
    PAPER2_STAGE1 / "paper2_stage1_no_annotation_followups_v1_annotation_candidates.csv",
    PAPER2_STAGE1 / "composition_effect_by_sample_mask_full_v1.csv",
    PAPER2_STAGE1 / "composition_effect_by_sample_mask_v1.csv",
    PAPER2_STAGE2 / "paper2_stage2_hidden_by_composition_label.csv",
    PAPER2_STAGE3 / "paper2_stage3_mechanism_summary_paper2_mechanism_rank50_v1.csv",
    PAPER1_STAGE3 / "paperpack72_manifest_template.csv",
    PAPER1_STAGE3 / "paperpack72_primary_manifest.csv",
    PAPER1_STAGE3 / "paperpack72_strict_sensitivity_manifest.csv",
    PAPER1_STAGE3 / "paperpack81_annotated_pool.csv",
    *TYPE_LABEL_SOURCES,
    *EVIDENCE_MANIFEST_SOURCES,
]


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


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(data, dict) or not isinstance(data.get("shapes"), list):
        return None
    return data


def _norm_label(label: str) -> str:
    key = label.strip().lower().replace("-", "_").replace(" ", "_")
    compact = key.replace("_", "")
    if compact in {"regiona", "answera", "evidencea"}:
        return "region_A"
    if re.fullmatch(r"(region|answer|evidence)[b-z]", compact):
        return "region_B"
    if key in {"a", "region_a", "answer"}:
        return "region_A"
    if key in {"b", "region_b", "relate", "context"}:
        return "region_B"
    return label.strip()


def _image_filename_from_json(path: Path, data: dict[str, Any]) -> str:
    image_path = str(data.get("imagePath") or "").strip()
    if image_path:
        return Path(image_path).name
    return path.with_suffix(".jpg").name


def _pack_name(path: Path) -> str:
    for root in ANNOTATION_ROOTS:
        try:
            rel = path.relative_to(root)
        except ValueError:
            continue
        return rel.parts[0] if rel.parts else root.name
    return path.parent.name


def _metadata_index() -> tuple[dict[str, list[dict[str, str]]], dict[str, list[dict[str, str]]]]:
    by_sample: dict[str, list[dict[str, str]]] = defaultdict(list)
    by_image: dict[str, list[dict[str, str]]] = defaultdict(list)
    sample_to_image: dict[str, str] = {}

    for source in METADATA_SOURCES:
        for row in _read_csv(source):
            row = {k: (v if v is not None else "") for k, v in row.items()}
            row["_metadata_source"] = str(source)
            sample_id = row.get("sample_id", "").strip()
            if not sample_id:
                item_id = row.get("item_id", "")
                match = re.search(r"okvqa_val_\d+", item_id)
                sample_id = match.group(0) if match else ""
                if sample_id:
                    row["sample_id"] = sample_id
            image_filename = row.get("image_filename", "").strip()
            if sample_id and image_filename and sample_id not in sample_to_image:
                sample_to_image[sample_id] = image_filename

    for source in METADATA_SOURCES:
        for row in _read_csv(source):
            row = {k: (v if v is not None else "") for k, v in row.items()}
            row["_metadata_source"] = str(source)
            sample_id = row.get("sample_id", "").strip()
            if not sample_id:
                item_id = row.get("item_id", "")
                match = re.search(r"okvqa_val_\d+", item_id)
                sample_id = match.group(0) if match else ""
                if sample_id:
                    row["sample_id"] = sample_id
            image_filename = row.get("image_filename", "").strip() or sample_to_image.get(sample_id, "")
            if image_filename:
                row["image_filename"] = image_filename
            if sample_id:
                by_sample[sample_id].append(row)
            if image_filename:
                by_image[image_filename].append(row)
    return by_sample, by_image


def _compact_values(rows: list[dict[str, str]], key: str, limit: int = 8) -> str:
    seen: list[str] = []
    for row in rows:
        value = row.get(key, "").strip()
        if value and value not in seen:
            seen.append(value)
    if len(seen) > limit:
        return " | ".join(seen[:limit]) + f" | ...(+{len(seen) - limit})"
    return " | ".join(seen)


def _scan_labelme() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for root in ANNOTATION_ROOTS:
        if not root.exists():
            continue
        for path in sorted(root.rglob("*.json")):
            data = _load_json(path)
            if data is None:
                continue
            shapes = data.get("shapes", [])
            raw_counts = Counter(str(shape.get("label", "")).strip().lower() for shape in shapes)
            norm_counts = Counter(_norm_label(str(shape.get("label", ""))) for shape in shapes)
            raw_answer_count = raw_counts.get("answer", 0)
            answer_letter_labels = [
                label for label in raw_counts if re.fullmatch(r"answer[a-z]", label.replace("_", ""))
            ]
            answer_letter_shape_count = sum(raw_counts[label] for label in answer_letter_labels)
            has_ab_labels = norm_counts.get("region_A", 0) > 0 and norm_counts.get("region_B", 0) > 0
            legacy_multi_answer_candidate = (
                raw_answer_count >= 2 or len(answer_letter_labels) >= 2 or answer_letter_shape_count >= 2
            )
            paper2_interpretation = "unclassified"
            if legacy_multi_answer_candidate:
                paper2_interpretation = "legacy_multi_answer_candidate"
            elif has_ab_labels:
                paper2_interpretation = "has_A_B_or_answer_relate"
            elif norm_counts.get("region_A", 0) > 0 and norm_counts.get("region_B", 0) == 0:
                paper2_interpretation = "one_region_candidate"

            rows.append(
                {
                    "image_filename": _image_filename_from_json(path, data),
                    "image_stem": Path(_image_filename_from_json(path, data)).stem,
                    "labelme_json_path": str(path),
                    "annotation_pack": _pack_name(path),
                    "shape_count": len(shapes),
                    "raw_labels": ";".join(f"{k}:{v}" for k, v in sorted(raw_counts.items())),
                    "normalized_labels": ";".join(f"{k}:{v}" for k, v in sorted(norm_counts.items())),
                    "raw_answer_shape_count": raw_answer_count,
                    "answer_letter_labels": ",".join(sorted(answer_letter_labels)),
                    "answer_letter_shape_count": answer_letter_shape_count,
                    "region_A_shape_count": norm_counts.get("region_A", 0),
                    "region_B_shape_count": norm_counts.get("region_B", 0),
                    "legacy_multi_answer_candidate": int(legacy_multi_answer_candidate),
                    "has_ab_or_answer_relate": int(has_ab_labels),
                    "paper2_interpretation": paper2_interpretation,
                }
            )
    return rows


def _aggregate_by_image(labelme_rows: list[dict[str, Any]], metadata_by_image: dict[str, list[dict[str, str]]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in labelme_rows:
        grouped[row["image_filename"]].append(row)
    out: list[dict[str, Any]] = []
    for image_filename, rows in sorted(grouped.items()):
        metadata = metadata_by_image.get(image_filename, [])
        sample_ids = _compact_values(metadata, "sample_id", limit=12)
        models = _compact_values(metadata, "model_family", limit=4)
        out.append(
            {
                "image_filename": image_filename,
                "json_count": len(rows),
                "annotation_packs": " | ".join(sorted({str(row["annotation_pack"]) for row in rows})),
                "sample_ids": sample_ids,
                "model_families": models,
                "question_texts": _compact_values(metadata, "question_text", limit=4),
                "answer_texts": _compact_values(metadata, "answer_text", limit=8),
                "sample_types": _compact_values(metadata, "sample_type", limit=8),
                "legacy_visual_type_labels": _compact_values(metadata, "legacy_visual_type_label", limit=8),
                "legacy_knowledge_level_labels": _compact_values(metadata, "legacy_knowledge_level_label", limit=8),
                "max_raw_answer_shape_count": max(int(row["raw_answer_shape_count"]) for row in rows),
                "max_answer_letter_shape_count": max(int(row["answer_letter_shape_count"]) for row in rows),
                "has_legacy_multi_answer_candidate": max(int(row["legacy_multi_answer_candidate"]) for row in rows),
                "has_ab_or_answer_relate": max(int(row["has_ab_or_answer_relate"]) for row in rows),
                "paper2_interpretations": " | ".join(sorted({str(row["paper2_interpretation"]) for row in rows})),
                "representative_json_path": rows[-1]["labelme_json_path"],
            }
        )
    return out


def _aggregate_by_sample(
    metadata_by_sample: dict[str, list[dict[str, str]]],
    by_image_rows: list[dict[str, Any]],
    current_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    annotation_by_image = {row["image_filename"]: row for row in by_image_rows}
    current_by_image = {row["image_filename"]: row for row in current_rows}
    out: list[dict[str, Any]] = []
    for sample_id, rows in sorted(metadata_by_sample.items()):
        image_filenames = []
        for row in rows:
            image_filename = row.get("image_filename", "").strip()
            if image_filename and image_filename not in image_filenames:
                image_filenames.append(image_filename)
        ann_rows = [annotation_by_image[name] for name in image_filenames if name in annotation_by_image]
        current_statuses = [
            current_by_image[name]["paper2_annotation_status"] for name in image_filenames if name in current_by_image
        ]
        out.append(
            {
                "sample_id": sample_id,
                "image_filenames": " | ".join(image_filenames),
                "model_families": _compact_values(rows, "model_family", limit=4),
                "question_texts": _compact_values(rows, "question_text", limit=4),
                "answer_texts": _compact_values(rows, "answer_text", limit=8),
                "sample_types": _compact_values(rows, "sample_type", limit=8),
                "stage1_labels": _compact_values(rows, "stage1_label", limit=8),
                "visual_type_labels": _compact_values(rows, "visual_type_label", limit=8),
                "legacy_visual_type_labels": _compact_values(rows, "legacy_visual_type_label", limit=8),
                "knowledge_level_labels": _compact_values(rows, "knowledge_level_label", limit=8),
                "legacy_knowledge_level_labels": _compact_values(rows, "legacy_knowledge_level_label", limit=8),
                "question_types": _compact_values(rows, "question_type", limit=8),
                "visual_structures": _compact_values(rows, "visual_structure", limit=8),
                "image_dependence_labels": _compact_values(rows, "image_dependence", limit=8),
                "reasoning_operations": _compact_values(rows, "reasoning_operation", limit=8),
                "ambiguity_flags": _compact_values(rows, "ambiguity_flag", limit=8),
                "extra_evidence_labels": _compact_values(rows, "extra_evidence", limit=8),
                "paper2_followup_annotation_statuses": " | ".join(sorted(set(current_statuses))),
                "has_labelme_json": int(bool(ann_rows)),
                "has_legacy_multi_answer_candidate": max(
                    [int(row["has_legacy_multi_answer_candidate"]) for row in ann_rows] or [0]
                ),
                "has_ab_or_answer_relate": max([int(row["has_ab_or_answer_relate"]) for row in ann_rows] or [0]),
                "annotation_packs": " | ".join(
                    sorted(
                        {
                            pack
                            for ann in ann_rows
                            for pack in str(ann.get("annotation_packs", "")).split(" | ")
                            if pack
                        }
                    )
                ),
                "metadata_sources": _compact_values(rows, "_metadata_source", limit=12),
            }
        )
    return out


def _current_followup_status(metadata_by_image: dict[str, list[dict[str, str]]]) -> list[dict[str, Any]]:
    pack = ROOT / "annotation" / "stage1_followup_ab_evidence_pack_v2_all28"
    summary = pack / "labelme_mask_export_summary.csv"
    if not summary.exists():
        return []
    rows: list[dict[str, Any]] = []
    for row in _read_csv(summary):
        image_filename = row.get("image_filename", "")
        metadata = metadata_by_image.get(image_filename, [])
        status = row.get("status", "")
        if "missing_region_B" in status and "missing_region_A" not in status:
            paper2_status = "one_region_only_candidate"
        elif status == "ok":
            paper2_status = "multi_source_mask_ready"
        elif "missing_region_A" in status:
            paper2_status = "needs_review_missing_A"
        else:
            paper2_status = "needs_review"
        rows.append(
            {
                "image_filename": image_filename,
                "sample_ids": _compact_values(metadata, "sample_id", limit=12),
                "model_families": _compact_values(metadata, "model_family", limit=4),
                "question_texts": _compact_values(metadata, "question_text", limit=4),
                "answer_texts": _compact_values(metadata, "answer_text", limit=8),
                "export_status": status,
                "paper2_annotation_status": paper2_status,
                "region_A_area_px": row.get("region_A_area_px", ""),
                "region_B_area_px": row.get("region_B_area_px", ""),
                "region_A_union_B_area_px": row.get("region_A_union_B_area_px", ""),
                "mask_dir": row.get("mask_dir", ""),
            }
        )
    return rows


def _write_readme(
    labelme_rows: list[dict[str, Any]],
    by_image_rows: list[dict[str, Any]],
    current_rows: list[dict[str, Any]],
) -> None:
    multi = [row for row in by_image_rows if int(row["has_legacy_multi_answer_candidate"]) == 1]
    one_region = [row for row in current_rows if row["paper2_annotation_status"] == "one_region_only_candidate"]
    text = f"""# Paper2 Annotation Registry

Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

This folder is a local-only index of annotation assets and sample metadata. It is intentionally under `annotation/`, so it is ignored by Git.

## Key Files

- `annotation_registry_labelme_jsons.csv`: every LabelMe JSON discovered under the Paper1/Paper2 annotation roots.
- `annotation_registry_by_image.csv`: one row per image with joined question/answer/type metadata when available.
- `paper2_sample_registry.csv`: one row per sample id with question/answer/type labels, composition labels, and annotation status.
- `legacy_multi_answer_candidates.csv`: old annotations where multiple `answer` shapes or `answera/answerb/...` labels suggest multi-evidence candidates.
- `current_followup_annotation_status.csv`: current Paper2 follow-up export status. A missing B is interpreted as `one_region_only_candidate`, not an annotation error.
- `one_region_candidates_from_current_export.csv`: current follow-up images where only A was drawn.

## Counts

- LabelMe JSON files scanned: `{len(labelme_rows)}`
- Unique images with LabelMe JSONs: `{len(by_image_rows)}`
- Legacy multi-answer candidate images: `{len(multi)}`
- Current one-region-only candidates: `{len(one_region)}`

## Paper2 Semantics

For Paper2 A/B evidence, `region_A` and `region_B` should be two distinct answer-supporting evidence sources. Masking only A or only B should ideally still leave the answer inferable; masking A and B together should break the evidence.

If an image has no real B region, it should be treated as `one_region_only_candidate`, useful as a one-region dominated control rather than forced into the compositional A/B set.
"""
    (OUT_DIR / "README.md").write_text(text, encoding="utf-8")


def main() -> int:
    metadata_by_sample, metadata_by_image = _metadata_index()
    labelme_rows = _scan_labelme()
    by_image_rows = _aggregate_by_image(labelme_rows, metadata_by_image)
    current_rows = _current_followup_status(metadata_by_image)
    sample_rows = _aggregate_by_sample(metadata_by_sample, by_image_rows, current_rows)
    legacy_multi = [row for row in by_image_rows if int(row["has_legacy_multi_answer_candidate"]) == 1]
    one_region = [row for row in current_rows if row["paper2_annotation_status"] == "one_region_only_candidate"]

    _write_csv(
        OUT_DIR / "annotation_registry_labelme_jsons.csv",
        labelme_rows,
        [
            "image_filename",
            "image_stem",
            "labelme_json_path",
            "annotation_pack",
            "shape_count",
            "raw_labels",
            "normalized_labels",
            "raw_answer_shape_count",
            "answer_letter_labels",
            "answer_letter_shape_count",
            "region_A_shape_count",
            "region_B_shape_count",
            "legacy_multi_answer_candidate",
            "has_ab_or_answer_relate",
            "paper2_interpretation",
        ],
    )
    _write_csv(
        OUT_DIR / "annotation_registry_by_image.csv",
        by_image_rows,
        [
            "image_filename",
            "json_count",
            "annotation_packs",
            "sample_ids",
            "model_families",
            "question_texts",
            "answer_texts",
            "sample_types",
            "legacy_visual_type_labels",
            "legacy_knowledge_level_labels",
            "max_raw_answer_shape_count",
            "max_answer_letter_shape_count",
            "has_legacy_multi_answer_candidate",
            "has_ab_or_answer_relate",
            "paper2_interpretations",
            "representative_json_path",
        ],
    )
    _write_csv(
        OUT_DIR / "legacy_multi_answer_candidates.csv",
        legacy_multi,
        [
            "image_filename",
            "json_count",
            "annotation_packs",
            "sample_ids",
            "model_families",
            "question_texts",
            "answer_texts",
            "sample_types",
            "max_raw_answer_shape_count",
            "max_answer_letter_shape_count",
            "paper2_interpretations",
            "representative_json_path",
        ],
    )
    _write_csv(
        OUT_DIR / "paper2_sample_registry.csv",
        sample_rows,
        [
            "sample_id",
            "image_filenames",
            "model_families",
            "question_texts",
            "answer_texts",
            "sample_types",
            "stage1_labels",
            "visual_type_labels",
            "legacy_visual_type_labels",
            "knowledge_level_labels",
            "legacy_knowledge_level_labels",
            "question_types",
            "visual_structures",
            "image_dependence_labels",
            "reasoning_operations",
            "ambiguity_flags",
            "extra_evidence_labels",
            "paper2_followup_annotation_statuses",
            "has_labelme_json",
            "has_legacy_multi_answer_candidate",
            "has_ab_or_answer_relate",
            "annotation_packs",
            "metadata_sources",
        ],
    )
    _write_csv(
        OUT_DIR / "current_followup_annotation_status.csv",
        current_rows,
        [
            "image_filename",
            "sample_ids",
            "model_families",
            "question_texts",
            "answer_texts",
            "export_status",
            "paper2_annotation_status",
            "region_A_area_px",
            "region_B_area_px",
            "region_A_union_B_area_px",
            "mask_dir",
        ],
    )
    _write_csv(
        OUT_DIR / "one_region_candidates_from_current_export.csv",
        one_region,
        [
            "image_filename",
            "sample_ids",
            "model_families",
            "question_texts",
            "answer_texts",
            "export_status",
            "paper2_annotation_status",
            "region_A_area_px",
            "region_B_area_px",
            "region_A_union_B_area_px",
            "mask_dir",
        ],
    )
    _write_readme(labelme_rows, by_image_rows, current_rows)
    decision = {
        "status": "annotation_registry_ready",
        "output_dir": str(OUT_DIR),
        "labelme_json_rows": len(labelme_rows),
        "unique_images": len(by_image_rows),
        "legacy_multi_answer_candidate_images": len(legacy_multi),
        "sample_registry_rows": len(sample_rows),
        "current_followup_rows": len(current_rows),
        "current_one_region_candidates": len(one_region),
    }
    (OUT_DIR / "annotation_registry_decision.json").write_text(
        json.dumps(decision, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(decision, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
