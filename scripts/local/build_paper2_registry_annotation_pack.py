#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import html
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
BRIDGING_ROOT = ROOT.parent
ANNOTATION = ROOT / "annotation"
REGISTRY = ANNOTATION / "_registry"
SAMPLE_REGISTRY = REGISTRY / "paper2_sample_registry.csv"
READY_POOL = REGISTRY / "paper2_ab_ready_pool_by_image_v1.csv"
DEFAULT_EXCLUDE_PACKS = [
    ANNOTATION / "stage1_followup_ab_evidence_pack_v2",
    ANNOTATION / "stage1_followup_ab_evidence_pack_v2_expanded",
    ANNOTATION / "stage1_followup_ab_evidence_pack_v2_all28",
    ANNOTATION / "stage1_followup_ab_evidence_pack_v3_more",
    ANNOTATION / "stage1_legacy_multi_answer_ab_pack_v1",
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


def _split_values(value: str) -> list[str]:
    return [part.strip() for part in value.split(" | ") if part.strip()]


def _excluded_images(extra_packs: list[Path]) -> set[str]:
    excluded: set[str] = set()
    for row in _read_csv(READY_POOL):
        image_filename = row.get("image_filename", "").strip()
        if image_filename:
            excluded.add(image_filename)
    for pack in [*DEFAULT_EXCLUDE_PACKS, *extra_packs]:
        for blank_name in [
            "stage1_followup_annotation_blank.csv",
            "stage1_more_annotation_blank.csv",
            "legacy_multi_answer_ab_manifest.csv",
            "registry_annotation_blank.csv",
        ]:
            blank = pack / blank_name
            for row in _read_csv(blank):
                image_filename = row.get("image_filename", "").strip()
                if image_filename:
                    excluded.add(image_filename)
        images_dir = pack / "images"
        if images_dir.exists():
            for path in images_dir.glob("COCO_val2014_*.*"):
                if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
                    excluded.add(path.name)
    return excluded


def _find_image(image_filename: str) -> Path | None:
    if not image_filename:
        return None
    for root in [ANNOTATION, BRIDGING_ROOT / "annotation"]:
        if not root.exists():
            continue
        matches = sorted(root.rglob(image_filename))
        if matches:
            return matches[0]
    return None


def _score(row: dict[str, str]) -> tuple[int, str]:
    score = 0
    reasons: list[str] = []
    text = " ".join(
        [
            row.get("sample_types", ""),
            row.get("stage1_labels", ""),
            row.get("visual_type_labels", ""),
            row.get("legacy_visual_type_labels", ""),
            row.get("question_types", ""),
            row.get("visual_structures", ""),
            row.get("image_dependence_labels", ""),
            row.get("reasoning_operations", ""),
            row.get("extra_evidence_labels", ""),
        ]
    ).lower()

    if "multi_evidence_supported" in text:
        score += 20
        reasons.append("old_multi_evidence_supported")
    if "multi_region" in text:
        score += 12
        reasons.append("multi_region_label")
    if "split" in text or "multi" in row.get("visual_structures", "").lower():
        score += 8
        reasons.append("split_or_multi_visual_structure")
    if "strong" in row.get("image_dependence_labels", "").lower():
        score += 6
        reasons.append("strong_image_dependence")
    if "relation_reasoning" in text:
        score += 5
        reasons.append("relation_reasoning")
    if "comparison_reasoning" in text:
        score += 4
        reasons.append("comparison_reasoning")
    if "text_object_binding" in text:
        score += 4
        reasons.append("text_object_binding")
    if "local_context_inference" in text:
        score += 3
        reasons.append("local_context_inference")
    if row.get("has_labelme_json", "") in {"1", "1.0", "True", "true"}:
        score += 2
        reasons.append("has_old_labelme_json")
    if row.get("has_ab_or_answer_relate", "") in {"1", "1.0", "True", "true"}:
        score += 2
        reasons.append("has_old_answer_relate")
    if row.get("has_legacy_multi_answer_candidate", "") in {"1", "1.0", "True", "true"}:
        score += 3
        reasons.append("legacy_multi_answer_candidate")
    if not reasons:
        reasons.append("fallback_registry_candidate")
    return score, ";".join(reasons)


def _select_rows(max_images: int, excluded: set[str]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    candidates: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    seen_images: set[str] = set()
    for row in _read_csv(SAMPLE_REGISTRY):
        image_names = _split_values(row.get("image_filenames", ""))
        if not image_names:
            continue
        image_filename = image_names[0]
        if image_filename in excluded or image_filename in seen_images:
            continue
        if not row.get("question_texts", "").strip() or not row.get("answer_texts", "").strip():
            continue
        image_path = _find_image(image_filename)
        if image_path is None:
            missing.append({**row, "image_filename": image_filename, "blocked_reason": "missing_local_image"})
            continue
        score, reasons = _score(row)
        candidates.append({**row, "image_filename": image_filename, "source_image_path": str(image_path), "selection_score": score, "selection_reasons": reasons})
        seen_images.add(image_filename)

    candidates.sort(key=lambda row: (int(row["selection_score"]), row.get("sample_id", "")), reverse=True)
    return candidates[:max_images], missing


def _write_helpers(out_dir: Path) -> None:
    (out_dir / "launch_labelme.ps1").write_text(
        "\n".join(
            [
                "$packDir = Split-Path -Parent $MyInvocation.MyCommand.Path",
                '$imagesDir = Join-Path $packDir "images"',
                '$pythonExe = "E:\\code\\conda\\python.exe"',
                'if (-not (Test-Path $pythonExe)) { $pythonExe = "python" }',
                'Write-Host "Launching LabelMe on $imagesDir"',
                'Write-Host "Use labels: region_A and region_B"',
                'Start-Process -FilePath $pythonExe -ArgumentList @("-m", "labelme", $imagesDir)',
            ]
        ),
        encoding="utf-8",
    )
    (out_dir / "export_labelme_masks.ps1").write_text(
        "\n".join(
            [
                "$packDir = Split-Path -Parent $MyInvocation.MyCommand.Path",
                "$repoRoot = Resolve-Path (Join-Path $packDir \"..\\..\")",
                '$script = Join-Path $repoRoot "scripts\\local\\export_paper2_followup_labelme_masks.py"',
                '$pythonExe = "E:\\code\\conda\\python.exe"',
                'if (-not (Test-Path $pythonExe)) { $pythonExe = "python" }',
                'Write-Host "Exporting LabelMe JSON to masks under $packDir\\exported_masks"',
                "& $pythonExe $script --pack-dir $packDir",
            ]
        ),
        encoding="utf-8",
    )


def _page(rows: list[dict[str, Any]]) -> str:
    cards = []
    for row in rows:
        cards.append(
            f"""
<section class="card">
  <h2>{html.escape(row['sample_id'])}</h2>
  <img src="{html.escape(Path(row['image_path']).as_posix())}" />
  <p><b>Question:</b> {html.escape(row['question_text'])}</p>
  <p><b>Answer:</b> {html.escape(row['answer_text'])}</p>
  <p><b>Type:</b> {html.escape(row.get('sample_types', ''))}</p>
  <p><b>Score:</b> {html.escape(str(row.get('selection_score', '')))} / {html.escape(row.get('selection_reasons', ''))}</p>
  <p class="todo">Use <code>region_A</code> and <code>region_B</code>. If only one true evidence source exists, draw only <code>region_A</code>.</p>
</section>"""
        )
    return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Paper2 Registry A/B Annotation Pack</title>
  <style>
    body {{ font-family: Georgia, serif; margin: 24px; background: #f6f1e8; color: #221b14; }}
    .card {{ background: white; border: 1px solid #d7c6aa; border-radius: 12px; padding: 18px; margin: 18px 0; box-shadow: 0 2px 10px #0001; }}
    img {{ max-width: 760px; width: 100%; border-radius: 8px; display: block; margin: 12px 0; }}
    code {{ background: #f2eadc; padding: 2px 5px; border-radius: 4px; }}
    .todo {{ color: #6d3e00; }}
  </style>
</head>
<body>
  <h1>Paper2 Registry A/B Annotation Pack</h1>
  <p>A/B rule: A alone or B alone should ideally still preserve enough evidence; A+B together should remove the answer evidence. If there is only one evidence source, draw only A and let the exporter mark it as one-region.</p>
  {''.join(cards)}
</body>
</html>
"""


def _guide(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Paper2 Registry A/B Annotation Pack",
        "",
        "Use LabelMe labels:",
        "",
        "- `region_A`: one answer-supporting evidence source.",
        "- `region_B`: a second distinct answer-supporting evidence source, only if it genuinely exists.",
        "",
        "If the image only has one usable evidence source, draw only `region_A`. That sample is useful as a one-region dominated control.",
        "",
    ]
    for idx, row in enumerate(rows, start=1):
        lines.extend(
            [
                f"## {idx}. {row['sample_id']}",
                "",
                f"![image]({Path(row['image_path']).as_posix()})",
                "",
                f"- **Question:** {row['question_text']}",
                f"- **Answer:** `{row['answer_text']}`",
                f"- **Type:** `{row.get('sample_types', '')}`",
                f"- **Visual labels:** `{row.get('visual_type_labels', '') or row.get('legacy_visual_type_labels', '')}`",
                f"- **Selection score:** `{row.get('selection_score', '')}`",
                f"- **Selection reasons:** `{row.get('selection_reasons', '')}`",
                "",
            ]
        )
    return "\n".join(lines)


def build_pack(out_name: str, max_images: int, extra_exclude_packs: list[Path]) -> dict[str, Any]:
    out_dir = ANNOTATION / out_name
    excluded = _excluded_images(extra_exclude_packs)
    selected, missing = _select_rows(max_images=max_images, excluded=excluded)
    images_dir = out_dir / "images"
    mask_root = out_dir / "exported_masks"
    out_rows: list[dict[str, Any]] = []
    for row in selected:
        src = Path(row["source_image_path"])
        dst = images_dir / row["image_filename"]
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        mask_dir = mask_root / dst.stem
        mask_dir.mkdir(parents=True, exist_ok=True)
        out_rows.append(
            {
                "sample_id": row.get("sample_id", ""),
                "image_filename": dst.name,
                "image_path": str(Path("images") / dst.name),
                "question_text": row.get("question_texts", ""),
                "answer_text": row.get("answer_texts", ""),
                "sample_types": row.get("sample_types", ""),
                "stage1_labels": row.get("stage1_labels", ""),
                "visual_type_labels": row.get("visual_type_labels", ""),
                "legacy_visual_type_labels": row.get("legacy_visual_type_labels", ""),
                "knowledge_level_labels": row.get("knowledge_level_labels", ""),
                "legacy_knowledge_level_labels": row.get("legacy_knowledge_level_labels", ""),
                "question_types": row.get("question_types", ""),
                "visual_structures": row.get("visual_structures", ""),
                "image_dependence_labels": row.get("image_dependence_labels", ""),
                "reasoning_operations": row.get("reasoning_operations", ""),
                "extra_evidence_labels": row.get("extra_evidence_labels", ""),
                "selection_score": row.get("selection_score", ""),
                "selection_reasons": row.get("selection_reasons", ""),
                "mask_dir": str(Path("exported_masks") / dst.stem),
                "region_A_mask_path": str(Path("exported_masks") / dst.stem / "region_A.png"),
                "region_B_mask_path": str(Path("exported_masks") / dst.stem / "region_B.png"),
                "region_A_union_B_mask_path": str(Path("exported_masks") / dst.stem / "region_A_union_B.png"),
                "annotation_status": "needs_region_masks_or_one_region_label",
            }
        )

    fields = [
        "sample_id",
        "image_filename",
        "image_path",
        "question_text",
        "answer_text",
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
        "extra_evidence_labels",
        "selection_score",
        "selection_reasons",
        "mask_dir",
        "region_A_mask_path",
        "region_B_mask_path",
        "region_A_union_B_mask_path",
        "annotation_status",
    ]
    _write_csv(out_dir / "registry_annotation_blank.csv", out_rows, fields)
    _write_csv(out_dir / "registry_annotation_missing_images.csv", missing, sorted({k for row in missing for k in row}) if missing else ["blocked_reason"])
    (out_dir / "registry_annotation_ui.html").write_text(_page(out_rows), encoding="utf-8")
    (out_dir / "QUESTION_ANSWER_LABELME_GUIDE.md").write_text(_guide(out_rows), encoding="utf-8")
    _write_helpers(out_dir)
    decision = {
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "status": "paper2_registry_annotation_pack_ready" if out_rows else "blocked_no_rows",
        "output_dir": str(out_dir),
        "selected_rows": len(out_rows),
        "selected_images": len({row["image_filename"] for row in out_rows}),
        "excluded_images": len(excluded),
        "missing_images": len(missing),
        "max_images": max_images,
    }
    (out_dir / "registry_annotation_pack_decision.json").write_text(
        json.dumps(decision, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(decision, indent=2, ensure_ascii=False))
    return decision


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a Paper2 annotation pack from the unified sample registry.")
    parser.add_argument("--out-name", default="stage1_followup_ab_evidence_pack_v4_registry60")
    parser.add_argument("--max-images", type=int, default=60)
    parser.add_argument("--exclude-pack", action="append", default=[])
    args = parser.parse_args()
    decision = build_pack(
        out_name=args.out_name,
        max_images=args.max_images,
        extra_exclude_packs=[Path(value).expanduser().resolve() for value in args.exclude_pack],
    )
    return 0 if decision["selected_rows"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
