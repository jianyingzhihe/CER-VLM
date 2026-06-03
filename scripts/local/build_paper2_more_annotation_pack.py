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
STAGE1 = ROOT / "doc" / "experiments" / "paper2" / "stage1"
PAIRED = STAGE1 / "paper2_stage1_behavior_viability_v1_paired.csv"
STRONG = STAGE1 / "paper2_stage1_behavior_viability_v1_strong_cases.csv"
VIABILITY = [
    STAGE1 / "stage1_composition_screen_viability_v1_gemma.csv",
    STAGE1 / "stage1_composition_screen_viability_v1_qwen.csv",
    STAGE1 / "stage1_composition_screen_viability_v1_manifest.csv",
]
DEFAULT_EXCLUDE_PACKS = [
    ANNOTATION / "stage1_followup_ab_evidence_pack_v2_all28",
    ANNOTATION / "stage1_followup_ab_evidence_pack_v2_expanded",
    ANNOTATION / "stage1_followup_ab_evidence_pack_v2",
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


def _image_index() -> dict[str, dict[str, str]]:
    index: dict[str, dict[str, str]] = {}
    for source in VIABILITY:
        for row in _read_csv(source):
            sample_id = row.get("sample_id", "").strip()
            if not sample_id or sample_id in index:
                continue
            image_filename = row.get("image_filename", "").strip()
            if not image_filename:
                continue
            index[sample_id] = {
                "image_filename": image_filename,
                "local_image_path": row.get("local_image_path", "").strip(),
                "original_image_path": row.get("original_image_path", "").strip(),
            }
    return index


def _find_image(sample_id: str, image_filename: str, indexed_paths: dict[str, str]) -> Path | None:
    for key in ["local_image_path", "original_image_path"]:
        value = indexed_paths.get(key, "").strip()
        if value:
            path = Path(value)
            if path.exists():
                return path
    if image_filename:
        roots = [
            ANNOTATION,
            ROOT,
            BRIDGING_ROOT / "annotation",
        ]
        for root in roots:
            if not root.exists():
                continue
            matches = sorted(root.rglob(image_filename))
            if matches:
                return matches[0]
    suffix = sample_id.replace("okvqa_val_", "")
    for root in [ANNOTATION, BRIDGING_ROOT / "annotation"]:
        if not root.exists():
            continue
        for path in sorted(root.rglob("images/COCO_val2014_*.jpg")):
            digits = path.stem.replace("COCO_val2014_", "").lstrip("0") or "0"
            if digits == suffix.lstrip("0"):
                return path
    return None


def _excluded_images(extra_packs: list[Path]) -> set[str]:
    excluded: set[str] = set()
    for pack in [*DEFAULT_EXCLUDE_PACKS, *extra_packs]:
        blank = pack / "stage1_followup_annotation_blank.csv"
        if blank.exists():
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


def _score(row: dict[str, str]) -> float:
    try:
        return float(row.get("strong_case_score", "") or 0.0)
    except ValueError:
        return 0.0


def _candidate_rows(min_score: float) -> list[dict[str, str]]:
    rows = _read_csv(PAIRED)
    if not rows:
        rows = _read_csv(STRONG)
    out: list[dict[str, str]] = []
    for row in rows:
        if row.get("format_ok_both", "1") not in {"", "1", "1.0", "True", "true"}:
            continue
        if row.get("decoded_changed", "0") not in {"1", "1.0", "True", "true"}:
            continue
        if _score(row) < min_score:
            continue
        out.append(row)
    return sorted(out, key=_score, reverse=True)


def _select_unique_images(rows: list[dict[str, str]], max_images: int, excluded: set[str]) -> list[dict[str, Any]]:
    index = _image_index()
    selected: list[dict[str, Any]] = []
    seen_images: set[str] = set()
    for row in rows:
        sample_id = row.get("sample_id", "").strip()
        image_info = index.get(sample_id, {})
        image_filename = image_info.get("image_filename", "")
        if not image_filename or image_filename in excluded or image_filename in seen_images:
            continue
        image = _find_image(sample_id, image_filename, image_info)
        if image is None:
            continue
        selected.append({**row, "image_filename": image_filename, "source_image_path": str(image)})
        seen_images.add(image_filename)
        if len(seen_images) >= max_images:
            break
    return selected


def _page(rows: list[dict[str, Any]]) -> str:
    cards = []
    for row in rows:
        cards.append(
            f"""
<section class="card">
  <h2>{html.escape(row['sample_id'])} / {html.escape(row['model_family'])}</h2>
  <img src="{html.escape(Path(row['image_path']).as_posix())}" />
  <p><b>Question:</b> {html.escape(row['question_text'])}</p>
  <p><b>Answer:</b> {html.escape(row['answer_text'])}</p>
  <p><b>Type:</b> {html.escape(row['sample_type'])}</p>
  <p><b>Clean predicted:</b> {html.escape(row.get('clean_predicted_answer', ''))}</p>
  <p><b>Wrong-image predicted:</b> {html.escape(row.get('wrong_predicted_answer', ''))}</p>
  <p><b>Score:</b> {html.escape(str(row.get('strong_case_score', '')))}</p>
  <p class="todo">Label with <code>region_A</code> and optionally <code>region_B</code>.</p>
</section>"""
        )
    return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Paper2 More A/B Evidence Candidates</title>
  <style>
    body {{ font-family: Georgia, serif; margin: 24px; background: #f6f1e8; color: #221b14; }}
    .card {{ background: white; border: 1px solid #d7c6aa; border-radius: 12px; padding: 18px; margin: 18px 0; box-shadow: 0 2px 10px #0001; }}
    img {{ max-width: 760px; width: 100%; border-radius: 8px; display: block; margin: 12px 0; }}
    code {{ background: #f2eadc; padding: 2px 5px; border-radius: 4px; }}
    .todo {{ color: #6d3e00; }}
  </style>
</head>
<body>
  <h1>Paper2 More A/B Evidence Candidates</h1>
  <p>Draw <code>region_A</code> and <code>region_B</code> only when the image has two distinct answer-supporting evidence sources. If there is only one real evidence source, draw <code>region_A</code> and treat the sample as one-region dominated.</p>
  {''.join(cards)}
</body>
</html>
"""


def _guide(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Paper2 More A/B Evidence Candidates",
        "",
        "LabelMe labels:",
        "",
        "- `region_A`: one answer-supporting evidence source.",
        "- `region_B`: second distinct answer-supporting evidence source, only if it truly exists.",
        "",
        "Paper2 A/B rule: masking A alone or B alone should ideally still allow the answer; masking A+B together should break it.",
        "",
        "If the image only has one usable evidence region, draw only `region_A`. That is useful: it marks the sample as one-region dominated.",
        "",
    ]
    for idx, row in enumerate(rows, start=1):
        lines.extend(
            [
                f"## {idx}. {row['sample_id']} / {row['model_family']}",
                "",
                f"![image]({Path(row['image_path']).as_posix()})",
                "",
                f"- **Question:** {row['question_text']}",
                f"- **Answer:** `{row['answer_text']}`",
                f"- **Type:** `{row['sample_type']}`",
                f"- **Clean predicted:** `{row.get('clean_predicted_answer', '')}`",
                f"- **Wrong-image predicted:** `{row.get('wrong_predicted_answer', '')}`",
                f"- **Strong-case score:** `{row.get('strong_case_score', '')}`",
                "",
            ]
        )
    return "\n".join(lines)


def _write_helpers(out_dir: Path) -> None:
    (out_dir / "launch_labelme.ps1").write_text(
        "\n".join(
            [
                "$packDir = Split-Path -Parent $MyInvocation.MyCommand.Path",
                '$imagesDir = Join-Path $packDir "images"',
                '$pythonExe = "E:\\code\\conda\\python.exe"',
                "if (-not (Test-Path $pythonExe)) { $pythonExe = \"python\" }",
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
                "if (-not (Test-Path $pythonExe)) { $pythonExe = \"python\" }",
                'Write-Host "Exporting LabelMe JSON to masks under $packDir\\exported_masks"',
                "& $pythonExe $script --pack-dir $packDir",
            ]
        ),
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a larger Paper2 A/B annotation pack from Stage1 behavior rows.")
    parser.add_argument("--out-name", default="stage1_followup_ab_evidence_pack_v3_more")
    parser.add_argument("--max-images", type=int, default=24)
    parser.add_argument("--min-score", type=float, default=0.0)
    parser.add_argument("--exclude-pack", action="append", default=[])
    args = parser.parse_args()

    extra_packs = [Path(value).expanduser().resolve() for value in args.exclude_pack]
    excluded = _excluded_images(extra_packs)
    selected = _select_unique_images(_candidate_rows(args.min_score), args.max_images, excluded)
    out_dir = ANNOTATION / args.out_name
    images_dir = out_dir / "images"
    mask_root = out_dir / "exported_masks"
    rows: list[dict[str, Any]] = []
    for row in selected:
        src = Path(row["source_image_path"])
        dst = images_dir / row["image_filename"]
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        mask_dir = mask_root / dst.stem
        mask_dir.mkdir(parents=True, exist_ok=True)
        rows.append(
            {
                **row,
                "image_path": str(Path("images") / dst.name),
                "mask_dir": str(Path("exported_masks") / dst.stem),
                "region_A_mask_path": str(Path("exported_masks") / dst.stem / "region_A.png"),
                "region_B_mask_path": str(Path("exported_masks") / dst.stem / "region_B.png"),
                "region_A_union_B_mask_path": str(Path("exported_masks") / dst.stem / "region_A_union_B.png"),
                "annotation_status": "needs_region_masks_or_one_region_label",
            }
        )

    fields = [
        "model_family",
        "sample_id",
        "sample_type",
        "question_text",
        "answer_text",
        "clean_predicted_answer",
        "wrong_predicted_answer",
        "decoded_changed",
        "clean_exact_answer",
        "strong_case_score",
        "image_filename",
        "image_path",
        "mask_dir",
        "region_A_mask_path",
        "region_B_mask_path",
        "region_A_union_B_mask_path",
        "annotation_status",
    ]
    _write_csv(out_dir / "stage1_more_annotation_blank.csv", rows, fields)
    (out_dir / "stage1_more_annotation_ui.html").write_text(_page(rows), encoding="utf-8")
    (out_dir / "QUESTION_ANSWER_LABELME_GUIDE.md").write_text(_guide(rows), encoding="utf-8")
    _write_helpers(out_dir)
    decision = {
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "status": "paper2_more_annotation_pack_ready" if rows else "blocked_no_new_images",
        "output_dir": str(out_dir),
        "selected_rows": len(rows),
        "selected_images": len({row["image_filename"] for row in rows}),
        "excluded_images": len(excluded),
        "max_images": args.max_images,
        "min_score": args.min_score,
    }
    (out_dir / "stage1_more_annotation_pack_decision.json").write_text(
        json.dumps(decision, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(decision, indent=2, ensure_ascii=False))
    return 0 if rows else 2


if __name__ == "__main__":
    raise SystemExit(main())
