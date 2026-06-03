#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import html
import json
import re
import shutil
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
BRIDGING_ROOT = ROOT.parent
ANNOTATION = ROOT / "annotation"
REGISTRY = ANNOTATION / "_registry"
DEFAULT_SOURCE = BRIDGING_ROOT / "doc" / "5.9" / "factorization_okvqa5046_t200_results" / "B_direct_eval.csv"
COCO_URL = "http://images.cocodataset.org/val2014/{image_filename}"


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


def _norm_question(question: str) -> str:
    question = re.sub(r"\s*Reply with only one short sentence.*$", "", question).strip()
    return re.sub(r"\s+", " ", question)


def _excluded_sets() -> tuple[set[str], set[str], set[str]]:
    samples: set[str] = set()
    images: set[str] = set()
    questions: set[str] = set()
    registry_files = [
        REGISTRY / "paper2_sample_registry.csv",
        REGISTRY / "paper2_ab_ready_pool_by_image_v1.csv",
        REGISTRY / "paper2_ab_ready_pool_by_sample_v1.csv",
        REGISTRY / "paper2_one_region_or_not_splittable_pool_by_image_v1.csv",
    ]
    for path in registry_files:
        for row in _read_csv(path):
            for key in ["sample_id", "sample_ids"]:
                for value in str(row.get(key, "")).split(" | "):
                    if value.strip():
                        samples.add(value.strip())
            for key in ["image_filename", "image_filenames"]:
                for value in str(row.get(key, "")).split(" | "):
                    if value.strip():
                        images.add(value.strip())
            for key in ["question_text", "question_texts"]:
                value = str(row.get(key, "")).strip()
                if value:
                    questions.add(value.lower())

    for path in ANNOTATION.rglob("*.csv"):
        if path.name not in {
            "stage1_followup_annotation_blank.csv",
            "stage1_more_annotation_blank.csv",
            "registry_annotation_blank.csv",
            "legacy_multi_answer_ab_manifest.csv",
            "unseen_okvqa_annotation_blank.csv",
        }:
            continue
        for row in _read_csv(path):
            for key in ["sample_id", "sample_ids"]:
                for value in str(row.get(key, "")).split(" | "):
                    if value.strip():
                        samples.add(value.strip())
            image = str(row.get("image_filename", "")).strip()
            if image:
                images.add(image)
            question = str(row.get("question_text", "") or row.get("question_texts", "")).strip()
            if question:
                questions.add(question.lower())
    return samples, images, questions


def _image_filename(row: dict[str, str]) -> str:
    image_id = row.get("image_id", "").strip()
    if image_id.isdigit():
        return f"COCO_val2014_{int(image_id):012d}.jpg"
    image_path = row.get("image_path", "").strip()
    if image_path:
        return Path(image_path).name
    return ""


def _score(question: str, answer: str, predicted: str, correct: str) -> tuple[int, str]:
    q = question.lower()
    score = 0
    reasons: list[str] = []
    phrase_weights = [
        ("sign", 8),
        ("brand", 8),
        ("language", 8),
        ("which", 6),
        ("shown", 5),
        ("picture", 5),
        ("image", 5),
        ("what kind", 5),
        ("what type", 5),
        ("how many", 5),
        ("why", 4),
        ("where", 4),
        ("color", 4),
        ("pattern", 4),
        ("sport", 3),
        ("animal", 3),
        ("person", 3),
        ("name", 3),
    ]
    for phrase, weight in phrase_weights:
        if phrase in q:
            score += weight
            reasons.append(phrase.replace(" ", "_"))
    if correct in {"1", "1.0", "true", "True"}:
        score += 4
        reasons.append("old_model_correct")
    if predicted and answer and predicted.strip().lower() == answer.strip().lower():
        score += 4
        reasons.append("predicted_exact_answer")
    if len(answer.split()) <= 3:
        score += 2
        reasons.append("short_answer")
    if not reasons:
        reasons.append("unseen_fallback")
    return score, ";".join(reasons)


def _select(source: Path, max_images: int) -> list[dict[str, Any]]:
    excluded_samples, excluded_images, excluded_questions = _excluded_sets()
    rows: list[dict[str, Any]] = []
    seen_images: set[str] = set()
    seen_questions: set[str] = set()
    for row in _read_csv(source):
        sample_id = row.get("sample_id", "").strip()
        image_filename = _image_filename(row)
        question = _norm_question(row.get("question", ""))
        answer = row.get("gold_answer", "").strip() or row.get("majority_answer", "").strip()
        if not sample_id or not image_filename or not question or not answer:
            continue
        if sample_id in excluded_samples or image_filename in excluded_images or question.lower() in excluded_questions:
            continue
        if image_filename in seen_images or question.lower() in seen_questions:
            continue
        score, reasons = _score(question, answer, row.get("predicted_answer", ""), row.get("correct", ""))
        rows.append(
            {
                "sample_id": sample_id,
                "question_id": row.get("question_id", ""),
                "image_id": row.get("image_id", ""),
                "image_filename": image_filename,
                "question_text": question,
                "answer_text": answer,
                "majority_answer": row.get("majority_answer", ""),
                "old_predicted_answer": row.get("predicted_answer", ""),
                "old_correct": row.get("correct", ""),
                "old_vqa_score": row.get("vqa_score", ""),
                "selection_score": score,
                "selection_reasons": reasons,
                "image_url": COCO_URL.format(image_filename=image_filename),
            }
        )
        seen_images.add(image_filename)
        seen_questions.add(question.lower())
    rows.sort(key=lambda item: (int(item["selection_score"]), item["sample_id"]), reverse=True)
    return rows[:max_images]


def _download(url: str, dst: Path) -> tuple[bool, str]:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and dst.stat().st_size > 0:
        return True, "cached"
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(request, timeout=30) as response:
            tmp = dst.with_suffix(dst.suffix + ".tmp")
            with tmp.open("wb") as handle:
                shutil.copyfileobj(response, handle)
            tmp.replace(dst)
        return True, "downloaded"
    except Exception as exc:  # pragma: no cover - defensive for network hiccups.
        return False, str(exc)


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
  <p><b>Old predicted:</b> {html.escape(row.get('old_predicted_answer', ''))}</p>
  <p><b>Score:</b> {html.escape(str(row.get('selection_score', '')))} / {html.escape(row.get('selection_reasons', ''))}</p>
</section>"""
        )
    return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Paper2 Unseen OKVQA A/B Annotation Pack</title>
  <style>
    body {{ font-family: Georgia, serif; margin: 24px; background: #f6f1e8; color: #221b14; }}
    .card {{ background: white; border: 1px solid #d7c6aa; border-radius: 12px; padding: 18px; margin: 18px 0; box-shadow: 0 2px 10px #0001; }}
    img {{ max-width: 760px; width: 100%; border-radius: 8px; display: block; margin: 12px 0; }}
  </style>
</head>
<body>
  <h1>Paper2 Unseen OKVQA A/B Annotation Pack</h1>
  <p>All selected sample ids, questions, and images are excluded from previous Paper2/Paper1 annotation registries and annotation packs.</p>
  {''.join(cards)}
</body>
</html>
"""


def _guide(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Paper2 Unseen OKVQA A/B Annotation Pack",
        "",
        "These samples were selected after excluding all sample ids, questions, and image filenames already present in the annotation registry/packs.",
        "",
        "Use LabelMe labels:",
        "",
        "- `region_A`: one answer-supporting evidence source.",
        "- `region_B`: a second distinct answer-supporting evidence source, only if it genuinely exists.",
        "",
        "If the image only has one usable evidence source, draw only `region_A`; if it cannot be split, leave it unannotated. Both are useful one-region/not-splittable controls.",
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
                f"- **Old predicted:** `{row.get('old_predicted_answer', '')}`",
                f"- **Selection score:** `{row.get('selection_score', '')}`",
                f"- **Selection reasons:** `{row.get('selection_reasons', '')}`",
                "",
            ]
        )
    return "\n".join(lines)


def build_pack(source: Path, out_name: str, max_images: int) -> dict[str, Any]:
    out_dir = ANNOTATION / out_name
    images_dir = out_dir / "images"
    mask_root = out_dir / "exported_masks"
    selected = _select(source, max_images=max_images)
    out_rows: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []
    for row in selected:
        dst = images_dir / row["image_filename"]
        ok, status = _download(row["image_url"], dst)
        if not ok:
            failed.append({**row, "download_error": status})
            continue
        mask_dir = mask_root / dst.stem
        mask_dir.mkdir(parents=True, exist_ok=True)
        out_rows.append(
            {
                **row,
                "image_path": str(Path("images") / dst.name),
                "download_status": status,
                "mask_dir": str(Path("exported_masks") / dst.stem),
                "region_A_mask_path": str(Path("exported_masks") / dst.stem / "region_A.png"),
                "region_B_mask_path": str(Path("exported_masks") / dst.stem / "region_B.png"),
                "region_A_union_B_mask_path": str(Path("exported_masks") / dst.stem / "region_A_union_B.png"),
                "annotation_status": "needs_region_masks_or_not_splittable",
            }
        )
    fields = [
        "sample_id",
        "question_id",
        "image_id",
        "image_filename",
        "image_path",
        "question_text",
        "answer_text",
        "majority_answer",
        "old_predicted_answer",
        "old_correct",
        "old_vqa_score",
        "selection_score",
        "selection_reasons",
        "image_url",
        "download_status",
        "mask_dir",
        "region_A_mask_path",
        "region_B_mask_path",
        "region_A_union_B_mask_path",
        "annotation_status",
    ]
    _write_csv(out_dir / "unseen_okvqa_annotation_blank.csv", out_rows, fields)
    _write_csv(out_dir / "unseen_okvqa_download_failed.csv", failed, sorted({k for row in failed for k in row}) if failed else ["download_error"])
    (out_dir / "unseen_okvqa_annotation_ui.html").write_text(_page(out_rows), encoding="utf-8")
    (out_dir / "QUESTION_ANSWER_LABELME_GUIDE.md").write_text(_guide(out_rows), encoding="utf-8")
    _write_helpers(out_dir)
    decision = {
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "status": "paper2_unseen_okvqa_annotation_pack_ready" if out_rows else "blocked_no_rows",
        "output_dir": str(out_dir),
        "source_csv": str(source),
        "selected_requested": max_images,
        "selected_rows": len(out_rows),
        "selected_images": len({row["image_filename"] for row in out_rows}),
        "download_failed": len(failed),
    }
    (out_dir / "unseen_okvqa_annotation_pack_decision.json").write_text(
        json.dumps(decision, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(decision, indent=2, ensure_ascii=False))
    return decision


def main() -> int:
    parser = argparse.ArgumentParser(description="Build an annotation pack from OKVQA samples unseen in prior annotation registries.")
    parser.add_argument("--source-csv", default=str(DEFAULT_SOURCE))
    parser.add_argument("--out-name", default="stage1_followup_ab_evidence_pack_v5_unseen60")
    parser.add_argument("--max-images", type=int, default=60)
    args = parser.parse_args()
    decision = build_pack(Path(args.source_csv).expanduser().resolve(), args.out_name, args.max_images)
    return 0 if decision["selected_rows"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
