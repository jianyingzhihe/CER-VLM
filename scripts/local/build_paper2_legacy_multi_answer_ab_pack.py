#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

from PIL import Image, ImageChops, ImageDraw


ROOT = Path(__file__).resolve().parents[2]
ANNOTATION = ROOT / "annotation"
REGISTRY = ANNOTATION / "_registry"
DEFAULT_SOURCE = REGISTRY / "legacy_raw_answer_multi_region_extra_not_current.csv"


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


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _find_image(json_path: Path, image_filename: str, data: dict[str, Any]) -> Path | None:
    candidates = []
    image_path = str(data.get("imagePath") or "").strip()
    if image_path:
        candidates.append((json_path.parent / image_path).resolve())
    candidates.extend(
        [
            json_path.with_suffix(".jpg"),
            json_path.with_suffix(".jpeg"),
            json_path.with_suffix(".png"),
            json_path.parent / image_filename,
        ]
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _points(shape: dict[str, Any]) -> list[tuple[float, float]]:
    return [(float(x), float(y)) for x, y in shape.get("points", [])]


def _draw(mask: Image.Image, shape: dict[str, Any]) -> None:
    pts = _points(shape)
    if not pts:
        return
    draw = ImageDraw.Draw(mask)
    shape_type = str(shape.get("shape_type") or "polygon").lower()
    if shape_type == "rectangle" and len(pts) >= 2:
        (x1, y1), (x2, y2) = pts[:2]
        draw.rectangle((min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)), fill=255)
    elif shape_type == "circle" and len(pts) >= 2:
        (x1, y1), (x2, y2) = pts[:2]
        r = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
        draw.ellipse((x1 - r, y1 - r, x1 + r, y1 + r), fill=255)
    elif shape_type in {"line", "linestrip"} and len(pts) >= 2:
        draw.line(pts, fill=255, width=5)
    elif shape_type == "point":
        x, y = pts[0]
        draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=255)
    else:
        draw.polygon(pts, fill=255)


def _area(mask: Image.Image) -> int:
    return sum(1 for px in mask.getdata() if px > 0)


def _answer_shapes(data: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        shape
        for shape in data.get("shapes", [])
        if str(shape.get("label", "")).strip().lower() == "answer"
    ]


def build_pack(source_csv: Path, out_name: str, include_current_ready: bool) -> dict[str, Any]:
    source_rows = _read_csv(source_csv)
    out_dir = ANNOTATION / out_name
    images_dir = out_dir / "images"
    mask_root = out_dir / "exported_masks"
    labelme_dir = out_dir / "source_labelme_json"
    out_rows: list[dict[str, Any]] = []
    blocked: list[dict[str, Any]] = []

    for row in source_rows:
        if not include_current_ready and row.get("current_status") == "already_current_AB_ready":
            continue
        json_path = Path(row.get("representative_json_path", ""))
        if not json_path.exists():
            blocked.append({**row, "blocked_reason": "missing_representative_json"})
            continue
        data = _load_json(json_path)
        answer_shapes = _answer_shapes(data)
        if len(answer_shapes) < 2:
            blocked.append({**row, "blocked_reason": "fewer_than_two_raw_answer_shapes"})
            continue
        image_filename = row.get("image_filename", "") or Path(str(data.get("imagePath", ""))).name
        image_path = _find_image(json_path, image_filename, data)
        if image_path is None:
            blocked.append({**row, "blocked_reason": "missing_image"})
            continue

        image = Image.open(image_path).convert("RGB")
        width, height = image.size
        region_a = Image.new("L", (width, height), 0)
        region_b = Image.new("L", (width, height), 0)
        _draw(region_a, answer_shapes[0])
        for shape in answer_shapes[1:]:
            _draw(region_b, shape)
        union = ImageChops.lighter(region_a, region_b)

        dst_image = images_dir / image_filename
        dst_image.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(image_path, dst_image)
        dst_json = labelme_dir / json_path.name
        dst_json.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(json_path, dst_json)

        mask_dir = mask_root / dst_image.stem
        mask_dir.mkdir(parents=True, exist_ok=True)
        region_a.save(mask_dir / "region_A.png")
        region_b.save(mask_dir / "region_B.png")
        union.save(mask_dir / "region_A_union_B.png")

        out_rows.append(
            {
                **row,
                "image_filename": dst_image.name,
                "image_path": str(Path("images") / dst_image.name),
                "mask_dir": str(Path("exported_masks") / dst_image.stem),
                "region_A_mask_path": str(Path("exported_masks") / dst_image.stem / "region_A.png"),
                "region_B_mask_path": str(Path("exported_masks") / dst_image.stem / "region_B.png"),
                "region_A_union_B_mask_path": str(Path("exported_masks") / dst_image.stem / "region_A_union_B.png"),
                "region_A_area_px": _area(region_a),
                "region_B_area_px": _area(region_b),
                "region_A_union_B_area_px": _area(union),
                "answer_shape_count": len(answer_shapes),
                "conversion_rule": "first_raw_answer_shape_to_region_A_remaining_raw_answer_shapes_to_region_B",
                "source_labelme_json_copy": str(Path("source_labelme_json") / json_path.name),
                "annotation_status": "legacy_multi_answer_converted_to_AB",
            }
        )

    fields = [
        "image_filename",
        "sample_ids",
        "question_texts",
        "answer_texts",
        "sample_types",
        "max_raw_answer_shape_count",
        "answer_shape_count",
        "annotation_packs",
        "current_status",
        "has_stage1_behavior_rows",
        "has_mask_full_composition_rows",
        "image_path",
        "mask_dir",
        "region_A_mask_path",
        "region_B_mask_path",
        "region_A_union_B_mask_path",
        "region_A_area_px",
        "region_B_area_px",
        "region_A_union_B_area_px",
        "conversion_rule",
        "representative_json_path",
        "source_labelme_json_copy",
        "annotation_status",
    ]
    _write_csv(out_dir / "legacy_multi_answer_ab_manifest.csv", out_rows, fields)
    _write_csv(
        out_dir / "legacy_multi_answer_ab_blocked.csv",
        blocked,
        sorted({key for item in blocked for key in item.keys()}) if blocked else ["blocked_reason"],
    )

    guide_lines = [
        "# Legacy Multi-Answer A/B Pack",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "Conversion rule:",
        "",
        "- First raw LabelMe `answer` shape -> `region_A`.",
        "- Remaining raw LabelMe `answer` shapes -> `region_B`.",
        "- `region_A_union_B` is generated as the union of A and B.",
        "- Old `relate` shapes are preserved in copied LabelMe JSONs but are not used for Paper2 A/B masks.",
        "",
        "This pack is local-only under `annotation/` and is not tracked by Git.",
        "",
        "## Rows",
        "",
        f"- Converted rows: `{len(out_rows)}`",
        f"- Blocked rows: `{len(blocked)}`",
        "",
    ]
    for idx, row in enumerate(out_rows, start=1):
        guide_lines.extend(
            [
                f"## {idx}. {row.get('sample_ids', '')}",
                "",
                f"![image]({Path(row['image_path']).as_posix()})",
                "",
                f"- **Question:** {row.get('question_texts', '')}",
                f"- **Answer:** `{row.get('answer_texts', '')}`",
                f"- **Answer shapes:** `{row.get('answer_shape_count', '')}`",
                f"- **A area:** `{row.get('region_A_area_px', '')}`",
                f"- **B area:** `{row.get('region_B_area_px', '')}`",
                "",
            ]
        )
    (out_dir / "README_LEGACY_MULTI_ANSWER_AB.md").write_text("\n".join(guide_lines), encoding="utf-8")

    decision = {
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "status": "legacy_multi_answer_ab_pack_ready" if out_rows else "blocked_no_converted_rows",
        "source_csv": str(source_csv),
        "output_dir": str(out_dir),
        "converted_rows": len(out_rows),
        "unique_images": len({row["image_filename"] for row in out_rows}),
        "blocked_rows": len(blocked),
        "with_stage1_behavior_rows": sum(int(row.get("has_stage1_behavior_rows") or 0) for row in out_rows),
        "with_mask_full_composition_rows": sum(int(row.get("has_mask_full_composition_rows") or 0) for row in out_rows),
        "include_current_ready": include_current_ready,
        "conversion_rule": "first raw answer shape -> region_A; remaining raw answer shapes -> region_B",
    }
    (out_dir / "legacy_multi_answer_ab_decision.json").write_text(
        json.dumps(decision, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(decision, indent=2, ensure_ascii=False))
    return decision


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert old multi-answer LabelMe annotations into Paper2 A/B masks.")
    parser.add_argument("--source-csv", default=str(DEFAULT_SOURCE))
    parser.add_argument("--out-name", default="stage1_legacy_multi_answer_ab_pack_v1")
    parser.add_argument("--include-current-ready", action="store_true")
    args = parser.parse_args()
    decision = build_pack(Path(args.source_csv).expanduser().resolve(), args.out_name, args.include_current_ready)
    return 0 if decision["converted_rows"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
