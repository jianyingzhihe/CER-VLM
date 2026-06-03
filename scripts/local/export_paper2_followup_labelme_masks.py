#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any

from PIL import Image, ImageChops, ImageDraw


ALIASES = {
    "region_a": "region_A",
    "a": "region_A",
    "answer": "region_A",
    "region_b": "region_B",
    "b": "region_B",
    "relate": "region_B",
    "context": "region_B",
}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _norm_label(label: str) -> str:
    key = label.strip().lower().replace("-", "_").replace(" ", "_")
    compact = key.replace("_", "")
    if compact in {"regiona", "answera", "evidencea"}:
        return "region_A"
    if re.fullmatch(r"(region|answer|evidence)[b-z]", compact):
        return "region_B"
    return ALIASES.get(key, label.strip())


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


def _find_image(images_dir: Path, stem: str) -> Path | None:
    for ext in [".jpg", ".jpeg", ".png", ".webp", ".bmp"]:
        candidate = images_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def _area(mask: Image.Image) -> int:
    return sum(1 for px in mask.getdata() if px > 0)


def export_pack(pack_dir: Path) -> dict[str, Any]:
    images_dir = pack_dir / "images"
    out_root = pack_dir / "exported_masks"
    out_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for json_path in sorted(images_dir.glob("*.json")):
        image_path = _find_image(images_dir, json_path.stem)
        if image_path is None:
            missing.append(f"{json_path.name}:missing_image")
            continue
        image = Image.open(image_path).convert("RGB")
        width, height = image.size
        data = _read_json(json_path)
        masks = {
            "region_A": Image.new("L", (width, height), 0),
            "region_B": Image.new("L", (width, height), 0),
        }
        labels_seen: list[str] = []
        for shape in data.get("shapes", []):
            label = _norm_label(str(shape.get("label", "")))
            labels_seen.append(label)
            if label in masks:
                _draw(masks[label], shape)

        union = ImageChops.lighter(masks["region_A"], masks["region_B"])
        out_dir = out_root / image_path.stem
        out_dir.mkdir(parents=True, exist_ok=True)
        masks["region_A"].save(out_dir / "region_A.png")
        masks["region_B"].save(out_dir / "region_B.png")
        union.save(out_dir / "region_A_union_B.png")

        row = {
            "image_filename": image_path.name,
            "json_filename": json_path.name,
            "labels_seen": ",".join(labels_seen),
            "region_A_area_px": _area(masks["region_A"]),
            "region_B_area_px": _area(masks["region_B"]),
            "region_A_union_B_area_px": _area(union),
            "status": "ok",
            "mask_dir": str(out_dir),
        }
        if row["region_A_area_px"] == 0:
            row["status"] = "missing_region_A"
        if row["region_B_area_px"] == 0:
            row["status"] = "missing_region_B" if row["status"] == "ok" else row["status"] + ";missing_region_B"
        rows.append(row)

    summary = pack_dir / "labelme_mask_export_summary.csv"
    fields = [
        "image_filename",
        "json_filename",
        "labels_seen",
        "region_A_area_px",
        "region_B_area_px",
        "region_A_union_B_area_px",
        "status",
        "mask_dir",
    ]
    with summary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    decision = {
        "pack_dir": str(pack_dir),
        "json_files": len(rows),
        "missing": missing,
        "ok_rows": sum(1 for row in rows if row["status"] == "ok"),
        "summary_csv": str(summary),
        "status": "ok" if rows and all(row["status"] == "ok" for row in rows) else "needs_review",
    }
    (pack_dir / "labelme_mask_export_decision.json").write_text(
        json.dumps(decision, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return decision


def main() -> int:
    parser = argparse.ArgumentParser(description="Export Paper2 follow-up LabelMe JSON to region_A/B/union masks.")
    parser.add_argument("--pack-dir", required=True)
    args = parser.parse_args()
    decision = export_pack(Path(args.pack_dir).expanduser().resolve())
    print(json.dumps(decision, indent=2, ensure_ascii=False))
    return 0 if decision["json_files"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
