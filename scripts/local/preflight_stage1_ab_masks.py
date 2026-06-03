#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path
from typing import Any

from PIL import Image, ImageChops, ImageDraw


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PACK_ROOT = PROJECT_ROOT / "annotation" / "stage1_ab_evidence_pack_v1"
RESULTS_ROOT = PROJECT_ROOT / "doc" / "experiments" / "stage1" / "results"


PRIMARY_MASKS = [
    ("region_A_mask_path", "region_A.png"),
    ("region_B_mask_path", "region_B.png"),
]
RANDOM_MASKS = [
    ("region_A_mask_path", "random_A_size_mask_path", "random_A_size.png"),
    ("region_B_mask_path", "random_B_size_mask_path", "random_B_size.png"),
    ("region_A_union_B_mask_path", "random_union_size_mask_path", "random_union_size.png"),
]


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


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


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _mask(path: Path, size: tuple[int, int]) -> Image.Image:
    return Image.open(path).convert("L").resize(size)


def _area(mask: Image.Image) -> int:
    return sum(1 for value in mask.getdata() if value > 0)


def _iou(a: Image.Image, b: Image.Image) -> float:
    aa = a.convert("1")
    bb = b.convert("1")
    inter = _area(ImageChops.logical_and(aa, bb).convert("L"))
    union = _area(ImageChops.logical_or(aa, bb).convert("L"))
    return inter / union if union else 0.0


def _bbox_size(mask: Image.Image) -> tuple[int, int]:
    bbox = mask.convert("L").getbbox()
    if bbox is None:
        return (1, 1)
    return max(1, bbox[2] - bbox[0]), max(1, bbox[3] - bbox[1])


def _random_rect_mask(reference: Image.Image, avoid: Image.Image, max_iou: float, seed: int) -> Image.Image:
    rng = random.Random(seed)
    width, height = reference.size
    rect_w, rect_h = _bbox_size(reference)
    rect_w = min(rect_w, width)
    rect_h = min(rect_h, height)
    best: Image.Image | None = None
    best_iou = 999.0
    for _ in range(200):
        x0 = rng.randint(0, max(0, width - rect_w))
        y0 = rng.randint(0, max(0, height - rect_h))
        candidate = Image.new("L", reference.size, 0)
        ImageDraw.Draw(candidate).rectangle([x0, y0, x0 + rect_w, y0 + rect_h], fill=255)
        overlap = _iou(candidate, avoid)
        if overlap < best_iou:
            best = candidate
            best_iou = overlap
        if overlap <= max_iou:
            return candidate
    assert best is not None
    return best


def _ensure_union(row: dict[str, str], size: tuple[int, int]) -> tuple[bool, str]:
    union_path = Path(row["region_A_union_B_mask_path"])
    if union_path.exists():
        return True, "provided"
    a_path = Path(row["region_A_mask_path"])
    b_path = Path(row["region_B_mask_path"])
    if not a_path.exists() or not b_path.exists():
        return False, "missing_region_A_or_B"
    union_path.parent.mkdir(parents=True, exist_ok=True)
    union = ImageChops.lighter(_mask(a_path, size), _mask(b_path, size))
    union.save(union_path)
    return True, "generated_from_A_B"


def preflight(pack_manifest: Path, tag: str, generate_random: bool) -> dict[str, Any]:
    rows = _read_csv(pack_manifest)
    valid: list[dict[str, Any]] = []
    blocked: list[dict[str, Any]] = []
    for row in rows:
        reasons: list[str] = []
        image_path = Path(row.get("local_image_path", ""))
        wrong_image_path = Path(row.get("wrong_image_path", ""))
        if not image_path.exists():
            reasons.append("image_missing")
            size = (1, 1)
        else:
            size = Image.open(image_path).size
        if not wrong_image_path.exists():
            reasons.append("wrong_image_missing")
        for key, _name in PRIMARY_MASKS:
            if not Path(row.get(key, "")).exists():
                reasons.append(f"{key}_missing")
        union_ok, union_source = _ensure_union(row, size)
        if not union_ok:
            reasons.append("region_A_union_B_mask_path_missing")
        if not reasons:
            union = _mask(Path(row["region_A_union_B_mask_path"]), size)
            union_frac = _area(union) / max(1, size[0] * size[1])
            if union_frac <= 0.0:
                reasons.append("union_mask_empty")
            if union_frac >= 0.80:
                reasons.append("union_mask_too_large")
            if generate_random:
                for src_key, dst_key, _name in RANDOM_MASKS:
                    src = _mask(Path(row[src_key]), size)
                    random_mask = _random_rect_mask(src, union, max_iou=0.10, seed=hash((row["sample_id"], dst_key)) & 0xFFFF)
                    dst = Path(row[dst_key])
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    random_mask.save(dst)
            for _src_key, dst_key, _name in RANDOM_MASKS:
                if not Path(row.get(dst_key, "")).exists():
                    reasons.append(f"{dst_key}_missing")
        out = dict(row)
        out["union_source"] = union_source
        out["preflight_status"] = "valid" if not reasons else "blocked"
        out["blocked_reason"] = "|".join(reasons)
        if reasons:
            blocked.append(out)
        else:
            valid.append(out)

    fields = list(rows[0].keys()) if rows else []
    for extra in ["union_source", "preflight_status", "blocked_reason"]:
        if extra not in fields:
            fields.append(extra)
    out_manifest = RESULTS_ROOT / f"stage1_behavior_manifest_{tag}.csv"
    blocked_path = RESULTS_ROOT / f"blocked_annotation_{tag}.csv"
    _write_csv(out_manifest, valid, fields)
    _write_csv(blocked_path, blocked, fields)
    status = "ready_for_gpu1_smoke" if valid else "blocked_missing_ab_masks"
    summary = {
        "created_at": _now(),
        "status": status,
        "tag": tag,
        "pack_manifest": str(pack_manifest),
        "valid_rows": len(valid),
        "blocked_rows": len(blocked),
        "behavior_manifest": str(out_manifest),
        "blocked_annotation": str(blocked_path),
        "generate_random": generate_random,
    }
    _write_json(RESULTS_ROOT / f"stage1_preflight_{tag}_decision.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Preflight Paper2 Stage1 A/B masks before GPU behavior screening.")
    parser.add_argument("--pack-manifest", default=str(PACK_ROOT / "stage1_candidate_manifest.csv"))
    parser.add_argument("--tag", default="v1")
    parser.add_argument("--no-generate-random", action="store_true")
    args = parser.parse_args()
    summary = preflight(Path(args.pack_manifest), args.tag, generate_random=not args.no_generate_random)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
