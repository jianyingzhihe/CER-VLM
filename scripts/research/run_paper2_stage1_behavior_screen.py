#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import time
from pathlib import Path
from typing import Any

import torch
from PIL import Image


MASK_NAMES = {
    "mask_A": "region_A.png",
    "mask_B": "region_B.png",
    "mask_A_union_B": "region_A_union_B.png",
    "random_A_size": "random_A_size.png",
    "random_B_size": "random_B_size.png",
    "random_union_size": "random_union_size.png",
}


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


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


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _parse_csv(raw: str) -> list[str]:
    return [part.strip() for part in raw.split(",") if part.strip()]


def _normalize_answer(text: str) -> str:
    value = (text or "").strip().lower()
    value = re.sub(r"^the answer is\s*", "", value)
    value = value.splitlines()[0] if value else ""
    value = re.split(r"[.;,\n]", value)[0].strip()
    value = re.sub(r"[^a-z0-9\u4e00-\u9fff ]+", "", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def _first_param_device(model) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _prompt(question: str) -> str:
    return f"{question} Reply with only one short sentence in exactly this format: The answer is <short answer>."


def _apply_mask(image: Image.Image, mask: Image.Image, fill_rgb: tuple[int, int, int] = (128, 128, 128)) -> Image.Image:
    base = image.convert("RGB")
    fill = Image.new("RGB", base.size, fill_rgb)
    return Image.composite(fill, base, mask.convert("L"))


def _target_candidates(tokenizer, answer: str) -> list[int]:
    variants = [answer, " " + answer, answer.capitalize(), " " + answer.capitalize()]
    out: list[int] = []
    seen: set[int] = set()
    for variant in variants:
        try:
            ids = tokenizer(variant, add_special_tokens=False)["input_ids"]
        except Exception:
            ids = tokenizer.encode(variant, add_special_tokens=False)
        if not ids:
            continue
        token_id = int(ids[0])
        if token_id not in seen:
            seen.add(token_id)
            out.append(token_id)
    return out


def _rank_and_top(logits: torch.Tensor, tokenizer, candidate_ids: list[int]) -> dict[str, Any]:
    row = logits[0, -1].float()
    probs = torch.softmax(row, dim=-1)
    best = None
    for token_id in candidate_ids:
        token_logit = float(row[int(token_id)].item())
        rank = int((row > row[int(token_id)]).sum().item() + 1)
        item = {
            "target_token_id": int(token_id),
            "target_token": tokenizer.decode([int(token_id)]),
            "target_logit": token_logit,
            "target_prob": float(probs[int(token_id)].item()),
            "target_rank": rank,
        }
        if best is None or rank < best["target_rank"]:
            best = item
    assert best is not None
    return best


def _top_wrong_token(logits: torch.Tensor, tokenizer, target_token_id: int) -> dict[str, Any]:
    row = logits[0, -1].float()
    scores = row.detach().clone()
    scores[int(target_token_id)] = -float("inf")
    token_id = int(torch.argmax(scores).item())
    return {
        "wrong_token_id": token_id,
        "wrong_token": tokenizer.decode([token_id]),
        "wrong_logit": float(row[token_id].item()),
        "wrong_rank": int((row > row[token_id]).sum().item() + 1),
    }


def _gemma_inputs(processor, image: Image.Image, question: str, answer_prefix: str, device: torch.device) -> dict[str, Any]:
    messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": question}]}]
    try:
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) + answer_prefix
        inputs = processor(text=[text], images=[image], return_tensors="pt")
    except Exception:
        inputs = processor(text=f"<start_of_image> {question}{answer_prefix}", images=image, return_tensors="pt")
    return {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}


def _qwen_inputs(processor, image: Image.Image, image_path: str, question: str, answer_prefix: str, device: torch.device) -> dict[str, Any]:
    messages = [{"role": "user", "content": [{"type": "image", "image": image_path}, {"type": "text", "text": question}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) + answer_prefix
    inputs = processor(text=[text], images=[image], return_tensors="pt")
    return {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}


def _condition_image(row: dict[str, str], condition: str, image_root: Path, mask_root: Path) -> tuple[Image.Image | None, str]:
    image_path = image_root / Path(row["image_filename"]).name
    if condition == "wrong_image":
        image_path = image_root / Path(row.get("wrong_image_filename") or row["image_filename"]).name
    if not image_path.exists():
        return None, f"missing_image:{image_path}"
    image = Image.open(image_path).convert("RGB")
    if condition in MASK_NAMES:
        mask_path = mask_root / Path(row["image_filename"]).stem / MASK_NAMES[condition]
        if not mask_path.exists():
            return None, f"missing_mask:{mask_path}"
        mask = Image.open(mask_path).convert("L").resize(image.size)
        image = _apply_mask(image, mask)
    return image, str(image_path)


def _score_condition(
    *,
    model_family: str,
    model,
    processor,
    tokenizer,
    device: torch.device,
    row: dict[str, str],
    condition: str,
    image_root: Path,
    mask_root: Path,
    answer_prefix: str,
    max_new_tokens: int,
) -> dict[str, Any]:
    image, image_path_or_reason = _condition_image(row, condition, image_root, mask_root)
    base = {
        "model_family": model_family,
        "sample_id": row.get("sample_id", ""),
        "sample_type": row.get("sample_type", ""),
        "condition": condition,
        "question_text": row.get("question_text", ""),
        "answer_text": row.get("answer_text", ""),
        "image_filename": row.get("image_filename", ""),
    }
    if image is None:
        return {**base, "status": "blocked", "blocked_reason": image_path_or_reason}
    answer = row.get("answer_text", "")
    target_ids = _target_candidates(tokenizer, answer)
    if not target_ids:
        return {**base, "status": "blocked", "blocked_reason": "target_tokenization_empty"}
    question = _prompt(row.get("question_text", ""))
    if model_family == "gemma":
        inputs = _gemma_inputs(processor, image, question, answer_prefix, device)
    else:
        inputs = _qwen_inputs(processor, image, image_path_or_reason, question, answer_prefix, device)
    with torch.inference_mode():
        outputs = model(**inputs, use_cache=False)
    target = _rank_and_top(outputs.logits, tokenizer, target_ids)
    wrong = _top_wrong_token(outputs.logits, tokenizer, int(target["target_token_id"]))
    generated_text = ""
    try:
        with torch.inference_mode():
            generated = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        prompt_len = int(inputs["input_ids"].shape[1])
        generated_text = tokenizer.decode(generated[0, prompt_len:], skip_special_tokens=True).strip()
    except Exception as exc:  # noqa: BLE001
        generated_text = f"[generation_error:{type(exc).__name__}]"
    predicted = _normalize_answer(generated_text)
    return {
        **base,
        "status": "ok",
        "blocked_reason": "",
        "target_token_id": target["target_token_id"],
        "target_token": target["target_token"],
        "target_logit": target["target_logit"],
        "target_prob": target["target_prob"],
        "target_rank": target["target_rank"],
        "wrong_token_id": wrong["wrong_token_id"],
        "wrong_token": wrong["wrong_token"],
        "wrong_logit": wrong["wrong_logit"],
        "wrong_rank": wrong["wrong_rank"],
        "target_minus_wrong_margin": float(target["target_logit"]) - float(wrong["wrong_logit"]),
        "generated_text": generated_text,
        "predicted_answer": predicted,
        "format_ok": int(bool(predicted)),
        "empty_answer": int(not bool(predicted)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Paper2 Stage1 behavior screen for composition candidates.")
    parser.add_argument("--model-family", choices=["gemma", "qwen"], required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--mask-root", required=True)
    parser.add_argument("--conditions", default="clean,wrong_image")
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=4)
    parser.add_argument("--answer-prefix", default="The answer is ")
    parser.add_argument("--out-csv", required=True)
    parser.add_argument("--out-json", required=True)
    args = parser.parse_args()

    rows = _read_csv(Path(args.manifest))
    if args.max_rows > 0:
        rows = rows[: args.max_rows]
    conditions = _parse_csv(args.conditions)
    payload = {
        "created_at": _now(),
        "script": "run_paper2_stage1_behavior_screen.py",
        "args": vars(args),
        "row_count": len(rows),
        "conditions": conditions,
        "status": "started",
    }
    _write_json(Path(args.out_json), payload)

    if args.model_family == "gemma":
        from transformers import AutoProcessor, Gemma3ForConditionalGeneration

        processor = AutoProcessor.from_pretrained(args.model_name, local_files_only=True)
        tokenizer = processor.tokenizer
        model = Gemma3ForConditionalGeneration.from_pretrained(
            args.model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            local_files_only=True,
            attn_implementation="eager",
        )
    else:
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        processor = AutoProcessor.from_pretrained(args.model_name, local_files_only=True)
        tokenizer = processor.tokenizer
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            args.model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            local_files_only=True,
            attn_implementation="eager",
        )
    model.eval()
    device = _first_param_device(model)

    out_rows: list[dict[str, Any]] = []
    for idx, row in enumerate(rows, 1):
        print(f"[paper2-stage1] {args.model_family} row {idx}/{len(rows)} sample={row.get('sample_id')}", flush=True)
        for condition in conditions:
            out_rows.append(
                _score_condition(
                    model_family=args.model_family,
                    model=model,
                    processor=processor,
                    tokenizer=tokenizer,
                    device=device,
                    row=row,
                    condition=condition,
                    image_root=Path(args.image_root),
                    mask_root=Path(args.mask_root),
                    answer_prefix=args.answer_prefix,
                    max_new_tokens=args.max_new_tokens,
                )
            )
    fields = [
        "model_family",
        "sample_id",
        "sample_type",
        "condition",
        "status",
        "blocked_reason",
        "question_text",
        "answer_text",
        "image_filename",
        "target_token_id",
        "target_token",
        "target_logit",
        "target_prob",
        "target_rank",
        "wrong_token_id",
        "wrong_token",
        "wrong_logit",
        "wrong_rank",
        "target_minus_wrong_margin",
        "generated_text",
        "predicted_answer",
        "format_ok",
        "empty_answer",
    ]
    _write_csv(Path(args.out_csv), out_rows, fields)
    payload.update(
        {
            "status": "completed",
            "output_rows": len(out_rows),
            "ok_rows": sum(row.get("status") == "ok" for row in out_rows),
            "blocked_rows": sum(row.get("status") != "ok" for row in out_rows),
            "completed_at": _now(),
        }
    )
    _write_json(Path(args.out_json), payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
