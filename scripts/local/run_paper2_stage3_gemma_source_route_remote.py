#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import importlib.util
import io
import json
import posixpath
import re
import socket
import sys
import time
from pathlib import Path
from typing import Any

from PIL import Image


PAPER2_ROOT = Path(__file__).resolve().parents[2]
BRIDGING_ROOT = PAPER2_ROOT.parent
REMOTE_ROOT = "/root/autodl-tmp/tca-reasoning/circuit_tracer_vlm"
REMOTE_STAGE = "/root/autodl-tmp/tca-reasoning/paper2_stage3_gemma_source_route"
REMOTE_ASSETS = f"{REMOTE_STAGE}/assets"
STAGE3 = PAPER2_ROOT / "doc" / "experiments" / "paper2" / "stage3_mechanism"
ASSETS = PAPER2_ROOT / "annotation" / "stage3_gemma_source_route_assets_v1"
PREFIX = "paper2_stage3_gemma_source_route"
CONDITIONS = ["mask_A", "mask_B", "mask_A_union_B", "random_union_size"]
MASK_NAMES = {
    "mask_A": "region_A.png",
    "mask_B": "region_B.png",
    "mask_A_union_B": "region_A_union_B.png",
    "random_union_size": "random_union_size.png",
}


def _load_base_runner():
    path = BRIDGING_ROOT / "scripts" / "local" / "run_stage2g_cross_model_remote.py"
    spec = importlib.util.spec_from_file_location("stage2g_runner", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _csv_text(rows: list[dict[str, Any]], fields: list[str]) -> str:
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=fields)
    writer.writeheader()
    for row in rows:
        writer.writerow({field: row.get(field, "") for field in fields})
    return buf.getvalue()


def _put_text(base, sftp, remote: str, text: str) -> None:
    base._mkdir_p(sftp, posixpath.dirname(remote))
    with sftp.file(remote, "w") as handle:
        handle.write(text.replace("\r\n", "\n"))


def _apply_mask(image: Image.Image, mask: Image.Image) -> Image.Image:
    base = image.convert("RGB")
    fill = Image.new("RGB", base.size, (128, 128, 128))
    return Image.composite(fill, base, mask.convert("L").resize(base.size))


def _prepare_condition_assets(tag: str) -> tuple[dict[str, Path], dict[str, Path], Path]:
    prompt_runs = STAGE3 / f"paper2_stage3_mechanism_gemma_prompt_runs_{tag}.csv"
    rows = _read_csv(prompt_runs)
    if len(rows) != 1:
        raise RuntimeError(f"expected exactly 1 Gemma prompt row, got {len(rows)} from {prompt_runs}")
    row = rows[0]
    image = Image.open(row["local_image_path"]).convert("RGB")
    stem = Path(row["image_filename"]).stem
    stage1_mask_dir = PAPER2_ROOT / "annotation" / "stage1_mask_behavior_assets_v1" / "exported_masks" / stem
    if not stage1_mask_dir.exists():
        raise FileNotFoundError(stage1_mask_dir)
    out_img_dir = ASSETS / "images" / tag
    out_img_dir.mkdir(parents=True, exist_ok=True)
    clean_path = out_img_dir / f"{stem}__clean.jpg"
    image.save(clean_path)
    condition_images: dict[str, Path] = {}
    for condition, mask_name in MASK_NAMES.items():
        mask_path = stage1_mask_dir / mask_name
        if not mask_path.exists() or mask_path.stat().st_size == 0:
            raise FileNotFoundError(mask_path)
        out_path = out_img_dir / f"{stem}__{condition}.jpg"
        _apply_mask(image, Image.open(mask_path)).save(out_path)
        condition_images[condition] = out_path

    fields = ["sample_id", "question", "image_path", "gold_answer", "notes"]
    question = (
        f"{row['question_text'].strip()} Reply exactly in the format: "
        "The answer is <short answer>."
    )
    local_manifest_dir = ASSETS / "manifests" / tag
    local_manifest_dir.mkdir(parents=True, exist_ok=True)
    manifests: dict[str, Path] = {}
    for condition, image_path in {"clean": clean_path, **condition_images}.items():
        manifest = local_manifest_dir / f"{PREFIX}_{tag}_{condition}_manifest.csv"
        manifest.write_text(
            _csv_text(
                [
                    {
                        "sample_id": row["sample_id"],
                        "question": question,
                        "image_path": str(image_path),
                        "gold_answer": row["answer_text"],
                        "notes": f"answer={row['answer_text']}",
                    }
                ],
                fields,
            ),
            encoding="utf-8",
        )
        manifests[condition] = manifest
    return manifests, {"clean": clean_path, **condition_images}, prompt_runs


def _upload_research_scripts(base, sftp) -> None:
    for name in [
        "run_batch_eval.py",
        "run_batch_answer_aligned_attribute.py",
        "trace_compare_ab_controlled.py",
        "run_answer_aligned_intervention_smoke.py",
    ]:
        local = BRIDGING_ROOT / "vlm-circuit-tracing" / "circuit_tracer_vlm" / "scripts" / "research" / name
        remote = f"{REMOTE_ROOT}/scripts/research/{name}"
        base._put_file(sftp, local, remote)
        sftp.chmod(remote, 0o755)


def _remoteize_manifest(local_manifest: Path, local_image: Path) -> str:
    rows = _read_csv(local_manifest)
    remote_image = f"{REMOTE_ASSETS}/images/{local_image.name}"
    for row in rows:
        row["image_path"] = remote_image
    return _csv_text(rows, ["sample_id", "question", "image_path", "gold_answer", "notes"])


def _upload_assets(base, sftp, tag: str, manifests: dict[str, Path], images: dict[str, Path]) -> None:
    base._mkdir_p(sftp, REMOTE_STAGE)
    for _, image_path in images.items():
        base._put_file(sftp, image_path, f"{REMOTE_ASSETS}/images/{image_path.name}")
    for condition, manifest in manifests.items():
        _put_text(
            base,
            sftp,
            f"{REMOTE_STAGE}/{PREFIX}_{tag}_{condition}_manifest.csv",
            _remoteize_manifest(manifest, images[condition]),
        )
    valid = [{"sample_id": "okvqa_val_1295955", "bucket": "paper2_gemma_case"}]
    _put_text(
        base,
        sftp,
        f"{REMOTE_STAGE}/{PREFIX}_{tag}_valid_samples.csv",
        _csv_text(valid, ["sample_id", "bucket"]),
    )


def _remote_script(tag: str, mode: str, max_feature_nodes: int) -> str:
    conditions = " ".join(CONDITIONS)
    retry = ",".join(str(x) for x in [max_feature_nodes, 4, 2] if x > 0)
    return f"""#!/usr/bin/env bash
set -euo pipefail
cd {REMOTE_ROOT}
source scripts/server/dev.sh
if [ -f /etc/network_turbo ]; then source /etc/network_turbo; fi
export PYTHONPATH={REMOTE_ROOT}:${{PYTHONPATH:-}}
export HF_HOME=/root/autodl-tmp/tca-reasoning/data/hf_cache
export HUGGINGFACE_HUB_CACHE=/root/autodl-tmp/tca-reasoning/data/hf_cache/hub

STAGE={REMOTE_STAGE}
PREFIX={PREFIX}
TAG={tag}
MODE={mode}
VALID="$STAGE/${{PREFIX}}_${{TAG}}_valid_samples.csv"

echo "DATE"; date '+%Y-%m-%d %H:%M:%S %Z %z'
echo "GPU"; nvidia-smi --query-gpu=name,memory.used,memory.free,utilization.gpu --format=csv,noheader || true
echo "DISK"; df -h /root/autodl-tmp
echo "CGROUP"; cat /sys/fs/cgroup/memory.current 2>/dev/null || true; cat /sys/fs/cgroup/memory.max 2>/dev/null || true

run_condition() {{
  local COND="$1"
  local STEM="${{PREFIX}}_${{MODE}}_${{TAG}}_${{COND}}"
  local RUN_ROOT="$STAGE/$STEM"
  local COMPARE="$RUN_ROOT/compare"
  rm -rf "$RUN_ROOT"
  mkdir -p "$COMPARE"
  echo "=== condition $COND ==="

  .venv/bin/python -u scripts/research/run_batch_eval.py \\
    --manifest "$STAGE/${{PREFIX}}_${{TAG}}_clean_manifest.csv" \\
    --output-csv "$RUN_ROOT/eval_clean.csv" \\
    --transcoder-set tianhux2/gemma3-4b-it-plt \\
    --device cuda --correct-rule strict_gold --no-resume

  .venv/bin/python -u scripts/research/run_batch_eval.py \\
    --manifest "$STAGE/${{PREFIX}}_${{TAG}}_${{COND}}_manifest.csv" \\
    --output-csv "$RUN_ROOT/eval_condition.csv" \\
    --transcoder-set tianhux2/gemma3-4b-it-plt \\
    --device cuda --correct-rule strict_gold --no-resume

  .venv/bin/python -u scripts/research/run_batch_answer_aligned_attribute.py \\
    --eval-csv "$RUN_ROOT/eval_condition.csv" \\
    --output-dir "$RUN_ROOT/graphs_condition" \\
    --transcoder-set tianhux2/gemma3-4b-it-plt \\
    --selected-csv "$VALID" \\
    --answer-source gold \\
    --metadata-csv "$RUN_ROOT/answer_aligned_meta_a.csv" \\
    --max-feature-nodes {max_feature_nodes} \\
    --retry-feature-nodes {retry} \\
    --exec-mode subprocess

  .venv/bin/python -u scripts/research/run_batch_answer_aligned_attribute.py \\
    --eval-csv "$RUN_ROOT/eval_clean.csv" \\
    --output-dir "$RUN_ROOT/graphs_clean" \\
    --transcoder-set tianhux2/gemma3-4b-it-plt \\
    --selected-csv "$VALID" \\
    --answer-source gold \\
    --metadata-csv "$RUN_ROOT/answer_aligned_meta_b.csv" \\
    --max-feature-nodes {max_feature_nodes} \\
    --retry-feature-nodes {retry} \\
    --exec-mode subprocess

  .venv/bin/python -u scripts/research/trace_compare_ab_controlled.py \\
    --pt-dir-a "$RUN_ROOT/graphs_condition" \\
    --pt-dir-b "$RUN_ROOT/graphs_clean" \\
    --bucket-csv "$VALID" \\
    --out-dir "$COMPARE" \\
    --buckets paper2_gemma_case \\
    --per-bucket 1 \\
    --topk-per-node 16 --beam-per-depth 16 --coverage 0.85 --max-depth 4 --min-abs-weight 0

  .venv/bin/python -u scripts/research/run_answer_aligned_intervention_smoke.py \\
    --run-root "$RUN_ROOT" \\
    --compare-dir "$COMPARE" \\
    --bucket paper2_gemma_case \\
    --run both \\
    --transcoder-set tianhux2/gemma3-4b-it-plt \\
    --sample-ids-csv "$VALID" \\
    --max-samples 1 \\
    --top-features-per-sample 1 \\
    --max-pos-buffer 2 \\
    --out-csv "$RUN_ROOT/intervention.csv"

  .venv/bin/python - "$STEM" "$RUN_ROOT" "$COMPARE" <<'PY'
import csv, json, sys
from pathlib import Path
stem, run_root, compare = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
def count(path):
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8", newline="") as f:
        return sum(1 for _ in csv.DictReader(f))
payload = {{
    "status": "pass",
    "stem": stem,
    "sample_compare_rows": count(compare / "sample_compare_controlled.csv"),
    "intervention_rows": count(run_root / "intervention.csv"),
}}
(Path("{REMOTE_STAGE}") / f"{{stem}}_decision.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
PY

  cp "$COMPARE/sample_compare_controlled.csv" "$STAGE/${{STEM}}_sample_compare_controlled.csv"
  cp "$COMPARE/bucket_summary_controlled.csv" "$STAGE/${{STEM}}_bucket_summary_controlled.csv" 2>/dev/null || true
  cp "$RUN_ROOT/intervention.csv" "$STAGE/${{STEM}}_intervention.csv"
  rm -rf "$RUN_ROOT/graphs_condition" "$RUN_ROOT/graphs_clean"
  echo "done $COND"
  df -h /root/autodl-tmp
}}

for COND in {conditions}; do
  run_condition "$COND"
done
echo "ALL_DONE"
"""


def _connect():
    base = _load_base_runner()
    sys.path.insert(0, str(BRIDGING_ROOT / ".tmp_paramiko"))
    import paramiko

    host, port, password = base._load_connection()
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(hostname=host, port=port, username="root", password=password, timeout=20, banner_timeout=20, auth_timeout=20)
    return base, client


def _status_command(tag: str, mode: str) -> str:
    return f"""
echo DATE; date '+%Y-%m-%d %H:%M:%S %Z %z'
echo PROCS
ps -eo pid,ppid,stat,etime,pcpu,pmem,args | grep -E 'paper2_stage3_gemma_source_route|run_batch_answer_aligned_attribute|trace_compare_ab_controlled' | grep -v grep || true
echo FILES
ls -lh {REMOTE_STAGE}/{PREFIX}_{mode}_{tag}_* 2>/dev/null || true
echo LOG
tail -n 120 {REMOTE_STAGE}/logs/{PREFIX}_{mode}_{tag}.log 2>/dev/null || true
echo GPU; nvidia-smi --query-gpu=memory.used,memory.free,utilization.gpu --format=csv,noheader 2>/dev/null || true
echo DISK; df -h /root/autodl-tmp
"""


def _fetch_outputs(sftp, tag: str, mode: str) -> None:
    STAGE3.mkdir(parents=True, exist_ok=True)
    for condition in CONDITIONS:
        stem = f"{PREFIX}_{mode}_{tag}_{condition}"
        for suffix in [
            "decision.json",
            "sample_compare_controlled.csv",
            "bucket_summary_controlled.csv",
            "intervention.csv",
        ]:
            remote = f"{REMOTE_STAGE}/{stem}_{suffix}"
            local = STAGE3 / f"{stem}_{suffix}"
            try:
                sftp.get(remote, str(local))
                print(f"fetched {local.name}", flush=True)
            except FileNotFoundError:
                print(f"missing {remote}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Paper2 Gemma compact source-route case probe on gpu1.")
    parser.add_argument("--tag", default="paper2_mechanism_rank50_v1")
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--max-feature-nodes", type=int, default=8)
    parser.add_argument("--detach", action="store_true")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--fetch-only", action="store_true")
    parser.add_argument("--timeout-seconds", type=int, default=86400)
    args = parser.parse_args()

    base, client = _connect()
    if args.status:
        _, stdout, _ = client.exec_command(_status_command(args.tag, args.mode), get_pty=True)
        print(stdout.read().decode("utf-8", errors="replace"))
        client.close()
        return 0

    sftp = client.open_sftp()
    if args.fetch_only:
        _fetch_outputs(sftp, args.tag, args.mode)
        sftp.close()
        client.close()
        return 0

    manifests, images, _ = _prepare_condition_assets(args.tag)
    _upload_research_scripts(base, sftp)
    _upload_assets(base, sftp, args.tag, manifests, images)
    base._mkdir_p(sftp, f"{REMOTE_STAGE}/logs")
    remote_script = f"{REMOTE_STAGE}/run_{PREFIX}_{args.mode}_{args.tag}.sh"
    with sftp.file(remote_script, "w") as handle:
        handle.write(_remote_script(args.tag, args.mode, args.max_feature_nodes).replace("\r\n", "\n"))
    sftp.chmod(remote_script, 0o755)
    sftp.close()

    if args.detach:
        cmd = f"nohup bash {remote_script} > {REMOTE_STAGE}/logs/{PREFIX}_{args.mode}_{args.tag}.log 2>&1 & echo $!"
        _, stdout, _ = client.exec_command(cmd)
        print(f"REMOTE_PID={stdout.read().decode('utf-8', errors='replace').strip()}")
        client.close()
        return 0

    stdin, stdout, stderr = client.exec_command(f"bash {remote_script}", get_pty=True)
    stdout.channel.settimeout(0.0)
    stderr.channel.settimeout(0.0)
    start = time.time()
    while not stdout.channel.exit_status_ready():
        try:
            data = stdout.channel.recv(8192)
            if data:
                print(data.decode("utf-8", errors="replace"), end="")
        except socket.timeout:
            pass
        time.sleep(0.5)
        if time.time() - start > args.timeout_seconds:
            stdout.channel.close()
            raise TimeoutError("remote Paper2 Gemma source-route command exceeded timeout")
    while stdout.channel.recv_ready():
        print(stdout.channel.recv(8192).decode("utf-8", errors="replace"), end="")
    status = stdout.channel.recv_exit_status()
    print(f"\nREMOTE_EXIT_STATUS={status}")
    sftp = client.open_sftp()
    _fetch_outputs(sftp, args.tag, args.mode)
    sftp.close()
    client.close()
    return int(status)


if __name__ == "__main__":
    raise SystemExit(main())
