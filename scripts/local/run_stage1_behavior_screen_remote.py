#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import socket
import subprocess
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
BRIDGING_ROOT = PROJECT_ROOT.parent
REMOTE_ROOT = "/root/autodl-tmp/tca-reasoning/circuit_tracer_vlm"
REMOTE_STAGE = "/root/autodl-tmp/tca-reasoning/paper2_stage1_composition"
REMOTE_ASSETS = f"{REMOTE_STAGE}/assets"
LOCAL_RESULTS = PROJECT_ROOT / "doc" / "experiments" / "stage1" / "results"
PREFIX = "stage1_composition_screen"


def _load_base_runner():
    spec = importlib.util.spec_from_file_location(
        "stage2g_runner",
        BRIDGING_ROOT / "scripts" / "local" / "run_stage2g_cross_model_remote.py",
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _stem(mode: str, tag: str) -> str:
    return f"{PREFIX}_{mode}_{tag}"


def _manifest_for(tag: str, viability: bool) -> Path:
    if viability:
        return PROJECT_ROOT / "annotation" / "stage1_ab_evidence_pack_v1" / "stage1_candidate_manifest.csv"
    return LOCAL_RESULTS / f"stage1_behavior_manifest_{tag}.csv"


def _put_if_exists(base, sftp, local: Path, remote: str) -> bool:
    if not local.exists():
        return False
    base._put_file(sftp, local, remote)
    return True


def _upload_assets(base, sftp, manifest: Path, remote_manifest_name: str) -> dict[str, int]:
    rows = _read_csv(manifest)
    base._mkdir_p(sftp, f"{REMOTE_ASSETS}/images")
    base._mkdir_p(sftp, f"{REMOTE_ASSETS}/exported_masks")
    base._put_file(sftp, manifest, f"{REMOTE_STAGE}/{remote_manifest_name}")
    uploaded = 0
    seen: set[str] = set()
    for row in rows:
        for key in ["local_image_path", "wrong_image_path"]:
            image = Path(row.get(key, ""))
            image_name = image.name
            if image_name and image_name not in seen and _put_if_exists(base, sftp, image, f"{REMOTE_ASSETS}/images/{image_name}"):
                uploaded += 1
                seen.add(image_name)
        stem = Path(row.get("image_filename", "")).stem
        mask_dir = Path(row.get("mask_dir", ""))
        for name in ["region_A.png", "region_B.png", "region_A_union_B.png", "random_A_size.png", "random_B_size.png", "random_union_size.png"]:
            _put_if_exists(base, sftp, mask_dir / name, f"{REMOTE_ASSETS}/exported_masks/{stem}/{name}")
    return {"rows": len(rows), "uploaded_images": uploaded}


def _remote_script(mode: str, tag: str, manifest_name: str, conditions: str, max_rows: int, max_new_tokens: int) -> str:
    stem = _stem(mode, tag)
    row_arg = f"--max-rows {max_rows}" if max_rows > 0 else ""
    return f"""#!/usr/bin/env bash
set -e
cd {REMOTE_ROOT}
source scripts/server/dev.sh
if [ -f /etc/network_turbo ]; then source /etc/network_turbo; fi
export PYTHONPATH={REMOTE_ROOT}:${{PYTHONPATH:-}}
export HF_HOME=/root/autodl-tmp/tca-reasoning/data/hf_cache
export HUGGINGFACE_HUB_CACHE=/root/autodl-tmp/tca-reasoning/data/hf_cache/hub
STAGE={REMOTE_STAGE}
ASSET_ROOT={REMOTE_ASSETS}
GEMMA_MODEL=$(ls -d /root/autodl-tmp/tca-reasoning/data/hf_cache/hub/models--google--gemma-3-4b-it/snapshots/* 2>/dev/null | head -n 1 || true)
QWEN_MODEL=$(ls -d /root/autodl-tmp/tca-reasoning/data/hf_cache/hub/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/* 2>/dev/null | head -n 1 || true)
mkdir -p "$STAGE"

echo '--- Paper2 Stage1 behavior screen preflight ---'
date '+%Y-%m-%d %H:%M:%S %Z %z'
df -h /root/autodl-tmp
free -h || true
nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu --format=csv,noheader || true

.venv/bin/python -m py_compile scripts/research/run_paper2_stage1_behavior_screen.py

if [ -n "$GEMMA_MODEL" ]; then
  echo '--- Paper2 Stage1 Gemma behavior screen ---'
  .venv/bin/python -u scripts/research/run_paper2_stage1_behavior_screen.py \\
    --model-family gemma \\
    --model-name "$GEMMA_MODEL" \\
    --manifest "$STAGE/{manifest_name}" \\
    --image-root "$ASSET_ROOT/images" \\
    --mask-root "$ASSET_ROOT/exported_masks" \\
    --conditions {conditions} \\
    {row_arg} \\
    --max-new-tokens {max_new_tokens} \\
    --out-csv "$STAGE/{stem}_gemma.csv" \\
    --out-json "$STAGE/{stem}_gemma.json"
else
  echo 'GEMMA_MODEL_CACHE_MISSING'
fi

if [ -n "$QWEN_MODEL" ]; then
  echo '--- Paper2 Stage1 Qwen behavior screen ---'
  .venv/bin/python -u scripts/research/run_paper2_stage1_behavior_screen.py \\
    --model-family qwen \\
    --model-name "$QWEN_MODEL" \\
    --manifest "$STAGE/{manifest_name}" \\
    --image-root "$ASSET_ROOT/images" \\
    --mask-root "$ASSET_ROOT/exported_masks" \\
    --conditions {conditions} \\
    {row_arg} \\
    --max-new-tokens {max_new_tokens} \\
    --out-csv "$STAGE/{stem}_qwen.csv" \\
    --out-json "$STAGE/{stem}_qwen.json"
else
  echo 'QWEN_MODEL_CACHE_MISSING'
fi

echo '--- Paper2 Stage1 behavior screen done ---'
df -h /root/autodl-tmp
nvidia-smi --query-gpu=memory.used,memory.free,utilization.gpu --format=csv,noheader || true
"""


def _status_command(mode: str, tag: str) -> str:
    stem = _stem(mode, tag)
    return f"""
echo DATE
date '+%Y-%m-%d %H:%M:%S %Z %z'
echo PROCS
ps -eo pid,ppid,stat,etime,pcpu,pmem,args | grep -E 'paper2_stage1_composition|run_paper2_stage1_behavior_screen' | grep -v grep || true
echo FILES
ls -lh {REMOTE_STAGE}/{stem}* 2>/dev/null || true
echo LOGS
ls -lh {REMOTE_STAGE}/logs/*paper2_stage1* 2>/dev/null || true
tail -n 80 {REMOTE_STAGE}/logs/*paper2_stage1* 2>/dev/null || true
echo GPU
nvidia-smi --query-gpu=memory.used,memory.free,utilization.gpu --format=csv,noheader 2>/dev/null || true
echo DISK
df -h /root/autodl-tmp
"""


def _fetch(sftp, mode: str, tag: str) -> None:
    stem = _stem(mode, tag)
    LOCAL_RESULTS.mkdir(parents=True, exist_ok=True)
    for name in [f"{stem}_gemma.csv", f"{stem}_gemma.json", f"{stem}_qwen.csv", f"{stem}_qwen.json"]:
        try:
            sftp.get(f"{REMOTE_STAGE}/{name}", str(LOCAL_RESULTS / name))
            print(f"fetched {name}", flush=True)
        except FileNotFoundError:
            print(f"missing {name}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Paper2 Stage1 behavior screen on AutoDL/gpu1.")
    parser.add_argument("--mode", choices=["viability", "smoke", "full"], default="viability")
    parser.add_argument("--tag", default="v1")
    parser.add_argument("--conditions", default="")
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=4)
    parser.add_argument("--timeout-seconds", type=int, default=43200)
    parser.add_argument("--detach", action="store_true")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--fetch-only", action="store_true")
    parser.add_argument("--skip-analyze", action="store_true")
    args = parser.parse_args()
    base = _load_base_runner()
    sys.path.insert(0, str(BRIDGING_ROOT / ".tmp_paramiko"))
    import paramiko

    host, port, password = base._load_connection()
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(hostname=host, port=port, username="root", password=password, timeout=20, banner_timeout=20, auth_timeout=20)
    sftp = client.open_sftp()

    if args.status:
        stdin, stdout, stderr = client.exec_command(_status_command(args.mode, args.tag), get_pty=True)
        print(stdout.read().decode("utf-8", errors="replace"))
        err = stderr.read().decode("utf-8", errors="replace")
        if err:
            print(err, file=sys.stderr)
        sftp.close()
        client.close()
        return 0
    if args.fetch_only:
        _fetch(sftp, args.mode, args.tag)
        sftp.close()
        client.close()
        if not args.skip_analyze:
            subprocess.run(
                [
                    sys.executable,
                    str(PROJECT_ROOT / "scripts" / "local" / "analyze_stage1_composition_screen.py"),
                    "--mode",
                    args.mode,
                    "--tag",
                    args.tag,
                ],
                check=False,
            )
        return 0

    local_script = PROJECT_ROOT / "scripts" / "research" / "run_paper2_stage1_behavior_screen.py"
    remote_script_path = f"{REMOTE_ROOT}/scripts/research/run_paper2_stage1_behavior_screen.py"
    base._put_file(sftp, local_script, remote_script_path)
    sftp.chmod(remote_script_path, 0o755)
    manifest = _manifest_for(args.tag, viability=args.mode == "viability")
    if not manifest.exists():
        raise FileNotFoundError(f"manifest not found: {manifest}")
    remote_manifest_name = f"{PREFIX}_{args.mode}_{args.tag}_manifest.csv"
    upload_info = _upload_assets(base, sftp, manifest, remote_manifest_name)
    print(f"uploaded assets: {json.dumps(upload_info, ensure_ascii=False)}", flush=True)
    conditions = args.conditions or ("clean,wrong_image" if args.mode == "viability" else "clean,mask_A,mask_B,mask_A_union_B,random_A_size,random_B_size,random_union_size,wrong_image")
    max_rows = args.max_rows
    if args.mode == "smoke" and max_rows == 0:
        max_rows = 4
    remote_run = f"{REMOTE_STAGE}/run_paper2_stage1_{args.mode}_{args.tag}.sh"
    with sftp.file(remote_run, "w") as handle:
        handle.write(_remote_script(args.mode, args.tag, remote_manifest_name, conditions, max_rows, args.max_new_tokens).replace("\r\n", "\n"))
    sftp.chmod(remote_run, 0o755)
    sftp.close()

    if args.detach:
        log_dir = f"{REMOTE_STAGE}/logs"
        stamp = time.strftime("%Y%m%d_%H%M%S")
        cmd = f"mkdir -p {log_dir}; nohup bash {remote_run} > {log_dir}/paper2_stage1_{args.mode}_{args.tag}_{stamp}.log 2>&1 < /dev/null & echo $!"
        _stdin, stdout, stderr = client.exec_command(cmd, get_pty=False)
        print(stdout.read().decode("utf-8", errors="replace"))
        err = stderr.read().decode("utf-8", errors="replace")
        if err:
            print(err, file=sys.stderr)
        client.close()
        return 0

    stdin, stdout, stderr = client.exec_command(f"bash {remote_run}", get_pty=True)
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
        try:
            data = stderr.channel.recv_stderr(8192)
            if data:
                print(data.decode("utf-8", errors="replace"), end="")
        except socket.timeout:
            pass
        time.sleep(0.5)
        if time.time() - start > args.timeout_seconds:
            stdout.channel.close()
            raise TimeoutError("remote Paper2 Stage1 behavior screen timed out")
    exit_status = stdout.channel.recv_exit_status()
    print(f"\nREMOTE_EXIT_STATUS={exit_status}")
    sftp = client.open_sftp()
    _fetch(sftp, args.mode, args.tag)
    sftp.close()
    client.close()
    return int(exit_status)


if __name__ == "__main__":
    raise SystemExit(main())
