#!/usr/bin/env python3
"""
GCE Experiment Runner for Parameter Golf.

Provisions an a3-highgpu-8g instance, runs a single experiment via torchrun,
monitors for completion/failure/preemption, uploads results, and tears down.

Usage:
    python infra/gce_run_experiment.py run \
        --name "T1-04-qkgain-4.0" \
        --script experiment1.py \
        --env QK_GAIN_INIT=4.0 SEED=1337

    python infra/gce_run_experiment.py run \
        --name "T1-07-xsa-all" \
        --script experiment1.py \
        --env XSA_LAST_N=11 \
        --zone us-central1-a

Requires: gcloud CLI, infra/gce_config.yaml, infra/gce_provision.py
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional

import yaml

from gce_provision import (
    InstanceInfo,
    delete_instance,
    find_and_create,
    create_instance,
    get_instance_status,
    load_config,
    scp_to_instance,
    ssh_exec,
    wait_for_ssh,
)


class ExperimentStatus(Enum):
    PENDING = "pending"
    PROVISIONING = "provisioning"
    RUNNING = "running"
    UPLOADING = "uploading"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    PREEMPTED = "preempted"
    EARLY_KILLED = "early_killed"


@dataclass
class ExperimentResult:
    name: str
    status: ExperimentStatus
    instance_name: str = ""
    instance_zone: str = ""
    provisioning_model: str = ""
    start_time: str = ""
    end_time: str = ""
    final_bpb: float = 0.0
    step_1000_bpb: float = 0.0
    output_gcs_uri: str = ""
    error: str = ""
    wallclock_seconds: float = 0.0
    preemption_count: int = 0

    def to_dict(self) -> dict:
        d = asdict(self)
        d["status"] = self.status.value
        return d


# ---------------------------------------------------------------------------
# Training scripts to sync
# ---------------------------------------------------------------------------

SYNC_FILES = [
    "experiment1.py",
    "experiment2.py",
    "train_gpt.py",
    "kernels.py",
]


# ---------------------------------------------------------------------------
# Code sync
# ---------------------------------------------------------------------------

def sync_code(instance: InstanceInfo, config: dict) -> bool:
    """Sync latest training scripts to the instance via scp.

    This ensures the golden image's code is up-to-date without
    rebuilding the image for every code change.
    """
    # Find the repo directory on the instance
    ssh_user = config.get("ssh_user", "ray")
    result = ssh_exec(instance, config,
        f"for d in /home/{ssh_user}/parameter-golf /workspace/parameter-golf /root/parameter-golf; do "
        f"test -d $d && echo $d && break; done")
    repo_dir = result.stdout.strip()
    if not repo_dir:
        # Fallback: create it
        repo_dir = f"/home/{ssh_user}/parameter-golf"
        ssh_exec(instance, config, f"mkdir -p {repo_dir}")

    # SCP the training scripts
    local_files = [f for f in SYNC_FILES if Path(f).exists()]
    if not local_files:
        print("  Warning: no training scripts found locally to sync")
        return False

    print(f"  Syncing {len(local_files)} files to {repo_dir}/...")
    ok = scp_to_instance(instance, config, local_files, f"{repo_dir}/")
    if ok:
        print("  Code sync complete")
    else:
        print("  Code sync failed")
    return ok


def ensure_data(instance: InstanceInfo, config: dict) -> bool:
    """Ensure training data is available on the instance.

    Checks for existing data shards. If we have at least 1 train + 1 val
    shard, proceed immediately (training works with any shard count).
    Only attempts download if we have zero data.
    """
    repo_dir = find_repo_dir(instance, config)
    data_dir = f"{repo_dir}/data/datasets/fineweb10B_sp1024"

    # Check how many shards exist
    result = ssh_exec(instance, config,
        f"ls {data_dir}/fineweb_train_*.bin 2>/dev/null | wc -l")
    train_count = int(result.stdout.strip() or "0")

    result2 = ssh_exec(instance, config,
        f"ls {data_dir}/fineweb_val_*.bin 2>/dev/null | wc -l")
    val_count = int(result2.stdout.strip() or "0")

    # Check tokenizer
    result3 = ssh_exec(instance, config,
        f"ls {repo_dir}/data/tokenizers/fineweb_1024_bpe.model 2>/dev/null | wc -l")
    has_tokenizer = int(result3.stdout.strip() or "0") >= 1

    if train_count >= 1 and val_count >= 1 and has_tokenizer:
        print(f"  Training data ready: {train_count} train + {val_count} val shards")
        return True

    # If we have train shards but are missing tokenizer, try to grab it from GCS (fast)
    if train_count >= 1 and not has_tokenizer:
        print(f"  Have {train_count} train shards, fetching tokenizer from GCS...")
        data_bucket = config.get("data_bucket", "parameter-golf-data")
        tok_dir = f"{repo_dir}/data/tokenizers"
        tok_gcs = f"gs://{data_bucket}/tokenizers/fineweb_1024_bpe.model"
        r = ssh_exec(instance, config,
            f"mkdir -p {tok_dir} && gsutil cp {tok_gcs} {tok_dir}/fineweb_1024_bpe.model",
            timeout=60)
        if r.returncode == 0:
            print("  Tokenizer fetched from GCS. Proceeding (val shards optional for smoke).")
            return True
        print(f"  GCS tokenizer fetch failed: {r.stderr[:200]}. Trying HuggingFace...")

    if train_count >= 1:
        # Have train data, try quick tokenizer + val download
        data_bucket = config.get("data_bucket", "parameter-golf-data")
        tok_dir = f"{repo_dir}/data/tokenizers"
        val_dir = f"{repo_dir}/data/datasets/fineweb10B_sp1024"
        # Try to get val shard + tokenizer from GCS
        r = ssh_exec(instance, config,
            f"mkdir -p {tok_dir} {val_dir} && "
            f"gsutil cp gs://{data_bucket}/tokenizers/fineweb_1024_bpe.model {tok_dir}/ 2>/dev/null; "
            f"gsutil cp 'gs://{data_bucket}/datasets/fineweb10B_sp1024/fineweb_val_000000001.bin' {val_dir}/ 2>/dev/null; "
            f"ls {tok_dir}/fineweb_1024_bpe.model",
            timeout=90)
        if r.returncode == 0:
            print("  Got tokenizer from GCS. Proceeding.")
            return True
        print(f"  Could not get tokenizer from GCS. Training may fail without tokenizer.")
        # Return True anyway — let training fail gracefully rather than blocking here
        return True

    # Only download if we have no data at all
    print(f"  Missing data (train={train_count}, val={val_count}, tok={has_tokenizer}). Downloading minimal set...")
    result = ssh_exec(instance, config,
        f"python3 {repo_dir}/data/cached_challenge_fineweb.py --variant sp1024 --train-shards 1",
        timeout=180)

    if result.returncode == 0:
        print("  Minimal data download complete")
        return True

    print(f"  Data download failed. Cannot proceed.")
    return False


# ---------------------------------------------------------------------------
# Training execution
# ---------------------------------------------------------------------------

def _build_env_string(env: dict[str, str]) -> str:
    """Build export VAR=val string for shell execution."""
    parts = []
    for k, v in env.items():
        # Shell-safe quoting
        safe_v = str(v).replace("'", "'\\''")
        parts.append(f"export {k}='{safe_v}'")
    return " && ".join(parts) if parts else "true"


def find_repo_dir(instance: InstanceInfo, config: dict) -> str:
    """Find the parameter-golf repo directory on the instance."""
    ssh_user = config.get("ssh_user", "ray")
    result = ssh_exec(instance, config,
        f"for d in /home/{ssh_user}/parameter-golf /workspace/parameter-golf /root/parameter-golf; do "
        f"test -f $d/experiment1.py && echo $d && break; done")
    repo_dir = result.stdout.strip()
    return repo_dir or f"/home/{ssh_user}/parameter-golf"


def start_training(
    instance: InstanceInfo,
    config: dict,
    script: str,
    env: dict[str, str],
    experiment_name: str,
) -> bool:
    """Start torchrun in a detached tmux session on the instance.

    Using tmux ensures training survives SSH disconnections.
    Returns True if the training started successfully.
    """
    repo_dir = find_repo_dir(instance, config)
    env_str = _build_env_string(env)

    # Add standard env vars
    std_env = {
        "RUN_ID": experiment_name,
        "DATA_PATH": f"{repo_dir}/data/datasets/fineweb10B_sp1024/",
        "TOKENIZER_PATH": f"{repo_dir}/data/tokenizers/fineweb_1024_bpe.model",
    }
    std_env_str = _build_env_string(std_env)

    train_cmd = (
        f"cd {repo_dir} && "
        f"{std_env_str} && "
        f"{env_str} && "
        f"torchrun --standalone --nproc_per_node=8 {script} "
        f"2>&1 | tee /tmp/training_output.log; "
        f"echo \"EXIT_CODE=$?\" >> /tmp/training_output.log"
    )

    # Kill any existing tmux training session
    ssh_exec(instance, config, "tmux kill-session -t training 2>/dev/null || true")

    # Start in tmux
    tmux_cmd = f"tmux new-session -d -s training '{train_cmd}'"
    print(f"  Starting training: torchrun --nproc_per_node=8 {script}")
    result = ssh_exec(instance, config, tmux_cmd, timeout=30)

    if result.returncode != 0:
        print(f"  Failed to start training: {result.stderr}")
        return False

    # Verify tmux session is running
    time.sleep(2)
    check = ssh_exec(instance, config, "tmux has-session -t training 2>/dev/null && echo running || echo stopped")
    if "running" in check.stdout:
        print("  Training started in tmux session 'training'")
        return True
    else:
        print("  Training session did not start properly")
        return False


# ---------------------------------------------------------------------------
# Training log parsing
# ---------------------------------------------------------------------------

def _parse_log(log_text: str) -> dict:
    """Parse training output log for key metrics.

    Returns dict with keys: final_bpb, step_1000_bpb, current_step, completed, error
    """
    result = {
        "final_bpb": None,
        "step_1000_bpb": None,
        "current_step": 0,
        "completed": False,
        "error": None,
        "exit_code": None,
    }

    # Find step numbers (format: "step XXXX |" or "step=XXXX")
    step_matches = re.findall(r'step[= ]+(\d+)', log_text, re.IGNORECASE)
    if step_matches:
        result["current_step"] = max(int(s) for s in step_matches)

    # Find BPB values at specific steps
    # Look for patterns like "val_bpb: 1.2345" or "bits per byte: 1.2345" or "BPB: 1.2345"
    bpb_pattern = r'(?:val_bpb|bits per byte|BPB)[:\s=]+([0-9]+\.[0-9]+)'
    bpb_matches = re.findall(bpb_pattern, log_text, re.IGNORECASE)

    # Also look for "val loss:" patterns and convert
    val_loss_pattern = r'val[_ ]loss[:\s=]+([0-9]+\.[0-9]+)'
    val_losses = re.findall(val_loss_pattern, log_text, re.IGNORECASE)

    # Try to find step 1000 BPB
    # Look for lines around step 1000
    for line in log_text.split('\n'):
        if re.search(r'step[= ]+10[0-2][0-9]', line, re.IGNORECASE):
            m = re.search(bpb_pattern, line, re.IGNORECASE)
            if m:
                result["step_1000_bpb"] = float(m.group(1))
            elif not result["step_1000_bpb"]:
                m = re.search(val_loss_pattern, line, re.IGNORECASE)
                if m:
                    # val_loss is in nats, BPB = val_loss / ln(2) * bytes_per_token
                    # but in this codebase, val_bpb is reported directly
                    result["step_1000_bpb"] = float(m.group(1))

    # Final BPB is the last reported BPB
    if bpb_matches:
        result["final_bpb"] = float(bpb_matches[-1])
    elif val_losses:
        result["final_bpb"] = float(val_losses[-1])

    # Check for final results markers
    if "final_int8_zlib_roundtrip" in log_text or "final_model" in log_text.lower():
        result["completed"] = True

    # Check for explicit completion
    exit_match = re.search(r'EXIT_CODE=(\d+)', log_text)
    if exit_match:
        result["exit_code"] = int(exit_match.group(1))
        result["completed"] = True

    # Check for errors
    error_patterns = [
        r'(RuntimeError:.*)',
        r'(CUDA out of memory.*)',
        r'(OutOfMemoryError.*)',
        r'(NCCL error.*)',
        r'(Traceback \(most recent call last\))',
    ]
    for pat in error_patterns:
        m = re.search(pat, log_text, re.IGNORECASE)
        if m:
            result["error"] = m.group(1)[:200]
            break

    return result


# ---------------------------------------------------------------------------
# Monitoring
# ---------------------------------------------------------------------------

def monitor_training(
    instance: InstanceInfo,
    config: dict,
    experiment_name: str,
    best_step_1000_bpb: float | None = None,
) -> tuple[ExperimentStatus, dict]:
    """Poll the instance for training progress.

    Returns (status, parsed_log_dict).
    """
    poll_interval = config.get("poll_interval_seconds", 30)
    early_kill_step = config.get("early_kill_step", 1000)
    threshold_pct = config.get("early_kill_threshold_pct", 5)
    max_wallclock = config.get("max_wallclock_seconds", 660)

    start_time = time.time()
    last_step = 0
    parsed = {}

    while True:
        elapsed = int(time.time() - start_time)

        # 1. Check if instance is still running (detect SPOT preemption)
        status = get_instance_status(
            instance.name, instance.zone, config["project"],
        )
        if status in ("TERMINATED", "STOPPING", "SUSPENDED", "NOT_FOUND"):
            print(f"  [{elapsed:4d}s] Instance {status} — likely preempted")
            return ExperimentStatus.PREEMPTED, parsed

        # 2. Check if tmux training session is still alive
        tmux_check = ssh_exec(instance, config,
            "tmux has-session -t training 2>/dev/null && echo running || echo stopped",
            timeout=15)
        training_alive = "running" in tmux_check.stdout

        # 3. Read the training log
        log_result = ssh_exec(instance, config,
            "cat /tmp/training_output.log 2>/dev/null || echo ''",
            timeout=15)
        log_text = log_result.stdout

        # 4. Parse the log
        parsed = _parse_log(log_text)
        current_step = parsed["current_step"]

        # Progress reporting
        if current_step != last_step:
            bpb_str = f" BPB={parsed['final_bpb']:.4f}" if parsed["final_bpb"] else ""
            print(f"  [{elapsed:4d}s] Step {current_step}{bpb_str}")
            last_step = current_step

        # 5. Early-kill check at step 1000
        if (parsed["step_1000_bpb"] and best_step_1000_bpb
                and current_step >= early_kill_step):
            threshold = best_step_1000_bpb * (1 + threshold_pct / 100)
            if parsed["step_1000_bpb"] > threshold:
                print(
                    f"  EARLY KILL: step {early_kill_step} BPB "
                    f"{parsed['step_1000_bpb']:.4f} > threshold "
                    f"{threshold:.4f} (best={best_step_1000_bpb:.4f} + {threshold_pct}%)"
                )
                # Kill the training
                ssh_exec(instance, config, "tmux kill-session -t training 2>/dev/null || true")
                return ExperimentStatus.EARLY_KILLED, parsed

        # 6. Check if training completed
        if not training_alive:
            if parsed["completed"]:
                if parsed.get("exit_code", 1) == 0 or parsed["final_bpb"]:
                    print(f"  [{elapsed:4d}s] Training completed")
                    return ExperimentStatus.SUCCEEDED, parsed
                else:
                    error = parsed.get("error", "unknown error")
                    print(f"  [{elapsed:4d}s] Training failed: {error}")
                    return ExperimentStatus.FAILED, parsed
            elif parsed.get("error"):
                print(f"  [{elapsed:4d}s] Training crashed: {parsed['error']}")
                return ExperimentStatus.FAILED, parsed
            else:
                # Training session ended but no clear completion signal
                # Check if there are final results
                if parsed["final_bpb"]:
                    print(f"  [{elapsed:4d}s] Training appears complete (BPB={parsed['final_bpb']:.4f})")
                    return ExperimentStatus.SUCCEEDED, parsed
                print(f"  [{elapsed:4d}s] Training session ended without clear result")
                return ExperimentStatus.FAILED, parsed

        # 7. Timeout check
        if elapsed > max_wallclock + 300:  # 5 min grace period over wallclock
            print(f"  [{elapsed:4d}s] Wallclock timeout exceeded")
            ssh_exec(instance, config, "tmux kill-session -t training 2>/dev/null || true")
            if parsed["final_bpb"]:
                return ExperimentStatus.SUCCEEDED, parsed
            return ExperimentStatus.FAILED, parsed

        time.sleep(poll_interval)


# ---------------------------------------------------------------------------
# Result upload
# ---------------------------------------------------------------------------

def upload_results(
    instance: InstanceInfo,
    config: dict,
    experiment_name: str,
) -> str:
    """Upload results from instance to GCS. Returns GCS URI prefix."""
    bucket = config["output_bucket"]
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    gcs_prefix = f"gs://{bucket}/experiments/{experiment_name}/{timestamp}"
    repo_dir = find_repo_dir(instance, config)

    print(f"  Uploading results to {gcs_prefix}/...")

    # Upload model files
    ssh_exec(instance, config,
        f"cd {repo_dir} && "
        f"gsutil -m cp final_model.* '{gcs_prefix}/' 2>/dev/null || echo 'No model files'",
        timeout=120)

    # Upload logs
    ssh_exec(instance, config,
        f"cd {repo_dir} && "
        f"gsutil -m cp -r logs/ '{gcs_prefix}/logs/' 2>/dev/null || echo 'No logs dir'",
        timeout=60)

    # Upload training output
    ssh_exec(instance, config,
        f"gsutil cp /tmp/training_output.log '{gcs_prefix}/' 2>/dev/null || echo 'No training log'",
        timeout=30)

    print(f"  Upload complete: {gcs_prefix}/")
    return gcs_prefix


# ---------------------------------------------------------------------------
# Full experiment lifecycle
# ---------------------------------------------------------------------------

def run_experiment(
    name: str,
    script: str,
    env: dict[str, str],
    config: dict,
    max_retries: int = 3,
    best_step_1000_bpb: float | None = None,
    fixed_zone: str | None = None,
) -> ExperimentResult:
    """Full lifecycle of a single experiment.

    1. Provision instance (find available zone)
    2. Wait for SSH
    3. Sync latest code
    4. Start training via tmux
    5. Monitor until completion/failure/preemption
    6. Upload results to GCS
    7. Delete instance

    On SPOT preemption: retry in a different zone (up to max_retries).
    On early-kill: return immediately.
    Instance is always cleaned up via try/finally.
    """
    result = ExperimentResult(
        name=name,
        status=ExperimentStatus.PENDING,
        start_time=datetime.now().isoformat(),
    )
    exclude_zones: list[str] = []

    for attempt in range(max_retries + 1):
        instance: InstanceInfo | None = None
        try:
            # 1. Provision
            result.status = ExperimentStatus.PROVISIONING
            print(f"\n{'='*60}")
            print(f" Experiment: {name} (attempt {attempt + 1}/{max_retries + 1})")
            print(f" Script: {script}")
            print(f" Env: {env}")
            print(f"{'='*60}")

            if fixed_zone:
                instance = create_instance(name, fixed_zone, config, spot=config.get("prefer_spot", True))
            else:
                instance = find_and_create(name, config, exclude_zones=exclude_zones)

            if instance is None:
                result.status = ExperimentStatus.FAILED
                result.error = "No instance could be provisioned in any zone"
                result.end_time = datetime.now().isoformat()
                return result

            result.instance_name = instance.name
            result.instance_zone = instance.zone
            result.provisioning_model = instance.provisioning_model

            # 2. Wait for SSH
            if not wait_for_ssh(instance, config):
                result.status = ExperimentStatus.FAILED
                result.error = "SSH timeout"
                result.end_time = datetime.now().isoformat()
                continue

            # 3. Sync code
            sync_code(instance, config)

            # 3.5. Ensure training data is available
            if not ensure_data(instance, config):
                result.status = ExperimentStatus.FAILED
                result.error = "Training data not available"
                result.end_time = datetime.now().isoformat()
                continue

            # 4. Start training
            result.status = ExperimentStatus.RUNNING
            if not start_training(instance, config, script, env, name):
                result.status = ExperimentStatus.FAILED
                result.error = "Failed to start training"
                result.end_time = datetime.now().isoformat()
                continue

            # 5. Monitor
            status, parsed = monitor_training(
                instance, config, name,
                best_step_1000_bpb=best_step_1000_bpb,
            )

            result.final_bpb = parsed.get("final_bpb") or 0.0
            result.step_1000_bpb = parsed.get("step_1000_bpb") or 0.0
            result.error = parsed.get("error") or ""

            if status == ExperimentStatus.PREEMPTED:
                result.preemption_count += 1
                exclude_zones.append(instance.zone)
                print(f"  Preempted in {instance.zone}. Retrying...")
                # Delete the stopped instance before retrying
                delete_instance(instance.name, instance.zone, config["project"])
                instance = None
                continue

            if status == ExperimentStatus.EARLY_KILLED:
                result.status = ExperimentStatus.EARLY_KILLED
                result.end_time = datetime.now().isoformat()
                result.wallclock_seconds = (
                    datetime.fromisoformat(result.end_time)
                    - datetime.fromisoformat(result.start_time)
                ).total_seconds()
                return result

            # 6. Upload results
            if status == ExperimentStatus.SUCCEEDED:
                result.status = ExperimentStatus.UPLOADING
                try:
                    gcs_uri = upload_results(instance, config, name)
                    result.output_gcs_uri = gcs_uri
                except Exception as e:
                    print(f"  Upload failed: {e}")
                    result.output_gcs_uri = ""

            result.status = status
            result.end_time = datetime.now().isoformat()
            result.wallclock_seconds = (
                datetime.fromisoformat(result.end_time)
                - datetime.fromisoformat(result.start_time)
            ).total_seconds()
            return result

        except KeyboardInterrupt:
            print("\n  Interrupted by user")
            result.status = ExperimentStatus.FAILED
            result.error = "Interrupted"
            result.end_time = datetime.now().isoformat()
            raise
        except Exception as e:
            print(f"  Unexpected error: {e}")
            result.error = str(e)
            result.status = ExperimentStatus.FAILED
        finally:
            # Always clean up the instance
            if instance is not None:
                try:
                    delete_instance(instance.name, instance.zone, config["project"])
                except Exception as e:
                    print(f"  Warning: cleanup failed for {instance.name}: {e}")

    # All retries exhausted
    if result.status == ExperimentStatus.PROVISIONING:
        result.status = ExperimentStatus.FAILED
        result.error = f"All {max_retries + 1} attempts failed"
    result.end_time = datetime.now().isoformat()
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_env_args(env_args: list[str]) -> dict[str, str]:
    """Parse KEY=VALUE pairs from CLI --env arguments."""
    result = {}
    for item in env_args:
        if "=" not in item:
            raise ValueError(f"Invalid env format: {item!r} (expected KEY=VALUE)")
        key, value = item.split("=", 1)
        result[key] = value
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Parameter Golf — GCE Experiment Runner",
    )
    parser.add_argument("--config", default="infra/gce_config.yaml", help="GCE config path")
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="Run a single experiment")
    p_run.add_argument("--name", required=True, help="Experiment name")
    p_run.add_argument("--script", default="experiment1.py", help="Training script")
    p_run.add_argument("--env", nargs="*", default=[], help="KEY=VALUE env overrides")
    p_run.add_argument("--zone", default=None, help="Force specific zone")
    p_run.add_argument("--retries", type=int, default=3, help="Max preemption retries")
    p_run.add_argument("--best-bpb", type=float, default=None,
                       help="Best step-1000 BPB for early-kill threshold")

    args = parser.parse_args()
    config = load_config(args.config)

    if args.command == "run":
        env = _parse_env_args(args.env)
        result = run_experiment(
            name=args.name,
            script=args.script,
            env=env,
            config=config,
            max_retries=args.retries,
            best_step_1000_bpb=args.best_bpb,
            fixed_zone=args.zone,
        )
        print(f"\n{'='*60}")
        print(f" Result: {result.name}")
        print(f" Status: {result.status.value}")
        if result.final_bpb:
            print(f" Final BPB: {result.final_bpb:.4f}")
        if result.step_1000_bpb:
            print(f" Step 1000 BPB: {result.step_1000_bpb:.4f}")
        if result.output_gcs_uri:
            print(f" Output: {result.output_gcs_uri}")
        if result.error:
            print(f" Error: {result.error}")
        print(f" Wallclock: {result.wallclock_seconds:.0f}s")
        print(f" Preemptions: {result.preemption_count}")
        print(f"{'='*60}")

        # Write result to JSON
        result_path = Path("infra") / f"result-{args.name}.json"
        result_path.write_text(json.dumps(result.to_dict(), indent=2))
        print(f"\nResult saved to: {result_path}")

        sys.exit(0 if result.status == ExperimentStatus.SUCCEEDED else 1)


if __name__ == "__main__":
    main()
