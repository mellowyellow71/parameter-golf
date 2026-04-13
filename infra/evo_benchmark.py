#!/usr/bin/env python3
"""
Evo benchmark wrapper for Parameter Golf.

Dispatches experiment1.py to a GCE 8xH100 instance, trains the model,
and returns the BPB score via evo's inline instrumentation contract.

All noisy output goes to stderr. Only the final {"score": ...} JSON hits stdout.

Usage (called by evo):
    python infra/evo_benchmark.py --target experiment1.py

Environment (set by evo):
    EVO_TRACES_DIR     - directory for task trace files
    EVO_EXPERIMENT_ID  - unique experiment identifier
"""
from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# --- Inline evo instrumentation (zero dependencies) ---
_TRACES_DIR = Path(os.environ["EVO_TRACES_DIR"]) if os.environ.get("EVO_TRACES_DIR") else None
_EXPERIMENT_ID = os.environ.get("EVO_EXPERIMENT_ID", "unknown")
_SCORES: dict[str, float] = {}
_STARTED_AT = datetime.now(timezone.utc).isoformat(timespec="seconds")

if _TRACES_DIR:
    _TRACES_DIR.mkdir(parents=True, exist_ok=True)


def log_task(task_id, score, *, summary=None, failure_reason=None, **extra):
    task_id = str(task_id)
    _SCORES[task_id] = score
    if _TRACES_DIR is None:
        return
    trace = {
        "experiment_id": _EXPERIMENT_ID,
        "task_id": task_id,
        "status": "passed" if failure_reason is None else "failed",
        "score": score,
        "ended_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    if summary:
        trace["summary"] = summary
    if failure_reason:
        trace["failure_reason"] = failure_reason
    trace.update(extra)
    (_TRACES_DIR / f"task_{task_id}.json").write_text(json.dumps(trace, indent=2))


def write_result(score=None):
    if score is None:
        score = sum(_SCORES.values()) / len(_SCORES) if _SCORES else 0.0
    result = {
        "score": round(score, 6),
        "tasks": dict(_SCORES),
        "started_at": _STARTED_AT,
        "ended_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    print(json.dumps(result, indent=2))


# --- Benchmark logic ---

def eprint(*args, **kwargs):
    """Print to stderr (keeps stdout clean for evo)."""
    print(*args, file=sys.stderr, **kwargs)


def run_benchmark(target_path: str) -> float:
    """Dispatch the target script to GCE H100 and return BPB.

    Returns the final BPB score, or 99.0 on failure.
    """
    # Add infra/ to path for imports
    sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

    from gce_provision import load_config, find_and_create, wait_for_ssh, scp_to_instance, ssh_exec, delete_instance
    from gce_run_experiment import _parse_log, find_repo_dir, ensure_data

    config = load_config("infra/gce_config.yaml")
    exp_name = f"evo-{_EXPERIMENT_ID}"
    instance = None

    try:
        # 1. Provision
        eprint(f"[evo] Provisioning 8xH100 instance for {exp_name}...")
        instance = find_and_create(exp_name, config)
        if instance is None:
            eprint("[evo] FAILED: No H100 instance available in any zone")
            return 99.0

        eprint(f"[evo] Instance: {instance.name} in {instance.zone}")

        # 2. Wait for SSH
        if not wait_for_ssh(instance, config):
            eprint("[evo] FAILED: SSH timeout")
            return 99.0

        # 3. Sync the target file (the modified experiment1.py)
        eprint(f"[evo] Syncing target: {target_path}")
        repo_dir = find_repo_dir(instance, config)

        # SCP the target file
        scp_to_instance(instance, config, [target_path], f"{repo_dir}/")
        # Also sync kernels.py if it exists
        if Path("kernels.py").exists():
            scp_to_instance(instance, config, ["kernels.py"], f"{repo_dir}/")

        # 4. Ensure training data (accepts whatever shards exist, no long download)
        eprint("[evo] Checking training data...")
        if not ensure_data(instance, config):
            eprint("[evo] FAILED: No training data available")
            return 99.0

        # 5. Run training via torchrun
        target_name = Path(target_path).name
        eprint(f"[evo] Starting training: torchrun --nproc_per_node=8 {target_name}")

        train_cmd = (
            f"cd {repo_dir} && "
            f"export DATA_PATH={repo_dir}/data/datasets/fineweb10B_sp1024/ && "
            f"export TOKENIZER_PATH={repo_dir}/data/tokenizers/fineweb_1024_bpe.model && "
            f"export RUN_ID={exp_name} && "
            f"torchrun --standalone --nproc_per_node=8 {target_name} "
            f"2>&1 | tee /tmp/training_output.log; "
            f"echo EXIT_CODE=$? >> /tmp/training_output.log"
        )

        # Start in tmux so it survives SSH drops
        ssh_exec(instance, config, "tmux kill-session -t training 2>/dev/null || true")
        ssh_exec(instance, config, f"tmux new-session -d -s training '{train_cmd}'", timeout=30)

        # 6. Monitor training
        eprint("[evo] Monitoring training...")
        poll_interval = 30
        max_wait = 900  # 15 minutes
        start = time.time()
        last_step = 0

        while time.time() - start < max_wait:
            # Check tmux alive
            check = ssh_exec(instance, config,
                "tmux has-session -t training 2>/dev/null && echo running || echo stopped",
                timeout=15)
            alive = "running" in check.stdout

            # Read log
            log_result = ssh_exec(instance, config, "cat /tmp/training_output.log 2>/dev/null", timeout=15)
            parsed = _parse_log(log_result.stdout)

            if parsed["current_step"] > last_step:
                bpb_str = f" BPB={parsed['final_bpb']:.4f}" if parsed.get("final_bpb") else ""
                eprint(f"[evo] Step {parsed['current_step']}{bpb_str}")
                last_step = parsed["current_step"]

            if not alive:
                break

            time.sleep(poll_interval)

        # 7. Get final log and parse BPB
        final_log = ssh_exec(instance, config, "cat /tmp/training_output.log 2>/dev/null", timeout=15)
        parsed = _parse_log(final_log.stdout)

        bpb = parsed.get("final_bpb") or 99.0
        error = parsed.get("error")

        if error:
            eprint(f"[evo] Training error: {error}")
        if bpb < 99:
            eprint(f"[evo] Final BPB: {bpb:.6f}")
        else:
            eprint("[evo] No BPB result obtained")

        return bpb

    except Exception as e:
        eprint(f"[evo] Exception: {e}")
        return 99.0
    finally:
        # Always cleanup
        if instance:
            eprint(f"[evo] Deleting instance {instance.name}...")
            try:
                delete_instance(instance.name, instance.zone, config["project"])
            except Exception as e:
                eprint(f"[evo] Cleanup warning: {e}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Evo benchmark for Parameter Golf")
    parser.add_argument("--target", default="experiment1.py", help="Target script to evaluate")
    args = parser.parse_args()

    # Run the benchmark (all output to stderr)
    bpb = run_benchmark(args.target)

    # Log the task result
    log_task(
        "train_bpb",
        score=bpb,
        summary=f"BPB={bpb:.6f}" if bpb < 99 else "failed",
        failure_reason=None if bpb < 99 else "training_failed",
    )

    # Write final result to stdout (the only stdout output)
    write_result(score=bpb)


if __name__ == "__main__":
    main()
