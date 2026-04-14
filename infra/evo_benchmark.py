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
    """Run the target through the funnel: smoke (1xH100) → qualify (8xH100 short).

    Returns the qualify step-1000 BPB estimate, or 99.0 on failure.
    Uses the efficient funnel pipeline instead of expensive full runs.
    """
    sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

    from gce_provision import load_config
    from funnel import run_smoke, run_qualify

    config = load_config("infra/gce_config.yaml")
    exp_name = f"evo-{_EXPERIMENT_ID}"

    # Copy the target file to the repo root so funnel's sync_code picks it up
    import shutil
    target_name = Path(target_path).name
    repo_copy = Path(target_name)
    if Path(target_path).resolve() != repo_copy.resolve():
        shutil.copy2(target_path, repo_copy)
        eprint(f"[evo] Copied target to {repo_copy}")

    env = {"VOCAB_SIZE": "1024", "COMPRESSOR": "lzma"}

    try:
        # Go straight to qualify (8xH100, 3min) — smoke can't discriminate
        eprint(f"[evo] Running qualify on 8xH100 (3min, step-1000 BPB)...")
        _real_stdout = sys.stdout
        sys.stdout = sys.stderr
        try:
            qualify = run_qualify(exp_name, target_name, env, config)
        finally:
            sys.stdout = _real_stdout

        if qualify.status != "pass":
            eprint(f"[evo] QUALIFY FAIL: {qualify.error or 'unknown'}")
            return 99.0

        # Report BPB (the real metric), not train_loss
        bpb = qualify.step_1000_bpb
        if bpb is None:
            # Fallback: use val_loss converted estimate
            bpb = qualify.val_loss_1000 / 1.6309 if qualify.val_loss_1000 else 99.0
        eprint(f"[evo] Qualify BPB: {bpb:.6f} at step {qualify.last_step}")
        return bpb

    except Exception as e:
        eprint(f"[evo] Exception: {e}")
        return 99.0


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
