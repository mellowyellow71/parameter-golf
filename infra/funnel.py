#!/usr/bin/env python3
"""
Multi-stage experiment funnel for Parameter Golf.

Stage 1 (Smoke):   1xH100 (a3-highgpu-1g), 300s, ~$0.25 — verify training starts
Stage 2 (Qualify):  8xH100 (a3-highgpu-8g), 180s, ~$2 — get val_loss@step1000
Stage 3 (Full):     8xH100 (a3-highgpu-8g), 600s, ~$5-8 — final BPB with GPTQ

Usage:
    python infra/funnel.py smoke --script winning_base_decoded.py --env "QK_GAIN_INIT=5.25"
    python infra/funnel.py qualify --script winning_base_decoded.py --env "QK_GAIN_INIT=5.25"
    python infra/funnel.py full --script winning_base_decoded.py --env "QK_GAIN_INIT=5.25"
    python infra/funnel.py auto --script winning_base_decoded.py  # runs all stages
    python infra/funnel.py status

Requires: gcloud CLI, infra/gce_config.yaml, infra/gce_provision.py
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

from gce_provision import (
    InstanceInfo,
    delete_instance,
    find_and_create,
    create_instance,
    load_config,
    scp_to_instance,
    ssh_exec,
    wait_for_ssh,
)
from gce_run_experiment import (
    _build_env_string,
    _parse_log,
    find_repo_dir,
    sync_code,
    ensure_data,
    upload_results,
)
from scaling_laws import ScalingPredictor

# ---------------------------------------------------------------------------
# State persistence
# ---------------------------------------------------------------------------

STATE_PATH = Path("infra/funnel_state.json")


def _load_state() -> dict:
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text())
    return {"experiments": {}}


def _save_state(state: dict) -> None:
    STATE_PATH.write_text(json.dumps(state, indent=2) + "\n")


def _update_stage(exp_name: str, stage: str, data: dict) -> None:
    """Update a single stage result in persisted state."""
    state = _load_state()
    if exp_name not in state["experiments"]:
        state["experiments"][exp_name] = {}
    state["experiments"][exp_name][stage] = {
        **data,
        "timestamp": datetime.now().isoformat(),
    }
    _save_state(state)


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class SmokeResult:
    status: str  # "pass", "fail", "error"
    train_loss_last: float | None = None
    last_step: int = 0
    loss_decreased: bool = False
    error: str = ""
    instance_name: str = ""
    instance_zone: str = ""


@dataclass
class QualifyResult:
    status: str  # "pass", "fail", "error"
    step_1000_bpb: float | None = None
    val_loss_1000: float | None = None
    last_step: int = 0
    error: str = ""
    instance_name: str = ""
    instance_zone: str = ""


@dataclass
class FullResult:
    status: str  # "done", "fail", "error"
    final_bpb: float | None = None
    artifact_size: int | None = None  # bytes
    last_step: int = 0
    error: str = ""
    gcs_uri: str = ""
    instance_name: str = ""
    instance_zone: str = ""


# ---------------------------------------------------------------------------
# Log parsing helpers (supplement gce_run_experiment._parse_log)
# ---------------------------------------------------------------------------

def _parse_train_losses(log_text: str) -> list[tuple[int, float]]:
    """Extract (step, train_loss) pairs from the training log.

    Expected format: '{step}/{iterations} train_loss: {value:.4f} ...'
    """
    pattern = r'(\d+)/\d+\s+train_loss:\s+([0-9]+\.[0-9]+)'
    return [(int(m.group(1)), float(m.group(2)))
            for m in re.finditer(pattern, log_text)]


def _parse_val_metrics(log_text: str) -> list[tuple[int, float, float]]:
    """Extract (step, val_loss, val_bpb) triples from the training log.

    Expected format: '{step}/{iterations} val_loss: {value:.4f} val_bpb: {value:.4f}'
    """
    pattern = r'(\d+)/\d+\s+val_loss:\s+([0-9]+\.[0-9]+)\s+val_bpb:\s+([0-9]+\.[0-9]+)'
    return [(int(m.group(1)), float(m.group(2)), float(m.group(3)))
            for m in re.finditer(pattern, log_text)]


# ---------------------------------------------------------------------------
# Common provisioning + run flow
# ---------------------------------------------------------------------------

def _provision_and_prepare(
    name: str,
    script: str,
    config: dict,
    machine_type: str | None = None,
) -> tuple[InstanceInfo | None, str | None]:
    """Provision instance, wait for SSH, sync code, ensure data.

    Returns (instance, error_message). instance is None on failure.
    """
    instance = find_and_create(name, config, machine_type=machine_type)
    if instance is None:
        return None, "No instance could be provisioned in any zone"

    if not wait_for_ssh(instance, config):
        delete_instance(instance.name, instance.zone, config["project"])
        return None, "SSH timeout"

    sync_code(instance, config)

    if not ensure_data(instance, config):
        delete_instance(instance.name, instance.zone, config["project"])
        return None, "Training data not available"

    # SCP the specific script if not in the standard SYNC_FILES list
    repo_dir = find_repo_dir(instance, config)
    if Path(script).exists():
        scp_to_instance(instance, config, [script], f"{repo_dir}/")

    return instance, None


def _start_torchrun(
    instance: InstanceInfo,
    config: dict,
    script: str,
    env: dict[str, str],
    experiment_name: str,
    nproc: int,
) -> bool:
    """Start torchrun in a tmux session with the given nproc_per_node.

    Returns True if training started successfully.
    """
    repo_dir = find_repo_dir(instance, config)
    env_str = _build_env_string(env)

    ssh_user = config.get("ssh_user", "ray")
    std_env = {
        "RUN_ID": experiment_name,
    }
    std_env_str = _build_env_string(std_env)

    train_cmd = (
        f"cd {repo_dir} && "
        f"{std_env_str} && "
        f"{env_str} && "
        f"torchrun --standalone --nproc_per_node={nproc} {script} "
        f"2>&1 | tee /tmp/training_output.log; "
        f"echo \"EXIT_CODE=$?\" >> /tmp/training_output.log"
    )

    # Kill any existing tmux training session
    ssh_exec(instance, config, "tmux kill-session -t training 2>/dev/null || true")

    # Start in tmux
    tmux_cmd = f"tmux new-session -d -s training '{train_cmd}'"
    print(f"  Starting training: torchrun --nproc_per_node={nproc} {script}")
    result = ssh_exec(instance, config, tmux_cmd, timeout=30)

    if result.returncode != 0:
        print(f"  Failed to start training: {result.stderr}")
        return False

    # Verify tmux session is running
    time.sleep(3)
    check = ssh_exec(instance, config,
        "tmux has-session -t training 2>/dev/null && echo running || echo stopped")
    if "running" in check.stdout:
        print("  Training started in tmux session 'training'")
        return True
    else:
        # Grab early crash info
        log_check = ssh_exec(instance, config,
            "tail -30 /tmp/training_output.log 2>/dev/null || echo 'no log'")
        print(f"  Training session did not start properly")
        print(f"  Last log output:\n{log_check.stdout[-500:]}")
        return False


def _poll_until_done(
    instance: InstanceInfo,
    config: dict,
    max_seconds: int,
    poll_interval: int = 15,
) -> str:
    """Poll training log until tmux session ends or timeout.

    Returns the full log text.
    """
    start = time.time()
    last_step = 0

    while True:
        elapsed = int(time.time() - start)

        # Check if tmux session is still alive
        tmux_check = ssh_exec(instance, config,
            "tmux has-session -t training 2>/dev/null && echo running || echo stopped",
            timeout=15)
        training_alive = "running" in tmux_check.stdout

        # Read log
        log_result = ssh_exec(instance, config,
            "cat /tmp/training_output.log 2>/dev/null || echo ''",
            timeout=15)
        log_text = log_result.stdout

        # Progress reporting
        train_losses = _parse_train_losses(log_text)
        if train_losses:
            current_step = train_losses[-1][0]
            if current_step != last_step:
                loss_str = f" train_loss={train_losses[-1][1]:.4f}"
                print(f"  [{elapsed:4d}s] Step {current_step}{loss_str}")
                last_step = current_step

        # Check for errors in log
        parsed = _parse_log(log_text)
        if parsed.get("error"):
            print(f"  [{elapsed:4d}s] Error detected: {parsed['error']}")

        # Done conditions
        if not training_alive:
            print(f"  [{elapsed:4d}s] Training session ended")
            # Read final log
            final_result = ssh_exec(instance, config,
                "cat /tmp/training_output.log 2>/dev/null || echo ''",
                timeout=15)
            return final_result.stdout

        # Grace period: max_seconds + 120s buffer for GPTQ/cleanup
        if elapsed > max_seconds + 120:
            print(f"  [{elapsed:4d}s] Timeout exceeded, killing training")
            ssh_exec(instance, config,
                "tmux kill-session -t training 2>/dev/null || true")
            time.sleep(2)
            final_result = ssh_exec(instance, config,
                "cat /tmp/training_output.log 2>/dev/null || echo ''",
                timeout=15)
            return final_result.stdout

        time.sleep(poll_interval)


# ---------------------------------------------------------------------------
# Stage 1: Smoke test (1xH100, 300s)
# ---------------------------------------------------------------------------

def run_smoke(
    name: str,
    script: str,
    env: dict[str, str],
    config: dict,
) -> SmokeResult:
    """Run a smoke test on a3-highgpu-1g (1xH100).

    Verifies the training script starts, loss decreases, and no crashes.
    Uses VOCAB_SIZE=1024 with existing sp1024 data to avoid download issues.
    """
    smoke_name = f"smoke-{name}"
    wallclock = config.get("smoketest_wallclock_seconds", 300)
    poll_interval = config.get("smoketest_poll_seconds", 15)
    machine_type = config.get("smoketest_machine_type", "a3-highgpu-1g")

    print(f"\n{'='*60}")
    print(f"  SMOKE TEST: {name}")
    print(f"  Machine: {machine_type} (1xH100)")
    print(f"  Wallclock: {wallclock}s")
    print(f"  Script: {script}")
    print(f"{'='*60}")

    # Smoke env: use sp1024 data, short wallclock
    smoke_env = {
        "MAX_WALLCLOCK_SECONDS": str(wallclock),
        "VOCAB_SIZE": "1024",
        "VAL_LOSS_EVERY": "500",       # validate more often in smoke
        "TRAIN_LOG_EVERY": "50",        # log more often
    }
    smoke_env.update(env)

    instance = None
    try:
        # Provision
        instance, err = _provision_and_prepare(
            smoke_name, script, config, machine_type=machine_type)
        if instance is None:
            result = SmokeResult(status="error", error=err or "provisioning failed")
            _update_stage(name, "smoke", {"status": result.status, "error": result.error})
            return result

        # Start training with nproc=1
        if not _start_torchrun(instance, config, script, smoke_env, smoke_name, nproc=1):
            result = SmokeResult(
                status="fail", error="Training failed to start",
                instance_name=instance.name, instance_zone=instance.zone)
            _update_stage(name, "smoke", {"status": "fail", "error": result.error})
            return result

        # Poll until done
        log_text = _poll_until_done(instance, config, wallclock, poll_interval)

        # Analyze results
        train_losses = _parse_train_losses(log_text)
        parsed = _parse_log(log_text)

        last_step = train_losses[-1][0] if train_losses else 0
        last_loss = train_losses[-1][1] if train_losses else None
        first_loss = train_losses[0][1] if train_losses else None

        loss_decreased = (first_loss is not None and last_loss is not None
                          and last_loss < first_loss)

        error = parsed.get("error") or ""

        # Determine pass/fail
        if error and last_step < 10:
            status = "fail"
        elif last_step == 0:
            status = "fail"
            error = error or "No training steps completed"
        elif not loss_decreased and last_step > 20:
            status = "fail"
            error = error or f"Loss did not decrease: {first_loss:.4f} -> {last_loss:.4f}"
        else:
            status = "pass"

        result = SmokeResult(
            status=status,
            train_loss_last=last_loss,
            last_step=last_step,
            loss_decreased=loss_decreased,
            error=error,
            instance_name=instance.name,
            instance_zone=instance.zone,
        )

        # Persist
        _update_stage(name, "smoke", {
            "status": result.status,
            "train_loss_last": result.train_loss_last,
            "last_step": result.last_step,
            "loss_decreased": result.loss_decreased,
            "error": result.error,
        })

        return result

    except KeyboardInterrupt:
        print("\n  Interrupted by user")
        result = SmokeResult(status="error", error="Interrupted")
        _update_stage(name, "smoke", {"status": "error", "error": "Interrupted"})
        raise
    except Exception as e:
        print(f"  Unexpected error: {e}")
        result = SmokeResult(status="error", error=str(e))
        _update_stage(name, "smoke", {"status": "error", "error": str(e)})
        return result
    finally:
        if instance is not None:
            try:
                delete_instance(instance.name, instance.zone, config["project"])
            except Exception as e:
                print(f"  Warning: cleanup failed for {instance.name}: {e}")


# ---------------------------------------------------------------------------
# Stage 2: Qualify (8xH100, 180s)
# ---------------------------------------------------------------------------

def run_qualify(
    name: str,
    script: str,
    env: dict[str, str],
    config: dict,
) -> QualifyResult:
    """Run a qualify test on a3-highgpu-8g (8xH100).

    Gets val_loss and val_bpb around step 1000 for correlation with final BPB.
    Uses full VOCAB_SIZE=8192 (the real tokenizer).
    """
    qualify_name = f"qual-{name}"
    wallclock = config.get("qualify_wallclock_seconds", 180)
    poll_interval = config.get("poll_interval_seconds", 30)

    print(f"\n{'='*60}")
    print(f"  QUALIFY: {name}")
    print(f"  Machine: a3-highgpu-8g (8xH100)")
    print(f"  Wallclock: {wallclock}s")
    print(f"  Script: {script}")
    print(f"{'='*60}")

    # Qualify env: short wallclock, validate at step 1000
    qualify_env = {
        "MAX_WALLCLOCK_SECONDS": str(wallclock),
        "VAL_LOSS_EVERY": "1000",
        "TRAIN_LOG_EVERY": "100",
    }
    qualify_env.update(env)

    instance = None
    try:
        # Provision (standard 8xH100)
        instance, err = _provision_and_prepare(qualify_name, script, config)
        if instance is None:
            result = QualifyResult(status="error", error=err or "provisioning failed")
            _update_stage(name, "qualify", {"status": result.status, "error": result.error})
            return result

        # Start training with nproc=8
        if not _start_torchrun(instance, config, script, qualify_env, qualify_name, nproc=8):
            result = QualifyResult(
                status="fail", error="Training failed to start",
                instance_name=instance.name, instance_zone=instance.zone)
            _update_stage(name, "qualify", {"status": "fail", "error": result.error})
            return result

        # Poll until done
        log_text = _poll_until_done(instance, config, wallclock, poll_interval)

        # Analyze results
        train_losses = _parse_train_losses(log_text)
        val_metrics = _parse_val_metrics(log_text)
        parsed = _parse_log(log_text)

        last_step = train_losses[-1][0] if train_losses else 0
        error = parsed.get("error") or ""

        # Find val metrics closest to step 1000
        step_1000_bpb = None
        val_loss_1000 = None
        if val_metrics:
            # Pick the entry closest to step 1000
            best = min(val_metrics, key=lambda x: abs(x[0] - 1000))
            val_loss_1000 = best[1]
            step_1000_bpb = best[2]
            print(f"  Val metrics at step {best[0]}: "
                  f"val_loss={val_loss_1000:.4f} val_bpb={step_1000_bpb:.4f}")

        # Also use _parse_log as fallback
        if step_1000_bpb is None and parsed.get("step_1000_bpb"):
            step_1000_bpb = parsed["step_1000_bpb"]

        # Determine pass/fail
        if error and last_step < 100:
            status = "fail"
        elif step_1000_bpb is not None:
            status = "pass"
        elif last_step >= 800:
            # Ran long enough but no val metrics — still useful
            status = "pass"
        else:
            status = "fail"
            error = error or f"Only reached step {last_step}, no val metrics"

        result = QualifyResult(
            status=status,
            step_1000_bpb=step_1000_bpb,
            val_loss_1000=val_loss_1000,
            last_step=last_step,
            error=error,
            instance_name=instance.name,
            instance_zone=instance.zone,
        )

        _update_stage(name, "qualify", {
            "status": result.status,
            "step_1000_bpb": result.step_1000_bpb,
            "val_loss_1000": result.val_loss_1000,
            "last_step": result.last_step,
            "error": result.error,
        })

        return result

    except KeyboardInterrupt:
        print("\n  Interrupted by user")
        _update_stage(name, "qualify", {"status": "error", "error": "Interrupted"})
        raise
    except Exception as e:
        print(f"  Unexpected error: {e}")
        result = QualifyResult(status="error", error=str(e))
        _update_stage(name, "qualify", {"status": "error", "error": str(e)})
        return result
    finally:
        if instance is not None:
            try:
                delete_instance(instance.name, instance.zone, config["project"])
            except Exception as e:
                print(f"  Warning: cleanup failed for {instance.name}: {e}")


# ---------------------------------------------------------------------------
# Stage 3: Full run (8xH100, 600s)
# ---------------------------------------------------------------------------

def run_full(
    name: str,
    script: str,
    env: dict[str, str],
    config: dict,
) -> FullResult:
    """Run the full experiment on a3-highgpu-8g (8xH100).

    Full 600s training with GPTQ quantization and eval.
    """
    full_name = f"full-{name}"
    wallclock = config.get("max_wallclock_seconds", 660)
    poll_interval = config.get("poll_interval_seconds", 30)

    print(f"\n{'='*60}")
    print(f"  FULL RUN: {name}")
    print(f"  Machine: a3-highgpu-8g (8xH100)")
    print(f"  Wallclock: {wallclock}s (competition: 600s)")
    print(f"  Script: {script}")
    print(f"{'='*60}")

    # Full env: standard wallclock
    full_env = {
        "MAX_WALLCLOCK_SECONDS": "600",
    }
    full_env.update(env)

    instance = None
    try:
        # Provision
        instance, err = _provision_and_prepare(full_name, script, config)
        if instance is None:
            result = FullResult(status="error", error=err or "provisioning failed")
            _update_stage(name, "full", {"status": result.status, "error": result.error})
            return result

        # Start training with nproc=8
        if not _start_torchrun(instance, config, script, full_env, full_name, nproc=8):
            result = FullResult(
                status="fail", error="Training failed to start",
                instance_name=instance.name, instance_zone=instance.zone)
            _update_stage(name, "full", {"status": "fail", "error": result.error})
            return result

        # Poll until done — full run with GPTQ needs extra buffer
        log_text = _poll_until_done(instance, config, 660, poll_interval)

        # Analyze results
        parsed = _parse_log(log_text)
        train_losses = _parse_train_losses(log_text)
        val_metrics = _parse_val_metrics(log_text)

        last_step = train_losses[-1][0] if train_losses else 0
        error = parsed.get("error") or ""

        final_bpb = parsed.get("final_bpb")
        # Prefer the last val_bpb if available
        if val_metrics:
            last_val = val_metrics[-1]
            if final_bpb is None or last_val[0] > last_step * 0.9:
                final_bpb = last_val[2]

        # Check artifact size on instance
        artifact_size = None
        repo_dir = find_repo_dir(instance, config)
        size_check = ssh_exec(instance, config,
            f"stat -c%s {repo_dir}/final_model.int6.ptz 2>/dev/null || "
            f"stat -c%s {repo_dir}/final_model.pt 2>/dev/null || echo 0",
            timeout=10)
        try:
            artifact_size = int(size_check.stdout.strip().split('\n')[0])
        except (ValueError, IndexError):
            pass

        # Upload results
        gcs_uri = ""
        if final_bpb is not None or last_step > 0:
            try:
                gcs_uri = upload_results(instance, config, full_name)
            except Exception as e:
                print(f"  Upload failed: {e}")

        # Determine status
        if final_bpb is not None:
            status = "done"
        elif error:
            status = "fail"
        else:
            status = "fail"
            error = error or f"No final BPB, reached step {last_step}"

        result = FullResult(
            status=status,
            final_bpb=final_bpb,
            artifact_size=artifact_size,
            last_step=last_step,
            error=error,
            gcs_uri=gcs_uri,
            instance_name=instance.name,
            instance_zone=instance.zone,
        )

        _update_stage(name, "full", {
            "status": result.status,
            "final_bpb": result.final_bpb,
            "artifact_size": result.artifact_size,
            "last_step": result.last_step,
            "error": result.error,
            "gcs_uri": result.gcs_uri,
        })

        return result

    except KeyboardInterrupt:
        print("\n  Interrupted by user")
        _update_stage(name, "full", {"status": "error", "error": "Interrupted"})
        raise
    except Exception as e:
        print(f"  Unexpected error: {e}")
        result = FullResult(status="error", error=str(e))
        _update_stage(name, "full", {"status": "error", "error": str(e)})
        return result
    finally:
        if instance is not None:
            try:
                delete_instance(instance.name, instance.zone, config["project"])
            except Exception as e:
                print(f"  Warning: cleanup failed for {instance.name}: {e}")


# ---------------------------------------------------------------------------
# Auto pipeline: smoke -> qualify -> full
# ---------------------------------------------------------------------------

def run_auto(
    name: str,
    script: str,
    env: dict[str, str],
    config: dict,
    smoke_loss_threshold: float = 8.0,
) -> dict:
    """Run the full funnel: smoke -> qualify -> full.

    Stops early if smoke fails or qualify BPB is too bad.
    Returns a summary dict.
    """
    summary = {"name": name, "stages_run": []}

    # Stage 1: Smoke
    print("\n" + "=" * 60)
    print("  AUTO PIPELINE: Stage 1 — Smoke Test")
    print("=" * 60)
    smoke = run_smoke(name, script, env, config)
    summary["smoke"] = {
        "status": smoke.status,
        "train_loss_last": smoke.train_loss_last,
        "last_step": smoke.last_step,
    }
    summary["stages_run"].append("smoke")

    if smoke.status != "pass":
        print(f"\n  SMOKE FAILED: {smoke.error}")
        print("  Pipeline stopped. Fix the issue and retry.")
        summary["final_status"] = "smoke_failed"
        return summary

    print(f"\n  SMOKE PASSED: step {smoke.last_step}, "
          f"train_loss={smoke.train_loss_last:.4f}")

    # Gate: if smoke loss is absurdly high, skip qualify
    if (smoke.train_loss_last is not None
            and smoke.train_loss_last > smoke_loss_threshold):
        print(f"  Smoke train_loss {smoke.train_loss_last:.4f} > "
              f"threshold {smoke_loss_threshold:.1f}, skipping qualify")
        summary["final_status"] = "smoke_loss_too_high"
        return summary

    # Stage 2: Qualify
    print("\n" + "=" * 60)
    print("  AUTO PIPELINE: Stage 2 — Qualify (step-1000 BPB)")
    print("=" * 60)
    qualify = run_qualify(name, script, env, config)
    summary["qualify"] = {
        "status": qualify.status,
        "step_1000_bpb": qualify.step_1000_bpb,
        "val_loss_1000": qualify.val_loss_1000,
        "last_step": qualify.last_step,
    }
    summary["stages_run"].append("qualify")

    if qualify.status != "pass":
        print(f"\n  QUALIFY FAILED: {qualify.error}")
        summary["final_status"] = "qualify_failed"
        return summary

    if qualify.step_1000_bpb is not None:
        print(f"\n  QUALIFY PASSED: step_1000_bpb={qualify.step_1000_bpb:.4f}")
    else:
        print(f"\n  QUALIFY PASSED: reached step {qualify.last_step} (no BPB metric)")

    # Gate: use scaling law regression to predict final BPB and decide
    if qualify.step_1000_bpb is not None:
        predictor = ScalingPredictor.load()
        decision = predictor.decide_qualify(qualify.step_1000_bpb)
        pred, lo, hi = predictor.predict(qualify.step_1000_bpb)

        print(f"  Scaling law prediction: final_bpb={pred:.4f} [{lo:.4f}, {hi:.4f}]")
        print(f"  Decision: {decision.action} — {decision.reason}")

        summary["qualify"]["predicted_final_bpb"] = pred
        summary["qualify"]["prediction_interval"] = [lo, hi]
        summary["qualify"]["decision"] = decision.action

        if decision.action == "kill":
            print(f"  Pipeline stopped: qualify prediction not competitive")
            summary["final_status"] = "qualify_bpb_not_competitive"
            return summary
    else:
        # Fallback: check qualify BPB against known best (from state file)
        state = _load_state()
        best_qualify_bpb = None
        for exp_data in state["experiments"].values():
            q = exp_data.get("qualify", {})
            if q.get("status") == "pass" and q.get("step_1000_bpb") is not None:
                if best_qualify_bpb is None or q["step_1000_bpb"] < best_qualify_bpb:
                    best_qualify_bpb = q["step_1000_bpb"]

        if (best_qualify_bpb is not None
                and qualify.val_loss_1000 is not None
                and qualify.val_loss_1000 > best_qualify_bpb * 1.05):
            print(f"  Qualify val_loss {qualify.val_loss_1000:.4f} is >5% worse than "
                  f"best {best_qualify_bpb:.4f}, skipping full run")
            summary["final_status"] = "qualify_bpb_not_competitive"
            return summary

    # Stage 3: Full
    print("\n" + "=" * 60)
    print("  AUTO PIPELINE: Stage 3 — Full Run")
    print("=" * 60)
    full = run_full(name, script, env, config)
    summary["full"] = {
        "status": full.status,
        "final_bpb": full.final_bpb,
        "artifact_size": full.artifact_size,
        "last_step": full.last_step,
        "gcs_uri": full.gcs_uri,
    }
    summary["stages_run"].append("full")

    if full.status == "done":
        print(f"\n  FULL RUN COMPLETE: final_bpb={full.final_bpb:.4f}")
        if full.artifact_size:
            print(f"  Artifact size: {full.artifact_size:,} bytes "
                  f"({full.artifact_size / 1024 / 1024:.1f} MB)")

        # Auto-calibrate scaling law predictor with paired data
        if qualify.step_1000_bpb is not None and full.final_bpb is not None:
            predictor = ScalingPredictor.load()
            seed = int(env.get("SEED", 1337))
            predictor.add_calibration_point(name, seed,
                                            qualify.step_1000_bpb, full.final_bpb)
            predictor.save()
            print(f"  Scaling law updated: {predictor.fit.n_points} calibration points, "
                  f"R²={predictor.fit.r_squared:.3f}")

        summary["final_status"] = "done"
    else:
        print(f"\n  FULL RUN FAILED: {full.error}")
        summary["final_status"] = "full_failed"

    return summary


# ---------------------------------------------------------------------------
# Status display
# ---------------------------------------------------------------------------

def show_status(config: dict) -> None:
    """Display the current funnel state for all experiments."""
    state = _load_state()
    experiments = state.get("experiments", {})

    if not experiments:
        print("No experiments in funnel state.")
        return

    print(f"\n{'NAME':<30s} {'SMOKE':<12s} {'QUALIFY':<18s} {'FULL':<18s}")
    print("-" * 80)

    for exp_name, stages in sorted(experiments.items()):
        smoke = stages.get("smoke", {})
        qualify = stages.get("qualify", {})
        full = stages.get("full", {})

        smoke_str = smoke.get("status", "-")
        if smoke.get("train_loss_last"):
            smoke_str += f" ({smoke['train_loss_last']:.2f})"

        qualify_str = qualify.get("status", "-")
        if qualify.get("step_1000_bpb"):
            qualify_str += f" ({qualify['step_1000_bpb']:.4f})"

        full_str = full.get("status", "-")
        if full.get("final_bpb"):
            full_str += f" ({full['final_bpb']:.4f})"

        print(f"{exp_name:<30s} {smoke_str:<12s} {qualify_str:<18s} {full_str:<18s}")

    print()


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
        description="Parameter Golf — Multi-stage Experiment Funnel",
    )
    parser.add_argument("--config", default="infra/gce_config.yaml", help="GCE config path")
    sub = parser.add_subparsers(dest="command", required=True)

    # smoke
    p_smoke = sub.add_parser("smoke", help="Run smoke test (1xH100, 300s)")
    p_smoke.add_argument("--name", required=True, help="Experiment name")
    p_smoke.add_argument("--script", default="winning_base_decoded.py", help="Training script")
    p_smoke.add_argument("--env", nargs="*", default=[], help="KEY=VALUE env overrides")

    # qualify
    p_qualify = sub.add_parser("qualify", help="Run qualify test (8xH100, 180s)")
    p_qualify.add_argument("--name", required=True, help="Experiment name")
    p_qualify.add_argument("--script", default="winning_base_decoded.py", help="Training script")
    p_qualify.add_argument("--env", nargs="*", default=[], help="KEY=VALUE env overrides")

    # full
    p_full = sub.add_parser("full", help="Run full experiment (8xH100, 600s)")
    p_full.add_argument("--name", required=True, help="Experiment name")
    p_full.add_argument("--script", default="winning_base_decoded.py", help="Training script")
    p_full.add_argument("--env", nargs="*", default=[], help="KEY=VALUE env overrides")

    # auto
    p_auto = sub.add_parser("auto", help="Run full funnel: smoke -> qualify -> full")
    p_auto.add_argument("--name", required=True, help="Experiment name")
    p_auto.add_argument("--script", default="winning_base_decoded.py", help="Training script")
    p_auto.add_argument("--env", nargs="*", default=[], help="KEY=VALUE env overrides")
    p_auto.add_argument("--smoke-threshold", type=float, default=8.0,
                        help="Max smoke train_loss to proceed to qualify")

    # status
    sub.add_parser("status", help="Show funnel status for all experiments")

    args = parser.parse_args()
    config = load_config(args.config)

    if args.command == "smoke":
        env = _parse_env_args(args.env)
        result = run_smoke(args.name, args.script, env, config)
        _print_smoke_result(result)
        sys.exit(0 if result.status == "pass" else 1)

    elif args.command == "qualify":
        env = _parse_env_args(args.env)
        result = run_qualify(args.name, args.script, env, config)
        _print_qualify_result(result)
        sys.exit(0 if result.status == "pass" else 1)

    elif args.command == "full":
        env = _parse_env_args(args.env)
        result = run_full(args.name, args.script, env, config)
        _print_full_result(result)
        sys.exit(0 if result.status == "done" else 1)

    elif args.command == "auto":
        env = _parse_env_args(args.env)
        summary = run_auto(args.name, args.script, env, config,
                           smoke_loss_threshold=args.smoke_threshold)
        _print_auto_summary(summary)
        sys.exit(0 if summary.get("final_status") == "done" else 1)

    elif args.command == "status":
        show_status(config)


def _print_smoke_result(result: SmokeResult) -> None:
    print(f"\n{'='*60}")
    print(f"  SMOKE RESULT: {result.status.upper()}")
    if result.train_loss_last is not None:
        print(f"  Train loss (last): {result.train_loss_last:.4f}")
    print(f"  Steps completed: {result.last_step}")
    print(f"  Loss decreased: {result.loss_decreased}")
    if result.error:
        print(f"  Error: {result.error}")
    print(f"{'='*60}")


def _print_qualify_result(result: QualifyResult) -> None:
    print(f"\n{'='*60}")
    print(f"  QUALIFY RESULT: {result.status.upper()}")
    if result.step_1000_bpb is not None:
        print(f"  Step-1000 BPB: {result.step_1000_bpb:.4f}")
    if result.val_loss_1000 is not None:
        print(f"  Step-1000 val_loss: {result.val_loss_1000:.4f}")
    print(f"  Steps completed: {result.last_step}")
    if result.error:
        print(f"  Error: {result.error}")
    print(f"{'='*60}")


def _print_full_result(result: FullResult) -> None:
    print(f"\n{'='*60}")
    print(f"  FULL RESULT: {result.status.upper()}")
    if result.final_bpb is not None:
        print(f"  Final BPB: {result.final_bpb:.4f}")
    if result.artifact_size is not None:
        mb = result.artifact_size / 1024 / 1024
        print(f"  Artifact: {result.artifact_size:,} bytes ({mb:.1f} MB)")
    print(f"  Steps completed: {result.last_step}")
    if result.gcs_uri:
        print(f"  GCS: {result.gcs_uri}")
    if result.error:
        print(f"  Error: {result.error}")
    print(f"{'='*60}")


def _print_auto_summary(summary: dict) -> None:
    print(f"\n{'='*60}")
    print(f"  AUTO PIPELINE SUMMARY: {summary['name']}")
    print(f"  Final status: {summary.get('final_status', 'unknown')}")
    print(f"  Stages run: {', '.join(summary.get('stages_run', []))}")
    if "smoke" in summary:
        s = summary["smoke"]
        print(f"  Smoke: {s['status']} "
              f"(step {s.get('last_step', '?')}, "
              f"loss={s.get('train_loss_last', '?')})")
    if "qualify" in summary:
        q = summary["qualify"]
        bpb_str = f"{q['step_1000_bpb']:.4f}" if q.get("step_1000_bpb") else "N/A"
        print(f"  Qualify: {q['status']} "
              f"(step {q.get('last_step', '?')}, bpb={bpb_str})")
    if "full" in summary:
        f = summary["full"]
        bpb_str = f"{f['final_bpb']:.4f}" if f.get("final_bpb") else "N/A"
        print(f"  Full: {f['status']} "
              f"(step {f.get('last_step', '?')}, bpb={bpb_str})")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
