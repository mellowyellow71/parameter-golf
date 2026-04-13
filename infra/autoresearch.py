#!/usr/bin/env python3
"""
Autoresearch Loop for Parameter Golf.

Autonomous research orchestrator that:
1. Runs pre-defined Tier 1-3 strategies in priority order
2. Analyzes results to identify promising hyperparameter dimensions
3. Generates new hypotheses (interpolation, combination, perturbation)
4. Dispatches experiments to GCE H100 instances
5. Runs continuously until manually stopped

Usage:
    python infra/autoresearch.py                    # Start the loop
    python infra/autoresearch.py --max-cycles 10    # Run 10 cycles then stop
    python infra/autoresearch.py --parallel 2       # 2 concurrent experiments
    python infra/autoresearch.py --dry-run          # Print what would run
    python infra/autoresearch.py status             # Print current state

Requires: gcloud CLI, infra/gce_config.yaml, infra/gce_batch.py
"""
from __future__ import annotations

import argparse
import fcntl
import json
import logging
import signal
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

from gce_batch import (
    STRATEGIES,
    StrategyConfig,
    expand_to_experiments,
    load_state,
    save_state,
    update_state_with_result,
    run_single_in_batch,
)
from gce_provision import load_config
from gce_run_experiment import ExperimentResult, ExperimentStatus

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("autoresearch")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

AUTORESEARCH_STATE_PATH = Path("infra/autoresearch_state.json")
LOOP_SLEEP_SECONDS = 60
MAX_GENERATED_PER_CYCLE = 5
H100_SPOT_COST_PER_HOUR = 24.48  # a3-highgpu-8g SPOT approximate

# Searchable hyperparameter dimensions with ranges and step sizes.
# Must stay in sync with experiment1.py Hyperparameters class.
SEARCHABLE_DIMENSIONS: dict[str, dict] = {
    "QK_GAIN_INIT":       {"default": 1.5,   "min": 0.5,  "max": 10.0, "type": "float", "step": 0.5},
    "MLP_MULT":           {"default": 3.0,   "min": 2.0,  "max": 5.0,  "type": "float", "step": 0.5},
    "NUM_LAYERS":         {"default": 11,    "min": 8,    "max": 15,   "type": "int",   "step": 1},
    "MODEL_DIM":          {"default": 512,   "min": 384,  "max": 768,  "type": "int",   "step": 64},
    "MATRIX_LR":          {"default": 0.025, "min": 0.01, "max": 0.05, "type": "float", "step": 0.005},
    "SCALAR_LR":          {"default": 0.025, "min": 0.01, "max": 0.05, "type": "float", "step": 0.005},
    "MUON_MOMENTUM":      {"default": 0.99,  "min": 0.90, "max": 0.999,"type": "float", "step": 0.01},
    "MUON_WD":            {"default": 0.04,  "min": 0.01, "max": 0.15, "type": "float", "step": 0.01},
    "XSA_LAST_N":         {"default": 4,     "min": 0,    "max": 11,   "type": "int",   "step": 1},
    "WARMDOWN_ITERS":     {"default": 3500,  "min": 2000, "max": 6000, "type": "int",   "step": 500},
    "SWA_EVERY":          {"default": 50,    "min": 10,   "max": 200,  "type": "int",   "step": 10},
    "LOGIT_SOFTCAP":      {"default": 30.0,  "min": 15.0, "max": 60.0, "type": "float", "step": 5.0},
    "MUON_BACKEND_STEPS": {"default": 5,     "min": 2,    "max": 7,    "type": "int",   "step": 1},
}

# Dimensions that shouldn't be combined (they interact too strongly)
COMBINATION_CONFLICTS = [
    {"MUON_MOMENTUM", "MUON_BACKEND_STEPS"},
    {"MATRIX_LR", "SCALAR_LR"},
]

_shutdown_requested = False


def _sigint_handler(signum, frame):
    global _shutdown_requested
    _shutdown_requested = True
    log.info("Shutdown requested. Finishing current experiment...")


# ---------------------------------------------------------------------------
# Autoresearch state management
# ---------------------------------------------------------------------------

def load_autoresearch_state() -> dict:
    if AUTORESEARCH_STATE_PATH.exists():
        with open(AUTORESEARCH_STATE_PATH) as f:
            fcntl.flock(f, fcntl.LOCK_SH)
            try:
                return json.load(f)
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)
    return {
        "autoresearch_version": 1,
        "cycle_count": 0,
        "phase": "ANALYZE",
        "started_at": datetime.now().isoformat(),
        "last_cycle_at": None,
        "generated_strategies": [],
        "dimension_analysis": {},
        "early_killed_patterns": [],
        "total_gpu_hours": 0.0,
    }


def save_autoresearch_state(state: dict) -> None:
    state["last_cycle_at"] = datetime.now().isoformat()
    tmp = AUTORESEARCH_STATE_PATH.with_suffix(".tmp")
    with open(tmp, "w") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            json.dump(state, f, indent=2)
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)
    tmp.rename(AUTORESEARCH_STATE_PATH)


# ---------------------------------------------------------------------------
# ANALYZE: Extract insights from completed experiments
# ---------------------------------------------------------------------------

def _get_env_for_experiment(exp_name: str) -> dict[str, str]:
    """Look up the env vars for a predefined or generated experiment."""
    # Check predefined strategies
    base = exp_name.rsplit("-seed", 1)[0] if "-seed" in exp_name else exp_name
    for s in STRATEGIES:
        if s.name == base:
            return dict(s.env)
    return {}


def _extract_dimension_values(experiments: dict) -> dict[str, list[tuple[float, float]]]:
    """Extract (hyperparam_value, bpb) pairs for each varied dimension."""
    dim_values: dict[str, list[tuple[float, float]]] = {}

    for exp_name, data in experiments.items():
        if data.get("status") != "succeeded":
            continue
        bpb = data.get("final_bpb", 0)
        if not bpb or bpb <= 0:
            continue

        env = _get_env_for_experiment(exp_name)
        for dim_name, dim_info in SEARCHABLE_DIMENSIONS.items():
            if dim_name in env:
                try:
                    val = float(env[dim_name])
                except (ValueError, TypeError):
                    continue
                dim_values.setdefault(dim_name, []).append((val, bpb))

    return dim_values


def _classify_shape(pairs: list[tuple[float, float]]) -> str:
    """Classify the response curve shape for a dimension."""
    if len(pairs) < 2:
        return "insufficient_data"
    sorted_pairs = sorted(pairs, key=lambda x: x[0])
    vals = [p[0] for p in sorted_pairs]
    bpbs = [p[1] for p in sorted_pairs]

    bpb_range = max(bpbs) - min(bpbs)
    if bpb_range < 0.001:
        return "flat"

    # Check if best is at an interior point (concave peak)
    best_idx = bpbs.index(min(bpbs))
    if 0 < best_idx < len(bpbs) - 1:
        return "concave_peak"
    elif best_idx == 0:
        return "monotonic_up"  # BPB increases with dimension value
    else:
        return "monotonic_down"


def analyze_results(batch_state: dict, ar_state: dict) -> dict:
    """Analyze completed experiments and update dimension analysis."""
    experiments = batch_state.get("experiments", {})
    dim_values = _extract_dimension_values(experiments)

    analysis = {}
    for dim_name, pairs in dim_values.items():
        if not pairs:
            continue
        sorted_pairs = sorted(pairs, key=lambda x: x[1])  # sort by BPB
        best_val, best_bpb = sorted_pairs[0]
        all_bpbs = [p[1] for p in pairs]
        analysis[dim_name] = {
            "tested_values": sorted(set(p[0] for p in pairs)),
            "best_value": best_val,
            "best_bpb": best_bpb,
            "sensitivity": max(all_bpbs) - min(all_bpbs),
            "shape": _classify_shape(pairs),
            "n_samples": len(pairs),
        }

    ar_state["dimension_analysis"] = analysis

    # Update early-killed patterns
    for exp_name, data in experiments.items():
        if data.get("status") == "early_killed":
            env = _get_env_for_experiment(exp_name)
            if env and not any(
                p.get("env_pattern") == env for p in ar_state["early_killed_patterns"]
            ):
                ar_state["early_killed_patterns"].append({
                    "env_pattern": env,
                    "experiment": exp_name,
                    "reason": f"early-killed, step1000_bpb={data.get('step_1000_bpb', '?')}",
                })

    return ar_state


# ---------------------------------------------------------------------------
# HYPOTHESIZE: Generate new experiment ideas
# ---------------------------------------------------------------------------

def _is_similar_to_early_killed(env: dict, patterns: list[dict]) -> bool:
    for pat in patterns:
        pat_env = pat.get("env_pattern", {})
        if all(env.get(k) == v for k, v in pat_env.items()):
            return True
    return False


def _existing_names(batch_state: dict, ar_state: dict) -> set[str]:
    names = set(batch_state.get("experiments", {}).keys())
    for gen in ar_state.get("generated_strategies", []):
        names.add(gen["name"])
    return names


def _make_value(val: float, dim_info: dict) -> str:
    if dim_info["type"] == "int":
        return str(int(round(val)))
    return str(round(val, 4))


def _generate_interpolation(analysis: dict, existing: set[str]) -> list[dict]:
    """Bisect between best two adjacent tested values for concave peak dimensions."""
    results = []
    for dim_name, info in analysis.items():
        if info["shape"] != "concave_peak" or info["n_samples"] < 3:
            continue
        dim_def = SEARCHABLE_DIMENSIONS.get(dim_name)
        if not dim_def:
            continue

        tested = sorted(info["tested_values"])
        best_val = info["best_value"]
        best_idx = tested.index(best_val) if best_val in tested else -1
        if best_idx < 0:
            continue

        # Try midpoints with neighbors
        for neighbor_idx in [best_idx - 1, best_idx + 1]:
            if 0 <= neighbor_idx < len(tested):
                mid = (tested[best_idx] + tested[neighbor_idx]) / 2
                if dim_def["type"] == "int":
                    mid = round(mid)
                    if mid in tested:
                        continue
                mid = max(dim_def["min"], min(dim_def["max"], mid))
                val_str = _make_value(mid, dim_def)
                name = f"AUTO-interp-{dim_name.lower()}-{val_str}-seed1337"
                if name not in existing:
                    results.append({
                        "name": name,
                        "origin": "interpolation",
                        "hypothesis": f"Bisect {dim_name} between {tested[best_idx]} and {tested[neighbor_idx]}",
                        "env": {dim_name: val_str},
                        "script": "experiment1.py",
                        "seeds": [1337, 42],
                        "priority": 80,
                        "status": "pending",
                    })
    return results


def _generate_combinations(analysis: dict, batch_state: dict, existing: set[str]) -> list[dict]:
    """Combine best values from top-2 most sensitive independent dimensions."""
    # Sort by sensitivity
    ranked = sorted(
        [(name, info) for name, info in analysis.items() if info["sensitivity"] > 0.002],
        key=lambda x: -x[1]["sensitivity"],
    )
    if len(ranked) < 2:
        return []

    # Take top 2, check for conflicts
    for i in range(len(ranked)):
        for j in range(i + 1, len(ranked)):
            dim_a, info_a = ranked[i]
            dim_b, info_b = ranked[j]
            # Check conflicts
            if any(dim_a in c and dim_b in c for c in COMBINATION_CONFLICTS):
                continue
            val_a = _make_value(info_a["best_value"], SEARCHABLE_DIMENSIONS[dim_a])
            val_b = _make_value(info_b["best_value"], SEARCHABLE_DIMENSIONS[dim_b])
            name = f"AUTO-combo-{dim_a.lower()}-{val_a}-{dim_b.lower()}-{val_b}-seed1337"
            if name not in existing:
                return [{
                    "name": name,
                    "origin": "combination",
                    "hypothesis": f"Combine best {dim_a}={val_a} with best {dim_b}={val_b}",
                    "env": {dim_a: val_a, dim_b: val_b},
                    "script": "experiment1.py",
                    "seeds": [1337, 42, 7],
                    "priority": 70,
                    "status": "pending",
                }]
    return []


def _generate_perturbations(analysis: dict, batch_state: dict, existing: set[str]) -> list[dict]:
    """Perturb the best config by +/- one step on each dimension."""
    # Find the overall best experiment
    best_exp = None
    best_bpb = float("inf")
    for name, data in batch_state.get("experiments", {}).items():
        if data.get("status") == "succeeded" and data.get("final_bpb", 99) < best_bpb:
            best_bpb = data["final_bpb"]
            best_exp = name

    if not best_exp:
        return []

    best_env = _get_env_for_experiment(best_exp)
    results = []

    for dim_name, dim_def in SEARCHABLE_DIMENSIONS.items():
        current = float(best_env.get(dim_name, dim_def["default"]))
        step = dim_def["step"]

        for delta in [step, -step]:
            new_val = current + delta
            new_val = max(dim_def["min"], min(dim_def["max"], new_val))
            if dim_def["type"] == "int":
                new_val = round(new_val)
            if abs(new_val - current) < 1e-6:
                continue

            val_str = _make_value(new_val, dim_def)
            direction = "up" if delta > 0 else "down"
            name = f"AUTO-perturb-{dim_name.lower()}-{val_str}-seed1337"
            if name not in existing:
                env = dict(best_env)
                env[dim_name] = val_str
                results.append({
                    "name": name,
                    "origin": "perturbation",
                    "hypothesis": f"Perturb {dim_name} {direction} from {current} to {new_val}",
                    "env": env,
                    "script": "experiment1.py",
                    "seeds": [1337],
                    "priority": 50,
                    "status": "pending",
                })

    return results


def generate_hypotheses(batch_state: dict, ar_state: dict, max_new: int = MAX_GENERATED_PER_CYCLE) -> list[dict]:
    """Generate new experiment hypotheses based on analysis."""
    analysis = ar_state.get("dimension_analysis", {})
    existing = _existing_names(batch_state, ar_state)
    early_killed = ar_state.get("early_killed_patterns", [])

    all_new = []

    # Priority 1: Interpolation
    all_new.extend(_generate_interpolation(analysis, existing))
    # Priority 2: Combination
    all_new.extend(_generate_combinations(analysis, batch_state, existing))
    # Priority 3: Perturbation
    all_new.extend(_generate_perturbations(analysis, batch_state, existing))

    # Filter out early-killed-similar configs
    filtered = [
        h for h in all_new
        if not _is_similar_to_early_killed(h["env"], early_killed)
    ]

    # Limit
    return filtered[:max_new]


# ---------------------------------------------------------------------------
# SELECT: Pick next experiments to run
# ---------------------------------------------------------------------------

def select_next_experiments(
    batch_state: dict,
    ar_state: dict,
    max_to_select: int = 1,
) -> list[tuple[str, StrategyConfig, int]]:
    """Select next experiments: predefined first, then generated."""
    completed = set(batch_state.get("experiments", {}).keys())
    selected = []

    # Phase 1: Predefined strategies (sorted by tier asc, priority desc)
    all_predefined = sorted(STRATEGIES, key=lambda s: (s.tier, -s.priority))
    for strategy in all_predefined:
        if len(selected) >= max_to_select:
            break
        for seed in strategy.seeds:
            if len(selected) >= max_to_select:
                break
            key = f"{strategy.name}-seed{seed}"
            if key in completed:
                continue
            # Skip if first seed early-killed
            first_key = f"{strategy.name}-seed{strategy.seeds[0]}"
            if (first_key in completed
                    and batch_state["experiments"][first_key].get("status") == "early_killed"
                    and seed != strategy.seeds[0]):
                continue
            selected.append((key, strategy, seed))

    # Phase 2: Generated hypotheses
    for gen in sorted(ar_state.get("generated_strategies", []), key=lambda g: -g.get("priority", 0)):
        if len(selected) >= max_to_select:
            break
        if gen.get("status") != "pending":
            continue
        for seed in gen.get("seeds", [1337]):
            if len(selected) >= max_to_select:
                break
            key = f"{gen['name']}-seed{seed}" if f"-seed{seed}" not in gen["name"] else gen["name"]
            if key in completed:
                continue
            config = StrategyConfig(
                name=gen["name"],
                tier=0,
                script=gen.get("script", "experiment1.py"),
                env=gen.get("env", {}),
                seeds=gen.get("seeds", [1337]),
                description=gen.get("hypothesis", "auto-generated"),
                priority=gen.get("priority", 0),
            )
            selected.append((key, config, seed))

    return selected


# ---------------------------------------------------------------------------
# DISPATCH + MONITOR
# ---------------------------------------------------------------------------

def dispatch_and_monitor(
    experiments: list[tuple[str, StrategyConfig, int]],
    config: dict,
    batch_state: dict,
    ar_state: dict,
    parallel: int = 1,
) -> list[ExperimentResult]:
    """Dispatch experiments using existing GCE infrastructure."""
    results = []
    for key, strategy, seed in experiments:
        if _shutdown_requested:
            log.info("Shutdown requested, stopping dispatch")
            break

        log.info(f"Dispatching: {key}")
        log.info(f"  Env: {strategy.env}")
        log.info(f"  Description: {strategy.description}")

        result = run_single_in_batch(key, strategy, seed, config, batch_state)
        results.append(result)

        # Update GPU hours
        if result.wallclock_seconds > 0:
            ar_state["total_gpu_hours"] += result.wallclock_seconds / 3600

        # Mark generated strategy as done
        for gen in ar_state.get("generated_strategies", []):
            if gen["name"] == strategy.name:
                gen["status"] = result.status.value
                break

        log.info(f"  Result: {result.status.value}" +
                 (f" BPB={result.final_bpb:.4f}" if result.final_bpb else ""))

    return results


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run_autoresearch_loop(
    config: dict,
    max_cycles: int | None = None,
    parallel: int = 1,
    dry_run: bool = False,
) -> None:
    """Main autoresearch loop."""
    signal.signal(signal.SIGINT, _sigint_handler)
    signal.signal(signal.SIGTERM, _sigint_handler)

    ar_state = load_autoresearch_state()
    log.info(f"Autoresearch starting (cycle {ar_state['cycle_count']})")

    while not _shutdown_requested:
        cycle = ar_state["cycle_count"] + 1
        if max_cycles and cycle > max_cycles:
            log.info(f"Max cycles ({max_cycles}) reached. Stopping.")
            break

        log.info(f"\n{'='*60}")
        log.info(f" CYCLE {cycle}")
        log.info(f"{'='*60}")

        # Load latest batch state
        batch_state = load_state()

        # Phase 1: ANALYZE
        ar_state["phase"] = "ANALYZE"
        log.info("[ANALYZE] Analyzing completed experiments...")
        ar_state = analyze_results(batch_state, ar_state)
        n_analyzed = len(ar_state.get("dimension_analysis", {}))
        log.info(f"  {n_analyzed} dimensions analyzed")

        # Phase 2: HYPOTHESIZE
        ar_state["phase"] = "HYPOTHESIZE"
        log.info("[HYPOTHESIZE] Generating new hypotheses...")
        new_hypotheses = generate_hypotheses(batch_state, ar_state)
        if new_hypotheses:
            ar_state["generated_strategies"].extend(new_hypotheses)
            for h in new_hypotheses:
                log.info(f"  NEW: {h['name']} ({h['origin']}): {h['hypothesis']}")
        else:
            log.info("  No new hypotheses generated")

        # Phase 3: SELECT
        ar_state["phase"] = "SELECT"
        log.info("[SELECT] Picking next experiments...")
        next_exps = select_next_experiments(batch_state, ar_state, max_to_select=parallel)
        if not next_exps:
            log.info("  No experiments to run. All strategies exhausted.")
            log.info(f"  Waiting {LOOP_SLEEP_SECONDS}s before retrying...")
            save_autoresearch_state(ar_state)
            time.sleep(LOOP_SLEEP_SECONDS)
            continue

        for key, strat, seed in next_exps:
            log.info(f"  Selected: {key}")

        if dry_run:
            log.info("[DRY RUN] Would dispatch these experiments. Skipping.")
            ar_state["cycle_count"] = cycle
            save_autoresearch_state(ar_state)
            continue

        # Phase 4: DISPATCH + MONITOR
        ar_state["phase"] = "DISPATCH"
        log.info("[DISPATCH] Running experiments...")
        results = dispatch_and_monitor(next_exps, config, batch_state, ar_state, parallel)

        # Phase 5: RECORD
        ar_state["phase"] = "RECORD"
        ar_state["cycle_count"] = cycle
        save_autoresearch_state(ar_state)

        # Summary
        succeeded = sum(1 for r in results if r.status == ExperimentStatus.SUCCEEDED)
        failed = sum(1 for r in results if r.status == ExperimentStatus.FAILED)
        killed = sum(1 for r in results if r.status == ExperimentStatus.EARLY_KILLED)
        best = min((r.final_bpb for r in results if r.final_bpb), default=0)
        log.info(f"[RECORD] Cycle {cycle}: {succeeded} ok, {failed} fail, {killed} killed" +
                 (f", best BPB={best:.4f}" if best else ""))
        log.info(f"  Total GPU hours: {ar_state['total_gpu_hours']:.1f}h "
                 f"(~${ar_state['total_gpu_hours'] * H100_SPOT_COST_PER_HOUR:.0f})")

    # Final save
    save_autoresearch_state(ar_state)
    log.info("Autoresearch loop stopped.")


def print_status():
    """Print current autoresearch state."""
    ar_state = load_autoresearch_state()
    batch_state = load_state()

    total = len(batch_state.get("experiments", {}))
    succeeded = sum(1 for d in batch_state.get("experiments", {}).values() if d.get("status") == "succeeded")
    best = batch_state.get("best_final_bpb")
    gen = len(ar_state.get("generated_strategies", []))
    pending_gen = sum(1 for g in ar_state.get("generated_strategies", []) if g.get("status") == "pending")
    dims = len(ar_state.get("dimension_analysis", {}))

    print(f"\n{'='*60}")
    print(f" Autoresearch Status")
    print(f"{'='*60}")
    print(f"  Cycle:          {ar_state['cycle_count']}")
    print(f"  Phase:          {ar_state['phase']}")
    print(f"  Started:        {ar_state.get('started_at', '?')}")
    print(f"  Last cycle:     {ar_state.get('last_cycle_at', '?')}")
    print(f"  GPU hours:      {ar_state['total_gpu_hours']:.1f}h (~${ar_state['total_gpu_hours'] * H100_SPOT_COST_PER_HOUR:.0f})")
    print(f"\n  Experiments:    {total} total, {succeeded} succeeded")
    print(f"  Best BPB:       {best:.4f}" if best else "  Best BPB:       -")
    print(f"  Dimensions:     {dims} analyzed")
    print(f"  Hypotheses:     {gen} generated, {pending_gen} pending")
    print(f"  Early-kills:    {len(ar_state.get('early_killed_patterns', []))}")

    if ar_state.get("dimension_analysis"):
        print(f"\n  Dimension Analysis:")
        for name, info in sorted(ar_state["dimension_analysis"].items(), key=lambda x: -x[1].get("sensitivity", 0)):
            print(f"    {name:25s} best={info['best_value']:>8} sens={info['sensitivity']:.4f} shape={info['shape']}")

    print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Parameter Golf — Autoresearch Loop")
    parser.add_argument("command", nargs="?", default="run", choices=["run", "status", "reset"])
    parser.add_argument("--config", default="infra/gce_config.yaml")
    parser.add_argument("--max-cycles", type=int, default=None)
    parser.add_argument("--parallel", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.command == "status":
        print_status()
        return

    if args.command == "reset":
        AUTORESEARCH_STATE_PATH.unlink(missing_ok=True)
        print("Autoresearch state reset.")
        return

    config = load_config(args.config)
    run_autoresearch_loop(
        config,
        max_cycles=args.max_cycles,
        parallel=args.parallel,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
