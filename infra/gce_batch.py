#!/usr/bin/env python3
"""
Batch Experiment Pipeline for Parameter Golf.

Reads experiment configs, queues by tier priority, runs experiments
sequentially or in parallel, tracks status, and manages early-kill.

Usage:
    # Run all Tier 1 experiments sequentially
    python infra/gce_batch.py run --tier 1

    # Run specific strategies
    python infra/gce_batch.py run --strategies T1-04,T1-07,T1-02

    # Run with 2 parallel instances
    python infra/gce_batch.py run --tier 1 --parallel 2

    # Check status of all experiments
    python infra/gce_batch.py status

    # Resume from a previous run (skip completed experiments)
    python infra/gce_batch.py run --tier 1 --resume

Requires: gcloud CLI, infra/gce_config.yaml, infra/gce_provision.py, infra/gce_run_experiment.py
"""
from __future__ import annotations

import argparse
import fcntl
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

from gce_provision import load_config
from gce_run_experiment import ExperimentResult, ExperimentStatus, run_experiment


# ---------------------------------------------------------------------------
# Strategy configuration
# ---------------------------------------------------------------------------

@dataclass
class StrategyConfig:
    """Configuration for a single experiment strategy."""
    name: str
    tier: int
    script: str
    env: dict
    seeds: list[int] = field(default_factory=lambda: [1337])
    description: str = ""
    priority: int = 0  # Higher = run first within tier


# ---------------------------------------------------------------------------
# Strategy definitions from pgolf_research_plan.md
# Env vars map to experiment1.py Hyperparameters class.
# ---------------------------------------------------------------------------

STRATEGIES: list[StrategyConfig] = [
    # ===== TIER 1: HIGH PRIORITY (28 strategies) =====

    # T1-01: Scylla Tokenizer -- requires separate tokenizer setup, placeholder
    StrategyConfig(
        name="T1-01-scylla-tokenizer", tier=1, priority=28,
        script="experiment1.py",
        env={"VOCAB_SIZE": "998"},
        seeds=[1337, 42, 7],
        description="Scylla tokenizer + current best stack (requires Scylla data variant)",
    ),

    # T1-02: SLOT on current base -- eval-time method, needs modified eval code
    StrategyConfig(
        name="T1-02-slot-basic", tier=1, priority=27,
        script="experiment1.py",
        env={"SLOT_ENABLED": "1", "SLOT_LR": "0.008", "SLOT_STEPS": "16"},
        seeds=[1337, 42, 7],
        description="SLOT (Selective Logit Offset Tuning) basic implementation",
    ),

    # T1-03: Per-Sample SLOT Delta
    StrategyConfig(
        name="T1-03-slot-persample", tier=1, priority=26,
        script="experiment1.py",
        env={"SLOT_ENABLED": "1", "SLOT_PER_SAMPLE": "1", "SLOT_LR": "0.008", "SLOT_STEPS": "16"},
        seeds=[1337, 42, 7],
        description="Per-sample SLOT delta with scored-position masking",
    ),

    # T1-04: QK-Gain Scaling (sweep)
    StrategyConfig(
        name="T1-04-qkgain-2.0", tier=1, priority=25,
        script="experiment1.py",
        env={"QK_GAIN_INIT": "2.0"},
        seeds=[1337, 42, 7],
        description="QK-Gain scaling at 2.0",
    ),
    StrategyConfig(
        name="T1-04-qkgain-3.0", tier=1, priority=25,
        script="experiment1.py",
        env={"QK_GAIN_INIT": "3.0"},
        seeds=[1337, 42, 7],
        description="QK-Gain scaling at 3.0",
    ),
    StrategyConfig(
        name="T1-04-qkgain-4.0", tier=1, priority=25,
        script="experiment1.py",
        env={"QK_GAIN_INIT": "4.0"},
        seeds=[1337, 42, 7],
        description="QK-Gain scaling at 4.0",
    ),
    StrategyConfig(
        name="T1-04-qkgain-5.0", tier=1, priority=25,
        script="experiment1.py",
        env={"QK_GAIN_INIT": "5.0"},
        seeds=[1337, 42, 7],
        description="QK-Gain scaling at 5.0",
    ),
    StrategyConfig(
        name="T1-04-qkgain-6.0", tier=1, priority=25,
        script="experiment1.py",
        env={"QK_GAIN_INIT": "6.0"},
        seeds=[1337, 42, 7],
        description="QK-Gain scaling at 6.0",
    ),

    # T1-05: MLP 3.5x + Mixed Int5/Int6
    StrategyConfig(
        name="T1-05-mlp35x-mixed-quant", tier=1, priority=24,
        script="experiment1.py",
        env={"MLP_MULT": "3.5", "QAT_ENABLED": "1"},
        seeds=[1337, 42, 7],
        description="MLP 3.5x with autopsy-informed mixed int5/int6 quantization",
    ),

    # T1-06: Brotli-11 compression -- systems, no training change
    StrategyConfig(
        name="T1-06-brotli-compression", tier=1, priority=23,
        script="experiment1.py",
        env={},
        seeds=[1337],
        description="Brotli-11 + byte-shuffle compression (post-training only)",
    ),

    # T1-07: XSA-All (11 layers)
    StrategyConfig(
        name="T1-07-xsa-all-11", tier=1, priority=22,
        script="experiment1.py",
        env={"XSA_LAST_N": "11"},
        seeds=[1337, 42, 7],
        description="XSA on all 11 layers with GPTQ",
    ),
    StrategyConfig(
        name="T1-07-xsa-6", tier=1, priority=22,
        script="experiment1.py",
        env={"XSA_LAST_N": "6"},
        seeds=[1337, 42, 7],
        description="XSA on last 6 layers",
    ),
    StrategyConfig(
        name="T1-07-xsa-8", tier=1, priority=22,
        script="experiment1.py",
        env={"XSA_LAST_N": "8"},
        seeds=[1337, 42, 7],
        description="XSA on last 8 layers",
    ),

    # T1-08: Coprime-Stride Multi-Shard Data Loader -- systems
    StrategyConfig(
        name="T1-08-coprime-stride", tier=1, priority=21,
        script="experiment1.py",
        env={},
        seeds=[1337],
        description="Coprime-stride data loading for better GPU utilization",
    ),

    # T1-09: IFNSO optimizer speedup -- systems
    StrategyConfig(
        name="T1-09-ifnso", tier=1, priority=20,
        script="experiment1.py",
        env={"MUON_BACKEND_STEPS": "3"},
        seeds=[1337, 42],
        description="Iteration-Free Newton-Schulz (3 steps instead of 5)",
    ),

    # T1-10: Legal Score-First TTT
    StrategyConfig(
        name="T1-10-ttt-scorefirst", tier=1, priority=19,
        script="experiment1.py",
        env={"TTT_ENABLED": "1", "TTT_LR": "0.002", "TTT_EPOCHS": "3"},
        seeds=[1337, 42, 7],
        description="Legal score-first TTT with GPTQ in training budget",
    ),

    # T1-13: WD 0.085 + MLP 4x simplification
    StrategyConfig(
        name="T1-13-wd085-mlp4x", tier=1, priority=16,
        script="experiment1.py",
        env={"MUON_WD": "0.085", "ADAM_WD": "0.085", "MLP_MULT": "4.0", "NUM_LAYERS": "10"},
        seeds=[1337, 42, 7],
        description="High weight decay + MLP 4x simplification approach",
    ),

    # T1-14: Mousse Optimizer
    StrategyConfig(
        name="T1-14-mousse", tier=1, priority=15,
        script="experiment1.py",
        env={"MUON_MOMENTUM": "0.95"},
        seeds=[1337, 42, 7],
        description="Mousse optimizer (curvature-aware Muon variant)",
    ),

    # T1-19: 1-sqrt Cooldown Shape
    StrategyConfig(
        name="T1-19-sqrt-cooldown", tier=1, priority=10,
        script="experiment1.py",
        env={"WARMDOWN_ITERS": "4000"},
        seeds=[1337, 42, 7],
        description="1-sqrt cooldown shape (longer warmdown)",
    ),

    # T1-20: Batch Size Warmup
    StrategyConfig(
        name="T1-20-batch-warmup", tier=1, priority=9,
        script="experiment1.py",
        env={"TRAIN_BATCH_TOKENS": "262144"},
        seeds=[1337, 42],
        description="Start with smaller batch (262K), ramp to 786K",
    ),

    # T1-21: Self-Generated GPTQ Calibration
    StrategyConfig(
        name="T1-21-gptq-selfgen", tier=1, priority=8,
        script="experiment1.py",
        env={"GPTQ_ENABLED": "1", "GPTQ_N_BATCHES": "64"},
        seeds=[1337, 42, 7],
        description="Self-generated GPTQ calibration (AR method, 64 batches)",
    ),

    # T1-22: Training-Data GPTQ Calibration
    StrategyConfig(
        name="T1-22-gptq-traindata", tier=1, priority=7,
        script="experiment1.py",
        env={"GPTQ_ENABLED": "1", "GPTQ_N_BATCHES": "32"},
        seeds=[1337, 42, 7],
        description="GPTQ calibration from training data (within budget)",
    ),

    # T1-24: Shallow Depth Recurrence
    StrategyConfig(
        name="T1-24-depth-recurrence", tier=1, priority=5,
        script="experiment1.py",
        env={"NUM_LAYERS": "13", "MLP_MULT": "2.5"},
        seeds=[1337, 42, 7],
        description="Shallow depth recurrence (2-pass, layers 4-5)",
    ),

    # T1-25: EMA Decay Tuning
    StrategyConfig(
        name="T1-25-ema-tuning-high", tier=1, priority=4,
        script="experiment1.py",
        env={"SWA_EVERY": "25"},
        seeds=[1337, 42],
        description="EMA decay tuning: more frequent averaging",
    ),
    StrategyConfig(
        name="T1-25-ema-tuning-low", tier=1, priority=4,
        script="experiment1.py",
        env={"SWA_EVERY": "100"},
        seeds=[1337, 42],
        description="EMA decay tuning: less frequent averaging",
    ),

    # T1-28: N-gram Backoff Cache
    StrategyConfig(
        name="T1-28-ngram-cache", tier=1, priority=1,
        script="experiment1.py",
        env={},
        seeds=[1337, 42, 7],
        description="N-gram backoff cache with Laplace normalization (eval-time)",
    ),

    # ===== TIER 2: MEDIUM PRIORITY (selected key strategies) =====

    # T2-01: Variance-Adaptive Muon
    StrategyConfig(
        name="T2-01-muon-vs", tier=2, priority=34,
        script="experiment1.py",
        env={"MUON_MOMENTUM": "0.97", "MUON_BETA2": "0.99"},
        seeds=[1337, 42],
        description="Variance-Adaptive Muon (Muon-VS)",
    ),

    # T2-04: FlashSigmoid Attention
    StrategyConfig(
        name="T2-04-flash-sigmoid", tier=2, priority=31,
        script="experiment1.py",
        env={},
        seeds=[1337, 42],
        description="FlashSigmoid attention (requires code modification)",
    ),

    # T2-06: CAGE QAT Gradient
    StrategyConfig(
        name="T2-06-cage-qat", tier=2, priority=29,
        script="experiment1.py",
        env={"QAT_ENABLED": "1", "LATE_QAT_THRESHOLD": "0.10"},
        seeds=[1337, 42],
        description="CAGE QAT Gradient (curvature-aware STE)",
    ),

    # T2-10: P2/Focal Loss for Token Weighting
    StrategyConfig(
        name="T2-10-focal-loss", tier=2, priority=25,
        script="experiment1.py",
        env={},
        seeds=[1337, 42],
        description="P2/Focal loss for token weighting (requires code mod)",
    ),

    # T2-11: Cross-Layer KV Sharing
    StrategyConfig(
        name="T2-11-kv-sharing", tier=2, priority=24,
        script="experiment1.py",
        env={"NUM_KV_HEADS": "2"},
        seeds=[1337, 42],
        description="Cross-layer KV sharing (MLKV/CLA, fewer KV heads)",
    ),

    # T2-14: Compute-Optimal QAT Scheduling
    StrategyConfig(
        name="T2-14-qat-scheduling-010", tier=2, priority=21,
        script="experiment1.py",
        env={"QAT_ENABLED": "1", "LATE_QAT_THRESHOLD": "0.10"},
        seeds=[1337, 42],
        description="QAT starting at 10% of training",
    ),
    StrategyConfig(
        name="T2-14-qat-scheduling-020", tier=2, priority=21,
        script="experiment1.py",
        env={"QAT_ENABLED": "1", "LATE_QAT_THRESHOLD": "0.20"},
        seeds=[1337, 42],
        description="QAT starting at 20% of training",
    ),

    # T2-20: Seesaw LR
    StrategyConfig(
        name="T2-20-seesaw-lr", tier=2, priority=15,
        script="experiment1.py",
        env={"MATRIX_LR": "0.030", "SCALAR_LR": "0.020"},
        seeds=[1337, 42],
        description="Seesaw LR: matrix_lr > scalar_lr",
    ),

    # T2-21: Layer-Wise FFN Scaling
    StrategyConfig(
        name="T2-21-nonuniform-mlp", tier=2, priority=14,
        script="experiment1.py",
        env={"MLP_MULT": "3.5", "NUM_LAYERS": "11"},
        seeds=[1337, 42],
        description="Non-uniform MLP width (wider in middle layers)",
    ),

    # T2-24: MTP Auxiliary Loss
    StrategyConfig(
        name="T2-24-mtp-2head", tier=2, priority=11,
        script="experiment1.py",
        env={"MTP_NUM_HEADS": "2", "MTP_LOSS_WEIGHT": "0.2"},
        seeds=[1337, 42],
        description="Multi-token prediction with 2 heads",
    ),
    StrategyConfig(
        name="T2-24-mtp-3head", tier=2, priority=11,
        script="experiment1.py",
        env={"MTP_NUM_HEADS": "3", "MTP_LOSS_WEIGHT": "0.15"},
        seeds=[1337, 42],
        description="Multi-token prediction with 3 heads",
    ),

    # T2-26: WSM Checkpoint Merging
    StrategyConfig(
        name="T2-26-wsm-merge", tier=2, priority=9,
        script="experiment1.py",
        env={"SWA_ENABLED": "0", "LAWA_ENABLED": "1", "LAWA_K": "10", "LAWA_FREQ": "100"},
        seeds=[1337, 42],
        description="WSM checkpoint merging (LAWA replacing SWA)",
    ),

    # ===== TIER 3: EXPLORATORY (selected key strategies) =====

    # T3-03: Differential Attention
    StrategyConfig(
        name="T3-03-diff-attention", tier=3, priority=25,
        script="experiment1.py",
        env={},
        seeds=[1337],
        description="Differential attention (requires code modification)",
    ),

    # T3-07: Larger Model at 2-3 Bit Quantization
    StrategyConfig(
        name="T3-07-large-2bit", tier=3, priority=19,
        script="experiment1.py",
        env={"NUM_LAYERS": "15", "MODEL_DIM": "640", "MLP_MULT": "3.0"},
        seeds=[1337, 42],
        description="Larger model (15L, 640d) with aggressive 2-3 bit quantization",
    ),

    # T3-10: Window Attention Training
    StrategyConfig(
        name="T3-10-window-attn", tier=3, priority=16,
        script="experiment1.py",
        env={"TRAIN_SEQ_LEN": "4096", "EVAL_SEQ_LEN": "4096"},
        seeds=[1337, 42],
        description="Window attention with mixed seq_len (4096)",
    ),

    # T3-13: Late-Stage SAM
    StrategyConfig(
        name="T3-13-late-sam", tier=3, priority=13,
        script="experiment1.py",
        env={},
        seeds=[1337, 42],
        description="Late-stage Sharpness-Aware Minimization",
    ),

    # T3-23: Cyclic Muon Momentum
    StrategyConfig(
        name="T3-23-cyclic-muon", tier=3, priority=3,
        script="experiment1.py",
        env={"MUON_MOMENTUM": "0.92"},
        seeds=[1337, 42],
        description="Cyclic Muon momentum (triangle wave between 0.92-0.99)",
    ),

    # ---- LR Sweep (additional exploration) ----
    StrategyConfig(
        name="sweep-lr-020", tier=2, priority=5,
        script="experiment1.py",
        env={"MATRIX_LR": "0.020", "SCALAR_LR": "0.020"},
        seeds=[1337, 42],
        description="LR sweep: 0.020",
    ),
    StrategyConfig(
        name="sweep-lr-030", tier=2, priority=5,
        script="experiment1.py",
        env={"MATRIX_LR": "0.030", "SCALAR_LR": "0.030"},
        seeds=[1337, 42],
        description="LR sweep: 0.030",
    ),
    StrategyConfig(
        name="sweep-lr-035", tier=2, priority=5,
        script="experiment1.py",
        env={"MATRIX_LR": "0.035", "SCALAR_LR": "0.035"},
        seeds=[1337, 42],
        description="LR sweep: 0.035",
    ),

    # ---- Architecture sweeps ----
    StrategyConfig(
        name="sweep-13L-mlp2", tier=2, priority=4,
        script="experiment1.py",
        env={"NUM_LAYERS": "13", "MLP_MULT": "2.0"},
        seeds=[1337, 42],
        description="Architecture: 13 layers, MLP 2x",
    ),
    StrategyConfig(
        name="sweep-9L-mlp4", tier=2, priority=4,
        script="experiment1.py",
        env={"NUM_LAYERS": "9", "MLP_MULT": "4.0"},
        seeds=[1337, 42],
        description="Architecture: 9 layers, MLP 4x",
    ),
    StrategyConfig(
        name="sweep-12L-mlp3", tier=2, priority=4,
        script="experiment1.py",
        env={"NUM_LAYERS": "12", "MLP_MULT": "3.0"},
        seeds=[1337, 42],
        description="Architecture: 12 layers, MLP 3x",
    ),
]


# ---------------------------------------------------------------------------
# State management
# ---------------------------------------------------------------------------

STATE_PATH = Path("infra/batch_state.json")


def load_state() -> dict:
    """Load batch state from disk. Creates empty state if missing."""
    if STATE_PATH.exists():
        with open(STATE_PATH) as f:
            fcntl.flock(f, fcntl.LOCK_SH)
            try:
                return json.load(f)
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)
    return {
        "experiments": {},
        "best_step_1000_bpb": None,
        "best_final_bpb": None,
        "last_updated": datetime.now().isoformat(),
    }


def save_state(state: dict) -> None:
    """Atomically save batch state to disk."""
    state["last_updated"] = datetime.now().isoformat()
    tmp_path = STATE_PATH.with_suffix(".tmp")
    with open(tmp_path, "w") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            json.dump(state, f, indent=2)
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)
    tmp_path.rename(STATE_PATH)


def update_state_with_result(state: dict, exp_key: str, result: ExperimentResult) -> None:
    """Update state with experiment result and recalculate bests."""
    state["experiments"][exp_key] = result.to_dict()

    # Update best metrics
    if result.status == ExperimentStatus.SUCCEEDED:
        if result.final_bpb > 0:
            if state["best_final_bpb"] is None or result.final_bpb < state["best_final_bpb"]:
                state["best_final_bpb"] = result.final_bpb
        if result.step_1000_bpb > 0:
            if state["best_step_1000_bpb"] is None or result.step_1000_bpb < state["best_step_1000_bpb"]:
                state["best_step_1000_bpb"] = result.step_1000_bpb


# ---------------------------------------------------------------------------
# Strategy loading & filtering
# ---------------------------------------------------------------------------

def load_strategies(
    tier: int | None = None,
    names: list[str] | None = None,
) -> list[StrategyConfig]:
    """Load and filter strategies. Returns sorted by (tier ASC, priority DESC)."""
    strategies = STRATEGIES

    if tier is not None:
        strategies = [s for s in strategies if s.tier == tier]

    if names:
        name_set = set(names)
        # Match by exact name or prefix
        strategies = [
            s for s in strategies
            if s.name in name_set or any(s.name.startswith(n) for n in name_set)
        ]

    return sorted(strategies, key=lambda s: (s.tier, -s.priority))


def expand_to_experiments(
    strategies: list[StrategyConfig],
) -> list[tuple[str, StrategyConfig, int]]:
    """Expand strategies into per-seed experiments.

    Returns list of (experiment_key, strategy, seed) tuples.
    """
    experiments = []
    for strategy in strategies:
        for seed in strategy.seeds:
            key = f"{strategy.name}-seed{seed}"
            experiments.append((key, strategy, seed))
    return experiments


# ---------------------------------------------------------------------------
# Batch execution
# ---------------------------------------------------------------------------

def run_single_in_batch(
    exp_key: str,
    strategy: StrategyConfig,
    seed: int,
    config: dict,
    state: dict,
) -> ExperimentResult:
    """Run a single experiment within the batch pipeline."""
    env = dict(strategy.env)
    env["SEED"] = str(seed)

    best_bpb = state.get("best_step_1000_bpb")

    result = run_experiment(
        name=exp_key,
        script=strategy.script,
        env=env,
        config=config,
        best_step_1000_bpb=best_bpb,
    )

    # Update state
    update_state_with_result(state, exp_key, result)
    save_state(state)

    return result


def run_batch(
    strategies: list[StrategyConfig],
    config: dict,
    parallel: int = 1,
    resume: bool = False,
) -> list[ExperimentResult]:
    """Run a batch of experiments.

    For parallel=1: sequential, updating early-kill threshold after each.
    For parallel>1: ThreadPoolExecutor with N workers.
    """
    state = load_state()
    experiments = expand_to_experiments(strategies)

    if resume:
        # Skip already completed or early-killed experiments
        skip_statuses = {"succeeded", "early_killed"}
        experiments = [
            (key, strat, seed) for key, strat, seed in experiments
            if key not in state["experiments"]
            or state["experiments"][key].get("status") not in skip_statuses
        ]

    total = len(experiments)
    if total == 0:
        print("No experiments to run (all completed or filtered out).")
        return []

    print(f"\n{'='*60}")
    print(f" Batch: {total} experiments across {len(strategies)} strategies")
    print(f" Parallel: {parallel}")
    print(f" Resume: {resume}")
    if state["best_final_bpb"]:
        print(f" Current best BPB: {state['best_final_bpb']:.4f}")
    print(f"{'='*60}\n")

    results: list[ExperimentResult] = []

    if parallel <= 1:
        # Sequential execution
        for i, (key, strategy, seed) in enumerate(experiments):
            print(f"\n[{i+1}/{total}] {key}")
            print(f"  Strategy: {strategy.description}")

            # Check if first seed of this strategy was early-killed
            # If so, skip remaining seeds
            base_name = strategy.name
            first_seed_key = f"{base_name}-seed{strategy.seeds[0]}"
            if (first_seed_key in state["experiments"]
                    and state["experiments"][first_seed_key].get("status") == "early_killed"
                    and seed != strategy.seeds[0]):
                print(f"  Skipping: first seed was early-killed")
                continue

            # Reload state for latest best_bpb
            state = load_state()

            result = run_single_in_batch(key, strategy, seed, config, state)
            results.append(result)

            print(f"  -> {result.status.value}", end="")
            if result.final_bpb:
                print(f" (BPB={result.final_bpb:.4f})", end="")
            print()
    else:
        # Parallel execution
        with ThreadPoolExecutor(max_workers=parallel) as pool:
            futures = {}
            for key, strategy, seed in experiments:
                future = pool.submit(
                    run_single_in_batch,
                    key, strategy, seed, config, state,
                )
                futures[future] = key

            for future in as_completed(futures):
                key = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                    status = result.status.value
                    bpb = f" BPB={result.final_bpb:.4f}" if result.final_bpb else ""
                    print(f"  [{key}] {status}{bpb}")
                except Exception as e:
                    print(f"  [{key}] ERROR: {e}")

    return results


# ---------------------------------------------------------------------------
# Status display
# ---------------------------------------------------------------------------

def print_status(state: dict) -> None:
    """Print formatted experiment status table."""
    experiments = state.get("experiments", {})
    if not experiments:
        print("No experiments recorded yet.")
        return

    print(f"\n{'EXPERIMENT':<45s} {'STATUS':<15s} {'STEP1K BPB':>10s} {'FINAL BPB':>10s} {'TIME':>8s}")
    print("-" * 95)

    # Sort by status then name
    sorted_exps = sorted(experiments.items(), key=lambda x: (x[1].get("status", ""), x[0]))

    succeeded = 0
    failed = 0
    killed = 0

    for key, data in sorted_exps:
        status = data.get("status", "unknown")
        s1k = data.get("step_1000_bpb", 0)
        final = data.get("final_bpb", 0)
        wall = data.get("wallclock_seconds", 0)

        s1k_str = f"{s1k:.4f}" if s1k else "-"
        final_str = f"{final:.4f}" if final else "-"
        wall_str = f"{wall:.0f}s" if wall else "-"

        print(f"{key:<45s} {status:<15s} {s1k_str:>10s} {final_str:>10s} {wall_str:>8s}")

        if status == "succeeded":
            succeeded += 1
        elif status == "failed":
            failed += 1
        elif status == "early_killed":
            killed += 1

    total = len(experiments)
    print(f"\nTotal: {total} | Succeeded: {succeeded} | Failed: {failed} | Early-killed: {killed}")

    if state.get("best_final_bpb"):
        print(f"Best final BPB: {state['best_final_bpb']:.4f}")
    if state.get("best_step_1000_bpb"):
        print(f"Best step-1000 BPB: {state['best_step_1000_bpb']:.4f}")
    print(f"Last updated: {state.get('last_updated', 'unknown')}")


def print_strategies(strategies: list[StrategyConfig]) -> None:
    """Print a list of strategies with their details."""
    print(f"\n{'NAME':<40s} {'TIER':>4s} {'PRI':>4s} {'SEEDS':>5s} {'DESCRIPTION'}")
    print("-" * 100)
    for s in strategies:
        seeds = len(s.seeds)
        print(f"{s.name:<40s} {s.tier:>4d} {s.priority:>4d} {seeds:>5d}   {s.description[:50]}")
    total_runs = sum(len(s.seeds) for s in strategies)
    print(f"\n{len(strategies)} strategies, {total_runs} total experiment runs")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Parameter Golf — Batch Experiment Pipeline",
    )
    parser.add_argument("--config", default="infra/gce_config.yaml", help="GCE config path")
    sub = parser.add_subparsers(dest="command", required=True)

    # run
    p_run = sub.add_parser("run", help="Run experiments")
    p_run.add_argument("--tier", type=int, default=None, help="Run all strategies in a tier (1, 2, or 3)")
    p_run.add_argument("--strategies", default=None, help="Comma-separated strategy names")
    p_run.add_argument("--parallel", type=int, default=1, help="Number of parallel instances")
    p_run.add_argument("--resume", action="store_true", help="Skip completed experiments")

    # status
    sub.add_parser("status", help="Show experiment status")

    # list
    p_list = sub.add_parser("list", help="List available strategies")
    p_list.add_argument("--tier", type=int, default=None, help="Filter by tier")

    # reset
    p_reset = sub.add_parser("reset", help="Reset batch state (careful!)")
    p_reset.add_argument("--confirm", action="store_true", help="Confirm reset")

    args = parser.parse_args()
    config = load_config(args.config)

    if args.command == "run":
        names = args.strategies.split(",") if args.strategies else None
        strategies = load_strategies(tier=args.tier, names=names)
        if not strategies:
            print("No strategies matched the filter.")
            sys.exit(1)

        print_strategies(strategies)
        results = run_batch(
            strategies, config,
            parallel=args.parallel,
            resume=args.resume,
        )

        # Summary
        succeeded = sum(1 for r in results if r.status == ExperimentStatus.SUCCEEDED)
        failed = sum(1 for r in results if r.status == ExperimentStatus.FAILED)
        killed = sum(1 for r in results if r.status == ExperimentStatus.EARLY_KILLED)
        print(f"\n{'='*60}")
        print(f" Batch complete: {succeeded} succeeded, {failed} failed, {killed} early-killed")
        best = min((r.final_bpb for r in results if r.final_bpb), default=0)
        if best:
            print(f" Best BPB this batch: {best:.4f}")
        print(f"{'='*60}")

    elif args.command == "status":
        state = load_state()
        print_status(state)

    elif args.command == "list":
        strategies = load_strategies(tier=args.tier)
        print_strategies(strategies)

    elif args.command == "reset":
        if args.confirm:
            STATE_PATH.unlink(missing_ok=True)
            print("Batch state reset.")
        else:
            print("Pass --confirm to actually reset the state.")


if __name__ == "__main__":
    main()
