#!/usr/bin/env python3
"""
Mass architecture screening pipeline for Parameter Golf.

Screens many architectures through cheap 1xH100 smoke tests ($0.25 each),
ranks them by train_loss, promotes top performers to evo tree nodes,
then evo:optimize iterates on the best branches.

Usage:
    python infra/mass_screen.py screen              # Smoke test all strategies
    python infra/mass_screen.py screen --parallel 5  # 5 concurrent smoke tests
    python infra/mass_screen.py promote --top 5      # Promote top 5 to evo nodes
    python infra/mass_screen.py status               # Show rankings
    python infra/mass_screen.py add --name "my-idea" --script winning_base_decoded.py --env "QK_GAIN=6.0"

Requires: infra/funnel.py, infra/gce_provision.py
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent))

from funnel import run_smoke, SmokeResult
from gce_provision import load_config


# ---------------------------------------------------------------------------
# Strategy definitions — ALL architectures to screen
# ---------------------------------------------------------------------------

@dataclass
class Strategy:
    name: str
    script: str
    env: dict[str, str]
    description: str = ""
    tier: int = 1
    source: str = "research_plan"


# The winning base with different hyperparameter/architecture variations
# Each creates a different evo branch if promoted
STRATEGIES: list[Strategy] = [
    # === WINNING BASE VARIANTS (SP8192 architecture) ===
    Strategy("win-qkgain-525", "winning_base_decoded.py",
             {"QK_GAIN_INIT": "5.25"},
             "QK-Gain 5.25 (latest competition record)", tier=1, source="PR#1471"),
    Strategy("win-qkgain-550", "winning_base_decoded.py",
             {"QK_GAIN_INIT": "5.5"},
             "QK-Gain 5.5 (push further)", tier=1, source="interpolation"),
    Strategy("win-qkgain-475", "winning_base_decoded.py",
             {"QK_GAIN_INIT": "4.75"},
             "QK-Gain 4.75 (between 4.0 and 5.0)", tier=1, source="interpolation"),
    Strategy("win-parallel-5", "winning_base_decoded.py",
             {"PARALLEL_RESIDUAL_START": "5"},
             "Earlier parallel residuals (layer 5)", tier=1, source="architecture"),
    Strategy("win-parallel-9", "winning_base_decoded.py",
             {"PARALLEL_RESIDUAL_START": "9"},
             "Later parallel residuals (layer 9)", tier=2, source="architecture"),
    Strategy("win-loop-2-6", "winning_base_decoded.py",
             {"LOOP_START": "2", "LOOP_END": "6"},
             "Wider recurrence (layers 2-6 instead of 3-5)", tier=1, source="architecture"),
    Strategy("win-loop-4-5", "winning_base_decoded.py",
             {"LOOP_START": "4", "LOOP_END": "5"},
             "Narrower recurrence (layers 4-5 only)", tier=2, source="architecture"),
    Strategy("win-3loops", "winning_base_decoded.py",
             {"NUM_LOOPS": "3"},
             "3x depth recurrence (more virtual layers)", tier=1, source="architecture"),
    Strategy("win-lr-025", "winning_base_decoded.py",
             {"MATRIX_LR": "0.025", "SCALAR_LR": "0.023"},
             "Higher LR (compensate for SDPA fallback)", tier=1, source="optimization"),
    Strategy("win-lr-020", "winning_base_decoded.py",
             {"MATRIX_LR": "0.020", "SCALAR_LR": "0.018"},
             "Lower LR", tier=2, source="optimization"),
    Strategy("win-wd-010", "winning_base_decoded.py",
             {"MUON_WD": "0.10", "EMBED_WD": "0.09"},
             "Higher weight decay", tier=1, source="optimization"),
    Strategy("win-wd-008", "winning_base_decoded.py",
             {"MUON_WD": "0.08", "EMBED_WD": "0.07"},
             "Lower weight decay", tier=2, source="optimization"),
    Strategy("win-warmdown-080", "winning_base_decoded.py",
             {"WARMDOWN_FRAC": "0.80"},
             "Longer warmdown (80% vs 72%)", tier=1, source="optimization"),
    Strategy("win-warmdown-065", "winning_base_decoded.py",
             {"WARMDOWN_FRAC": "0.65"},
             "Shorter warmdown (65%)", tier=2, source="optimization"),
    Strategy("win-ema-0997", "winning_base_decoded.py",
             {"EMA_DECAY": "0.997"},
             "Higher EMA decay", tier=2, source="optimization"),
    Strategy("win-loop-early-025", "winning_base_decoded.py",
             {"ENABLE_LOOPING_AT": "0.25"},
             "Earlier recurrence activation (25% vs 35%)", tier=1, source="architecture"),
    Strategy("win-loop-late-045", "winning_base_decoded.py",
             {"ENABLE_LOOPING_AT": "0.45"},
             "Later recurrence activation (45%)", tier=2, source="architecture"),
    Strategy("win-mlp-35", "winning_base_decoded.py",
             {"MLP_MULT": "3.5"},
             "Narrower MLP (3.5x vs 4x, more steps in budget)", tier=1, source="architecture"),
    Strategy("win-mlp-45", "winning_base_decoded.py",
             {"MLP_MULT": "4.5"},
             "Wider MLP (4.5x, fewer steps but more capacity)", tier=2, source="architecture"),
    Strategy("win-no-skipgates", "winning_base_decoded.py",
             {"SKIP_GATES_ENABLED": "0"},
             "Disable skip gates (simpler architecture)", tier=2, source="ablation"),
    Strategy("win-no-parallel", "winning_base_decoded.py",
             {"PARALLEL_RESIDUAL_START": "99"},
             "Disable parallel residuals (all sequential)", tier=2, source="ablation"),
    Strategy("win-no-loop", "winning_base_decoded.py",
             {"NUM_LOOPS": "1"},
             "Disable depth recurrence (standard 11L)", tier=2, source="ablation"),

    # === COMBINATIONS (multi-change) ===
    Strategy("win-combo-qk525-wd010", "winning_base_decoded.py",
             {"QK_GAIN_INIT": "5.25", "MUON_WD": "0.10"},
             "QK-Gain 5.25 + higher WD", tier=1, source="combination"),
    Strategy("win-combo-qk525-lr025", "winning_base_decoded.py",
             {"QK_GAIN_INIT": "5.25", "MATRIX_LR": "0.025"},
             "QK-Gain 5.25 + higher LR", tier=1, source="combination"),
    Strategy("win-combo-3loop-parallel5", "winning_base_decoded.py",
             {"NUM_LOOPS": "3", "PARALLEL_RESIDUAL_START": "5"},
             "3x recurrence + earlier parallel", tier=1, source="combination"),
    Strategy("win-combo-wideloop-earlystart", "winning_base_decoded.py",
             {"LOOP_START": "2", "LOOP_END": "6", "ENABLE_LOOPING_AT": "0.25"},
             "Wide recurrence (2-6) + early activation", tier=1, source="combination"),
    Strategy("win-combo-mlp35-3loop", "winning_base_decoded.py",
             {"MLP_MULT": "3.5", "NUM_LOOPS": "3"},
             "Narrower MLP + 3x recurrence (more steps, more depth)", tier=1, source="combination"),

    # === OUR OLD ARCHITECTURE VARIANTS (experiment1.py SP1024) ===
    Strategy("old-baseline", "experiment1.py",
             {},
             "Original SP1024 baseline for comparison", tier=3, source="baseline"),
    Strategy("old-qk4-xsa11", "experiment1.py",
             {"QK_GAIN_INIT": "4.0", "XSA_LAST_N": "11"},
             "Best from old experiments (1.1527 BPB)", tier=3, source="evo_run0"),
]

# All strategies use VOCAB_SIZE=1024 + COMPRESSOR=lzma for smoke tests
SMOKE_ENV_DEFAULTS = {"VOCAB_SIZE": "1024", "COMPRESSOR": "lzma"}


# ---------------------------------------------------------------------------
# State management
# ---------------------------------------------------------------------------

STATE_PATH = Path("infra/screen_state.json")


def load_screen_state() -> dict:
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text())
    return {"results": {}, "last_updated": None}


def save_screen_state(state: dict) -> None:
    state["last_updated"] = datetime.now().isoformat()
    STATE_PATH.write_text(json.dumps(state, indent=2))


# ---------------------------------------------------------------------------
# Screening
# ---------------------------------------------------------------------------

def screen_strategy(strategy: Strategy, config: dict) -> dict:
    """Run a single smoke test and return the result."""
    env = dict(SMOKE_ENV_DEFAULTS)
    env.update(strategy.env)

    print(f"\n  [{strategy.name}] Smoke testing: {strategy.description}")
    try:
        result = run_smoke(strategy.name, strategy.script, env, config)
        return {
            "name": strategy.name,
            "status": result.status,
            "train_loss": result.train_loss_last,
            "last_step": result.last_step,
            "loss_decreased": result.loss_decreased,
            "error": result.error,
            "script": strategy.script,
            "env": strategy.env,
            "description": strategy.description,
            "tier": strategy.tier,
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        return {
            "name": strategy.name,
            "status": "error",
            "train_loss": None,
            "error": str(e),
            "script": strategy.script,
            "env": strategy.env,
            "description": strategy.description,
            "tier": strategy.tier,
            "timestamp": datetime.now().isoformat(),
        }


def run_mass_screen(
    strategies: list[Strategy],
    config: dict,
    parallel: int = 3,
    resume: bool = True,
) -> list[dict]:
    """Screen multiple strategies through smoke tests."""
    state = load_screen_state()

    # Filter out already-tested if resuming
    if resume:
        strategies = [s for s in strategies if s.name not in state["results"]
                      or state["results"][s.name].get("status") == "error"]

    if not strategies:
        print("All strategies already screened. Use --no-resume to re-run.")
        return list(state["results"].values())

    total = len(strategies)
    print(f"\n{'='*60}")
    print(f" Mass Screening: {total} strategies, {parallel} parallel")
    print(f"{'='*60}")

    results = []

    if parallel <= 1:
        for i, strategy in enumerate(strategies):
            print(f"\n[{i+1}/{total}]", end="")
            result = screen_strategy(strategy, config)
            results.append(result)
            state["results"][strategy.name] = result
            save_screen_state(state)
            _print_result(result)
    else:
        with ThreadPoolExecutor(max_workers=parallel) as pool:
            futures = {pool.submit(screen_strategy, s, config): s for s in strategies}
            for i, future in enumerate(as_completed(futures)):
                strategy = futures[future]
                try:
                    result = future.result()
                except Exception as e:
                    result = {"name": strategy.name, "status": "error", "train_loss": None, "error": str(e)}
                results.append(result)
                state["results"][strategy.name] = result
                save_screen_state(state)
                print(f"\n[{i+1}/{total}]", end="")
                _print_result(result)

    return results


def _print_result(r: dict):
    status = r.get("status", "?")
    loss = r.get("train_loss")
    name = r.get("name", "?")
    if status == "pass" and loss is not None:
        print(f"  PASS {name}: train_loss={loss:.4f} at step {r.get('last_step', '?')}")
    else:
        print(f"  {status.upper()} {name}: {r.get('error', '')[:60]}")


# ---------------------------------------------------------------------------
# Ranking and promotion
# ---------------------------------------------------------------------------

def get_rankings(state: dict) -> list[dict]:
    """Rank all passed strategies by train_loss (lower is better)."""
    passed = [
        r for r in state["results"].values()
        if r.get("status") == "pass" and r.get("train_loss") is not None
    ]
    return sorted(passed, key=lambda r: r["train_loss"])


def promote_to_evo(rankings: list[dict], top_n: int = 5) -> None:
    """Create evo experiment nodes for the top N strategies."""
    to_promote = rankings[:top_n]

    print(f"\n{'='*60}")
    print(f" Promoting top {len(to_promote)} strategies to evo tree")
    print(f"{'='*60}")

    for r in to_promote:
        name = r["name"]
        desc = r.get("description", "")
        loss = r.get("train_loss", "?")
        env_str = " ".join(f"{k}={v}" for k, v in r.get("env", {}).items())

        hypothesis = f"{name}: smoke train_loss={loss:.4f}. {desc}. env: {env_str}"

        # Create evo experiment
        cmd = ["evo", "new", "--parent", "exp_0000", "-m", hypothesis]
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0:
            # Parse the experiment ID
            import re
            m = re.search(r'"id":\s*"([^"]+)"', result.stdout)
            exp_id = m.group(1) if m else "?"

            # Record the smoke score so evo knows the baseline
            done_cmd = ["evo", "done", exp_id, "--score", str(loss)]
            subprocess.run(done_cmd, capture_output=True, text=True)

            print(f"  {exp_id}: {name} (loss={loss:.4f}) -> evo node")
        else:
            print(f"  FAILED to create evo node for {name}: {result.stderr[:80]}")


# ---------------------------------------------------------------------------
# Status display
# ---------------------------------------------------------------------------

def print_status():
    state = load_screen_state()
    rankings = get_rankings(state)
    total = len(state.get("results", {}))
    passed = len(rankings)
    failed = total - passed

    print(f"\n{'='*60}")
    print(f" Mass Screening Status")
    print(f" Total: {total} | Passed: {passed} | Failed: {failed}")
    print(f"{'='*60}")

    if rankings:
        print(f"\n{'RANK':>4} {'TRAIN_LOSS':>10} {'NAME':<35} {'DESCRIPTION'}")
        print("-" * 95)
        for i, r in enumerate(rankings):
            loss = r.get("train_loss", 0)
            name = r.get("name", "?")
            desc = r.get("description", "")[:40]
            marker = " <-- BEST" if i == 0 else ""
            print(f"{i+1:>4} {loss:>10.4f} {name:<35} {desc}{marker}")

    failed_list = [r for r in state.get("results", {}).values() if r.get("status") != "pass"]
    if failed_list:
        print(f"\nFailed/Error ({len(failed_list)}):")
        for r in failed_list:
            print(f"  {r.get('name', '?')}: {r.get('status', '?')} — {r.get('error', '')[:60]}")

    print(f"\nLast updated: {state.get('last_updated', 'never')}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Mass Architecture Screening")
    sub = parser.add_subparsers(dest="command", required=True)

    p_screen = sub.add_parser("screen", help="Smoke test all strategies")
    p_screen.add_argument("--parallel", type=int, default=3)
    p_screen.add_argument("--tier", type=int, default=None, help="Only screen this tier")
    p_screen.add_argument("--no-resume", action="store_true")

    p_promote = sub.add_parser("promote", help="Promote top strategies to evo")
    p_promote.add_argument("--top", type=int, default=5)

    sub.add_parser("status", help="Show rankings")

    p_add = sub.add_parser("add", help="Add a custom strategy")
    p_add.add_argument("--name", required=True)
    p_add.add_argument("--script", default="winning_base_decoded.py")
    p_add.add_argument("--env", nargs="*", default=[])
    p_add.add_argument("--description", default="")

    args = parser.parse_args()
    config = load_config()

    if args.command == "screen":
        strategies = STRATEGIES
        if args.tier is not None:
            strategies = [s for s in strategies if s.tier == args.tier]
        run_mass_screen(strategies, config, parallel=args.parallel, resume=not args.no_resume)
        print_status()

    elif args.command == "promote":
        state = load_screen_state()
        rankings = get_rankings(state)
        promote_to_evo(rankings, top_n=args.top)

    elif args.command == "status":
        print_status()

    elif args.command == "add":
        env = {}
        for item in args.env:
            k, v = item.split("=", 1)
            env[k] = v
        strategy = Strategy(args.name, args.script, env, args.description)
        STRATEGIES.append(strategy)
        result = screen_strategy(strategy, config)
        state = load_screen_state()
        state["results"][args.name] = result
        save_screen_state(state)
        _print_result(result)


if __name__ == "__main__":
    main()
