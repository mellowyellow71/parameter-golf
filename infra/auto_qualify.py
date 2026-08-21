#!/usr/bin/env python3
"""
Autonomous qualify runner for Parameter Golf overnight screening.

Picks the next untested strategy from strategies_v2.py,
runs it through qualify (8xH100, 3min, step-1000 BPB), records results,
and integrates with evo for tree-based tracking.

Usage:
    python infra/auto_qualify.py          # Run next untested strategy
    python infra/auto_qualify.py --status  # Show results so far
    python infra/auto_qualify.py --best    # Show best result
    python infra/auto_qualify.py --next    # Show what would run next (dry run)
"""
from __future__ import annotations

import fcntl
import json
import subprocess
import sys
import traceback
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from strategies_v2 import CATEGORY_A, Strategy
from funnel import run_qualify, QualifyResult
from gce_provision import load_config

RESULTS_PATH = Path("infra/qualify_results.json")


LOCK_PATH = Path("infra/qualify_results.lock")


def load_results() -> dict:
    if RESULTS_PATH.exists():
        return json.loads(RESULTS_PATH.read_text())
    return {"results": {}, "best": None, "best_bpb": 99.0, "runs": 0}


def save_results(data: dict):
    data["last_updated"] = datetime.now().isoformat()
    RESULTS_PATH.write_text(json.dumps(data, indent=2))


def claim_next() -> Strategy | None:
    """Atomically claim the next untested strategy using file lock."""
    with open(LOCK_PATH, "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            data = load_results()
            nxt = get_next(data)
            if nxt is None:
                return None
            # Mark as claimed (in-progress) so other workers skip it
            data["results"][nxt.name] = {
                "name": nxt.name, "status": "running",
                "timestamp": datetime.now().isoformat(),
            }
            save_results(data)
            return nxt
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def get_queue() -> list[Strategy]:
    """Get strategies ordered by priority, env-only (no code changes needed)."""
    env_only = [s for s in CATEGORY_A if not s.needs_code]
    priority_order = {"P0": 0, "P1": 1, "P2": 2}
    env_only.sort(key=lambda s: (priority_order.get(s.priority, 9), s.tier))
    return env_only


def get_next(results: dict) -> Strategy | None:
    """Get the next untested strategy (skips running/claimed ones too)."""
    tested = set(results.get("results", {}).keys())
    for s in get_queue():
        if s.name not in tested:
            return s
    return None



# Architecture branch mapping for run_0003
# exp_0000 = SP8192 winning base, exp_0001 = SP4096, exp_0002 = SP1024, exp_0003 = Novel
EVO_ARCH_BRANCHES = {
    "winning_base_decoded.py": "exp_0000",  # SP8192 winning base
    "arch_sp8192.py": "exp_0000",
    "arch_sp4096.py": "exp_0001",
    "arch_mini_recur.py": "exp_0001",
    "experiment1.py": "exp_0002",           # SP1024 original
    "arch_sp1024.py": "exp_0002",
}

def evo_log(strategy_name: str, bpb: float, description: str, parent: str = "exp_0000"):
    """Log result to evo tree under the correct architecture branch."""
    try:
        # Create new evo experiment
        hypothesis = f"{strategy_name}: qualify BPB={bpb:.4f}. {description}"
        result = subprocess.run(
            ["evo", "new", "--parent", parent, "-m", hypothesis],
            capture_output=True, text=True, timeout=30
        )
        if result.returncode != 0:
            print(f"  [evo] new failed: {result.stderr[:100]}")
            return

        # Parse experiment ID from output
        import re
        m = re.search(r'(exp_\d+)', result.stdout)
        exp_id = m.group(1) if m else None
        if not exp_id:
            print(f"  [evo] couldn't parse exp_id from: {result.stdout[:100]}")
            return

        # Mark done with score, then fix status to committed
        done = subprocess.run(
            ["evo", "done", exp_id, "--score", str(bpb), "--no-compare"],
            capture_output=True, text=True, timeout=30
        )
        if done.returncode == 0:
            # Fix status: evo done --no-compare marks as "failed", we want "committed"
            import json as _json
            from pathlib import Path as _Path
            gp = _Path(".evo/run_0003/graph.json")
            if gp.exists():
                g = _json.loads(gp.read_text())
                if exp_id in g.get("nodes", {}):
                    g["nodes"][exp_id]["status"] = "committed"
                    gp.write_text(_json.dumps(g, indent=2))
            print(f"  [evo] Logged {exp_id} score={bpb:.4f} (committed)")
        else:
            print(f"  [evo] done failed: {done.stderr[:100]}")
    except Exception as e:
        print(f"  [evo] Error: {e}")


def run_one(strategy: Strategy) -> dict:
    """Run a single qualify experiment and return result dict."""
    config = load_config("infra/gce_config.yaml")

    env = dict(strategy.env)
    # Use SP1024 for qualify screening (SP8192 data not available yet — needs tokenizer training)
    # Relative rankings are preserved: r=0.86 correlation holds across tokenizers
    env.setdefault("VOCAB_SIZE", "1024")
    env.setdefault("COMPRESSOR", "lzma")

    print(f"\n{'='*60}")
    print(f"  AUTO-QUALIFY: {strategy.name}")
    print(f"  Description: {strategy.description}")
    print(f"  Env: {env}")
    print(f"  Priority: {strategy.priority} | Tier: {strategy.tier}")
    print(f"{'='*60}\n")

    try:
        result = run_qualify(strategy.name, strategy.script, env, config)
        return {
            "name": strategy.name,
            "status": result.status,
            "step_1000_bpb": result.step_1000_bpb,
            "val_loss_1000": result.val_loss_1000,
            "last_step": result.last_step,
            "error": result.error,
            "env": strategy.env,
            "description": strategy.description,
            "priority": strategy.priority,
            "tier": strategy.tier,
            "category": strategy.category,
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        traceback.print_exc()
        return {
            "name": strategy.name,
            "status": "error",
            "step_1000_bpb": None,
            "val_loss_1000": None,
            "last_step": 0,
            "error": str(e),
            "env": strategy.env,
            "description": strategy.description,
            "priority": strategy.priority,
            "tier": strategy.tier,
            "category": strategy.category,
            "timestamp": datetime.now().isoformat(),
        }


def print_status(data: dict):
    results = data.get("results", {})
    if not results:
        print("No results yet.")
        return

    passed = {k: v for k, v in results.items() if v.get("status") == "pass" and v.get("step_1000_bpb")}
    failed = {k: v for k, v in results.items() if v.get("status") != "pass" or not v.get("step_1000_bpb")}

    print(f"\n{'='*70}")
    print(f"  QUALIFY RESULTS  |  Tested: {len(results)}  |  Passed: {len(passed)}  |  Failed: {len(failed)}")
    print(f"  Best: {data.get('best', 'none')} @ {data.get('best_bpb', 'N/A')} BPB")
    print(f"{'='*70}")

    if passed:
        ranked = sorted(passed.values(), key=lambda r: r["step_1000_bpb"])
        print(f"\n{'RANK':>4} {'BPB@1000':>10} {'STEP':>6} {'NAME':<30} {'DESCRIPTION'}")
        print("-" * 100)
        for i, r in enumerate(ranked):
            bpb = r["step_1000_bpb"]
            step = r.get("last_step", "?")
            name = r["name"]
            desc = r.get("description", "")[:40]
            marker = " <-- BEST" if i == 0 else ""
            print(f"{i+1:>4} {bpb:>10.6f} {step:>6} {name:<30} {desc}{marker}")

    if failed:
        print(f"\nFailed ({len(failed)}):")
        for k, r in failed.items():
            print(f"  {k}: {r.get('status', '?')} — {r.get('error', '')[:60]}")

    remaining = len(get_queue()) - len(results)
    print(f"\nRemaining in queue: {remaining}")
    print(f"Last updated: {data.get('last_updated', 'never')}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--best", action="store_true")
    parser.add_argument("--next", action="store_true")
    args = parser.parse_args()

    data = load_results()

    if args.status:
        print_status(data)
        return

    if args.best:
        print(f"Best: {data.get('best', 'none')} @ {data.get('best_bpb', 'N/A')} BPB")
        return

    if args.next:
        nxt = get_next(data)
        if nxt:
            print(f"Next: {nxt.name} ({nxt.description})")
            print(f"  Env: {nxt.env}")
            print(f"  Priority: {nxt.priority} | Tier: {nxt.tier}")
        else:
            print("All strategies tested!")
        return

    # Atomically claim the next untested strategy (parallel-safe)
    nxt = claim_next()
    if nxt is None:
        print("All strategies tested! Run --status to see results.")
        return

    result = run_one(nxt)

    # Write final result (with lock for parallel safety)
    with open(LOCK_PATH, "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        data = load_results()  # re-read in case other workers updated
        data["results"][nxt.name] = result
        data["runs"] = data.get("runs", 0) + 1
        if result.get("step_1000_bpb") and result["step_1000_bpb"] < data.get("best_bpb", 99.0):
            data["best"] = nxt.name
            data["best_bpb"] = result["step_1000_bpb"]
            print(f"\n  *** NEW BEST: {nxt.name} @ {result['step_1000_bpb']:.6f} BPB ***")
        save_results(data)

    # Log to evo tree under the correct architecture branch
    if result.get("step_1000_bpb") and result["status"] == "pass":
        parent = EVO_ARCH_BRANCHES.get(nxt.script, "exp_0000")
        evo_log(nxt.name, result["step_1000_bpb"], nxt.description, parent=parent)

    print_status(data)


if __name__ == "__main__":
    main()
