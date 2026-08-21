#!/usr/bin/env python3
"""
Parameter Golf Monitoring Dashboard.

Real-time monitoring of qualify screening, evo tree, and overnight loop.

Usage:
    python infra/dashboard.py                # Launch on localhost:7860
    python infra/dashboard.py --port 7861    # Custom port
    python infra/dashboard.py --share        # Public Gradio link

Requires: pip install gradio pandas plotly
"""
from __future__ import annotations

import json
import subprocess
from datetime import datetime
from pathlib import Path

import gradio as gr
import pandas as pd
import plotly.graph_objects as go

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

QUALIFY_PATH = Path("infra/qualify_results.json")
EVO_GRAPH_PATH = Path(".evo/run_0001/graph.json")
OVERNIGHT_LOG = Path("infra/overnight.log")
OVERNIGHT_PID = Path("infra/overnight.pid")


def load_qualify() -> dict:
    if QUALIFY_PATH.exists():
        return json.loads(QUALIFY_PATH.read_text())
    return {"results": {}}


def load_evo_graph() -> dict:
    if EVO_GRAPH_PATH.exists():
        return json.loads(EVO_GRAPH_PATH.read_text())
    return {"nodes": {}}


def get_loop_status() -> str:
    """Check if overnight loop is running."""
    if not OVERNIGHT_PID.exists():
        return "NOT RUNNING"
    pid = OVERNIGHT_PID.read_text().strip()
    try:
        result = subprocess.run(["kill", "-0", pid], capture_output=True)
        if result.returncode == 0:
            return f"RUNNING (PID {pid})"
    except Exception:
        pass
    return "STOPPED"


def get_instances() -> str:
    """Check running GCE instances."""
    try:
        r = subprocess.run(
            ["gcloud", "compute", "instances", "list",
             "--project=bryan-usage-0", "--filter=name~pgolf",
             "--format=table(name,zone,machineType,status)"],
            capture_output=True, text=True, timeout=15)
        return r.stdout.strip() or "No pgolf instances"
    except Exception as e:
        return f"Error: {e}"


# ---------------------------------------------------------------------------
# Qualify results tab
# ---------------------------------------------------------------------------

def qualify_leaderboard() -> tuple[pd.DataFrame, go.Figure, str]:
    data = load_qualify()
    results = data.get("results", {})

    if not results:
        empty_df = pd.DataFrame(columns=["Rank", "Name", "BPB@1000", "Val Loss", "Step", "Priority", "Description"])
        fig = go.Figure()
        fig.add_annotation(text="No results yet", showarrow=False, font_size=20)
        return empty_df, fig, "No results yet"

    # Build leaderboard
    rows = []
    for name, r in results.items():
        rows.append({
            "Name": name,
            "BPB@1000": r.get("step_1000_bpb"),
            "Val Loss": r.get("val_loss_1000"),
            "Step": r.get("last_step", 0),
            "Status": r.get("status", "?"),
            "Priority": r.get("priority", "?"),
            "Tier": r.get("tier", "?"),
            "Description": r.get("description", "")[:50],
            "Timestamp": r.get("timestamp", ""),
            "Env": str(r.get("env", {}))[:60],
        })

    df = pd.DataFrame(rows)
    passed = df[df["Status"] == "pass"].copy()
    failed = df[df["Status"] != "pass"].copy()

    if not passed.empty and passed["BPB@1000"].notna().any():
        passed = passed.sort_values("BPB@1000").reset_index(drop=True)
        passed.insert(0, "Rank", range(1, len(passed) + 1))
    else:
        passed = pd.DataFrame()

    # Bar chart
    fig = go.Figure()
    if not passed.empty and "BPB@1000" in passed.columns:
        chart_data = passed.nsmallest(30, "BPB@1000")
        colors = ["#4CAF50" if r["Priority"] == "P0" else "#2196F3" if r["Priority"] == "P1" else "#FF9800"
                  for _, r in chart_data.iterrows()]
        fig.add_trace(go.Bar(
            x=chart_data["Name"],
            y=chart_data["BPB@1000"],
            marker_color=colors,
            text=chart_data["BPB@1000"].round(4),
            textposition="outside",
        ))
        best = chart_data["BPB@1000"].min()
        fig.add_hline(y=best, line_dash="dash", line_color="red",
                      annotation_text=f"Best: {best:.4f}")
        fig.update_layout(
            title="Qualify BPB@Step1000 (Lower = Better)",
            xaxis_title="Strategy", yaxis_title="BPB",
            xaxis_tickangle=-45, height=500, margin=dict(b=150),
        )

    # Summary
    total = len(results)
    n_pass = len([r for r in results.values() if r.get("status") == "pass"])
    n_fail = total - n_pass
    best_name = data.get("best", "none")
    best_bpb = data.get("best_bpb", "N/A")
    summary = (
        f"**Tested:** {total} | **Passed:** {n_pass} | **Failed:** {n_fail}\n\n"
        f"**Best:** {best_name} @ **{best_bpb}** BPB\n\n"
        f"**Competition SOTA:** 1.0810 BPB\n\n"
        f"**Queue remaining:** ~{50 - total}\n\n"
        f"**Last updated:** {data.get('last_updated', 'never')}"
    )

    display_df = passed if not passed.empty else df
    return display_df, fig, summary


# ---------------------------------------------------------------------------
# Evo tree tab
# ---------------------------------------------------------------------------

def evo_tree_view() -> tuple[str, pd.DataFrame]:
    try:
        r = subprocess.run(["evo", "tree"], capture_output=True, text=True, timeout=10)
        tree_text = r.stdout if r.returncode == 0 else "evo tree command failed"
    except Exception:
        tree_text = "evo not available"

    # Also load graph.json for a table view
    graph = load_evo_graph()
    nodes = graph.get("nodes", {})
    if not nodes:
        return tree_text, pd.DataFrame(columns=["ID", "Score", "Status", "Hypothesis"])

    rows = []
    for nid, node in nodes.items():
        rows.append({
            "ID": nid,
            "Parent": node.get("parent", ""),
            "Score": node.get("score"),
            "Status": node.get("status", ""),
            "Hypothesis": (node.get("hypothesis", "") or "")[:80],
            "Created": node.get("created_at", "")[:19],
        })

    df = pd.DataFrame(rows)
    if "Score" in df.columns:
        df = df.sort_values("Score", na_position="last").reset_index(drop=True)
    return tree_text, df


# ---------------------------------------------------------------------------
# Live status tab
# ---------------------------------------------------------------------------

def live_status() -> tuple[str, str, str]:
    loop = get_loop_status()
    instances = get_instances()

    # Last 30 lines of overnight log
    log_tail = ""
    if OVERNIGHT_LOG.exists():
        lines = OVERNIGHT_LOG.read_text().splitlines()
        log_tail = "\n".join(lines[-40:])

    return f"**Loop:** {loop}", instances, log_tail


# ---------------------------------------------------------------------------
# Build dashboard
# ---------------------------------------------------------------------------

def build_app() -> gr.Blocks:
    with gr.Blocks(title="Parameter Golf Dashboard") as app:
        gr.Markdown("# Parameter Golf — Experiment Dashboard")
        gr.Markdown("Auto-refreshes every 30 seconds. Competition deadline: April 30, 2026.")

        with gr.Tab("Qualify Leaderboard"):
            summary_md = gr.Markdown()
            qualify_chart = gr.Plot()
            qualify_table = gr.DataFrame()
            refresh_btn = gr.Button("Refresh", variant="primary")
            refresh_btn.click(qualify_leaderboard, outputs=[qualify_table, qualify_chart, summary_md])

        with gr.Tab("Evo Tree"):
            evo_text = gr.Textbox(label="Evo Tree", lines=25, interactive=False)
            evo_table = gr.DataFrame(label="Experiments")
            evo_btn = gr.Button("Refresh", variant="primary")
            evo_btn.click(evo_tree_view, outputs=[evo_text, evo_table])

        with gr.Tab("Live Status"):
            loop_md = gr.Markdown()
            instances_text = gr.Textbox(label="GCE Instances", lines=5, interactive=False)
            log_text = gr.Textbox(label="Overnight Log (last 40 lines)", lines=20, interactive=False)
            status_btn = gr.Button("Refresh", variant="primary")
            status_btn.click(live_status, outputs=[loop_md, instances_text, log_text])

        # Auto-refresh on load
        app.load(qualify_leaderboard, outputs=[qualify_table, qualify_chart, summary_md])
        app.load(evo_tree_view, outputs=[evo_text, evo_table])
        app.load(live_status, outputs=[loop_md, instances_text, log_text])

    return app


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    app = build_app()
    app.launch(server_name="0.0.0.0", server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
