#!/usr/bin/env python3
"""
Gradio Dashboard for Parameter Golf experiments.

Real-time monitoring of experiment status, BPB comparisons, timeline,
hyperparameter heatmaps, and autoresearch progress.

Usage:
    python infra/dashboard.py                # Launch on localhost:7860
    python infra/dashboard.py --port 7861    # Custom port
    python infra/dashboard.py --share        # Public Gradio link

Requires: pip install gradio pandas plotly
"""
from __future__ import annotations

import argparse
import fcntl
import json
import re
from datetime import datetime
from pathlib import Path

import gradio as gr
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

BATCH_STATE_PATH = Path("infra/batch_state.json")
AUTORESEARCH_STATE_PATH = Path("infra/autoresearch_state.json")
H100_SPOT_COST_PER_HOUR = 24.48


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path) as f:
        fcntl.flock(f, fcntl.LOCK_SH)
        try:
            return json.load(f)
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def load_batch_state() -> dict:
    return _load_json(BATCH_STATE_PATH)


def load_autoresearch_state() -> dict:
    return _load_json(AUTORESEARCH_STATE_PATH)


def _extract_tier(name: str) -> str:
    m = re.match(r"^T(\d)-", name)
    if m:
        return f"Tier {m.group(1)}"
    if name.startswith("AUTO-"):
        return "Auto"
    if name.startswith("sweep-"):
        return "Sweep"
    return "Other"


def experiments_to_dataframe(state: dict) -> pd.DataFrame:
    """Convert batch_state experiments to a sorted DataFrame."""
    experiments = state.get("experiments", {})
    if not experiments:
        return pd.DataFrame(columns=[
            "Name", "Status", "Tier", "Final BPB", "Step1K BPB",
            "Time (s)", "Zone", "Model", "Preemptions", "Error",
        ])

    rows = []
    for name, data in experiments.items():
        rows.append({
            "Name": name,
            "Status": data.get("status", "unknown"),
            "Tier": _extract_tier(name),
            "Final BPB": data.get("final_bpb", None),
            "Step1K BPB": data.get("step_1000_bpb", None),
            "Time (s)": data.get("wallclock_seconds", None),
            "Zone": data.get("instance_zone", ""),
            "Model": data.get("provisioning_model", ""),
            "Preemptions": data.get("preemption_count", 0),
            "Error": (data.get("error", "") or "")[:80],
            "Start": data.get("start_time", ""),
            "End": data.get("end_time", ""),
        })

    df = pd.DataFrame(rows)
    # Sort: succeeded with best BPB first, then others
    df["_sort"] = df["Final BPB"].fillna(999)
    df = df.sort_values("_sort").drop(columns=["_sort"]).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Overview metrics
# ---------------------------------------------------------------------------

def compute_metrics(state: dict, ar_state: dict) -> dict:
    experiments = state.get("experiments", {})
    total = len(experiments)
    succeeded = sum(1 for d in experiments.values() if d.get("status") == "succeeded")
    failed = sum(1 for d in experiments.values() if d.get("status") == "failed")
    killed = sum(1 for d in experiments.values() if d.get("status") == "early_killed")
    running = sum(1 for d in experiments.values() if d.get("status") in ("running", "provisioning"))

    gpu_hours = ar_state.get("total_gpu_hours", 0)
    if not gpu_hours:
        gpu_hours = sum(
            d.get("wallclock_seconds", 0) / 3600
            for d in experiments.values()
        )

    return {
        "total": total,
        "succeeded": succeeded,
        "failed": failed,
        "killed": killed,
        "running": running,
        "success_rate": succeeded / total if total > 0 else 0,
        "best_bpb": state.get("best_final_bpb"),
        "best_name": min(
            ((n, d.get("final_bpb", 99)) for n, d in experiments.items() if d.get("final_bpb")),
            key=lambda x: x[1], default=("", None)
        )[0],
        "gpu_hours": gpu_hours,
        "cost_usd": gpu_hours * H100_SPOT_COST_PER_HOUR,
        "cycle": ar_state.get("cycle_count", 0),
        "phase": ar_state.get("phase", "N/A"),
    }


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------

def create_bpb_bar_chart(df: pd.DataFrame) -> go.Figure:
    succeeded = df[df["Status"] == "succeeded"].copy()
    if succeeded.empty:
        fig = go.Figure()
        fig.add_annotation(text="No completed experiments yet", showarrow=False, font_size=20)
        return fig

    succeeded = succeeded.nsmallest(30, "Final BPB")
    color_map = {"Tier 1": "#2196F3", "Tier 2": "#FF9800", "Tier 3": "#4CAF50", "Auto": "#9C27B0", "Sweep": "#795548", "Other": "#607D8B"}
    colors = [color_map.get(t, "#607D8B") for t in succeeded["Tier"]]

    fig = go.Figure(go.Bar(
        x=succeeded["Name"].str[:30],
        y=succeeded["Final BPB"],
        marker_color=colors,
        text=succeeded["Final BPB"].round(4),
        textposition="outside",
    ))
    best = succeeded["Final BPB"].min()
    fig.add_hline(y=best, line_dash="dash", line_color="red", annotation_text=f"Best: {best:.4f}")
    fig.update_layout(
        title="Final BPB by Experiment (Top 30)",
        xaxis_title="Experiment",
        yaxis_title="Bits Per Byte",
        xaxis_tickangle=-45,
        height=500,
        margin=dict(b=150),
    )
    return fig


def create_bpb_progression(df: pd.DataFrame) -> go.Figure:
    succeeded = df[(df["Status"] == "succeeded") & df["End"].notna() & (df["End"] != "")].copy()
    if succeeded.empty:
        fig = go.Figure()
        fig.add_annotation(text="No completed experiments yet", showarrow=False, font_size=20)
        return fig

    succeeded["end_dt"] = pd.to_datetime(succeeded["End"], errors="coerce")
    succeeded = succeeded.dropna(subset=["end_dt"]).sort_values("end_dt")
    succeeded["cummin_bpb"] = succeeded["Final BPB"].cummin()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=succeeded["end_dt"], y=succeeded["Final BPB"],
        mode="markers", name="Individual", marker=dict(size=8, opacity=0.5),
    ))
    fig.add_trace(go.Scatter(
        x=succeeded["end_dt"], y=succeeded["cummin_bpb"],
        mode="lines+markers", name="Best So Far", line=dict(color="red", width=3),
    ))
    fig.update_layout(
        title="BPB Progression Over Time",
        xaxis_title="Time",
        yaxis_title="Bits Per Byte",
        height=400,
    )
    return fig


def create_timeline(df: pd.DataFrame) -> go.Figure:
    has_times = df[df["Start"].notna() & (df["Start"] != "") & df["End"].notna() & (df["End"] != "")].copy()
    if has_times.empty:
        fig = go.Figure()
        fig.add_annotation(text="No experiment timeline data yet", showarrow=False, font_size=20)
        return fig

    has_times["start_dt"] = pd.to_datetime(has_times["Start"], errors="coerce")
    has_times["end_dt"] = pd.to_datetime(has_times["End"], errors="coerce")
    has_times = has_times.dropna(subset=["start_dt", "end_dt"]).tail(30)

    color_map = {"succeeded": "#4CAF50", "failed": "#F44336", "early_killed": "#FFC107",
                 "preempted": "#FF5722", "running": "#2196F3"}

    fig = go.Figure()
    for _, row in has_times.iterrows():
        color = color_map.get(row["Status"], "#9E9E9E")
        fig.add_trace(go.Bar(
            x=[row["end_dt"] - row["start_dt"]],
            y=[row["Name"][:25]],
            base=[row["start_dt"]],
            orientation="h",
            marker_color=color,
            showlegend=False,
            hovertext=f"{row['Name']}: {row['Status']} ({row['Time (s)']:.0f}s)" if row["Time (s)"] else "",
        ))

    fig.update_layout(
        title="Experiment Timeline (Last 30)",
        xaxis_title="Time",
        height=max(300, len(has_times) * 25),
        barmode="overlay",
    )
    return fig


def create_tier_boxplot(df: pd.DataFrame) -> go.Figure:
    succeeded = df[df["Status"] == "succeeded"].copy()
    if succeeded.empty:
        fig = go.Figure()
        fig.add_annotation(text="No completed experiments yet", showarrow=False, font_size=20)
        return fig

    fig = px.box(succeeded, x="Tier", y="Final BPB", color="Tier", points="all",
                 title="BPB Distribution by Tier")
    fig.update_layout(height=400)
    return fig


# ---------------------------------------------------------------------------
# Autoresearch info
# ---------------------------------------------------------------------------

def autoresearch_summary(ar_state: dict) -> str:
    if not ar_state:
        return "Autoresearch not started yet. Run `python infra/autoresearch.py` to begin."

    lines = [
        f"**Cycle:** {ar_state.get('cycle_count', 0)}",
        f"**Phase:** {ar_state.get('phase', 'N/A')}",
        f"**Started:** {ar_state.get('started_at', '?')}",
        f"**Last Cycle:** {ar_state.get('last_cycle_at', '?')}",
        f"**GPU Hours:** {ar_state.get('total_gpu_hours', 0):.1f}h "
        f"(~${ar_state.get('total_gpu_hours', 0) * H100_SPOT_COST_PER_HOUR:.0f})",
        "",
    ]

    # Dimension analysis
    dims = ar_state.get("dimension_analysis", {})
    if dims:
        lines.append("### Dimension Analysis")
        lines.append("| Dimension | Best Value | Sensitivity | Shape | Samples |")
        lines.append("|-----------|-----------|-------------|-------|---------|")
        for name, info in sorted(dims.items(), key=lambda x: -x[1].get("sensitivity", 0)):
            lines.append(
                f"| {name} | {info.get('best_value', '?')} | "
                f"{info.get('sensitivity', 0):.4f} | {info.get('shape', '?')} | "
                f"{info.get('n_samples', 0)} |"
            )
        lines.append("")

    # Generated hypotheses
    gen = ar_state.get("generated_strategies", [])
    if gen:
        pending = sum(1 for g in gen if g.get("status") == "pending")
        lines.append(f"### Generated Hypotheses ({len(gen)} total, {pending} pending)")
        for g in gen[-10:]:  # Last 10
            status = g.get("status", "?")
            icon = {"pending": "...", "succeeded": "OK", "failed": "FAIL", "early_killed": "KILL"}.get(status, "?")
            lines.append(f"- [{icon}] **{g['name']}** ({g.get('origin', '?')}): {g.get('hypothesis', '')}")
        lines.append("")

    # Early-killed patterns
    kills = ar_state.get("early_killed_patterns", [])
    if kills:
        lines.append(f"### Early-Kill Patterns ({len(kills)})")
        for k in kills[-5:]:
            lines.append(f"- `{k.get('env_pattern', {})}` — {k.get('reason', '')}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Dashboard layout
# ---------------------------------------------------------------------------

def build_dashboard() -> gr.Blocks:
    with gr.Blocks(title="Parameter Golf Dashboard") as app:
        gr.Markdown("# Parameter Golf Research Dashboard")

        # Overview metrics row
        with gr.Row():
            m_total = gr.Number(label="Total Experiments", value=0, interactive=False)
            m_succeeded = gr.Number(label="Succeeded", value=0, interactive=False)
            m_failed = gr.Number(label="Failed", value=0, interactive=False)
            m_killed = gr.Number(label="Early-Killed", value=0, interactive=False)
            m_best = gr.Number(label="Best BPB", value=0, precision=4, interactive=False)
            m_hours = gr.Number(label="GPU Hours", value=0, precision=1, interactive=False)
            m_cost = gr.Number(label="Est. Cost ($)", value=0, precision=0, interactive=False)

        with gr.Tabs():
            # Tab 1: Experiments table
            with gr.TabItem("Experiments"):
                with gr.Row():
                    filter_status = gr.Dropdown(
                        ["All", "succeeded", "failed", "early_killed", "running", "pending"],
                        value="All", label="Filter Status",
                    )
                    filter_tier = gr.Dropdown(
                        ["All", "Tier 1", "Tier 2", "Tier 3", "Auto", "Sweep"],
                        value="All", label="Filter Tier",
                    )
                exp_table = gr.Dataframe(
                    headers=["Name", "Status", "Tier", "Final BPB", "Step1K BPB", "Time (s)", "Zone", "Model", "Error"],
                    label="Experiments",
                    interactive=False,
                )

            # Tab 2: BPB Comparison
            with gr.TabItem("BPB Comparison"):
                bpb_bar = gr.Plot(label="BPB by Experiment")
                bpb_prog = gr.Plot(label="BPB Progression")

            # Tab 3: Timeline
            with gr.TabItem("Timeline"):
                timeline_plot = gr.Plot(label="Experiment Timeline")

            # Tab 4: Tier Analysis
            with gr.TabItem("Tier Analysis"):
                tier_box = gr.Plot(label="BPB by Tier")

            # Tab 5: Autoresearch
            with gr.TabItem("Autoresearch"):
                ar_md = gr.Markdown("Autoresearch not started yet.")

        last_updated = gr.Markdown("*Dashboard loading...*")

        # Refresh logic
        def refresh(status_filter, tier_filter):
            state = load_batch_state()
            ar_state = load_autoresearch_state()
            metrics = compute_metrics(state, ar_state)
            df = experiments_to_dataframe(state)

            # Apply filters
            filtered = df.copy()
            if status_filter != "All":
                filtered = filtered[filtered["Status"] == status_filter]
            if tier_filter != "All":
                filtered = filtered[filtered["Tier"] == tier_filter]

            display_cols = ["Name", "Status", "Tier", "Final BPB", "Step1K BPB", "Time (s)", "Zone", "Model", "Error"]
            table_data = filtered[display_cols] if not filtered.empty else pd.DataFrame(columns=display_cols)

            return (
                metrics["total"],
                metrics["succeeded"],
                metrics["failed"],
                metrics["killed"],
                metrics["best_bpb"] or 0,
                metrics["gpu_hours"],
                metrics["cost_usd"],
                table_data,
                create_bpb_bar_chart(df),
                create_bpb_progression(df),
                create_timeline(df),
                create_tier_boxplot(df),
                autoresearch_summary(ar_state),
                f"*Last updated: {datetime.now().strftime('%H:%M:%S')} | "
                f"Autoresearch cycle: {metrics['cycle']} ({metrics['phase']})*",
            )

        # Initial load
        outputs = [
            m_total, m_succeeded, m_failed, m_killed, m_best, m_hours, m_cost,
            exp_table, bpb_bar, bpb_prog, timeline_plot, tier_box,
            ar_md, last_updated,
        ]

        app.load(
            fn=refresh,
            inputs=[filter_status, filter_tier],
            outputs=outputs,
        )

        # Auto-refresh timer
        timer = gr.Timer(30)
        timer.tick(
            fn=refresh,
            inputs=[filter_status, filter_tier],
            outputs=outputs,
        )

        # Filter change triggers refresh
        filter_status.change(fn=refresh, inputs=[filter_status, filter_tier], outputs=outputs)
        filter_tier.change(fn=refresh, inputs=[filter_status, filter_tier], outputs=outputs)

    return app


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Parameter Golf Dashboard")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    dashboard = build_dashboard()
    dashboard.launch(server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
