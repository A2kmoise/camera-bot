"""
eval/plots.py
Generates publication-quality diagnostic plots and summary reports:
1. State timeline plot with event markers
2. Confidence decay curve vs theoretical exp curve
3. Continuity vs False-Temporary trade-off across T_half
4. Latency stacked bar chart
5. Summary metrics table in CSV and Markdown
"""

import argparse
from pathlib import Path
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


STATE_NUM_MAP = {
    "Unknown": 0,
    "Temporary": 1,
    "Confirmed": 2,
}

STATE_COLORS = {
    "Unknown": "#7f8c8d",     # Grey
    "Temporary": "#f39c12",   # Amber / Orange
    "Confirmed": "#27ae60",   # Green
}


def plot_state_timeline(
    log_df: pd.DataFrame,
    output_path: str = "reports/timeline.png",
    title: str = "State Timeline & Confidence",
):
    """
    Plot state transitions over time with event markers and confidence overlay.
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig, (ax_state, ax_conf) = plt.subplots(2, 1, figsize=(12, 6), sharex=True, gridspec_kw={"height_ratios": [1, 1.2]})

    t = log_df["timestamp"].values
    states = log_df["state"].values
    confs = log_df["confidence"].values
    state_nums = [STATE_NUM_MAP.get(s, 0) for s in states]

    # Subplot 1: Discrete state steps
    ax_state.step(t, state_nums, where="post", color="#2980b9", linewidth=2.0)
    ax_state.set_yticks([0, 1, 2])
    ax_state.set_yticklabels(["Unknown", "Temporary", "Confirmed"], fontweight="bold")
    ax_state.set_ylabel("Identity State", fontweight="bold")
    ax_state.set_ylim(-0.3, 2.3)
    ax_state.grid(True, linestyle="--", alpha=0.5)

    # Color background zones by state
    for i in range(len(t) - 1):
        s = states[i]
        c = STATE_COLORS.get(s, "#7f8c8d")
        ax_state.axvspan(t[i], t[i + 1], color=c, alpha=0.18)

    # Subplot 2: Confidence curve
    ax_conf.plot(t, confs, color="#d35400", linewidth=2.0, label="Observed Confidence")
    ax_conf.set_xlabel("Time (seconds)", fontweight="bold")
    ax_conf.set_ylabel("Confidence", fontweight="bold")
    ax_conf.set_ylim(0.0, 1.05)
    ax_conf.grid(True, linestyle="--", alpha=0.5)

    # Floor line
    ax_conf.axhline(0.30, color="#c0392b", linestyle=":", linewidth=1.5, label="Floor (0.30)")
    ax_conf.legend(loc="upper right")

    # Event markers
    if "event_marker" in log_df.columns:
        markers = log_df[log_df["event_marker"].notna() & (log_df["event_marker"] != "") & (log_df["event_marker"] != "None")]
        for _, m_row in markers.iterrows():
            m_t = m_row["timestamp"]
            m_name = m_row["event_marker"]
            for ax in (ax_state, ax_conf):
                ax.axvline(m_t, color="#8e44ad", linestyle="--", alpha=0.8, linewidth=1.2)
            ax_state.text(m_t, 2.15, f" {m_name}", rotation=90, verticalalignment="top", fontsize=8, color="#6c3483", fontweight="bold")

    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"[Plot] Saved state timeline to {output_path}")


def plot_confidence_decay(
    log_df: pd.DataFrame,
    t_half: float = 2.0,
    floor: float = 0.30,
    output_path: str = "reports/decay_curve.png",
):
    """
    Plot observed confidence decay intervals vs the theoretical exponential decay curve.
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(9, 5))

    lam = np.log(2.0) / max(t_half, 0.001)

    # Theoretical curve
    t_theory = np.linspace(0.0, t_half * 3.5, 200)
    conf_theory = np.exp(-lam * t_theory)
    plt.plot(t_theory, conf_theory, "k--", linewidth=2.2, label=f"Theoretical exp(-ln2*t/{t_half:.1f}s)")

    # Find longest continuous Temporary decay run in log
    temp_intervals = []
    current_run = []
    for _, row in log_df.iterrows():
        if row["state"] == "Temporary":
            current_run.append(row)
        else:
            if len(current_run) > 3:
                temp_intervals.append(current_run)
            current_run = []
    if current_run:
        temp_intervals.append(current_run)

    if temp_intervals:
        best_run = max(temp_intervals, key=len)
        run_df = pd.DataFrame(best_run)
        t_start = run_df["timestamp"].iloc[0]
        c_start = max(run_df["confidence"].iloc[0], 0.01)
        dt = run_df["timestamp"].values - t_start
        norm_conf = run_df["confidence"].values / c_start

        plt.plot(dt, norm_conf, "o-", color="#e67e22", markersize=4, label="Observed Logged Decay", alpha=0.85)

    # Half-life marker
    plt.axvline(t_half, color="#2980b9", linestyle=":", label=f"T_half = {t_half:.1f}s (Conf = 0.50)")
    plt.axhline(0.50, color="#2980b9", linestyle=":")
    plt.axhline(floor, color="#c0392b", linestyle="--", label=f"Confidence Floor = {floor:.2f}")

    plt.title(f"Confidence Decay vs Theoretical Half-Life (T_half = {t_half:.1f}s)", fontsize=13, fontweight="bold")
    plt.xlabel("Elapsed Time Without Face dt (seconds)", fontweight="bold")
    plt.ylabel("Normalized Confidence", fontweight="bold")
    plt.ylim(0.0, 1.05)
    plt.xlim(0.0, max(t_half * 3.0, 4.0))
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"[Plot] Saved confidence decay plot to {output_path}")


def plot_continuity_vs_false_temporary(
    sweep_results: List[Dict[str, any]],
    output_path: str = "reports/tradeoff_curve.png",
):
    """
    Plot Continuity Rate vs False-Temporary Rate across different T_half parameters.
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5))

    df = pd.DataFrame(sweep_results)
    if "t_half" not in df.columns or "continuity_rate" not in df.columns:
        print("[Plot] Sweep results missing required columns.")
        return

    df = df.sort_values(by="t_half")
    t_halfs = df["t_half"].values
    continuity = df["continuity_rate"].values
    false_temp = df.get("false_temporary_rate", [0.0] * len(df)).values

    fig, ax1 = plt.subplots(figsize=(8, 5))

    color = "#27ae60"
    ax1.set_xlabel("T_half Decay Time (seconds)", fontweight="bold")
    ax1.set_ylabel("Identity Continuity Rate", color=color, fontweight="bold")
    ax1.plot(t_halfs, continuity, "o-", color=color, linewidth=2.5, markersize=7, label="Continuity Rate")
    ax1.tick_params(axis="y", labelcolor=color)
    ax1.set_ylim(0.0, 1.05)
    ax1.grid(True, linestyle="--", alpha=0.5)

    ax2 = ax1.twinx()
    color2 = "#c0392b"
    ax2.set_ylabel("False-Temporary Rate", color=color2, fontweight="bold")
    ax2.plot(t_halfs, false_temp, "s--", color=color2, linewidth=2.5, markersize=7, label="False-Temporary Rate")
    ax2.tick_params(axis="y", labelcolor=color2)
    ax2.set_ylim(-0.02, max(max(false_temp) * 1.5, 0.25))

    plt.title("Continuity vs False-Temporary Trade-Off across T_half", fontsize=13, fontweight="bold")
    fig.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"[Plot] Saved tradeoff plot to {output_path}")


def plot_latency_breakdown(
    stage_latencies: Dict[str, float],
    output_path: str = "reports/latency_breakdown.png",
):
    """
    Stacked bar chart of per-stage execution latency against 15 FPS / 30 FPS budget lines.
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(9, 4.5))

    stages = list(stage_latencies.keys())
    times = [stage_latencies[s] for s in stages]
    total_time = sum(times)

    colors = ["#3498db", "#9b59b6", "#e67e22", "#1abc9c", "#f1c40f", "#2ecc71", "#e74c3c"]
    colors = colors[:len(stages)]

    bottom = 0.0
    for s, t, c in zip(stages, times, colors):
        plt.barh(["Pipeline"], [t], left=[bottom], color=c, label=f"{s} ({t:.1f} ms)", edgecolor="white")
        bottom += t

    # Budget reference lines
    plt.axvline(33.3, color="#27ae60", linestyle="--", linewidth=1.8, label="30 FPS Budget (33.3 ms)")
    plt.axvline(66.7, color="#c0392b", linestyle="--", linewidth=1.8, label="15 FPS Floor (66.7 ms)")

    plt.xlabel("Latency per Frame (milliseconds)", fontweight="bold")
    plt.title(f"Per-Stage Execution Latency (Total: {total_time:.1f} ms -> {1000.0/max(total_time,1):.1f} FPS)", fontsize=13, fontweight="bold")
    plt.xlim(0.0, max(80.0, total_time * 1.25))
    plt.grid(True, linestyle="--", alpha=0.4, axis="x")
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"[Plot] Saved latency breakdown to {output_path}")


def _df_to_markdown(df: pd.DataFrame) -> str:
    """Format DataFrame as a clean Markdown table without external dependencies."""
    headers = [str(c) for c in df.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join([":---:" if c in ("t_half", "floor", "continuity_rate", "false_temporary_rate", "fps_mean") else ":---" for c in headers]) + " |",
    ]
    for _, row in df.iterrows():
        row_cells = []
        for c in df.columns:
            val = row[c]
            if isinstance(val, float):
                row_cells.append(f"{val:.3f}")
            else:
                row_cells.append(str(val))
        lines.append("| " + " | ".join(row_cells) + " |")
    return "\n".join(lines)


def save_summary_table(
    sweep_results: List[Dict[str, any]],
    output_csv: str = "reports/summary_metrics.csv",
    output_md: str = "reports/summary_metrics.md",
):
    """
    Save sweep metrics table to CSV and Markdown.
    """
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(sweep_results)
    df.to_csv(output_csv, index=False)

    md_content = _df_to_markdown(df)
    with open(output_md, "w", encoding="utf-8") as f:
        f.write("# Person Tracker Evaluation & Parameter Sweep Summary\n\n")
        f.write(md_content)
        f.write("\n")
    print(f"[Summary] Saved metrics summary to {output_csv} and {output_md}")


def main():
    parser = argparse.ArgumentParser(description="Generate diagnostic plots from pipeline logs")
    parser.add_argument("--log", type=str, required=True, help="Path to pipeline log CSV")
    parser.add_argument("--t-half", type=float, default=2.0, help="T_half decay time in seconds")
    parser.add_argument("--outdir", type=str, default="reports", help="Output directory for plots")
    args = parser.parse_args()

    df = pd.read_csv(args.log)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    plot_state_timeline(df, str(outdir / "timeline.png"))
    plot_confidence_decay(df, t_half=args.t_half, output_path=str(outdir / "decay_curve.png"))

    # Demo latency breakdown
    sample_latencies = {
        "person_track": 14.5,
        "face_detect": 8.2,
        "embedding": 4.1,
        "mapping": 0.8,
        "state_update": 0.3,
        "draw": 2.5,
    }
    plot_latency_breakdown(sample_latencies, str(outdir / "latency_breakdown.png"))


if __name__ == "__main__":
    main()
