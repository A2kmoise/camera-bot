"""
eval/metrics.py
Computes core quantitative metrics from pipeline logs and ground-truth label CSVs:
- Identity continuity rate
- Correct-identity rate
- ID switch count
- Recovery time (face reappears -> Confirmed)
- False-Temporary rate (Temporary on wrong person - most critical metric)
- Unknown-when-should-know rate (conservatism cost)
- Time-in-state breakdown
- Centring error statistics (MAE, RMS)
- FPS distribution (mean, p5) and per-stage latency
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import pandas as pd


def compute_metrics(
    log_csv_path: str,
    gt_csv_path: Optional[str] = None,
) -> Dict[str, any]:
    """
    Compute comprehensive tracking and identity evaluation metrics.
    """
    df_log = pd.read_csv(log_csv_path)

    # Clean and parse types
    df_log["timestamp"] = pd.to_numeric(df_log["timestamp"], errors="coerce").fillna(0.0)
    df_log["frame"] = pd.to_numeric(df_log["frame"], errors="coerce").fillna(0).astype(int)
    df_log["state"] = df_log["state"].astype(str).str.strip()
    df_log["name"] = df_log["name"].astype(str).str.strip()
    df_log["confidence"] = pd.to_numeric(df_log["confidence"], errors="coerce").fillna(0.0)
    df_log["centring_error"] = pd.to_numeric(df_log["centring_error"], errors="coerce").fillna(0.0)
    df_log["fps"] = pd.to_numeric(df_log["fps"], errors="coerce").fillna(0.0)

    # 1. Performance / Latency Metrics
    fps_vals = df_log["fps"].dropna().values
    fps_mean = float(np.mean(fps_vals)) if len(fps_vals) > 0 else 0.0
    fps_p5 = float(np.percentile(fps_vals, 5)) if len(fps_vals) > 0 else 0.0
    fps_median = float(np.median(fps_vals)) if len(fps_vals) > 0 else 0.0

    # 2. Time-in-State Breakdown
    total_log_rows = len(df_log)
    state_counts = df_log["state"].value_counts().to_dict()
    confirmed_ratio = state_counts.get("Confirmed", 0) / max(total_log_rows, 1)
    temporary_ratio = state_counts.get("Temporary", 0) / max(total_log_rows, 1)
    unknown_ratio = state_counts.get("Unknown", 0) / max(total_log_rows, 1)

    # 3. Centring Error Statistics
    err_vals = np.abs(df_log["centring_error"].dropna().values)
    centring_mae = float(np.mean(err_vals)) if len(err_vals) > 0 else 0.0
    centring_rms = float(np.sqrt(np.mean(err_vals ** 2))) if len(err_vals) > 0 else 0.0
    centring_max = float(np.max(err_vals)) if len(err_vals) > 0 else 0.0

    # Read run metadata if available for per-stage latency
    meta_path = Path(log_csv_path).with_name(f"{Path(log_csv_path).stem}_meta.json")
    meta_info = {}
    if meta_path.exists():
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta_info = json.load(f)
        except Exception:
            pass

    # 4. Ground-Truth Comparisons
    metrics = {
        "total_frames": total_log_rows,
        "fps_mean": fps_mean,
        "fps_p5": fps_p5,
        "fps_median": fps_median,
        "time_confirmed_pct": confirmed_ratio * 100.0,
        "time_temporary_pct": temporary_ratio * 100.0,
        "time_unknown_pct": unknown_ratio * 100.0,
        "centring_mae": centring_mae,
        "centring_rms": centring_rms,
        "centring_max": centring_max,
        "continuity_rate": 0.0,
        "correct_identity_rate": 0.0,
        "id_switch_count": 0,
        "false_temporary_rate": 0.0,
        "unknown_when_should_know_rate": 0.0,
        "mean_recovery_time_s": 0.0,
    }

    if gt_csv_path and Path(gt_csv_path).exists():
        df_gt = pd.read_csv(gt_csv_path)
        df_gt["frame"] = pd.to_numeric(df_gt["frame"], errors="coerce").fillna(0).astype(int)
        df_gt["gt_name"] = df_gt["gt_name"].astype(str).str.strip()
        df_gt["gt_face_visible"] = (
            df_gt["gt_face_visible"].astype(str).str.lower().isin(["1", "true", "yes"]).astype(int)
        )

        merged = pd.merge(df_log, df_gt, on="frame", how="inner", suffixes=("_pred", "_gt"))
        N = len(merged)
        if N > 0:
            # Identity Continuity Rate:
            # Fraction of frames where predicted identity is not Unknown and equals gt_name
            is_valid_id = (merged["state"] != "Unknown") & (merged["name"] == merged["gt_name"])
            continuity_rate = float(np.sum(is_valid_id) / N)

            # Correct Identity Rate:
            # Correct if name == gt_name when gt is known, or state == Unknown when gt is Unknown
            is_correct = (merged["name"] == merged["gt_name"]) | (
                (merged["gt_name"] == "Unknown") & (merged["state"] == "Unknown")
            )
            correct_id_rate = float(np.sum(is_correct) / N)

            # False-Temporary Rate (The Most Critical Metric!):
            # Fraction of frames in Temporary state where predicted name != gt_name
            temp_mask = merged["state"] == "Temporary"
            temp_count = np.sum(temp_mask)
            if temp_count > 0:
                false_temp = (merged["state"] == "Temporary") & (merged["name"] != merged["gt_name"])
                false_temp_rate = float(np.sum(false_temp) / temp_count)
            else:
                false_temp_rate = 0.0

            # Unknown-when-should-know rate (Conservatism cost):
            # Fraction of frames where gt_face_visible == 1 and gt_name != Unknown, but pred is Unknown
            should_know_mask = (merged["gt_face_visible"] == 1) & (merged["gt_name"] != "Unknown")
            should_know_count = np.sum(should_know_mask)
            if should_know_count > 0:
                unk_when_know = should_know_mask & (merged["state"] == "Unknown")
                unknown_when_should_know_rate = float(np.sum(unk_when_know) / should_know_count)
            else:
                unknown_when_should_know_rate = 0.0

            # ID switch count:
            # Transitions where name changes from Person A directly to Person B without Unknown
            id_switches = 0
            prev_name = None
            for _, row in merged.iterrows():
                curr_name = row["name"]
                if prev_name and curr_name not in (prev_name, "Unknown", "None") and prev_name not in ("Unknown", "None"):
                    id_switches += 1
                prev_name = curr_name

            # Recovery time:
            # Face reappears (gt_face_visible transitions 0 -> 1) until state becomes Confirmed
            recovery_times = []
            tracking_recovery = False
            reappear_t = 0.0

            for i in range(1, len(merged)):
                prev_row = merged.iloc[i - 1]
                curr_row = merged.iloc[i]

                # Face reappeared transition
                if prev_row["gt_face_visible"] == 0 and curr_row["gt_face_visible"] == 1:
                    tracking_recovery = True
                    reappear_t = curr_row["timestamp_pred"]

                if tracking_recovery and curr_row["state"] == "Confirmed":
                    rec_duration = max(0.0, curr_row["timestamp_pred"] - reappear_t)
                    recovery_times.append(rec_duration)
                    tracking_recovery = False

            mean_recovery_t = float(np.mean(recovery_times)) if recovery_times else 0.0

            metrics.update({
                "continuity_rate": continuity_rate,
                "correct_identity_rate": correct_id_rate,
                "false_temporary_rate": false_temp_rate,
                "unknown_when_should_know_rate": unknown_when_should_know_rate,
                "id_switch_count": id_switches,
                "mean_recovery_time_s": mean_recovery_t,
            })
    else:
        # If no explicit ground truth CSV, estimate continuity on active track
        # assuming the dominant identity is correct
        dominant_name = df_log[df_log["name"] != "Unknown"]["name"].mode()
        if not dominant_name.empty:
            dom = dominant_name[0]
            matched = (df_log["state"] != "Unknown") & (df_log["name"] == dom)
            metrics["continuity_rate"] = float(np.sum(matched) / max(len(df_log), 1))

    return metrics


def format_markdown_table(metrics_dict: Dict[str, any], title: str = "Evaluation Metrics") -> str:
    """Format metrics dictionary into a GitHub Markdown table."""
    lines = [
        f"### {title}",
        "",
        "| Metric | Value | Unit / Description |",
        "| :--- | :---: | :--- |",
        f"| **Identity Continuity Rate** | `{metrics_dict['continuity_rate']:.3f}` | Fraction correct & active identity |",
        f"| **Correct Identity Rate** | `{metrics_dict['correct_identity_rate']:.3f}` | Over all evaluated frames |",
        f"| **False-Temporary Rate** | `{metrics_dict['false_temporary_rate']:.3f}` | Temporary state on wrong person (lower is better) |",
        f"| **Unknown-When-Should-Know** | `{metrics_dict['unknown_when_should_know_rate']:.3f}` | Conservatism cost when face visible |",
        f"| **ID Switch Count** | `{metrics_dict['id_switch_count']}` | Swaps between known identities |",
        f"| **Mean Recovery Time** | `{metrics_dict['mean_recovery_time_s']:.3f} s` | Face reappears -> Confirmed |",
        f"| **Time in Confirmed** | `{metrics_dict['time_confirmed_pct']:.1f}%` | Active face recognition match |",
        f"| **Time in Temporary** | `{metrics_dict['time_temporary_pct']:.1f}%` | Honest decayed identity hold |",
        f"| **Time in Unknown** | `{metrics_dict['time_unknown_pct']:.1f}%` | Safe identity reset / lost |",
        f"| **Mean Processing FPS** | `{metrics_dict['fps_mean']:.1f}` | Target >= 15 FPS on CPU |",
        f"| **5th Percentile FPS (p5)** | `{metrics_dict['fps_p5']:.1f}` | Worst-case frame rate |",
        f"| **Centring Error (MAE)** | `{metrics_dict['centring_mae']:.3f}` | Normalized tracking offset |",
        f"| **Centring Error (RMS)** | `{metrics_dict['centring_rms']:.3f}` | Root-mean-square tracking offset |",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Compute tracking and identity evaluation metrics")
    parser.add_argument("--log", type=str, required=True, help="Path to pipeline log CSV")
    parser.add_argument("--gt", type=str, default=None, help="Path to ground truth labels CSV")
    parser.add_argument("--output", type=str, default=None, help="Output JSON/CSV path for metrics")
    args = parser.parse_args()

    results = compute_metrics(args.log, args.gt)
    md_table = format_markdown_table(results, title=f"Run Evaluation: {Path(args.log).stem}")
    print(md_table)

    if args.output:
        out_p = Path(args.output)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        if out_p.suffix == ".json":
            with open(out_p, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2)
        else:
            pd.DataFrame([results]).to_csv(out_p, index=False)
        print(f"\n[INFO] Saved metrics to {out_p}")


if __name__ == "__main__":
    main()
