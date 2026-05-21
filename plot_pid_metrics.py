#!/usr/bin/env python3

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d


def compute_thw_cost(thw, thw_safe=1.5, thw_danger=0.8):
    thw = np.asarray(thw, dtype=float)
    cost = np.zeros_like(thw)

    danger_mask = thw <= thw_danger
    caution_mask = (thw > thw_danger) & (thw < thw_safe)

    cost[danger_mask] = 1.0
    cost[caution_mask] = (
        (thw_safe - thw[caution_mask]) / (thw_safe - thw_danger)
    ) ** 2

    return cost, caution_mask, danger_mask


def summarize_pid_csv(csv_path, thw_safe=1.5, thw_danger=0.8):
    df = pd.read_csv(csv_path)

    if len(df) < 3:
        raise ValueError(f"CSV too short: {csv_path}")

    time = df["time_sec"].to_numpy()
    dt = np.median(np.diff(time))
    dt = max(float(dt), 1e-6)

    spacing_error = df["spacing_error_m"].to_numpy()
    relative_velocity = df["relative_velocity_mps"].to_numpy()
    follower_speed = df["follower_speed_mps"].to_numpy()
    distance = df["distance_m"].to_numpy()

    # 스무딩 적용
    acceleration = np.gradient(follower_speed, dt)
    acceleration = uniform_filter1d(acceleration, size=5)
    jerk = np.gradient(acceleration, dt)
    jerk = uniform_filter1d(jerk, size=5)

    thw = distance / np.maximum(follower_speed, 0.1)

    thw_cost, caution_mask, danger_mask = compute_thw_cost(
        thw, thw_safe=thw_safe, thw_danger=thw_danger,
    )

    control_efficiency_cost = np.mean(spacing_error ** 2 + relative_velocity ** 2)
    traffic_disturbance = np.mean(acceleration ** 2)
    driving_comfort_cost = np.mean(jerk ** 2)
    average_thw_cost = np.mean(thw_cost)
    caution_region_duration = max(0.0, float(np.sum(caution_mask) * dt))
    danger_region_duration = max(0.0, float(np.sum(danger_mask) * dt))

    return {
        "csv_path": str(csv_path),
        "control_efficiency_cost": control_efficiency_cost,
        "traffic_disturbance": traffic_disturbance,
        "driving_comfort_cost": driving_comfort_cost,
        "average_thw_cost": average_thw_cost,
        "caution_region_duration": caution_region_duration,
        "danger_region_duration": danger_region_duration,
    }


def plot_pid_metrics(summary, out_path):
    metrics = [
        ("control_efficiency_cost", "Average Control Efficiency Cost"),
        ("traffic_disturbance", "Average Traffic Disturbance"),
        ("driving_comfort_cost", "Average Driving Comfort Cost"),
        ("average_thw_cost", "Average THW Cost"),
        ("caution_region_duration", "Caution-Region Duration (s)"),
        ("danger_region_duration", "Danger-Region Duration (s)"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    axes = axes.ravel()

    for ax, (key, title) in zip(axes, metrics):
        value = summary[key]
        bars = ax.bar(["PID"], [value], color="steelblue", width=0.4)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Controller")
        ax.set_ylim(bottom=0)
        ax.bar_label(bars, fmt="%.4f", padding=3, fontsize=9)
        ax.grid(True, alpha=0.25)

    fig.suptitle("PID Baseline Performance and Safety Metrics", fontsize=14)
    fig.tight_layout()

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)

    print(f"Saved figure: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", nargs="+", required=True)
    parser.add_argument("--out", default="outputs/pid_metric_summary.png")
    parser.add_argument("--thw-safe", type=float, default=1.5)
    parser.add_argument("--thw-danger", type=float, default=0.8)

    args = parser.parse_args()

    summaries = [
        summarize_pid_csv(csv_path, thw_safe=args.thw_safe, thw_danger=args.thw_danger)
        for csv_path in args.csv
    ]

    summary_df = pd.DataFrame(summaries)
    mean_summary = summary_df.drop(columns=["csv_path"]).mean().to_dict()

    out_path = Path(args.out)
    summary_csv_path = out_path.with_suffix(".csv")

    summary_df.to_csv(summary_csv_path, index=False)
    print(f"Saved summary CSV: {summary_csv_path}")

    print("\nMean metrics:")
    for key, value in mean_summary.items():
        print(f"  {key}: {value:.4f}")

    plot_pid_metrics(mean_summary, out_path)


if __name__ == "__main__":
    main()