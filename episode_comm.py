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

    acceleration = np.gradient(follower_speed, dt)
    acceleration = uniform_filter1d(acceleration, size=5)
    jerk = np.gradient(acceleration, dt)
    jerk = uniform_filter1d(jerk, size=5)

    thw = distance / np.maximum(follower_speed, 0.1)
    thw_cost, caution_mask, danger_mask = compute_thw_cost(thw, thw_safe=thw_safe, thw_danger=thw_danger)

    return {
        "csv_path": str(csv_path),
        "control_efficiency_cost": np.mean(spacing_error ** 2 + relative_velocity ** 2),
        "traffic_disturbance": np.mean(acceleration ** 2),
        "driving_comfort_cost": np.mean(jerk ** 2),
        "average_thw_cost": np.mean(thw_cost),
        "caution_region_duration": max(0.0, float(np.sum(caution_mask) * dt)),
        "danger_region_duration": max(0.0, float(np.sum(danger_mask) * dt)),
    }


def plot_comparison(means, labels, colors, out_path):
    metrics = [
        ("control_efficiency_cost", "Control Efficiency Cost"),
        ("traffic_disturbance", "Traffic Disturbance"),
        ("driving_comfort_cost", "Driving Comfort Cost"),
        ("average_thw_cost", "Average THW Cost"),
        ("caution_region_duration", "Caution-Region Duration [s]"),
        ("danger_region_duration", "Danger-Region Duration [s]"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.ravel()

    x = np.arange(len(labels))
    width = 0.5

    for ax, (key, title) in zip(axes, metrics):
        values = [float(m[key]) for m in means]

        bars = ax.bar(x, values, width=width, color=colors, edgecolor="black", linewidth=0.7)

        ax.set_title(title, fontsize=11, fontweight="bold", pad=10)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=10)
        ax.set_ylim(bottom=0, top=max(max(values) * 1.3, 1e-6))
        ax.grid(True, axis="y", alpha=0.3, linestyle="--")

        for bar, val in zip(bars, values):
            if abs(val) < 0.001 and val != 0:
                label = f"{val:.2e}"
            else:
                label = f"{val:.4f}"
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + ax.get_ylim()[1] * 0.01,
                label,
                ha="center", va="bottom",
                fontsize=8, fontweight="bold"
            )

    fig.suptitle("PID Performance Comparison: Comm 100% vs 70% vs 50%", fontsize=14, fontweight="bold")
    fig.tight_layout()

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    print(f"Saved figure: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv-100", nargs="+", required=True, help="통신 100% CSV 파일들")
    parser.add_argument("--csv-70", nargs="+", required=True, help="통신 70% CSV 파일들")
    parser.add_argument("--csv-50", nargs="+", required=True, help="통신 50% CSV 파일들")
    parser.add_argument("--out", default="outputs/comparison.png")
    parser.add_argument("--thw-safe", type=float, default=1.5)
    parser.add_argument("--thw-danger", type=float, default=0.8)

    args = parser.parse_args()

    summaries_100 = [summarize_pid_csv(p, args.thw_safe, args.thw_danger) for p in args.csv_100]
    summaries_70 = [summarize_pid_csv(p, args.thw_safe, args.thw_danger) for p in args.csv_70]
    summaries_50 = [summarize_pid_csv(p, args.thw_safe, args.thw_danger) for p in args.csv_50]

    mean_100 = pd.DataFrame(summaries_100).drop(columns=["csv_path"]).mean().to_dict()
    mean_70 = pd.DataFrame(summaries_70).drop(columns=["csv_path"]).mean().to_dict()
    mean_50 = pd.DataFrame(summaries_50).drop(columns=["csv_path"]).mean().to_dict()

    means = [mean_100, mean_70, mean_50]
    labels = ["PID 100%", "PID 70%", "PID 50%"]
    colors = ["steelblue", "tomato", "mediumseagreen"]

    print("\nPID 100% Mean:"); [print(f"  {k}: {v:.6f}") for k, v in mean_100.items()]
    print("\nPID 70% Mean:");  [print(f"  {k}: {v:.6f}") for k, v in mean_70.items()]
    print("\nPID 50% Mean:");  [print(f"  {k}: {v:.6f}") for k, v in mean_50.items()]

    plot_comparison(means, labels, colors, args.out)


if __name__ == "__main__":
    main()