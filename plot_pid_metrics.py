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


def plot_pid_metrics(summary_df, out_path):
    metrics = [
        ("control_efficiency_cost", "Control efficiency cost"),
        ("traffic_disturbance", "Traffic disturbance"),
        ("driving_comfort_cost", "Driving comfort cost"),
        ("average_thw_cost", "Average THW cost"),
        ("caution_region_duration", "Caution-region duration [s]"),
        ("danger_region_duration", "Danger-region duration [s]"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(14, 7.5))
    axes = axes.ravel()

    for ax, (key, title) in zip(axes, metrics):
        values = summary_df[key].to_numpy()
        mean_value = float(values.mean())

        ax.boxplot(
            values,
            labels=["PID"],
            widths=0.4,
            patch_artist=True,
            boxprops=dict(facecolor="skyblue", color="steelblue", alpha=0.6),
            medianprops=dict(color="firebrick", linewidth=1.5),
        )

        x_jitter = np.random.normal(1, 0.04, size=len(values))
        ax.scatter(x_jitter, values, alpha=0.6, color="navy", edgecolor="black", s=25, zorder=3)

        ax.set_title(title, fontsize=11, fontweight="bold", pad=10)
        ax.set_xlabel("Controller", fontsize=9)

        y_max = max(values)
        ax.set_ylim(bottom=0, top=max(y_max * 1.3, 1e-5))
        ax.grid(True, axis="y", alpha=0.3, linestyle="--")

        if abs(mean_value) < 0.001 and mean_value != 0:
            label = f"Mean: {mean_value:.2e}"
        else:
            label = f"Mean: {mean_value:.4f}"

        ax.text(
            1, ax.get_ylim()[1] * 0.9, label,
            ha="center", va="center",
            fontsize=9, fontweight="bold", color="darkred",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gainsboro", alpha=0.8),
        )

    fig.suptitle("PID Baseline Performance and Safety Metrics (Distribution)", fontsize=14, fontweight="bold")
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

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    summary_csv_path = out_path.with_suffix(".csv")
    summary_df.to_csv(summary_csv_path, index=False)

    mean_summary = summary_df.drop(columns=["csv_path"]).mean().to_dict()

    print(f"Saved summary CSV: {summary_csv_path}")
    print("\nMean metrics:")
    for key, value in mean_summary.items():
        print(f"  {key}: {value:.6f}")

    plot_pid_metrics(summary_df, out_path)


if __name__ == "__main__":
    main()