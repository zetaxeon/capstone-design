#!/usr/bin/env python3

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def compute_thw_cost(thw, thw_safe=1.5, thw_danger=0.8):
    thw = np.asarray(thw, dtype=float)
    cost = np.zeros_like(thw)
    danger_mask = thw <= thw_danger
    caution_mask = (thw > thw_danger) & (thw < thw_safe)
    cost[danger_mask] = 1.0
    cost[caution_mask] = ((thw_safe - thw[caution_mask]) / (thw_safe - thw_danger)) ** 2
    return cost, caution_mask, danger_mask

def summarize_pid_csv(csv_path, thw_safe=1.5, thw_danger=0.8, smooth_window=5):
    df = pd.read_csv(csv_path)
    if len(df) < 3:
        return None

    time = df["time_sec"].to_numpy()
    dt = np.median(np.diff(time))
    dt = max(float(dt), 1e-6)

    spacing_error = df["spacing_error_m"].to_numpy()
    relative_velocity = df["relative_velocity_mps"].to_numpy()
    follower_speed = df["follower_speed_mps"].to_numpy()
    distance = df["distance_m"].to_numpy()

    acceleration = np.gradient(follower_speed, dt)
    acceleration = pd.Series(acceleration).rolling(window=smooth_window, center=True, min_periods=1).mean().to_numpy()

    jerk = np.gradient(acceleration, dt)
    jerk = pd.Series(jerk).rolling(window=smooth_window, center=True, min_periods=1).mean().to_numpy()

    thw = distance / np.maximum(follower_speed, 0.1)
    thw_cost, caution_mask, danger_mask = compute_thw_cost(thw, thw_safe=thw_safe, thw_danger=thw_danger)

    return {
        "control_efficiency_cost": np.mean(spacing_error ** 2 + relative_velocity ** 2),
        "traffic_disturbance": np.mean(acceleration ** 2),
        "driving_comfort_cost": np.mean(jerk ** 2),
        "average_thw_cost": np.mean(thw_cost),
        "caution_region_duration": np.sum(caution_mask) * dt,
        "danger_region_duration": np.sum(danger_mask) * dt,
    }

def collect_group_summary(csv_paths, thw_safe, thw_danger, smooth_window):
    summaries = []
    for p in csv_paths:
        res = summarize_pid_csv(p, thw_safe, thw_danger, smooth_window)
        if res is not None:
            summaries.append(res)
    df = pd.DataFrame(summaries)
    return df.mean().to_dict()

def main():
    parser = argparse.ArgumentParser(description="Compare 3 PID Comm Scenarios (100% vs 70% vs 50%)")
    parser.add_argument("--csv-100", nargs="+", required=True)
    parser.add_argument("--csv-70", nargs="+", required=True)
    parser.add_argument("--csv-50", nargs="+", required=True)
    parser.add_argument("--out", default="outputs/comparison.png")
    parser.add_argument("--thw-safe", type=float, default=1.5)
    parser.add_argument("--thw-danger", type=float, default=0.8)
    parser.add_argument("--smooth-window", type=int, default=5)
    args = parser.parse_args()

    # 데이터 취합
    mean_100 = collect_group_summary(args.csv_100, args.thw_safe, args.thw_danger, args.smooth_window)
    mean_70  = collect_group_summary(args.csv_70, args.thw_safe, args.thw_danger, args.smooth_window)
    mean_50  = collect_group_summary(args.csv_50, args.thw_safe, args.thw_danger, args.smooth_window)

    metrics = [
        ("control_efficiency_cost", "Control efficiency cost"),
        ("traffic_disturbance", "Traffic disturbance"),
        ("driving_comfort_cost", "Driving comfort cost"),
        ("average_thw_cost", "Average THW cost"),
        ("caution_region_duration", "Caution-region duration [s]"),
        ("danger_region_duration", "Danger-region duration [s]"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))
    axes = axes.ravel()

    labels = ["Comm 100%", "Comm 70%", "Comm 50%"]
    colors = ["#4C72B0", "#DD8452", "#55A868"]

    for ax, (key, title) in zip(axes, metrics):
        values = [float(mean_100[key]), float(mean_70[key]), float(mean_50[key])]
        
        # 그래프 그리기 (기존 bar_label 유발 요소를 완벽히 제거)
        bars = ax.bar(labels, values, color=colors, edgecolor='black', width=0.5)
        
        ax.set_title(title, fontsize=11, fontweight='bold', pad=12)
        
        # 글자가 위 천장에 가려지지 않도록 최대 높이 마진을 35%로 여유롭게 설정
        max_val = max(values)
        ax.set_ylim(0, max(max_val * 1.35, 1e-5))
        ax.grid(True, axis='y', alpha=0.25, linestyle="--")

        # [수정 핵심] 중복 없이 루프 딱 한 번만 돌며 깔끔하게 숫자 추가
        for bar in bars:
            h = bar.get_height()
            
            # 0.001보다 작은 아주 작은 실수 데이터(e-05 등) 분기 처리
            if 0 < abs(h) < 0.001:
                lbl = f"{h:.2e}"
            else:
                lbl = f"{h:.4f}"
                
            ax.text(
                bar.get_x() + bar.get_width() / 2.0, 
                h + (ax.get_ylim()[1] * 0.01), 
                lbl, 
                ha='center', 
                va='bottom', 
                fontsize=9.5, 
                fontweight='bold',
                color='black'
            )

    fig.suptitle("PID Performance Comparison across V2V Communication Success Rates", fontsize=15, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    print(f"\n[성공] 3중 비교 그래프 렌더링이 완료되었습니다: {out_path}")

if __name__ == "__main__":
    main()