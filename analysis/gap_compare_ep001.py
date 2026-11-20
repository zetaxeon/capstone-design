#!/usr/bin/env python3.10
"""F1 gap over time, PID vs PPO-Lag (s43), evaluation seed 1000 episode 1, 3-vehicle platoon.

Reads only the existing platoon evaluation logs (outputs/tunnel/eval3/, analysis/platoon_eval.sh);
no simulation. Solid = measured F1-leader distance (vehicle centres), dashed = that controller's
own target gap d* = 7 + 1.0 * v_F1. Shading = time the leader-F1 link is in the tunnel zone
(comm_in_tunnel of the PID run; the PPO-Lag run differs by < 0.3 s at p = 0.3). Vertical line =
leader hard brake start. This is the episode shown in outputs/video/scenario_ppolag_p03.mp4.

Two outputs:
  default  draft with one panel per in-tunnel p (0.3 and 0.1)   -> outputs/video/gap_compare_ep001.png
  --slide  p = 0.3 only, 1920x1080, large text, no title        -> outputs/video/gap_compare_p03.png

Usage: .venv/bin/python analysis/gap_compare_ep001.py [--slide] [--out PATH]
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import font_manager
from matplotlib.lines import Line2D

REPO = Path(__file__).resolve().parents[1]
EVAL = REPO / "outputs" / "tunnel" / "eval3"
STYLE = {  # same colours as analysis/report_figures.py
    "PID": {"color": "#2a78d6", "label": "PID (재튜닝)", "short": "PID", "cell": "pid",
            "file": "pid/pid_episode_001.csv"},
    "PPO-Lag": {"color": "#eb6834", "label": "PPO-Lag (s43)", "short": "PPO-Lag", "cell": "lag_s43",
                "file": "ppo_eval/ppo_eval_episode_001.csv"},
}
INK, MUTED, GRID, SHADE = "#1f1f1f", "#6b6b6b", "#e6e6e3", "#ecebe6"
FONT_PATH = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"  # as in report_figures.py
DRAFT = {"base": 10, "small": 9, "title": 11, "line": 2.0, "dash": 1.2, "marker": 9}
SLIDE = {"base": 24, "small": 22, "title": 0, "line": 4.0, "dash": 2.4, "marker": 16}


def setup_font(size):
    font_manager.fontManager.addfont(FONT_PATH)
    plt.rcParams.update({"font.family": font_manager.FontProperties(fname=FONT_PATH).get_name(),
                         "font.size": size, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED})


def load(ctrl, p):
    s = STYLE[ctrl]
    steps = pd.read_csv(EVAL / f"{s['cell']}_t{p}" / s["file"])
    summary = pd.read_csv(EVAL / f"{s['cell']}_t{p}" / "platoon_episodes.csv").iloc[0]
    return steps, summary


def panel(ax, p, t_range, fs, slide=False):
    data = {c: load(c, p) for c in STYLE}
    pid = data["PID"][0]
    tun = pid.loc[pid.comm_in_tunnel == 1, "time_sec"]
    ax.axvspan(tun.min(), tun.max(), color=SHADE, zorder=0, lw=0)
    ax.text(tun.min() + 0.2, 0.97, f"터널 (p = {p})", transform=ax.get_xaxis_transform(),
            va="top", ha="left", color=MUTED, fontsize=fs["small"])
    brake_t = float(data["PID"][1].leader_brake_time_sec)
    ax.axvline(brake_t, color=INK, lw=fs["line"] / 2, zorder=1)
    brake_label = "리더 급제동" if slide else f"리더 급제동 {brake_t:.2f} s"
    ax.text(brake_t + 0.2, 0.04, brake_label, transform=ax.get_xaxis_transform(),
            va="bottom", ha="left", color=INK, fontsize=fs["small"])
    for ctrl, s in STYLE.items():
        steps, summary = data[ctrl]
        w = steps[(steps.time_sec >= t_range[0]) & (steps.time_sec <= t_range[1])]
        ax.plot(w.time_sec, w.distance_m, color=s["color"], lw=fs["line"], label=f"{s['label']} 차간거리", zorder=3)
        ax.plot(w.time_sec, w.desired_distance_m, color=s["color"], lw=fs["dash"], ls=(0, (4, 3)), alpha=0.9,
                label=f"{s['label']} 목표 d* = 7 + v", zorder=2)
        if summary.platoon_collision:
            last = steps.iloc[-1]
            ax.plot(last.time_sec, last.distance_m, marker="X", ms=fs["marker"], color=s["color"],
                    markeredgecolor="white", markeredgewidth=1.5, zorder=4)
            who = "F2→F1" if summary.collision_f2_f1 else "F1→리더"
            ax.annotate(f"{s['label']} 에피소드 종료\n{who} 충돌 ({last.time_sec:.2f} s)",
                        (last.time_sec, last.distance_m), xytext=(10, -26), textcoords="offset points",
                        fontsize=fs["small"], color=INK, arrowprops={"arrowstyle": "-", "color": MUTED, "lw": 0.8})
    ax.set_xlim(*t_range)
    ax.set_ylim(bottom=0)
    ax.set_ylabel("차간거리 (m)" if slide else "F1–리더 거리 (m, 차량 중심 간)")
    ax.grid(axis="y", color=GRID, lw=fs["line"] / 2.5)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    if not slide:
        ax.set_title(f"터널 p = {p}", loc="left", fontsize=fs["title"], color=INK, fontweight="bold")


def draft(opts):
    setup_font(DRAFT["base"])
    fig, axes = plt.subplots(2, 1, figsize=(9, 7.2), sharex=True, constrained_layout=True)
    for ax, p in zip(axes, ("0.3", "0.1")):
        panel(ax, p, (opts.t0, opts.t1), DRAFT)
    axes[-1].set_xlabel("에피소드 시간 (s)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncol=2, frameon=False, fontsize=DRAFT["small"])
    out = opts.out or REPO / "outputs" / "video" / "gap_compare_ep001.png"
    fig.savefig(out, dpi=200, facecolor="white")
    print("wrote", out)


def slide(opts):
    setup_font(SLIDE["base"])
    fig, ax = plt.subplots(figsize=(16, 9), constrained_layout=True)
    fig.get_layout_engine().set(w_pad=0.25, h_pad=0.2)
    panel(ax, "0.3", (opts.t0, opts.t1), SLIDE, slide=True)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.13)  # headroom so the tunnel label clears the PID peak
    ax.set_xlabel("시간 (s)")
    ax.tick_params(width=1.5, length=6)
    handles = [Line2D([], [], color=s["color"], lw=SLIDE["line"], label=s["short"]) for s in STYLE.values()]
    handles.append(Line2D([], [], color=MUTED, lw=SLIDE["dash"], ls=(0, (4, 3)), label="목표 간격"))
    fig.legend(handles=handles, loc="outside upper center", ncol=3, frameon=False, fontsize=SLIDE["base"],
               handlelength=2.6, columnspacing=2.0)
    out = opts.out or REPO / "outputs" / "video" / "gap_compare_p03.png"
    fig.savefig(out, dpi=120, facecolor="white")  # 16 x 9 in at 120 dpi = 1920 x 1080
    print("wrote", out)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--slide", action="store_true", help="p = 0.3 only, 1920x1080 presentation figure")
    parser.add_argument("--out", default=None)
    parser.add_argument("--t0", type=float, default=5.0)
    parser.add_argument("--t1", type=float, default=35.0)
    opts = parser.parse_args()
    (slide if opts.slide else draft)(opts)


if __name__ == "__main__":
    main()
