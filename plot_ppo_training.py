#!/usr/bin/env python3
"""Plot PPO training curves from the metric logs written by episode_ppo_v2.py.

Reads <dir>/ppo_episodes.csv (one row per episode) and <dir>/ppo_updates.csv
(one row per PPO update) and saves ppo_episode_curves.png / ppo_update_curves.png
next to them.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"
SERIES = "#2a78d6"
REFERENCE = "#8a8984"
THW_SAFE_SEC = 1.2  # tau_safe in episode_ppo_v2.py
# Episodes where the follower never launches have THW_min ~35 s; clip so the converged band stays visible.
THW_MIN_YLIM = (0.0, 4.0)

EPISODE_PANELS = [
    ("episode_reward", "Episode reward (sum)", None, None),
    ("mean_cost", "Per-tick mean THW cost", "cost_limit", None),
    ("mean_action", "Mean action (+ throttle / - brake)", 0.0, None),
    ("mean_abs_jerk_mps3", "Mean |jerk| [m/s³]", None, None),
    ("max_follower_speed_mps", "Max follower speed [m/s]", None, None),
    ("mean_abs_spacing_error_m", "Mean |spacing error| [m]", None, None),
    ("sim_time_sec", "Episode length [s, sim time]", None, None),
    ("thw_min_s", "THW_min [s] (y clipped to 0-4)", "thw_safe", THW_MIN_YLIM),
]

UPDATE_PANELS = [
    ("mean_reward", "Rollout mean reward"),
    ("mean_episode_cost", "Per-tick mean cost used for lambda"),
    ("lambda", "Lagrange multiplier (lambda)"),
    ("entropy", "Policy entropy"),
    ("actor_loss", "Actor loss"),
    ("reward_value_loss", "Reward critic loss"),
]


def _style_axis(ax, title, xlabel):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, fontsize=11, color=TEXT_PRIMARY, loc="left", pad=8)
    ax.set_xlabel(xlabel, fontsize=9, color=TEXT_SECONDARY)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=8, length=0)
    ax.grid(True, axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)


def _plot_series(ax, x, y, window):
    ax.plot(x, y, color=SERIES, alpha=0.3, linewidth=1.0, label="per point")
    if len(y) >= 2:
        smoothed = y.rolling(window=min(window, len(y)), min_periods=1).mean()
        ax.plot(x, smoothed, color=SERIES, linewidth=2.0, label=f"moving avg ({window})")


def _save(fig, out_path, suptitle):
    fig.suptitle(suptitle, fontsize=13, color=TEXT_PRIMARY, x=0.01, ha="left")
    handles, labels = fig.axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper right", frameon=False, fontsize=9, labelcolor=TEXT_SECONDARY, ncol=3)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print(f"Saved figure: {out_path}")


def _draw_reference(ax, value, label, below=False):
    ax.axhline(value, color=REFERENCE, linewidth=1.0, linestyle="--")
    ax.annotate(
        label,
        xy=(1, value),
        xycoords=("axes fraction", "data"),
        xytext=(0, -12 if below else 4),
        textcoords="offset points",
        ha="right",
        fontsize=8,
        color=TEXT_SECONDARY,
    )


def _draw_cost_limit(ax, cost_limit):
    _draw_reference(ax, cost_limit, f"cost limit d={cost_limit:g}")


def plot_episodes(df, out_path, window, cost_limit):
    fig, axes = plt.subplots(2, 4, figsize=(20, 8), facecolor=SURFACE)
    for ax, (key, title, reference, ylim) in zip(axes.ravel(), EPISODE_PANELS):
        _style_axis(ax, title, "episode")
        if key not in df:
            ax.text(0.5, 0.5, f"{key} not logged", transform=ax.transAxes, ha="center", color=TEXT_SECONDARY)
            continue
        if reference == "cost_limit":
            _draw_cost_limit(ax, cost_limit)
        elif reference == "thw_safe":
            _draw_reference(ax, THW_SAFE_SEC, f"τ_safe = {THW_SAFE_SEC:g} s", below=True)
        elif reference is not None:
            ax.axhline(reference, color=REFERENCE, linewidth=1.0, linestyle="--")
        _plot_series(ax, df["episode"], df[key], window)
        if ylim is not None:
            ax.set_ylim(*ylim)
    _save(fig, out_path, f"PPO training by episode ({len(df)} episodes)")


def plot_updates(df, out_path, window, cost_limit):
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), facecolor=SURFACE)
    for ax, (key, title) in zip(axes.ravel(), UPDATE_PANELS):
        _style_axis(ax, title, "PPO update")
        if key == "mean_episode_cost":
            _draw_cost_limit(ax, cost_limit)
        _plot_series(ax, df["update"], df[key], window)
    _save(fig, out_path, f"PPO training by update ({len(df)} updates)")


def main():
    parser = argparse.ArgumentParser(description="Plot PPO training curves")
    parser.add_argument("--dir", default="outputs/ppo", help="Directory with ppo_episodes.csv / ppo_updates.csv")
    parser.add_argument("--window", type=int, default=10, help="Moving-average window (default: 10)")
    parser.add_argument(
        "--cost-limit",
        type=float,
        default=5.0 / 650.0,
        help="Per-tick mean cost limit d drawn on the cost panels (default: 5/650)",
    )
    args = parser.parse_args()

    metrics_dir = Path(args.dir)
    episodes_csv = metrics_dir / "ppo_episodes.csv"
    updates_csv = metrics_dir / "ppo_updates.csv"

    if not episodes_csv.exists() and not updates_csv.exists():
        raise SystemExit(f"No PPO metric logs in {metrics_dir}. Run episode_ppo_v2.py --controller ppo first.")

    if episodes_csv.exists():
        plot_episodes(pd.read_csv(episodes_csv), metrics_dir / "ppo_episode_curves.png", args.window, args.cost_limit)
    else:
        print(f"Skipping episode curves: {episodes_csv} not found")

    if updates_csv.exists():
        plot_updates(pd.read_csv(updates_csv), metrics_dir / "ppo_update_curves.png", args.window, args.cost_limit)
    else:
        print(f"Skipping update curves: {updates_csv} not found (no PPO update has run yet)")


if __name__ == "__main__":
    main()
