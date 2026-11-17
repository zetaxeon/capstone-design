#!/usr/bin/env python3
"""Report figures from the existing tunnel results (no simulation).

Usage: .venv/bin/python analysis/report_figures.py
Reads  outputs/tunnel/eval/ (PPO-Lag, PPO; two vehicles), outputs/tunnel/pid_retune/ (retuned PID,
       kp 0.4 / ki 0.01 / kd 0.6, evaluation seed 1000; two vehicles) and
       outputs/tunnel/summary/platoon_episodes.csv (three-vehicle platoon, from platoon_metrics.py).
Writes docs/results/figures/2026-11-17_tradeoff_scatter.png   in-tunnel spacing error vs collision rate
       docs/results/figures/2026-11-17_error_collision_by_p.png  both against the in-tunnel p
       outputs/tunnel/summary/report_figure_data.csv           the plotted numbers
PPO-Lag and PPO pool the three training seeds (300 episodes per point); PID has 100.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import font_manager

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tunnel_paper_metrics import episode_metrics  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
FIG = REPO / "docs/results/figures"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e3e2dd"
STYLE = {  # fixed order: colour and marker belong to the controller
    "PID": {"color": "#2a78d6", "marker": "o", "label": "PID (재튜닝)"},
    "PPO-Lag": {"color": "#eb6834", "marker": "s", "label": "PPO-Lag"},
    "PPO": {"color": "#1baf7a", "marker": "^", "label": "PPO"},
}
P_ORDER = [1.0, 0.9, 0.7, 0.5, 0.3, 0.1]
# Scatter labels for points that overlap: (p values, text offset in points, alignment).
LABEL_GROUPS = {
    "PID": [([1.0, 0.9, 0.7], (4, -19), "center"), ([0.5], (0, 11), "center"), ([0.3], (0, 11), "center")],
    "PPO-Lag": [([1.0, 0.9, 0.7, 0.5], (-6, -19), "left")],
    "PPO": [([1.0, 0.9, 0.7, 0.5], (4, 11), "right")],
}


def two_vehicle():
    rows = []
    learned = pd.read_csv(REPO / "outputs/tunnel/summary/paper_metrics_episodes.csv")
    for algo, name in (("lag", "PPO-Lag"), ("ppo", "PPO")):
        for p, g in learned[learned.algo == algo].groupby("p"):
            rows.append({"controller": name, "p": p, "episodes": len(g), "err_tunnel_m": g.abs_err_tunnel.mean(),
                         "collision_pct": 100 * g.collision.mean()})
    for cell in sorted((REPO / "outputs/tunnel/pid_retune").glob("kp0.4_ki0.01_kd0.6_t*")):
        if not cell.is_dir() or "seed" in cell.name:
            continue
        p = float(cell.name.rsplit("_t", 1)[1])
        eps = pd.read_csv(cell / "pid" / "pid_episodes.csv")
        err = [episode_metrics(pd.read_csv(f))["abs_err_tunnel"] for f in sorted((cell / "pid").glob("pid_episode_*.csv"))]
        rows.append({"controller": "PID", "p": p, "episodes": len(eps), "err_tunnel_m": sum(err) / len(err),
                     "collision_pct": 100 * eps.collision.mean()})
    return pd.DataFrame(rows)


def platoon():
    d = pd.read_csv(REPO / "outputs/tunnel/summary/platoon_episodes.csv")
    d["controller"] = d.algo.map({"pid": "PID", "lag": "PPO-Lag", "ppo": "PPO"})
    return (d.groupby(["controller", "p"]).agg(episodes=("episode", "size"), platoon_collision_pct=("platoon_collision", lambda x: 100 * x.mean()))
            .reset_index())


def setup_font():
    for path in ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",):
        if Path(path).exists():
            font_manager.fontManager.addfont(path)
            plt.rcParams["font.family"] = font_manager.FontProperties(fname=path).get_name()
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "xtick.color": MUTED,
                         "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.unicode_minus": False, "text.color": INK})


def scatter(two):
    fig, ax = plt.subplots(figsize=(7.4, 5.0))
    ax.grid(color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for name, st in STYLE.items():
        g = two[two.controller == name].set_index("p").loc[P_ORDER].reset_index()
        ax.plot(g.err_tunnel_m, g.collision_pct, color=st["color"], linewidth=2, zorder=2)
        ax.scatter(g.err_tunnel_m, g.collision_pct, s=64, color=st["color"], marker=st["marker"], edgecolor="white",
                   linewidth=1.5, zorder=3, label=st["label"])
        # Every point carries its p. Points that sit on top of each other share one label.
        groups, used = LABEL_GROUPS[name], set()
        for ps, offset, ha in groups:
            pts = g[g.p.isin(ps)]
            used.update(ps)
            ax.annotate("p " + " · ".join(f"{q:.1f}" for q in ps), (pts.err_tunnel_m.mean(), pts.collision_pct.mean()),
                        xytext=offset, textcoords="offset points", ha=ha, color=INK)
        for _, r in g[~g.p.isin(used)].iterrows():
            ax.annotate(f"p {r.p:.1f}", (r.err_tunnel_m, r.collision_pct), xytext=(9, 3), textcoords="offset points", color=INK)
    ax.set_xscale("log")
    ax.set_xticks([0.5, 1, 2, 5, 10, 20])
    ax.set_xticklabels(["0.5", "1", "2", "5", "10", "20"])
    ax.minorticks_off()
    ax.set_xlim(0.28, 25)
    ax.set_ylim(-5, 50)
    ax.set_xlabel("터널 안 평균 간격 오차 (m, 로그 축)")
    ax.set_ylabel("충돌률 (%)")
    ax.legend(loc="upper center", frameon=False, ncol=3, bbox_to_anchor=(0.5, 1.0))
    ax.set_title("간격 추종과 안전의 맞바꿈: 2대 (리더 + 추종차 1대), 점 옆 숫자는 터널 안 통신 성공률 p",
                 loc="left", fontsize=10.5, pad=14)
    fig.tight_layout()
    out = FIG / "2026-11-17_tradeoff_scatter.png"
    fig.savefig(out, dpi=170)
    return out


def by_p(two, three):
    x = {p: i for i, p in enumerate(P_ORDER)}
    fig, (a, b) = plt.subplots(1, 2, figsize=(10.2, 4.4))
    for ax in (a, b):
        ax.grid(axis="y", color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        ax.set_xticks(range(len(P_ORDER)))
        ax.set_xticklabels([f"{p:.1f}" for p in P_ORDER])
        ax.set_xlabel("터널 안 통신 성공률 p")
    for name, st in STYLE.items():
        g = two[two.controller == name].set_index("p").loc[P_ORDER].reset_index()
        xs = [x[p] for p in g.p]
        a.plot(xs, g.err_tunnel_m, color=st["color"], linewidth=2, marker=st["marker"], markersize=7,
               markeredgecolor="white", markeredgewidth=1.2)
        a.annotate(st["label"], (xs[-1], g.err_tunnel_m.iloc[-1]), xytext=(8, {"PPO-Lag": 5, "PPO": -7}.get(name, 0)),
                   textcoords="offset points", va="center", color=INK)
        b.plot(xs, g.collision_pct, color=st["color"], linewidth=2, marker=st["marker"], markersize=7,
               markeredgecolor="white", markeredgewidth=1.2, label=f"{st['label']}, 2대")
        t = three[three.controller == name].sort_values("p", ascending=False)
        b.plot([x[p] for p in t.p], t.platoon_collision_pct, color=st["color"], linewidth=2, linestyle=(0, (4, 2.5)),
               marker=st["marker"], markersize=7, markerfacecolor="white", markeredgecolor=st["color"], markeredgewidth=1.5,
               label=f"{st['label']}, 3대 군집")
    a.set_yscale("log")
    a.set_yticks([0.5, 1, 2, 5, 10, 20])
    a.set_yticklabels(["0.5", "1", "2", "5", "10", "20"])
    a.set_ylim(0.4, 25)
    a.set_xlim(-0.3, len(P_ORDER) - 1 + 1.25)
    a.set_ylabel("터널 안 평균 간격 오차 (m, 로그 축)")
    a.set_title("간격 오차 (2대)", loc="left", fontsize=10.5)
    b.set_ylabel("충돌률 (%)")
    b.set_ylim(-2.5, 56)
    b.set_title("충돌률: 실선 2대, 점선 3대 군집 (어느 쌍이든 충돌)", loc="left", fontsize=10.5)
    b.legend(loc="upper left", frameon=False, ncol=1)
    fig.tight_layout()
    out = FIG / "2026-11-17_error_collision_by_p.png"
    fig.savefig(out, dpi=170)
    return out


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    setup_font()
    two, three = two_vehicle(), platoon()
    data = two.merge(three[["controller", "p", "platoon_collision_pct"]], on=["controller", "p"], how="left")
    data = data.sort_values(["controller", "p"], ascending=[True, False]).round(3)
    data.to_csv(REPO / "outputs/tunnel/summary/report_figure_data.csv", index=False)
    print(data.to_string(index=False))
    print(scatter(two))
    print(by_p(two, three))


if __name__ == "__main__":
    main()
