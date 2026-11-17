#!/usr/bin/env python3
"""Paper-metric re-analysis of the tunnel evaluation logs (no CARLA needed).

Usage: .venv/bin/python analysis/tunnel_paper_metrics.py [EVAL_DIR] [--fig-dir DIR]
EVAL_DIR defaults to outputs/tunnel/eval. Prints markdown tables and writes
  outputs/tunnel/summary/paper_metrics_episodes.csv   one row per (cell, episode)
  DIR/collision_timeseries.png                        one representative collision episode

Per tick, from the logged true state (the same definitions as episode_ppo_v2.py):
  r_cont = 2 * spacing_error^2 + relative_velocity^2          eq. (10), Q = diag(2, 1)
  r_traf = a^2,  r_jerk = (a_t - a_(t-1))^2                    eq. (11)-(12), a = follower accel
  THW = distance / follower speed (infinite at or below 0.5 m/s)
  THW cost: 0 if THW >= 1.2; 1 if THW <= 1.0; ((1.2 - THW) / 0.2)^2 between      eq. (15)
  Caution region: 1.0 < THW < 1.2.  Danger region: THW <= 1.0.
"""
import argparse
import itertools
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd

TAU_SAFE, TAU_DANGER, THW_MIN_SPEED = 1.2, 1.0, 0.5
D_CLOSE, D_FAR = -4.0, 8.0
DT = 0.05
REPO = Path(__file__).resolve().parent.parent


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, mid - half), min(1.0, mid + half))


def episode_metrics(t):
    err, relv = t.spacing_error_m.to_numpy(), t.relative_velocity_mps.to_numpy()
    acc, v, dist = t.follower_accel_mps2.to_numpy(), t.follower_speed_mps.to_numpy(), t.distance_m.to_numpy()
    inside = t.follower_in_tunnel.to_numpy() == 1
    thw = np.where(v > THW_MIN_SPEED, dist / np.maximum(v, 1e-9), np.inf)
    cost = np.where(thw >= TAU_SAFE, 0.0, np.where(thw <= TAU_DANGER, 1.0, ((TAU_SAFE - thw) / (TAU_SAFE - TAU_DANGER)) ** 2))
    jerk = np.concatenate([[0.0], np.diff(acc) ** 2])
    out = {
        "ticks": len(t),
        "r_cont_sum": float((2 * err**2 + relv**2).sum()),
        "r_traf_sum": float((acc**2).sum()),
        "r_jerk_sum": float(jerk.sum()),
        "thw_cost_sum": float(cost.sum()),
        "caution_ticks": int(((thw > TAU_DANGER) & (thw < TAU_SAFE)).sum()),
        "danger_ticks": int((thw <= TAU_DANGER).sum()),
        "danger_ticks_tunnel": int(((thw <= TAU_DANGER) & inside).sum()),
        "t_since_comm_end_s": float(t.t_since_comm_s.iloc[-1]),
        "t_since_comm_max_last3s": float(t.t_since_comm_s.iloc[-60:].max()),
        "end_in_tunnel": int(t.comm_in_tunnel.iloc[-1]),
        "tunnel_ticks": int(inside.sum()),
        "abs_err_tunnel": float(np.abs(err[inside]).mean()) if inside.any() else np.nan,
        "abs_err_outside": float(np.abs(err[~inside]).mean()) if (~inside).any() else np.nan,
        "thw_min_tunnel": float(thw[inside].min()) if inside.any() else np.nan,
        "exited": 0, "err_at_exit": np.nan, "outside_band_at_exit": np.nan, "recover_s": np.nan,
    }
    if inside.any():
        last = int(np.flatnonzero(inside)[-1])
        if last + 1 < len(t):          # the follower left the tunnel before the episode ended
            after = err[last + 1:]
            ok = (after >= D_CLOSE) & (after <= D_FAR)
            out.update(exited=1, err_at_exit=float(after[0]), outside_band_at_exit=int(not ok[0]),
                       recover_s=float(np.argmax(ok) * DT) if ok.any() else np.nan)
    return out


def load(eval_dir):
    rows = []
    for cell in sorted(p for p in eval_dir.iterdir() if p.is_dir()):
        m = re.fullmatch(r"(pid|lag_s\d+|ppo_s\d+)_t([\d.]+)", cell.name)
        if not m:
            continue
        sub = "pid" if m.group(1) == "pid" else "ppo_eval"
        eps = pd.read_csv(cell / sub / f"{'pid' if sub == 'pid' else 'ppo'}_episodes.csv").set_index("episode")
        for f in sorted((cell / sub).glob(f"{sub}_episode_*.csv")):
            ep = int(f.stem.rsplit("_", 1)[1])
            e = eps.loc[ep]
            row = {"cell": cell.name, "ctrl": m.group(1), "algo": m.group(1).split("_")[0], "p": float(m.group(2)),
                   "episode": ep, "collision": int(e.collision), "leader_brake": int(e.leader_brake),
                   "end_reason": e.end_reason, "leader_target_speed_mps": float(e.leader_target_speed_mps),
                   "logged_thw_cost": float(e.get("episode_thw_cost", np.nan))}
            row.update(episode_metrics(pd.read_csv(f)))
            rows.append(row)
    return pd.DataFrame(rows)


def md(df, floatfmt=None):
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(lines)


def groups(d):
    """Yield (label, frame): each seed, then the pooled algorithm."""
    for ctrl in sorted(d.ctrl.unique()):
        yield ctrl, d[d.ctrl == ctrl]
    for algo in ("lag", "ppo"):
        if (d.algo == algo).any():
            yield f"{algo} pooled", d[d.algo == algo]


def step_table(d):
    out = []
    for label, g in groups(d):
        n = g.ticks.sum()
        out.append({
            "controller": label, "episodes": len(g), "steps": int(n),
            "r_cont": round(g.r_cont_sum.sum() / n, 2), "r_traf": round(g.r_traf_sum.sum() / n, 3),
            "r_jerk": round(g.r_jerk_sum.sum() / n, 4), "THW cost / step": f"{g.thw_cost_sum.sum() / n:.5f}",
            "THW cost / episode": round(g.thw_cost_sum.mean(), 3),
            "Caution steps / episode": round(g.caution_ticks.mean(), 2),
            "Danger steps / episode": round(g.danger_ticks.mean(), 2),
            "episodes with Danger": int((g.danger_ticks > 0).sum()),
        })
    return pd.DataFrame(out)


def perm_seed_test(a, b):
    """Exact two-sided permutation test on per-seed collision counts (difference of means)."""
    allv = list(a) + list(b)
    obs = abs(np.mean(a) - np.mean(b))
    hits = total = 0
    for idx in itertools.combinations(range(len(allv)), len(a)):
        x = [allv[i] for i in idx]
        y = [allv[i] for i in range(len(allv)) if i not in idx]
        total += 1
        hits += abs(np.mean(x) - np.mean(y)) >= obs - 1e-12
    return hits / total


def paired_scenario_test(d, p, rng):
    """Sign-flip permutation test over the 100 shared scenarios on (PPO-Lag count - PPO count)."""
    g = d[d.p == p]
    lag = g[g.algo == "lag"].groupby("episode").collision.sum()
    ppo = g[g.algo == "ppo"].groupby("episode").collision.sum()
    diff = (lag - ppo).to_numpy()
    obs = abs(diff.sum())
    signs = rng.choice([-1, 1], size=(20000, len(diff)))
    return float((np.abs((signs * diff).sum(axis=1)) >= obs).mean()), int((diff < 0).sum()), int((diff > 0).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("eval_dir", nargs="?", default=str(REPO / "outputs/tunnel/eval"))
    ap.add_argument("--fig-dir", default=str(REPO / "docs/results/figures"))
    args = ap.parse_args()
    d = load(Path(args.eval_dir))
    summary = REPO / "outputs/tunnel/summary"
    summary.mkdir(parents=True, exist_ok=True)
    d.to_csv(summary / "paper_metrics_episodes.csv", index=False)
    learned = d[d.algo != "pid"]

    print("## check: recomputed vs logged THW cost (learned controllers)")
    diff = (learned.thw_cost_sum - learned.logged_thw_cost).abs()
    print(f"episodes {len(learned)}, max |diff| {diff.max():.4f}, episodes with |diff| > 0.01: {(diff > 0.01).sum()}\n")

    for p in (1.0, 0.9):
        print(f"## 1. paper metrics, in-tunnel p = {p}")
        print(md(step_table(learned[learned.p == p])), "\n")

    print("## 2. THW cost and Danger region over p")
    rows = []
    for p in sorted(d.p.unique(), reverse=True):
        for label, g in list(groups(d[d.p == p])):
            n = g.ticks.sum()
            rows.append({"p": p, "controller": label, "episodes": len(g),
                         "THW cost / step": f"{g.thw_cost_sum.sum() / n:.5f}",
                         "THW cost / episode": round(g.thw_cost_sum.mean(), 2),
                         "Danger steps / episode": round(g.danger_ticks.mean(), 2),
                         "Danger s / episode": round(g.danger_ticks.mean() * DT, 3),
                         "of which in tunnel (steps)": round(g.danger_ticks_tunnel.mean(), 2),
                         "episodes with Danger": int((g.danger_ticks > 0).sum()),
                         "Caution steps / episode": round(g.caution_ticks.mean(), 2)})
    print(md(pd.DataFrame(rows)), "\n")

    print("## 3. collision rate, Wilson 95% CI")
    rows = []
    for p in sorted(d.p.unique(), reverse=True):
        for label, g in groups(d[d.p == p]):
            k, n = int(g.collision.sum()), len(g)
            lo, hi = wilson(k, n)
            rows.append({"p": p, "controller": label, "collisions": f"{k}/{n}", "rate": f"{100 * k / n:.1f}%",
                         "Wilson 95%": f"[{100 * lo:.1f}, {100 * hi:.1f}]"})
    print(md(pd.DataFrame(rows)), "\n")

    print("## 3b. PPO-Lag vs PPO")
    from scipy.stats import fisher_exact
    rng = np.random.default_rng(0)
    rows = []
    for p in sorted(learned.p.unique(), reverse=True):
        g = learned[learned.p == p]
        if not (g.algo == "ppo").any():
            continue
        lag = g[g.algo == "lag"].groupby("ctrl").collision.sum().tolist()
        ppo = g[g.algo == "ppo"].groupby("ctrl").collision.sum().tolist()
        fisher = fisher_exact([[sum(lag), 300 - sum(lag)], [sum(ppo), 300 - sum(ppo)]])[1]
        paired, n_lag_better, n_ppo_better = paired_scenario_test(learned, p, rng)
        gb = g[g.leader_brake == 1]
        rows.append({"p": p, "PPO-Lag per seed": lag, "PPO per seed": ppo,
                     "range PPO-Lag": f"{min(lag)}-{max(lag)}", "range PPO": f"{min(ppo)}-{max(ppo)}",
                     "pooled Fisher p": f"{fisher:.3f}", "paired-by-scenario p": f"{paired:.3f}",
                     "scenarios Lag better / PPO better": f"{n_lag_better} / {n_ppo_better}",
                     "seed-level permutation p": f"{perm_seed_test(lag, ppo):.2f}",
                     "brake episodes Lag vs PPO": f"{int(gb[gb.algo == 'lag'].collision.sum())} vs {int(gb[gb.algo == 'ppo'].collision.sum())} of 147"})
    print(md(pd.DataFrame(rows)), "\n")

    print("## 4. time since the last received packet at the collision tick")
    bins = [-0.001, 0.0, 0.5, 1.0, 2.0, 1e9]
    names = ["0 s", "0-0.5 s", "0.5-1 s", "1-2 s", "> 2 s"]
    rows = []
    for p in (0.3, 0.1):
        for algo, label in (("pid", "PID"), ("lag", "PPO-Lag"), ("ppo", "PPO")):
            g = d[(d.p == p) & (d.algo == algo) & (d.collision == 1)]
            if g.empty:
                continue
            x = g.t_since_comm_end_s
            cut = pd.cut(x, bins, labels=names).value_counts().reindex(names)
            row = {"p": p, "controller": label, "collisions": len(g), "in brake episodes": int(g.leader_brake.sum()),
                   "inside tunnel link": int(g.end_in_tunnel.sum()),
                   "median s": round(x.median(), 2), "IQR s": f"{x.quantile(.25):.2f}-{x.quantile(.75):.2f}", "max s": round(x.max(), 2),
                   "median of max in last 3 s": round(g.t_since_comm_max_last3s.median(), 2)}
            row.update({n: int(cut[n]) for n in names})
            rows.append(row)
    print(md(pd.DataFrame(rows)), "\n")
    # Reference: the same quantity over all in-tunnel ticks would need the tick files; the expected
    # value for i.i.d. loss at success probability p is DT * (1 - p) / p.
    print(f"expected time since last packet for i.i.d. loss: p=0.3 {DT * 0.7 / 0.3:.2f} s, p=0.1 {DT * 0.9 / 0.1:.2f} s\n")

    print("## 5. tunnel metrics")
    rows = []
    for p in sorted(d.p.unique(), reverse=True):
        for algo, label in (("pid", "PID"), ("lag", "PPO-Lag"), ("ppo", "PPO")):
            g = d[(d.p == p) & (d.algo == algo)]
            if g.empty:
                continue
            ex = g[g.exited == 1]
            out = ex[ex.outside_band_at_exit == 1]
            rec = out.recover_s.dropna()
            rows.append({"p": p, "controller": label, "episodes": len(g),
                         "|err| in tunnel m": round(g.abs_err_tunnel.mean(), 2),
                         "|err| outside m": round(g.abs_err_outside.mean(), 2),
                         "min THW in tunnel, mean s": round(g.thw_min_tunnel.mean(), 2),
                         "min THW in tunnel, worst s": round(g.thw_min_tunnel.min(), 2),
                         "left the tunnel": len(ex),
                         "outside [-4, 8] m at exit": len(out),
                         "recovery median s": round(rec.median(), 2) if len(rec) else "-",
                         "recovery max s": round(rec.max(), 2) if len(rec) else "-",
                         "never recovered": int(out.recover_s.isna().sum())})
    print(md(pd.DataFrame(rows)), "\n")

    figure(d, Path(args.eval_dir), Path(args.fig_dir))


def figure(d, eval_dir, fig_dir):
    """Representative collision: PPO-Lag, p = 0.1, brake episode, median time-since-packet."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    g = d[(d.p == 0.1) & (d.algo == "lag") & (d.collision == 1) & (d.leader_brake == 1)]
    pick = g.iloc[(g.t_since_comm_end_s - g.t_since_comm_end_s.median()).abs().argsort().iloc[0]]
    t = pd.read_csv(eval_dir / pick.cell / "ppo_eval" / f"ppo_eval_episode_{int(pick.episode):03d}.csv")
    brake_on = t.time_sec[t.leader_brake_event == 1].iloc[0]
    t = t[t.time_sec >= brake_on - 4.0]
    thw = np.where(t.follower_speed_mps > THW_MIN_SPEED, t.distance_m / t.follower_speed_mps.clip(lower=1e-9), np.nan)

    ink, muted, grid = "#1f2328", "#6b7280", "#e5e7eb"
    c_lead, c_fol = "#2a78d6", "#eb6834"
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": grid, "axes.labelcolor": muted, "xtick.color": muted,
                         "ytick.color": muted, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(4, 1, figsize=(7.2, 7.6), sharex=True, gridspec_kw={"height_ratios": [3, 3, 1.6, 3]})
    for a in ax:
        a.grid(axis="y", color=grid, linewidth=0.6)
        a.axvline(brake_on, color=muted, linewidth=0.8, linestyle=":")
        tun = t[t.comm_in_tunnel == 1]
        if len(tun):
            a.axvspan(tun.time_sec.iloc[0], tun.time_sec.iloc[-1], color="#f3f4f6", zorder=0)
    end = t.time_sec.iloc[-1]
    ax[0].plot(t.time_sec, t.leader_speed_mps, color=c_lead, linewidth=2, label="leader")
    ax[0].plot(t.time_sec, t.follower_speed_mps, color=c_fol, linewidth=2, label="follower")
    ax[0].legend(loc="lower left", frameon=False, labelcolor=ink)
    ax[0].set_ylabel("speed (m/s)")
    ax[0].set_ylim(-1, 24)
    ax[0].annotate("leader hard brake starts", (brake_on, 24), xytext=(4, -10), textcoords="offset points", color=muted)
    ax[1].plot(t.time_sec, t.distance_m, color=c_fol, linewidth=2)
    ax[1].axhline(5.0, color=muted, linewidth=0.8, linestyle="--")
    ax[1].annotate("5 m emergency brake", (t.time_sec.iloc[0], 5.0), xytext=(2, 3), textcoords="offset points", color=muted)
    ax[1].set_ylabel("gap (m)")
    ax[1].set_ylim(bottom=0)
    ok = t[t.comm_success == 1]
    ax[2].plot(t.time_sec, t.t_since_comm_s, color=c_fol, linewidth=2)
    ax[2].plot(ok.time_sec, np.zeros(len(ok)), "|", color=ink, markersize=9)
    ax[2].set_ylabel("time since\nlast packet (s)")
    ax[2].annotate("ticks = packet received", (t.time_sec.iloc[0], ax[2].get_ylim()[1]), xytext=(2, -9), textcoords="offset points", color=muted)
    ax[3].plot(t.time_sec, thw, color=c_fol, linewidth=2)
    for y, name, dy in ((TAU_SAFE, "τ_safe 1.2 s", 3), (TAU_DANGER, "τ_danger 1.0 s", -10)):
        ax[3].axhline(y, color=muted, linewidth=0.8, linestyle="--")
        ax[3].annotate(name, (t.time_sec.iloc[len(t) // 3], y), xytext=(2, dy), textcoords="offset points", color=muted)
    ax[3].set_ylim(0, 3)
    ax[3].set_ylabel("THW (s)")
    ax[3].set_xlabel("time in episode (s); shaded = V2V link in tunnel; the trace ends at the collision")
    fig.suptitle(f"Collision under tunnel loss: PPO-Lag {pick.ctrl.split('_')[1]}, in-tunnel p = 0.1, episode {int(pick.episode)}",
                 x=0.01, ha="left", fontsize=10, color=ink)
    fig.tight_layout()
    fig_dir.mkdir(parents=True, exist_ok=True)
    out = fig_dir / "2026-11-17_collision_timeseries.png"
    fig.savefig(out, dpi=160)
    print(f"figure: {out}  ({pick.cell} episode {int(pick.episode)}, t_since_comm at collision {pick.t_since_comm_end_s:.2f} s, "
          f"leader target {pick.leader_target_speed_mps} m/s, brake onset {brake_on:.2f} s, end {end:.2f} s)")


if __name__ == "__main__":
    main()
