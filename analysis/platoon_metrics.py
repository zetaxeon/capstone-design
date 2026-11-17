#!/usr/bin/env python3
"""Metrics of the three-vehicle platoon evaluation (leader, F1, F2), from the logs only.

Usage: .venv/bin/python analysis/platoon_metrics.py [EVAL3_DIR]
EVAL3_DIR defaults to outputs/tunnel/eval3 (cells <controller>_t<p>/ written by platoon_eval.sh).
Prints markdown tables and writes outputs/tunnel/summary/platoon_episodes.csv (one row per
cell and episode).

Collisions come from <cell>/platoon_episodes.csv (which pair touched). An episode ends at the
first collision, so an F1-leader collision removes F2's chance to collide in that episode; the
F2 rate is therefore also given over the episodes without an F1-leader collision.

Per-follower step metrics reuse tunnel_paper_metrics.episode_metrics (THW cost eq. 15, Caution
1.0 < THW < 1.2 s, Danger THW <= 1.0 s, in-tunnel |spacing error|). In F2's files the "leader"
columns describe its predecessor, F1.

Amplification, for each stage (F1 / leader and F2 / F1), follower over predecessor. Acceleration
is taken from the logged speeds, a(t) = (v(t + 0.25 s) - v(t - 0.25 s)) / 0.5 s, because the
simulator's own acceleration reading spikes to 20-45 m/s^2 for a tick when a vehicle comes to rest.
  tunnel   RMS of a over the steps the follower is inside the tunnel, pooled over episodes:
           sqrt(sum a_f^2 / sum a_pred^2); also split by episodes with / without a leader hard brake
  brake    peak deceleration (largest -a) in the 6 s after the leader's hard brake starts
           (hard-brake episodes only; median of the per-episode ratios)
A ratio above 1 means the disturbance grows toward the rear.
"""
import math
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tunnel_paper_metrics import DT, episode_metrics, md, wilson  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
BRAKE_WINDOW_SEC = 6.0
ACCEL_HALF_TICKS = 5      # central difference over +-0.25 s
STEP_KEYS = ("ticks", "thw_cost_sum", "caution_ticks", "danger_ticks", "abs_err_tunnel", "abs_err_outside",
             "thw_min_tunnel", "r_cont_sum", "r_traf_sum", "r_jerk_sum")


def speed_accel(v):
    """Central-difference acceleration from a speed series; NaN at the first and last 0.25 s."""
    a = np.full(len(v), np.nan)
    h = ACCEL_HALF_TICKS
    if len(v) > 2 * h:
        a[h:-h] = (v[2 * h:] - v[:-2 * h]) / (2 * h * DT)
    return a


def amplification(t):
    """Tunnel RMS sums and brake-window peak decelerations for one follower's tick file."""
    acc_f = speed_accel(t.follower_speed_mps.to_numpy())
    acc_p = speed_accel(t.leader_speed_mps.to_numpy())      # predecessor's speed
    ok = ~np.isnan(acc_f)
    inside = (t.follower_in_tunnel.to_numpy() == 1) & ok
    out = {
        "tun_sq_f": float((acc_f[inside] ** 2).sum()), "tun_sq_p": float((acc_p[inside] ** 2).sum()),
        "tun_n": int(inside.sum()), "peak_dec_f": np.nan, "peak_dec_p": np.nan,
    }
    brake = np.flatnonzero(t.leader_brake_event.to_numpy() == 1)
    if len(brake):
        t0 = t.time_sec.iloc[brake[0]]
        win = ((t.time_sec >= t0) & (t.time_sec <= t0 + BRAKE_WINDOW_SEC)).to_numpy() & ok
        if win.any():
            out["peak_dec_f"] = float(-acc_f[win].min())
            out["peak_dec_p"] = float(-acc_p[win].min())
    return out


def load(eval_dir):
    rows = []
    for cell in sorted(p for p in eval_dir.iterdir() if p.is_dir()):
        m = re.fullmatch(r"(pid|lag_s\d+|ppo_s\d+)_t([\d.]+)", cell.name)
        if not m or not (cell / "platoon_episodes.csv").exists():
            continue
        sub = "pid" if m.group(1) == "pid" else "ppo_eval"
        platoon = pd.read_csv(cell / "platoon_episodes.csv")
        for _, e in platoon.iterrows():
            ep = int(e.episode)
            row = {"cell": cell.name, "ctrl": m.group(1), "algo": m.group(1).split("_")[0], "p": float(m.group(2))}
            row.update(e.to_dict())
            for tag, folder in (("f1", cell / sub), ("f2", cell / "f2" / sub)):
                t = pd.read_csv(folder / f"{sub}_episode_{ep:03d}.csv")
                em = episode_metrics(t)
                row.update({f"{tag}_{k}": em[k] for k in STEP_KEYS})
                row.update({f"{tag}_{k}": v for k, v in amplification(t).items()})
                row[f"{tag}_max_abs_err"] = float(t.spacing_error_m.abs().max())
            rows.append(row)
    return pd.DataFrame(rows)


def groups(d):
    order = [("pid", "PID"), ("lag", "PPO-Lag"), ("ppo", "PPO")]
    for algo, label in order:
        g = d[d.algo == algo]
        if len(g):
            yield label, g


def per_seed(g, fn, fmt="{:.0f}"):
    seeds = sorted(g.ctrl.unique())
    if len(seeds) < 2:
        return ""
    return " / ".join(fmt.format(fn(g[g.ctrl == s])) for s in seeds)


def pct(k, n):
    lo, hi = wilson(k, n)
    return f"{100 * k / n:.1f}% [{100 * lo:.1f}, {100 * hi:.1f}]"


def main():
    eval_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO / "outputs/tunnel/eval3"
    d = load(eval_dir)
    out = REPO / "outputs/tunnel/summary"
    out.mkdir(parents=True, exist_ok=True)
    d.to_csv(out / "platoon_episodes.csv", index=False)
    ps = sorted(d.p.unique(), reverse=True)
    print(f"cells: {d.cell.nunique()}, episodes per cell: {sorted(d.groupby('cell').size().unique())}\n")

    print("## end reasons")
    print(md(d.groupby(["algo", "p", "end_reason"]).size().unstack(fill_value=0).reset_index()), "\n")

    print("## 1. collisions")
    rows = []
    for p in ps:
        for label, g in groups(d[d.p == p]):
            n = len(g)
            no_f1 = g[g.collision_f1_leader == 0]
            rows.append({
                "p": p, "controller": label, "episodes": n,
                "platoon collision": pct(int(g.platoon_collision.sum()), n),
                "per seed": per_seed(g, lambda x: x.platoon_collision.sum()),
                "F1-leader": int(g.collision_f1_leader.sum()),
                "F2-F1": int(g.collision_f2_f1.sum()),
                "other": int(g.collision_other.sum()),
                "F2-F1 given no F1-leader": f"{int(no_f1.collision_f2_f1.sum())}/{len(no_f1)} = {pct(int(no_f1.collision_f2_f1.sum()), len(no_f1))}",
                "per seed ": per_seed(g, lambda x: x[x.collision_f1_leader == 0].collision_f2_f1.sum()),
                "brake episodes: platoon collisions": f"{int(g[g.leader_brake == 1].platoon_collision.sum())}/{int((g.leader_brake == 1).sum())}",
                "emergency entry F1 / F2": f"{int(g.f1_emergency_entered.sum())} / {int(g.f2_emergency_entered.sum())}",
            })
    print(md(pd.DataFrame(rows)), "\n")

    print("## 2. safety and tracking per follower (mean per episode)")
    rows = []
    for p in ps:
        for label, g in groups(d[d.p == p]):
            row = {"p": p, "controller": label}
            for tag in ("f1", "f2"):
                row[f"{tag} THW cost"] = round(g[f"{tag}_thw_cost_sum"].mean(), 2)
                row[f"{tag} Danger steps"] = round(g[f"{tag}_danger_ticks"].mean(), 2)
                row[f"{tag} Caution steps"] = round(g[f"{tag}_caution_ticks"].mean(), 2)
                row[f"{tag} episodes with Danger"] = int((g[f"{tag}_danger_ticks"] > 0).sum())
                row[f"{tag} |err| in tunnel m"] = round(g[f"{tag}_abs_err_tunnel"].mean(), 2)
                row[f"{tag} min THW in tunnel (worst) s"] = round(g[f"{tag}_thw_min_tunnel"].min(), 2)
            row["f2 THW cost per seed"] = per_seed(g, lambda x: x.f2_thw_cost_sum.mean(), "{:.2f}")
            row["f2 |err| tunnel per seed"] = per_seed(g, lambda x: x.f2_abs_err_tunnel.mean(), "{:.2f}")
            rows.append(row)
    print(md(pd.DataFrame(rows)), "\n")

    print("## 3. amplification (follower / predecessor)")
    rows = []
    for p in ps:
        for label, g in groups(d[d.p == p]):
            row = {"p": p, "controller": label}
            for tag, name in (("f1", "F1/leader"), ("f2", "F2/F1")):
                pooled = lambda x: math.sqrt(x[f"{tag}_tun_sq_f"].sum() / x[f"{tag}_tun_sq_p"].sum())  # noqa: E731
                b = g[(g.leader_brake == 1) & g[f"{tag}_peak_dec_f"].notna() & (g[f"{tag}_peak_dec_p"] > 0.5)]
                ratio = b[f"{tag}_peak_dec_f"] / b[f"{tag}_peak_dec_p"]
                row[f"{name} tunnel RMS accel"] = round(pooled(g), 2)
                row[f"{name} per seed"] = per_seed(g, pooled, "{:.2f}")
                for flag, word in ((1, "brake"), (0, "no-brake")):
                    sub = g[g.leader_brake == flag]
                    row[f"{name} tunnel RMS, {word} episodes"] = round(pooled(sub), 2) if len(sub) else ""
                row[f"{name} brake peak decel (median)"] = round(ratio.median(), 2) if len(ratio) else ""
                row[f"{name} brake ratio > 1"] = f"{int((ratio > 1).sum())}/{len(ratio)}"
                row[f"{name} peak decel follower m/s2 (median)"] = round(b[f"{tag}_peak_dec_f"].median(), 2) if len(b) else ""
            rows.append(row)
    print(md(pd.DataFrame(rows)), "\n")

    print("## 3b. amplification, episodes without any collision only")
    rows = []
    for p in ps:
        for label, g in groups(d[(d.p == p) & (d.platoon_collision == 0)]):
            row = {"p": p, "controller": label, "episodes": len(g)}
            for tag, name in (("f1", "F1/leader"), ("f2", "F2/F1")):
                row[f"{name} tunnel RMS accel"] = round(math.sqrt(g[f"{tag}_tun_sq_f"].sum() / g[f"{tag}_tun_sq_p"].sum()), 2)
                nb = g[g.leader_brake == 0]
                row[f"{name} tunnel RMS, no-brake episodes"] = round(math.sqrt(nb[f"{tag}_tun_sq_f"].sum() / nb[f"{tag}_tun_sq_p"].sum()), 2) if len(nb) else ""
                b = g[(g.leader_brake == 1) & g[f"{tag}_peak_dec_f"].notna() & (g[f"{tag}_peak_dec_p"] > 0.5)]
                ratio = b[f"{tag}_peak_dec_f"] / b[f"{tag}_peak_dec_p"]
                row[f"{name} brake peak decel (median)"] = round(ratio.median(), 2) if len(ratio) else ""
            rows.append(row)
    print(md(pd.DataFrame(rows)), "\n")

    print("## 3c. tunnel RMS acceleration per vehicle (m/s^2), by leader hard brake")
    rows = []
    for p in ps:
        for label, g in groups(d[d.p == p]):
            row = {"p": p, "controller": label}
            for flag, word in ((0, "no brake"), (1, "brake")):
                sub = g[g.leader_brake == flag]
                if not len(sub):
                    continue
                row[f"{word}: leader"] = round(math.sqrt(sub.f1_tun_sq_p.sum() / sub.f1_tun_n.sum()), 2)
                row[f"{word}: F1"] = round(math.sqrt(sub.f1_tun_sq_f.sum() / sub.f1_tun_n.sum()), 2)
                row[f"{word}: F2"] = round(math.sqrt(sub.f2_tun_sq_f.sum() / sub.f2_tun_n.sum()), 2)
                row[f"{word}: F2 per seed"] = per_seed(sub, lambda x: math.sqrt(x.f2_tun_sq_f.sum() / x.f2_tun_n.sum()), "{:.2f}")
            rows.append(row)
    print(md(pd.DataFrame(rows).fillna("")), "\n")

    print("## 4. F1 against the two-vehicle evaluation (same seed, same leader, same L->F1 packets)")
    rows = []
    for cell, g in d.groupby("cell"):
        ctrl, p = g.ctrl.iloc[0], g.p.iloc[0]
        if ctrl == "pid":
            two = REPO / f"outputs/tunnel/pid_retune/kp0.4_ki0.01_kd0.6_t{p}/pid/pid_episodes.csv"
        else:
            two = REPO / f"outputs/tunnel/eval/{cell}/ppo_eval/ppo_episodes.csv"
        if not two.exists():
            continue
        t = pd.read_csv(two).set_index("episode").collision
        f1 = g.set_index("episode").collision_f1_leader
        both = pd.concat([t.rename("two"), f1.rename("three")], axis=1).dropna()
        rows.append({"cell": cell, "2-vehicle collisions": int(both.two.sum()), "3-vehicle F1-leader": int(both.three.sum()),
                     "same episodes": int((both.two == both.three).sum()),
                     "2-vehicle only": int(((both.two == 1) & (both.three == 0)).sum()),
                     "3-vehicle only": int(((both.two == 0) & (both.three == 1)).sum())})
    print(md(pd.DataFrame(rows)), "\n")

    print("## 6. the six evaluation metrics per follower")
    rows = []
    for p in ps:
        for label, g in groups(d[d.p == p]):
            for tag in ("f1", "f2"):
                n = g[f"{tag}_ticks"].sum()
                rows.append({
                    "p": p, "controller": label, "follower": tag.upper(),
                    "r_cont / step": round(g[f"{tag}_r_cont_sum"].sum() / n, 2),
                    "r_traf / step": round(g[f"{tag}_r_traf_sum"].sum() / n, 2),
                    "r_jerk / step": round(g[f"{tag}_r_jerk_sum"].sum() / n, 2),
                    "THW cost / step": f"{g[f'{tag}_thw_cost_sum'].sum() / n:.5f}",
                    "THW cost / episode": round(g[f"{tag}_thw_cost_sum"].mean(), 2),
                    "Caution steps / episode": round(g[f"{tag}_caution_ticks"].mean(), 2),
                    "Danger steps / episode": round(g[f"{tag}_danger_ticks"].mean(), 2),
                    "steps / episode": round(g[f"{tag}_ticks"].mean()),
                    "r_cont per seed": per_seed(g, lambda x: x[f"{tag}_r_cont_sum"].sum() / x[f"{tag}_ticks"].sum(), "{:.2f}"),
                    "r_jerk per seed": per_seed(g, lambda x: x[f"{tag}_r_jerk_sum"].sum() / x[f"{tag}_ticks"].sum(), "{:.2f}"),
                })
    print(md(pd.DataFrame(rows)), "\n")

    print("## 5. PPO-Lag vs PPO, platoon collisions")
    import itertools
    rows = []
    for p in ps:
        g = d[d.p == p]
        lag = g[g.algo == "lag"].groupby("ctrl").platoon_collision.sum().tolist()
        ppo = g[g.algo == "ppo"].groupby("ctrl").platoon_collision.sum().tolist()
        if len(lag) != 3 or len(ppo) != 3:
            continue
        allv = lag + ppo
        obs = abs(np.mean(lag) - np.mean(ppo))
        combos = list(itertools.combinations(range(6), 3))
        hits = sum(abs(np.mean([allv[i] for i in c]) - np.mean([allv[i] for i in range(6) if i not in c])) >= obs - 1e-12 for c in combos)
        a = g[g.algo == "lag"].groupby("episode").platoon_collision.sum()
        b = g[g.algo == "ppo"].groupby("episode").platoon_collision.sum()
        diff = (a - b).to_numpy()
        signs = np.random.default_rng(0).choice([-1, 1], size=(20000, len(diff)))
        paired = float((np.abs((signs * diff).sum(axis=1)) >= abs(diff.sum())).mean())
        rows.append({"p": p, "PPO-Lag per seed": lag, "PPO per seed": ppo, "seed-level permutation p": f"{hits / len(combos):.2f}",
                     "paired-by-scenario p": f"{paired:.3f}", "scenarios Lag better / PPO better": f"{int((diff < 0).sum())} / {int((diff > 0).sum())}"})
    print(md(pd.DataFrame(rows)), "\n")


if __name__ == "__main__":
    main()
