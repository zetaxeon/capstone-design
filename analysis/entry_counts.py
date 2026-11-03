"""Emergency-brake entry counts (ad-hoc analyses from 2026-11-03, collected here).

1. 20-episode cross-eval (outputs/xeval_20): entry (distance < 5.0 m, equals the override
   signature brake=1/throttle=0) vs the old near-miss count (min distance < 5.1 m).
2. 100-episode cross-eval (outputs/xeval_100): paired McNemar test, comm09 vs emcost, per p.
3. Training logs: entry rate per 500-episode phase, lambda, last 50 episodes, comm09 vs emcost.
Run from anywhere: python3 analysis/entry_counts.py
"""
import os, sys, glob
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]
os.chdir(REPO); sys.path.insert(0, str(REPO))
import pandas as pd
from scipy.stats import binomtest

print("== 1. 20-episode cross-eval: entry vs near-miss")
for cell in sorted(glob.glob("outputs/xeval_20/*_p*/")):
    entry = near = sig = 0
    for f in sorted(glob.glob(f"{cell}ppo_eval/ppo_eval_episode_*.csv")):
        d = pd.read_csv(f); e = d.distance_m < 5.0
        entry += e.any(); near += (d.distance_m < 5.1).any()
        sig += ((d.brake == 1.0) & (d.throttle == 0) & e).any()
    print(f"  {Path(cell).name:13s} entry {entry:2d}/20  near {near:2d}/20  override-signature {sig:2d}/20")

print("== 2. 100-episode cross-eval: paired entry, comm09 vs emcost (entry = logged emergency_entered)")
for p in ("1.0", "0.9", "0.5"):
    a = pd.read_csv(f"outputs/xeval_100/comm09_p{p}/ppo_eval/ppo_episodes.csv")
    b = pd.read_csv(f"outputs/xeval_100/emcost_p{p}/ppo_eval/ppo_episodes.csv")
    x = int(((a.emergency_entered == 1) & (b.emergency_entered == 0)).sum())
    y = int(((a.emergency_entered == 0) & (b.emergency_entered == 1)).sum())
    pv = binomtest(x, x + y).pvalue if x + y else 1.0
    print(f"  p={p}: comm09 {a.emergency_entered.sum()}/100, emcost {b.emergency_entered.sum()}/100, "
          f"only comm09 {x}, only emcost {y}, McNemar exact p={pv:.3f}")

print("== 3. Training logs")
for r in ("run_comm09_3000", "run_emcost_3000"):
    e = pd.read_csv(f"outputs/{r}/ppo/ppo_episodes.csv"); u = pd.read_csv(f"outputs/{r}/ppo/ppo_updates.csv")
    e["ent"] = e.emergency_entered if "emergency_entered" in e else (e.min_distance_m < 5.0).astype(int)
    e["ph"] = ((e.episode - 1) // 500) * 500 + 1
    nz = u[u["lambda"] > 0]
    t = e.tail(50); c = t[t.end_reason == "profile_completed"]
    print(f"  {r}: updates {len(u)}, lambda peak {u['lambda'].max():.2f}, last lambda>0 at ep "
          f"{int(nz.episode.iloc[-1]) if len(nz) else None}, collisions {e.collision.sum()}, entries {e.ent.sum()}")
    print("    entry % by phase:", (100 * e.groupby("ph").ent.mean()).round(1).to_dict())
    print(f"    last 50: completed {len(c)}, |err| {c.mean_abs_spacing_error_m.mean():.3f}, "
          f"jerk {pd.to_numeric(c.mean_abs_jerk_mps3).mean():.2f}, THW_min min {pd.to_numeric(c.thw_min_s).min():.3f}")
