"""Summarize the 20-episode cross-eval (outputs/xeval_20). Uses the old near-miss metric (min distance < 5.1 m); superseded by xeval100_summarize.py.
Run from anywhere: python3 analysis/xeval20_summarize.py"""
import os, sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]  # repo root; all paths below are relative to it
os.chdir(REPO); sys.path.insert(0, str(REPO))
import pandas as pd
S = "outputs/xeval_20"
out = {}
for ck in ("run1026", "comm09"):
    for p in ("1.0", "0.9", "0.5"):
        try:
            e = pd.read_csv(f"{S}/{ck}_p{p}/ppo_eval/ppo_episodes.csv")
        except FileNotFoundError:
            continue
        n = lambda c: pd.to_numeric(e[c], errors="coerce")
        out[f"{ck} p={p}"] = {
            "eps": len(e),
            "completed": int((e.end_reason == "profile_completed").sum()),
            "collision": int(e.collision.sum()),
            "mean|err| m": round(e.mean_abs_spacing_error_m.mean(), 3),
            "rmse m": round(e.rmse_spacing_error_m.mean(), 3),
            "|dv| m/s": round(e.mean_abs_relative_velocity_mps.mean(), 3),
            "jerk": round(n("mean_abs_jerk_mps3").mean(), 2),
            "THW_min min": round(n("thw_min_s").min(), 3),
            "THW_min mean": round(n("thw_min_s").mean(), 3),
            "cost/tick": round(e.mean_cost.mean(), 5),
            "ep cost sum": round(e.episode_cost.mean(), 3),
            "eps cost>d(5)": int((e.episode_cost > 5).sum()),
            "min dist": round(e.min_distance_m.min(), 2),
            "dist<5.1 eps": int((e.min_distance_m < 5.1).sum()),
            "comm fail": round(e.comm_failures.sum() / e.comm_ticks.sum(), 3),
            "max consec": int(e.comm_max_consecutive_failures.max()),
        }
pd.set_option("display.width", 250)
print(pd.DataFrame(out).to_string())
