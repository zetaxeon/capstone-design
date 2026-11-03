"""Compare training logs of run1026_3000 and run_comm09_3000 (first completion, lambda, 500-episode phases, last 50 episodes).
Run from anywhere: python3 analysis/compare_runs.py"""
import os, sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]  # repo root; all paths below are relative to it
os.chdir(REPO); sys.path.insert(0, str(REPO))
import sys
import pandas as pd
ROOT = "outputs"
RUNS = {"run1026 (p=1.0, 4d)": f"{ROOT}/run1026_3000/ppo", "comm09 (p=0.9, 7d)": f"{ROOT}/run_comm09_3000/ppo"}

def num(s):
    return pd.to_numeric(s, errors="coerce")

def window_stats(t):
    return {
        "eps": len(t),
        "completed": int((t.end_reason == "profile_completed").sum()),
        "collision": int(t.collision.sum()),
        "hard_viol": int((t.end_reason == "hard_violation").sum()),
        "mean|err| m": round(t.mean_abs_spacing_error_m.mean(), 3),
        "rmse m": round(t.rmse_spacing_error_m.mean(), 3),
        "jerk m/s3": round(num(t.mean_abs_jerk_mps3).mean(), 2),
        "THW_min s (min)": round(num(t.thw_min_s).min(), 3),
        "THW_min s (mean)": round(num(t.thw_min_s).mean(), 3),
        "cost/tick": round(t.mean_cost.mean(), 5),
        "min dist<5.1m": int((t.min_distance_m < 5.1).sum()),
        "mean action": round(t.mean_action.mean(), 3),
    }

rows, summary = {}, {}
for name, path in RUNS.items():
    e = pd.read_csv(f"{path}/ppo_episodes.csv")
    u = pd.read_csv(f"{path}/ppo_updates.csv")
    rows[name] = window_stats(e.tail(50))
    comp = e[e.end_reason == "profile_completed"].episode
    lam_peak = u["lambda"].max()
    peak_row = u.loc[u["lambda"].idxmax()]
    after = u[u["update"] > peak_row["update"]]
    zero = after[after["lambda"] == 0]
    roll = (e.end_reason == "profile_completed").rolling(50).mean()
    first90 = e.episode[roll >= 0.9]
    summary[name] = {
        "episodes": len(e),
        "updates": int(u["update"].max()),
        "sim time h": round(e.sim_time_sec.sum() / 3600, 2),
        "first completion ep": int(comp.iloc[0]) if len(comp) else None,
        "first ep with 50-ep completion>=90%": int(first90.iloc[0]) if len(first90) else None,
        "lambda peak (ep)": f"{lam_peak:.3f} ({int(peak_row['episode'])})",
        "lambda back to 0 at ep": int(zero.episode.iloc[0]) if len(zero) else None,
        "final lambda": round(u["lambda"].iloc[-1], 4),
        "final log_std": round(u.log_std.iloc[-1], 3),
        "total collisions": int(e.collision.sum()),
    }
    if "comm_failures" in e:
        summary[name]["comm fail rate"] = round(e.comm_failures.sum() / e.comm_ticks.sum(), 4)
        summary[name]["max consecutive fail"] = int(e.comm_max_consecutive_failures.max())
    for lo, hi in [(1, 500), (501, 1000), (1001, 1500), (1501, 2000), (2001, 2500), (2501, 3000)]:
        t = e[(e.episode >= lo) & (e.episode <= hi)]
        rows.setdefault(f"{name} {lo}-{hi}", {}).update(
            {"done%": round(100 * (t.end_reason == "profile_completed").mean(), 1),
             "coll": int(t.collision.sum()),
             "|err|": round(t.mean_abs_spacing_error_m[t.end_reason == "profile_completed"].mean(), 2),
             "jerk": round(num(t.mean_abs_jerk_mps3[t.end_reason == "profile_completed"]).mean(), 2)})

pd.set_option("display.width", 200)
print("== Summary"); print(pd.DataFrame(summary).to_string())
print("\n== Last 50 episodes"); print(pd.DataFrame({k: v for k, v in rows.items() if k in RUNS}).to_string())
print("\n== By phase (|err|, jerk over completed episodes only)")
print(pd.DataFrame({k: v for k, v in rows.items() if k not in RUNS}).T.to_string())
