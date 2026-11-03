"""Summarize the 100-episode cross-eval (outputs/xeval_100): entry rate with Wilson CI, paired emcost-comm09 differences, criterion 1.
Run from anywhere: python3 analysis/xeval100_summarize.py"""
import os, sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]  # repo root; all paths below are relative to it
os.chdir(REPO); sys.path.insert(0, str(REPO))
"""Summarize the 100-episode cross-eval. Emergency metric = entry (any tick with distance < 5.0 m,
i.e. an override tick). Near-miss counts are deliberately not reported."""
import sys, numpy as np, pandas as pd
X = "outputs/xeval_100"
CKS = ("run1026", "comm09", "emcost"); PS = ("1.0", "0.9", "0.5")

def wilson(k, n, z=1.96):
    if n == 0: return (np.nan, np.nan)
    p = k / n; d = 1 + z*z/n; c = (p + z*z/(2*n)) / d; h = z*np.sqrt(p*(1-p)/n + z*z/(4*n*n)) / d
    return (max(0, c-h), min(1, c+h))

def load(ck, p):
    try: e = pd.read_csv(f"{X}/{ck}_p{p}/ppo_eval/ppo_episodes.csv")
    except FileNotFoundError: return None
    # Entry = the logged override flag (unrounded distance < 5.0 m). min_distance_m in the CSV is
    # rounded to 1 mm, so a 4.9996 m entry shows as 5.0; check agreement only up to that rounding.
    e["entered"] = e.emergency_entered.astype(int)
    assert ((e.min_distance_m <= 5.0005) == (e.entered == 1)).all(), f"{ck} p={p}: entry flag mismatch"
    for c in ("mean_abs_jerk_mps3", "thw_min_s"): e[c] = pd.to_numeric(e[c], errors="coerce")
    return e

D = {(ck, p): load(ck, p) for ck in CKS for p in PS}
rows = {}
for (ck, p), e in D.items():
    if e is None: continue
    k, n = int(e.entered.sum()), len(e); lo, hi = wilson(k, n)
    rows[f"{ck} p={p}"] = {
        "eps": n, "completed": int((e.end_reason == "profile_completed").sum()), "collision": int(e.collision.sum()),
        "entry": f"{k}/{n} ({100*k/n:.0f}%, CI {100*lo:.0f}-{100*hi:.0f})",
        "mean|err| m": round(e.mean_abs_spacing_error_m.mean(), 3), "rmse m": round(e.rmse_spacing_error_m.mean(), 3),
        "|dv| m/s": round(e.mean_abs_relative_velocity_mps.mean(), 3), "jerk": round(e.mean_abs_jerk_mps3.mean(), 2),
        "THW_min min": round(e.thw_min_s.min(), 3), "THW_min mean": round(e.thw_min_s.mean(), 3),
        "THW cost/ep": round(e.episode_thw_cost.mean(), 3), "eps THW cost>5": int((e.episode_thw_cost > 5).sum()),
        "min dist": round(e.min_distance_m.min(), 2),
        "comm fail": round(e.comm_failures.sum() / e.comm_ticks.sum(), 3), "max consec": int(e.comm_max_consecutive_failures.max()),
    }
pd.set_option("display.width", 300); pd.set_option("display.max_columns", 50)
print(pd.DataFrame(rows).to_string())

def paired(a, b, col):
    """Mean of b-a over episodes (same leader profile and comm pattern per index), 95% bootstrap CI."""
    d = (b[col].values - a[col].values); rng = np.random.default_rng(0)
    bs = [rng.choice(d, len(d)).mean() for _ in range(5000)]
    return d.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5), 100*d.mean()/a[col].mean()

print("\nPaired differences, emcost - comm09 (same episode index = same leader speed and comm pattern):")
for p in PS:
    a, b = D[("comm09", p)], D[("emcost", p)]
    if a is None or b is None: continue
    assert len(a) == len(b)
    for col in ("mean_abs_spacing_error_m", "mean_abs_jerk_mps3", "thw_min_s"):
        m, lo, hi, pct = paired(a, b, col)
        print(f"  p={p} {col:26s} {m:+.3f} (95% CI {lo:+.3f}..{hi:+.3f}, {pct:+.1f}%)")

e = D[("emcost", "0.9")]
if e is not None:
    k, n = int(e.entered.sum()), len(e); lo, hi = wilson(k, n)
    print(f"\nCriterion 1 (emcost p=0.9 entry <= 10%): {k}/{n} = {100*k/n:.0f}% (95% CI {100*lo:.0f}-{100*hi:.0f}%) -> {'PASS' if k/n <= 0.10 else 'FAIL'}")
