"""Progress check of run_emcost_3000: episodes, last-100 entry rate, lambda.
Run from anywhere: python3 analysis/status_emcost.py"""
import os, sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]  # repo root; all paths below are relative to it
os.chdir(REPO); sys.path.insert(0, str(REPO))
import pandas as pd, re, subprocess, datetime
R="outputs/run_emcost_3000"
e=pd.read_csv(f"{R}/ppo/ppo_episodes.csv"); u=pd.read_csv(f"{R}/ppo/ppo_updates.csv")
log=open(f"{R}/train.log").read()
last=e.tail(100)
print(datetime.datetime.now().strftime("%T"), "| episodes", len(e), "| updates", len(u),
      "| errors", len(re.findall(r"(?i)traceback|error", log)),
      "| alive", subprocess.run(["ps","-p","60909"],capture_output=True).returncode==0)
print(f"last 100 eps: entry {last.emergency_entered.sum()}/{len(last)}, completed {(last.end_reason=='profile_completed').sum()}, collisions {last.collision.sum()}, "
      f"end reasons {last.end_reason.value_counts().to_dict()}")
done=last[last.end_reason=="profile_completed"]
if len(done): print(f"  completed eps: |err| {done.mean_abs_spacing_error_m.mean():.2f} m, jerk {pd.to_numeric(done.mean_abs_jerk_mps3).mean():.1f}, THW_min {pd.to_numeric(done.thw_min_s).min():.2f} s")
print(f"lambda now {u['lambda'].iloc[-1]:.3f} (peak {u['lambda'].max():.3f} at update {int(u.loc[u['lambda'].idxmax(),'update'])}), masked last 10 updates {u.masked_transitions.tail(10).sum()}")
c=e[e.end_reason=="profile_completed"]
print("first completion ep:", int(c.episode.iloc[0]) if len(c) else None, "| total entries", e.emergency_entered.sum())
