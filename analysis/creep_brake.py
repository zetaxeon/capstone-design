"""Brake vs throttle share in the post-stop creep phase (episodes 1501-3000) of training logs.
Run from anywhere: python3 analysis/creep_brake.py"""
import os, sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]  # repo root; all paths below are relative to it
os.chdir(REPO); sys.path.insert(0, str(REPO))
import pandas as pd, glob, numpy as np
R="outputs"
for run in ["run1026_3000","run_comm09_3000"]:
    rows=[]
    for fn in sorted(glob.glob(f"{R}/{run}/ppo/ppo_episode_*.csv")):
        ep=int(fn.split("_")[-1][:-4])
        if ep<=1500: continue
        d=pd.read_csv(fn, usecols=["leader_speed_mps","follower_speed_mps","distance_m","throttle","brake"])
        mv=np.flatnonzero(d.leader_speed_mps.values>0.3)
        if not len(mv): continue
        i0=mv[-1]+1
        st=np.flatnonzero((np.arange(len(d))>=i0)&(d.follower_speed_mps.values<0.1))
        if not len(st): continue
        c=d.iloc[st[0]:]; c=c[c.distance_m>=5.0]
        rows.append(dict(ep=ep,n=len(c),br=(c.brake>0).sum(),thr=(c.throttle>0).sum(),
            em=int((d.distance_m.iloc[st[0]:]<5.0).any())))
    r=pd.DataFrame(rows); t=r.sum()
    print(f"{run} eps 1501-3000 ({len(r)} eps): creep phase (after follower's first stop) policy brake {100*t.br/t.n:.1f}% / throttle {100*t.thr/t.n:.1f}% of ticks; eps with any creep brake {(r.br>0).sum()}; emergency eps {t.em}")
