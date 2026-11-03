"""Policy brake usage and emergency-override tick share per 500-episode phase in training logs.
Run from anywhere: python3 analysis/brake_usage.py"""
import os, sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]  # repo root; all paths below are relative to it
os.chdir(REPO); sys.path.insert(0, str(REPO))
import pandas as pd, glob, numpy as np
R="outputs"
for run in ["run1026_3000","run_comm09_3000"]:
    rows=[]
    for fn in sorted(glob.glob(f"{R}/{run}/ppo/ppo_episode_*.csv")):
        d=pd.read_csv(fn, usecols=["leader_speed_mps","follower_speed_mps","distance_m","throttle","brake"])
        ep=int(fn[-8:-4]) if fn[-8:-4].isdigit() else int(fn.split("_")[-1][:-4])
        em=d.distance_m<5.0
        pol=~em
        mv=np.flatnonzero(d.leader_speed_mps.values>0.3); i0=(mv[-1]+1) if len(mv) else 0
        stop=pol & (d.index>=i0)
        rows.append(dict(ep=ep, ticks=len(d), pol=pol.sum(), pol_brake=(pol&(d.brake>0)).sum(),
            pol_brake_strong=(pol&(d.brake>=0.3)).sum(), em=em.sum(),
            stop=stop.sum(), stop_brake=(stop&(d.brake>0)).sum(),
            em_ep=int(em.any())))
    r=pd.DataFrame(rows); r["phase"]=((r.ep-1)//500)*500+1
    g=r.groupby("phase").sum(numeric_only=True)
    out=pd.DataFrame({"policy brake tick %":100*g.pol_brake/g.pol,"brake>=0.3 %":100*g.pol_brake_strong/g.pol,
        "stop-phase brake %":100*g.stop_brake/g.stop,"emergency tick %":100*g.em/g.ticks,"eps w/ emergency":g.em_ep})
    t=r.sum(numeric_only=True)
    print(f"== {run}: {len(r)} eps")
    print(out.round(2).to_string())
    print("TOTAL policy brake %.2f%% | stop-phase brake %.2f%% | emergency ticks %.2f%% | eps w/ emergency %d"%(100*t.pol_brake/t.pol,100*t.stop_brake/t.stop,100*t.em/t.ticks,t.em_ep))
