"""Recompute episode costs from training tick logs under alternative emergency-cost definitions and replay the lambda update with the policy held fixed.
Run from anywhere: python3 analysis/cost_replay.py"""
import os, sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]  # repo root; all paths below are relative to it
os.chdir(REPO); sys.path.insert(0, str(REPO))
import pandas as pd, glob, numpy as np
R="outputs"
D_TICK=5/650; ETA=0.01*650
def thw_cost(dist,v):
    thw=np.where(v>0.5, dist/np.maximum(v,1e-9), np.inf)
    c=np.where(thw>=1.2,0.0,np.where(thw<=1.0,1.0,((1.2-thw)/0.2)**2)); return c
for run in ["run_comm09_3000","run1026_3000"]:
    E=pd.read_csv(f"{R}/{run}/ppo/ppo_episodes.csv")
    rows=[]
    for _,e in E.iterrows():
        d=pd.read_csv(f"{R}/{run}/ppo/ppo_episode_{int(e.episode):03d}.csv",usecols=["distance_m","follower_speed_mps"])
        c=thw_cost(d.distance_m.values,d.follower_speed_mps.values); em=d.distance_m.values<5.0
        rows.append(dict(ep=e.episode,upd=e.updates_so_far,ticks=len(d),thw=c.sum(),logged=e.episode_cost,
                         A=np.maximum(c,em).sum(),entry=int(em.any()),emt=em.sum()))
    r=pd.DataFrame(rows)
    print(f"===== {run}: THW cost recompute vs logged episode_cost max abs diff {np.abs(r.thw-r.logged).max():.4f}")
    variants={"current (THW)":r.thw,"A tick-unit":r.A,"B K=50 (r=0.1)":r.thw+50*r.entry,"B K=25 (r=0.2)":r.thw+25*r.entry}
    r["phase"]=((r.ep-1)//500)*500+1
    out={}
    for name,S in variants.items():
        r["c"]=S/r.ticks
        w=r.groupby("upd").c.mean()     # lambda input per update window
        lam=0;lams=[]
        for m in w.values: lam=max(0,lam+ETA*(m-D_TICK)); lams.append(lam)
        lam_s=pd.Series(lams,index=w.index)
        ph=r.groupby("upd").phase.first()
        out[name]=pd.DataFrame({"windows>d %":100*(w>D_TICK).groupby(ph).mean(),"replay λ end":lam_s.groupby(ph).last(),"replay λ max":lam_s.groupby(ph).max()})
        r[name]=S
    print("per-episode cost sum (median / 90th pct / max) and episodes over d (per-tick basis):")
    for name in variants:
        S=r[name]; over=(S/r.ticks>D_TICK).mean()*100
        print(f"  {name:16s} median {S.median():6.2f}  p90 {S.quantile(.9):6.2f}  max {S.max():7.2f}  eps over d {over:5.1f}%")
    print("emergency entry rate by phase:", (r.groupby("phase").entry.mean()*100).round(1).to_dict())
    print(pd.concat(out,axis=1).round(2).to_string())
