"""Final-stop phase per cross-eval cell (outputs/xeval_20): distance and speed at leader stop, creep throttle, emergency timing.
Run from anywhere: python3 analysis/stopphase.py"""
import os, sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]  # repo root; all paths below are relative to it
os.chdir(REPO); sys.path.insert(0, str(REPO))
import pandas as pd, glob, numpy as np
def load(cell):
    out=[]
    for f in sorted(glob.glob(f"outputs/xeval_20/{cell}/ppo_eval/ppo_eval_episode_*.csv")):
        d=pd.read_csv(f); d["ep"]=int(f[-7:-4])
        mv=d.index[d.leader_speed_mps>0.3]
        i_stop=mv[-1]+1 if len(mv) else 0
        d["after_stop"]=d.index>=i_stop
        d["t_rel"]=d.time_sec-d.time_sec.iloc[min(i_stop,len(d)-1)]
        out.append(d)
    return out
for cell in ["comm09_p1.0","comm09_p0.9","comm09_p0.5","run1026_p1.0","run1026_p0.9","run1026_p0.5"]:
    eps=load(cell); rows=[]
    for d in eps:
        s=d[d.after_stop]; r=s.iloc[0]
        em=s[s.distance_m<5.1]
        # creep: follower speed in the after-stop phase while distance<7
        rows.append(dict(d_stop=r.distance_m,vf_stop=r.follower_speed_mps,
            em=len(em)>0, t_em=em.t_rel.iloc[0] if len(em) else np.nan,
            thr_after=s.throttle[s.distance_m>=5.0].mean(),
            maxthr_after=s.throttle[s.distance_m>=5.0].max(),
            dur=s.t_rel.iloc[-1], d_end=d.distance_m.iloc[-1]))
    r=pd.DataFrame(rows)
    print(f"{cell:13s} em {r.em.sum():2d}/20 | at leader stop: dist {r.d_stop.mean():5.2f} vF {r.vf_stop.mean():4.2f} | after: mean thr {r.thr_after.mean():.3f} max thr {r.maxthr_after.mean():.3f} t_em {r.t_em.mean():4.1f}s dur {r.dur.mean():4.1f}s end dist {r.d_end.mean():.2f}")
