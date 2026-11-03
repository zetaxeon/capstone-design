"""Flag-induced action change (flag/t forced to 0) across leader phases, comm09 at p=0.9 and p=0.5.
Run from anywhere: python3 analysis/flag_global.py"""
import os, sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]  # repo root; all paths below are relative to it
os.chdir(REPO); sys.path.insert(0, str(REPO))
exec(open("analysis/counterfactual_flag.py").read().split("for cell in")[0])
for cell in ["comm09_p0.9","comm09_p0.5"]:
    rows=[]
    for fn in sorted(glob.glob(f"outputs/xeval_20/{cell}/ppo_eval/ppo_eval_episode_*.csv")):
        d=pd.read_csv(fn); a=act(obs_of(d)); a0=act(obs_of(d,flag=0,t=0.0))
        mv=d.index[d.leader_speed_mps>0.3]; i0=mv[-1]+1
        ph=np.where(d.index<i0, "leader moving", "leader stopped")
        ph=np.where((d.index<i0)&(d.leader_phase=="decelerate"), "leader decelerating", ph)
        f=d.comm_flag.values==1
        rows.append(pd.DataFrame(dict(phase=ph[f], diff=(a-a0)[f], v=d.follower_speed_mps.values[f])))
    r=pd.concat(rows)
    print(cell); print(r.groupby("phase")["diff"].agg(["count","mean","min","max"]).round(3).to_string())
    print("  stopped & v<2 m/s:", r[(r.phase=="leader stopped")&(r.v<2)]["diff"].mean().round(3))
