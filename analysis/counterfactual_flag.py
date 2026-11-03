"""Replay logged observations through the comm09 checkpoint offline; compare actions with flag_i/t_i forced to 0 in the creep phase.
Run from anywhere: python3 analysis/counterfactual_flag.py"""
import os, sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]  # repo root; all paths below are relative to it
os.chdir(REPO); sys.path.insert(0, str(REPO))
import sys, glob, torch, numpy as np, pandas as pd
from episode_ppo_v2 import _PPOLagPolicy, _normalize_ppo_observation
p = torch.load("outputs/run_comm09_3000/ppo_lag.pt", map_location="cpu", weights_only=False)
m = _PPOLagPolicy(7); m.load_state_dict(p["model"]); m.eval()
def act(obs):
    x = torch.tensor([_normalize_ppo_observation(o) for o in obs], dtype=torch.float32)
    with torch.no_grad(): return torch.tanh(m.actor_mean(x)).squeeze(-1).numpy()
def obs_of(d, flag=None, t=None):
    f = d.comm_flag.values if flag is None else np.full(len(d), flag)
    tt = d.t_since_comm_s.values if t is None else np.full(len(d), t)
    return np.stack([d.obs_spacing_error_m, d.obs_relative_velocity_mps, d.lead_accel_obs_mps2,
                     d.follower_speed_mps, d.lateral_offset_m, f, tt], 1)
for cell in ["comm09_p1.0", "comm09_p0.5"]:
    rec, creep = [], []
    for fn in sorted(glob.glob(f"outputs/xeval_20/{cell}/ppo_eval/ppo_eval_episode_*.csv")):
        d = pd.read_csv(fn)
        a = act(obs_of(d)); applied = d.throttle.values - d.brake.values
        # decision ticks: phase k with best agreement between reconstructed action and applied control
        ok = d.distance_m.values >= 5.0
        k = max(range(10), key=lambda k: np.mean(np.abs(a[(d.index % 10 == k) & ok] - applied[(d.index % 10 == k) & ok]) < 1e-3))
        dec = (d.index % 10 == k) & ok
        rec.append(np.mean(np.abs(a[dec] - applied[dec]) < 1e-3))
        # creep phase: leader stopped for good, follower re-launched, decision ticks
        mv = d.index[d.leader_speed_mps > 0.3]; i0 = mv[-1] + 1
        relaunch = d.index[(d.index > i0) & (d.throttle > 0)]
        if len(relaunch) == 0: continue
        s = dec & (d.index >= relaunch[0])
        a0 = act(obs_of(d[s], flag=0, t=0.0))
        creep.append(pd.DataFrame(dict(flag=d.comm_flag[s].values, t=d.t_since_comm_s[s].values, a=a[np.asarray(s)], a_noflag=a0,
                                       err=d.spacing_error_m[s].values, v=d.follower_speed_mps[s].values)))
    c = pd.concat(creep)
    print(f"{cell}: reconstruction match {np.mean(rec):.3f}, creep decisions {len(c)}")
    print("  mean action by flag:", c.groupby("flag").a.agg(["count","mean"]).round(3).to_dict("index"))
    print("  same decisions with flag=0,t=0 forced: mean %.3f" % c.a_noflag.mean())
    if (c.flag == 1).any():
        f1 = c[c.flag == 1]
        print("  flag=1 decisions: actual %.3f vs flag forced 0 %.3f (diff %+.3f)" % (f1.a.mean(), f1.a_noflag.mean(), (f1.a - f1.a_noflag).mean()))
        print("  by t:", f1.groupby(f1.t.round(2)).apply(lambda g: round((g.a - g.a_noflag).mean(), 3)).to_dict())
    print("  frac of creep decisions with v>0.4: %.2f" % (c.v > 0.4).mean())
