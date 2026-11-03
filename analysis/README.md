# analysis/

Analysis, evaluation, and aggregation scripts from 2026-11-03 (V2V comm model, emergency-brake entry cost, cross-evaluation).
All paths are relative to the repository root, and every script can be run from any directory.

Data the scripts read (local only; `outputs/` is in .gitignore):
- `outputs/run1026_3000/`, `outputs/run_comm09_3000/`, `outputs/run_emcost_3000/`: training runs
- `outputs/xeval_20/`: 20-episode cross-eval (run1026, comm09)
- `outputs/xeval_100/`: 100-episode cross-eval (run1026, comm09, emcost), `summary.txt`, per-cell logs and CSVs

Emergency metric: **entry** = any tick with distance < 5.0 m (= the override tick). Scripts marked *(near)* still use the old near-miss count (min distance < 5.1 m).
The `xeval*_run.sh` scripts need a running CARLA and delete/rewrite each cell folder. `status_emcost.py` checks the hard-coded training pid 60909.

| Script | Purpose |
|---|---|
| `brake_usage.py` | Policy brake usage and emergency-override tick share per 500-episode phase in training logs. |
| `compare_runs.py` | Compare training logs of run1026_3000 and run_comm09_3000 (first completion, lambda, 500-episode phases, last 50 episodes). *(near)* |
| `cost_replay.py` | Recompute episode costs from training tick logs under alternative emergency-cost definitions and replay the lambda update with the policy held fixed. |
| `counterfactual_flag.py` | Replay logged observations through the comm09 checkpoint offline; compare actions with flag_i/t_i forced to 0 in the creep phase. |
| `creep_brake.py` | Brake vs throttle share in the post-stop creep phase (episodes 1501-3000) of training logs. |
| `entry_counts.py` | Entry counts collected from the 11/03 ad-hoc checks: 20-episode entry vs near-miss, 100-episode McNemar (comm09 vs emcost), training-log entry rates and lambda. |
| `flag_global.py` | Flag-induced action change (flag/t forced to 0) across leader phases, comm09 at p=0.9 and p=0.5. |
| `route_dump.py` | Dump the planned routes from CARLA (needs a running CARLA) for the r_i route-offset diagnosis. |
| `route_trace.py` | Run episode_ppo_v2.py with r_i tracing; writes trace.json into the directory given as the last argument. |
| `status_emcost.py` | Progress check of run_emcost_3000: episodes, last-100 entry rate, lambda. |
| `stopphase.py` | Final-stop phase per cross-eval cell (outputs/xeval_20): distance and speed at leader stop, creep throttle, emergency timing. *(near)* |
| `xeval100_rerun_emcost.sh` | Rerun only the three emcost cells (used after the CARLA segfault on 11/03). |
| `xeval100_run.sh` | 100-episode cross-eval (run1026, comm09, emcost x p 1.0/0.9/0.5); optional arg: training pid to wait for. |
| `xeval100_summarize.py` | Summarize the 100-episode cross-eval (outputs/xeval_100): entry rate with Wilson CI, paired emcost-comm09 differences, criterion 1. |
| `xeval20_run.sh` | 20-episode cross-eval (run1026, comm09 x p 1.0/0.9/0.5). |
| `xeval20_summarize.py` | Summarize the 20-episode cross-eval (outputs/xeval_20). Uses the old near-miss metric (min distance < 5.1 m); superseded by xeval100_summarize.py. *(near)* |
