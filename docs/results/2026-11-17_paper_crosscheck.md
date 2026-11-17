# Cross-check against the reference paper (2026-11-17)

The three result documents of 2026-11-17 were first written from the code and from earlier notes. Afterwards the definitions they use were compared with the reference paper (not in this repository).

## Outcome

- **The definitions match.** The THW cost formula and its two thresholds, the names and bounds of the Caution and Danger regions, the six evaluation metrics (step-wise averages of r_cont, r_traf, r_jerk; THW cost; time steps in the Caution and Danger regions), the per-follower reporting, the [d_close, d_far] band, the leader target speeds and the 100 repetitions per cell are the ones the reference paper uses.
- **THW cost is reported two ways** in our tables, per step and summed per episode, so that either convention can be read off.
- **One statement was wrong and has been corrected.** `2026-11-16_tunnel_eval.md` said in its Limits that the vanilla PPO baseline is an assumption of this project. A standard PPO baseline is part of the reference comparison; only how ours is built (λ fixed at 0, everything else as PPO-Lag) is this project's choice. `implementation_notes.md` Sec. 6 was corrected the same way.
- **Known differences of this project** (map, episode length, communication conditions, distance definition, platoon size) are listed in `docs/implementation_notes.md` and in `CLAUDE.md`. Because of them, absolute values are not comparable with the reference paper; the comparison itself is kept outside this repository.

## Six metrics per follower at p = 0.9

Three-vehicle platoon runs, 100 episodes per cell, step-wise averages; PPO-Lag and PPO pool the three training seeds. r_traf and r_jerk use the simulator's acceleration reading, as the reward does. Source: `analysis/platoon_metrics.py`, Sec. 6 of its output.

| Controller | Follower | Control efficiency (r_cont) | Traffic disturbance (r_traf) | Driving comfort (r_jerk) | THW cost / episode | Caution steps | Danger steps |
|---|---|---|---|---|---|---|---|
| PID (retuned) | F1 | 15.68 | 7.54 | 2.51 | 0 | 0 | 0 |
| PID (retuned) | F2 | 16.24 | 5.89 | 2.27 | 0 | 0 | 0 |
| PPO | F1 | 8.30 | 9.61 | 2.48 | 0 | 0.09 | 0 |
| PPO | F2 | 8.09 | 9.98 | 2.47 | 0 | 0 | 0 |
| PPO-Lag | F1 | 8.70 | 9.93 | 2.56 | 0 | 0 | 0 |
| PPO-Lag | F2 | 9.19 | 10.90 | 2.92 | 0 | 0 | 0 |

Per seed (s42 / s43 / s44): r_cont PPO-Lag F1 8.07 / 9.13 / 8.91, F2 8.20 / 10.51 / 8.90; PPO F1 7.51 / 8.27 / 9.12, F2 7.49 / 8.41 / 8.37. With the old PID gains (two vehicles, 11-16) the same condition gave r_cont 31.2, r_traf 16.2, r_jerk 7.6, 11.1 Caution and 4.8 Danger steps.

What our own numbers show at p = 0.9:

- No controller enters the Danger region, and only PPO has any Caution step. The three controllers separate on safety only at p ≤ 0.3.
- Both learned controllers have about half the control-efficiency cost of PID (8–9 against 16); PPO is slightly lower than PPO-Lag, within the seed spread.
- From F1 to F2, PPO is flat, PPO-Lag grows slightly (r_cont +6%, r_traf +10%, r_jerk +14%), and PID's r_cont grows 4% while its r_traf and r_jerk fall. This agrees with the stage gains in `2026-11-17_platoon3_eval.md`.
