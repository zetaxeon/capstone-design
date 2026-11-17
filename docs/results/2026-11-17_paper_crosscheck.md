# Cross-check of today's analyses against the manuscript (2026-11-17)

The three result documents of 2026-11-17 were written from the code and from earlier notes, without opening the manuscript (`3월_논문.pdf`, 6 pages, not in the repository). This is the check against its text: Sec. II–III (model, Algorithm 1, eq. 1–15), Table I, Sec. IV (setup, six metrics, Fig. 5). The manuscript's own result values are not copied here.

## What matches

| Item | Manuscript | Our analysis | Verdict |
|---|---|---|---|
| THW cost | Eq. (15): 0 at THW ≥ τ_safe, 1 at THW ≤ τ_danger, ((τ_safe − THW)/(τ_safe − τ_danger))² between; τ_safe 1.2 s, τ_danger 1.0 s | Same formula and thresholds | Same |
| Region names | Sec. III-F: safe at THW ≥ τ_safe, dangerous at THW ≤ τ_danger, the region between is the caution zone | Caution 1.0 < THW < 1.2 s, Danger THW ≤ 1.0 s | Same (was inferred from eq. 15; the text confirms it) |
| Six metrics | Sec. IV-C: step-wise averages of r_cont, r_traf, r_jerk ("average control efficiency cost", "average traffic disturbance", "average driving comfort cost"); step-wise average of c_i ("average THW cost"); average number of time steps in the caution and danger regions ("Caution-region duration", "Danger-region duration"); lower is better | Step means of the three terms, THW cost, Caution and Danger steps per episode | Same definitions |
| r_traf, r_jerk | Eq. (11)–(12): a_i², (a_i(t) − a_i(t−1))² | Same, with a_i = measured acceleration in m/s² | Same form; see "a_i" below |
| r_cont | Eq. (10) with Q = diag(2, 1) | 2·Δd² + Δv² | Same, reading Δq as Δv (already marked [ASSUMED]) |
| Table I | T_s 0.05 s, h 1.0 s, d0 7.0 m, ω 0.1 / 0.25 / 0.25, d_far 8.0, d_close −4.0, α 20 / 10, d = 5, p_success 0.9 | Same in the code | Same |
| Recovery band | [d_close, d_far] is the range without r_penalty (eq. 13) | [−4, 8] m used for "back inside the band" | Same |
| Leader speeds | Target speed drawn from {15, 17.5, 20} m/s, cruise, then a sudden deceleration | Same three speeds | Same |
| Repetitions | Each evaluation scenario repeated 100 times, average reported | 100 episodes per cell | Same |
| Per-follower reporting | Fig. 5 reports every metric per follower | Platoon results are per follower | Same layout (table below) |
| Deployment | Algorithm 1 line 21 and Sec. III-B: train on Follower 1 only, deploy the trained policy to the remaining followers | `--followers 2`, same checkpoint | Same |
| Baselines | PPO-Lag is compared with a conventional PID and a standard PPO-based controller | PID, PPO (λ fixed at 0), PPO-Lag | Same set |

## What differs, and what had to be corrected

1. **A wrong statement in `2026-11-16_tunnel_eval.md` (corrected).** Its Limits said the vanilla PPO baseline is an assumption of this project. The manuscript does compare against a standard PPO-based controller. What is assumed is only how ours is built (λ fixed at 0, everything else as PPO-Lag). `implementation_notes.md` Sec. 6 is corrected the same way.
2. **"Average THW cost": per step in the text, per episode in the figure.** The text defines it as the step-wise average of c_i, which cannot exceed 1, but the values plotted in Fig. 5(d) are far above 1 and line up with the Danger-region duration plus a fraction of the Caution-region duration, i.e. with a sum over the episode. Our tables give both ("THW cost / step" and "THW cost / episode"); the per-episode value is the one on the figure's scale.
3. **Durations are not comparable in absolute size.** The manuscript's evaluation profile lasts roughly 280 s (Fig. 3b), about 5600 steps; our episodes last about 40 s (800–880 steps). Only zero / non-zero and the ordering of the controllers can be compared.
4. **Evaluation condition.** The manuscript uses one success probability, the same in training and evaluation (Table I: 0.9), with no location dependence and no collision statistic. Our p = 0.9 cells are the only ones in its condition; the tunnel zone, the sweep down to 0.1, the p = 1.0 reference, the leader's hard-brake event and every collision rate are additions of this project. (The manuscript's conclusion lists packet-loss models as future work.)
5. **Distance.** Eq. (3) is the path-wise distance along the reference path; the code uses the straight-line distance between vehicle centres. Already recorded as a [DEVIATION]; it applies to every spacing error, THW value and gap in today's documents. On the curved part of the tunnel route the straight line is slightly shorter than the path.
6. **a_i.** The manuscript uses a_i both for the longitudinal acceleration (eq. 1) and for the action in [−1, 1] (eq. 9). We use the measured acceleration (already [ASSUMED]). The sizes in Fig. 5(b)–(c) are well above 1, which an action in [−1, 1] cannot produce, so the acceleration reading is the consistent one.
7. **Platoon size.** Fig. 5 has three followers; ours has two.
8. **String stability.** The manuscript argues it from the growth of the six metrics with the follower index, not from a ratio. Our amplification ratios are this project's own measure; the table below is the like-for-like view.
9. **Map and framework.** Town04 highway and a ROS2 co-simulation with OmniSafe in the manuscript; Town03 tunnel route and the CARLA Python API here (already recorded).

## Our results in the manuscript's layout (Fig. 5), at its condition p = 0.9

Three-vehicle platoon runs, 100 episodes per cell, step-wise averages; PPO-Lag and PPO pool the three training seeds. r_traf and r_jerk use the simulator's acceleration reading, as the reward does.

| Controller | Follower | Control efficiency (r_cont) | Traffic disturbance (r_traf) | Driving comfort (r_jerk) | THW cost / episode | Caution steps | Danger steps |
|---|---|---|---|---|---|---|---|
| PID (retuned) | F1 | 15.68 | 7.54 | 2.51 | 0 | 0 | 0 |
| PID (retuned) | F2 | 16.24 | 5.89 | 2.27 | 0 | 0 | 0 |
| PPO | F1 | 8.30 | 9.61 | 2.48 | 0 | 0.09 | 0 |
| PPO | F2 | 8.09 | 9.98 | 2.47 | 0 | 0 | 0 |
| PPO-Lag | F1 | 8.70 | 9.93 | 2.56 | 0 | 0 | 0 |
| PPO-Lag | F2 | 9.19 | 10.90 | 2.92 | 0 | 0 | 0 |

Per seed (s42 / s43 / s44): r_cont PPO-Lag F1 8.07 / 9.13 / 8.91, F2 8.20 / 10.51 / 8.90; PPO F1 7.51 / 8.27 / 9.12, F2 7.49 / 8.41 / 8.37. With the old PID gains (two vehicles, 11-16) the same condition gave r_cont 31.2, r_traf 16.2, r_jerk 7.6, 11.1 Caution and 4.8 Danger steps.

Against the manuscript's qualitative findings:

- **"PPO-Lag stays out of the danger region": reproduced.** Zero Danger steps for both followers.
- **"PPO and PID enter the danger region": not reproduced at p = 0.9.** Our PPO and the retuned PID have zero Danger steps too; the untuned PID did enter it. In our environment the three controllers separate on safety only at p ≤ 0.3.
- **"PPO-Lag operates near τ_safe, with a large caution duration": not reproduced.** Our PPO-Lag has no Caution step at p = 0.9; its smallest THW stays above 1.28 s.
- **"PPO-Lag has the best control efficiency": partly.** Both learned controllers are far better than PID (about 8–9 against 16), but PPO is slightly better than PPO-Lag, within the seed spread.
- **"PPO-Lag pays in traffic disturbance and comfort": same direction, small.** PPO-Lag's r_traf and r_jerk are the highest or equal highest, by a few percent, and more so for F2.
- **"Metrics grow toward the rear for PPO and PID but not for PPO-Lag": not reproduced.** From F1 to F2 our PPO is flat, PPO-Lag grows slightly (r_cont +6%, r_traf +10%, r_jerk +14%), and PID's r_cont grows 4% while its r_traf and r_jerk fall. This agrees with the stage gains in `2026-11-17_platoon3_eval.md` (learned policies about 1.0, PID about 0.8).

None of this contradicts the manuscript: map, speed profile, episode length, distance definition and platoon size all differ. It does mean that the manuscript's headline separation between PPO-Lag and the baselines does not appear in this project's p = 0.9 runs.
