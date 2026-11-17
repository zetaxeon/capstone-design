# Three-vehicle platoon in the tunnel: leader + F1 + F2 (2026-11-17)

Question: when the Follower 1 controller is deployed unchanged on a second follower, as the paper does (Algorithm 1 line 21), does the platoon stay safe under tunnel packet loss, and do disturbances grow toward the rear? PPO-Lag, vanilla PPO and the retuned PID are compared. Design and assumptions: `docs/implementation_notes.md` Sec. 7.

## Setup

- **No retraining.** F2 runs the same checkpoint as F1 (PPO-Lag and PPO, training seeds 42 / 43 / 44, the checkpoints of the 11-16 tunnel experiment), or the same PID gains (kp 0.4, ki 0.01, kd 0.6).
- **Communication.** Predecessor-following: leader → F1 and F1 → F2, each link with its own independent packet losses at the same in-tunnel p; a link is degraded while either of its vehicles is in the tunnel. F2 receives F1's own true state.
- **Evaluation.** In-tunnel p 1.0 / 0.9 / 0.5 / 0.3 / 0.1, 100 episodes per cell, evaluation seed 1000, leader hard-brake probability 0.5 (49 of the 100 episodes). p = 1.0 is the no-loss reference (p 1.0 outside, no noise); the others use p 0.9 outside and noise std 0.1. 35 cells, all 100 episodes, no traceback, no retry, no timeout.
- **Code.** `episode_ppo_v2.py --followers 2`, run by `analysis/platoon_eval.sh`; metrics by `analysis/platoon_metrics.py`. Data: `outputs/tunnel/eval3/` (local only), tables in `outputs/tunnel/summary/platoon_metrics.md`.
- PPO-Lag and PPO rows pool the three seeds (300 episodes); the seeds drive the same 100 scenarios, so pooled intervals are somewhat too narrow and per-seed values (s42 / s43 / s44) are given.

### Regression check before the evaluation

- With `--followers 1` the changed runner is byte-identical to the previous commit (same server session, PPO-Lag seed 42, p = 0.3, 5 episodes, two runs of each).
- It reproduces the stored two-vehicle results byte for byte: PID at p = 1.0 and 0.3 (5 episodes each), PPO-Lag seed 42 at p = 0.9 and PPO seed 43 at p = 0.1 (10 episodes each).
- One stored episode does not reproduce: `lag_s42_t0.3` episode 5 differs from 8.2 s on, starting with a 1e-4 difference in the simulator's acceleration reading under identical control inputs. The previous commit, the system Python, a freshly started server and the second port all give the same result as the changed code, so this is a one-off of the 11-16 simulator run and not a code difference.
- In the platoon runs F1 drives exactly as in the two-vehicle runs over the common ticks. F1 – leader collisions happen in the same episodes in all 35 cells, except 10 episodes in which F2 ran into F1 first and ended the episode.

## 1. Platoon collision rate (primary metric)

An episode counts as a platoon collision when any pair touches. Wilson 95% interval in brackets.

| In-tunnel p | PID (100) | PPO-Lag (300) | per seed | PPO (300) | per seed |
|---|---|---|---|---|---|
| 1.0 | 0% [0, 3.7] | 0% [0, 1.3] | 0 / 0 / 0 | 0% [0, 1.3] | 0 / 0 / 0 |
| 0.9 | 0% [0, 3.7] | 0% [0, 1.3] | 0 / 0 / 0 | 0% [0, 1.3] | 0 / 0 / 0 |
| 0.5 | 0% [0, 3.7] | 0% [0, 1.3] | 0 / 0 / 0 | 0.7% [0.2, 2.4] | 0 / 0 / 2 |
| 0.3 | 0% [0, 3.7] | 3.7% [2.1, 6.4] | 2 / 0 / 9 | 8.3% [5.7, 12.0] | 2 / 4 / 19 |
| 0.1 | 2.0% [0.6, 7.0] | 44.0% [38.5, 49.7] | 43 / 34 / 55 | 51.0% [45.4, 56.6] | 47 / 49 / 57 |

Which pair (an episode ends at the first collision; no collision with anything else occurred):

| In-tunnel p | Controller | F1 – leader | F2 – F1 | F2 – F1 among episodes without an F1 – leader collision | per seed |
|---|---|---|---|---|---|
| 0.5 | PPO | 1 | 1 | 1 / 299 = 0.3% [0.1, 1.9] | 0 / 0 / 1 |
| 0.3 | PPO-Lag | 9 | 2 | 2 / 291 = 0.7% [0.2, 2.5] | 0 / 0 / 2 |
| 0.3 | PPO | 21 | 4 | 4 / 279 = 1.4% [0.6, 3.6] | 0 / 1 / 3 |
| 0.1 | PID | 2 | 0 | 0 / 98 = 0% [0, 3.8] | |
| 0.1 | PPO-Lag | 100 | 32 | 32 / 200 = 16.0% [11.6, 21.7] | 11 / 12 / 9 |
| 0.1 | PPO | 130 | 23 | 23 / 170 = 13.5% [9.2, 19.5] | 8 / 8 / 7 |

Two-vehicle collision rate for comparison (11-16 evaluation; PID from the retune): p = 0.3 PPO-Lag 3.3%, PPO 7.3%, PID 0%; p = 0.1 PPO-Lag 35.7%, PPO 43.7%, PID 2%.

- **Without loss the deployment is safe.** At p = 1.0 and 0.9 no pair collides in 1400 episodes, for any controller. F2's problems below are therefore a communication effect, not an effect of deploying the F1 policy behind another follower.
- **PPO-Lag stays collision-free down to p = 0.5** (900 episodes); PPO has 2 collisions at p = 0.5 (seed 44).
- **The second follower adds collisions only under heavy loss.** At p = 0.3 the platoon rate is close to the two-vehicle rate (PPO-Lag 3.7% vs 3.3%). At p = 0.1 it rises from 35.7% to 44.0% (PPO-Lag) and from 43.7% to 51.0% (PPO): in the episodes where F1 does not hit the leader, F2 hits F1 in 16% and 13.5%.
- **F2's collisions are not only hard-brake collisions.** 25 of PPO-Lag's 32 F2 – F1 collisions at p = 0.1 are in hard-brake episodes (the learned controllers' F2 – F1 collisions come a median 4.0 s after the leader's brake starts, their F1 – leader collisions 2.7 s after); 7 are in episodes without a hard brake (PPO: 9 of 23).
- **The retuned PID platoon has no F2 collision at any p**, and 2 platoon collisions in total (both F1 – leader at p = 0.1, the same two episodes as with two vehicles).
- **PPO-Lag vs PPO**: same picture as with two vehicles. On the same scenarios the three PPO-Lag networks collide less (paired test p < 0.001 at p = 0.3 and 0.1; PPO-Lag better in 14 and 19 scenarios, PPO in 0 and 1), but with the training seed as the unit the difference is not significant (p = 0.60 and 0.40), and three seeds per algorithm cannot show it.

## 2. Safety and tracking per follower

Mean per episode. THW cost is eq. (15) summed over the episode; Danger = THW ≤ 1.0 s, Caution = 1.0 < THW < 1.2 s, in steps of 0.05 s.

| In-tunnel p | Controller | THW cost F1 / F2 | Danger steps F1 / F2 | Caution steps F1 / F2 | Episodes with Danger F1 / F2 | \|err\| in tunnel F1 / F2 (m) |
|---|---|---|---|---|---|---|
| 1.0 | PID | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 2.54 / 2.62 |
| 1.0 | PPO-Lag | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0.73 / 0.87 |
| 1.0 | PPO | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0.52 / 0.62 |
| 0.9 | PID | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 2.64 / 2.71 |
| 0.9 | PPO-Lag | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0.73 / 0.87 |
| 0.9 | PPO | 0 / 0 | 0 / 0 | 0.09 / 0 | 0 / 0 | 0.53 / 0.62 |
| 0.5 | PID | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 3.31 / 3.39 |
| 0.5 | PPO-Lag | 0 / 0 | 0 / 0 | 0.01 / 0.02 | 0 / 0 | 0.67 / 0.82 |
| 0.5 | PPO | 0.14 / 0.11 | 0.06 / 0.06 | 0.82 / 0.20 | 1 / 2 | 0.55 / 0.62 |
| 0.3 | PID | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 4.71 / 4.80 |
| 0.3 | PPO-Lag | 0.66 / 0.23 | 0.49 / 0.14 | 0.57 / 0.45 | 11 / 3 | 0.89 / 0.98 |
| 0.3 | PPO | 1.77 / 0.66 | 1.24 / 0.39 | 2.09 / 1.11 | 26 / 10 | 0.79 / 0.69 |
| 0.1 | PID | 0.24 / 0 | 0.22 / 0 | 0.10 / 0 | 2 / 0 | 16.15 / 17.40 |
| 0.1 | PPO-Lag | 9.41 / 6.86 | 7.66 / 5.60 | 5.81 / 4.33 | 131 / 69 | 2.90 / 3.15 |
| 0.1 | PPO | 17.53 / 9.66 | 13.77 / 7.52 | 15.45 / 8.37 | 172 / 81 | 2.51 / 1.90 |

F2 per seed (s42 / s43 / s44): THW cost at p = 0.3 PPO-Lag 0.02 / 0 / 0.68, PPO 0 / 0.43 / 1.55; at p = 0.1 PPO-Lag 6.88 / 3.74 / 9.95, PPO 6.39 / 8.85 / 13.74. In-tunnel error at p = 1.0 PPO-Lag 0.73 / 0.83 / 1.06 m, PPO 0.58 / 0.68 / 0.61 m.

- **Down to p = 0.5 neither PPO-Lag follower enters the Danger region**, and the PID followers never leave the safe region down to p = 0.3.
- **F2 tracks slightly worse than F1 with every controller, already without loss**: +0.14 m (PPO-Lag), +0.10 m (PPO), +0.08 m (PID) at p = 1.0. This is the visible cost of following a follower instead of the scripted leader; for PPO-Lag and PID it stays about the same down to p = 0.3.
- **F2's THW cost and Danger stay are lower than F1's** at p ≤ 0.3. Part of that is censoring: an F1 – leader collision ends the episode before F2 gets close. It does not mean F2 is the safer position.
- **The PID platoon stretches under loss**: both followers drop back (4.7 / 4.8 m at p = 0.3, 16.2 / 17.4 m at p = 0.1), and F2's lag comes on top of F1's, so the tail of the platoon is about twice as far behind the leader's schedule as a single follower.

Emergency-brake entries (gap < 5 m) without loss show one deployment effect that is independent of communication: at p = 1.0 PPO seed 44 enters it in 100 of 100 episodes as F1 and 100 of 100 as F2, seed 43 in 100 and 25, at the final stop (the known creep of the vanilla-PPO policies). PPO-Lag: F1 4 episodes (seed 42), F2 none down to p = 0.5. PID: none. F1's counts are higher than in the two-vehicle evaluation because the episode now lasts until F2 has stopped, which gives F1 several more seconds to creep; they are not comparable with the 11-16 numbers.

## 3. Rear amplification

Follower over predecessor, for both stages. Acceleration is taken from the logged speeds (central difference over ±0.25 s); the simulator's acceleration reading spikes to 20–45 m/s² for one tick when a vehicle comes to rest and cannot be used for peaks. A ratio above 1 means the disturbance grows toward the rear.

**Episodes without any collision** (a collision cuts the trace and biases the ratio; at p ≥ 0.5 this is all or nearly all episodes):

| In-tunnel p | Controller | Episodes | Tunnel RMS accel: F1 / leader | F2 / F1 | Peak decel after hard brake: F1 / leader | F2 / F1 |
|---|---|---|---|---|---|---|
| 1.0 | PID | 100 | 0.81 | 0.86 | 0.76 | 0.74 |
| 1.0 | PPO-Lag | 300 | 0.95 | 1.03 | 0.95 | 0.99 |
| 1.0 | PPO | 300 | 0.95 | 1.01 | 0.97 | 1.00 |
| 0.9 | PID | 100 | 0.81 | 0.86 | 0.76 | 0.74 |
| 0.9 | PPO-Lag | 300 | 0.95 | 1.03 | 0.95 | 0.99 |
| 0.9 | PPO | 300 | 0.95 | 1.01 | 0.97 | 1.00 |
| 0.5 | PID | 100 | 0.84 | 0.88 | 0.73 | 0.75 |
| 0.5 | PPO-Lag | 300 | 0.95 | 1.02 | 0.95 | 0.99 |
| 0.5 | PPO | 298 | 0.95 | 1.01 | 0.96 | 1.00 |
| 0.3 | PID | 100 | 0.94 | 0.93 | 0.74 | 0.74 |
| 0.3 | PPO-Lag | 289 | 0.98 | 1.03 | 0.96 | 0.99 |
| 0.3 | PPO | 275 | 0.95 | 1.00 | 0.96 | 0.99 |
| 0.1 | PID | 98 | 1.35 | 0.97 | 0.64 | 0.74 |
| 0.1 | PPO-Lag | 168 | 1.40 | 1.08 | 0.96 | 0.95 |
| 0.1 | PPO | 147 | 1.43 | 0.97 | 0.94 | 0.98 |

Per seed, all episodes, tunnel RMS F2 / F1 at p = 1.0: PPO-Lag 0.99 / 1.06 / 1.02, PPO 1.02 / 1.02 / 0.99. Peak-deceleration ratio F2 / F1 above 1 in 66 of 147 hard-brake episodes for PPO-Lag and 75 of 147 for PPO at p = 1.0; in 0 of 49 for PID.

The ratios are dominated by the hard-brake episodes. Without a hard brake the leader barely accelerates, so a ratio to the leader means little; the absolute values show what loss does there (tunnel RMS acceleration in m/s², episodes without a hard brake, all episodes):

| In-tunnel p | Leader | PID F1 / F2 | PPO-Lag F1 / F2 | PPO F1 / F2 |
|---|---|---|---|---|
| 1.0 | 0.10 | 0.17 / 0.20 | 0.30 / 0.38 | 0.23 / 0.34 |
| 0.9 | 0.10 | 0.23 / 0.25 | 0.31 / 0.39 | 0.25 / 0.34 |
| 0.5 | 0.10 | 0.75 / 0.82 | 0.39 / 0.49 | 0.29 / 0.38 |
| 0.3 | 0.10 | 1.63 / 1.66 | 0.77 / 0.85 | 0.45 / 0.50 |
| 0.1 | 0.09–0.18 | 3.68 / 3.68 | 2.05 / 2.19 | 1.50 / 1.56 |

- **PID attenuates at both stages** (0.81 and 0.86 in tunnel RMS, 0.76 and 0.74 in peak deceleration, down to p = 0.5): each PID follower brakes more gently than the vehicle ahead.
- **The learned policies attenuate at the first stage and not at the second.** F1 / leader is about 0.95; F2 / F1 is 1.00–1.03 in RMS and 0.99–1.00 in peak deceleration, at every p down to 0.3 and for both algorithms. F2 reproduces F1's braking at full size, and in about half of the hard-brake episodes brakes harder. A gain of about 1 per stage is the boundary of string stability; with two followers it causes no collision, and this experiment cannot say what a longer platoon would do.
- **Loss creates the disturbance at F1; F2 passes it on.** Without a hard brake, F1's RMS acceleration rises with loss (PPO-Lag 0.30 → 0.77 → 2.05 m/s² at p 1.0 / 0.3 / 0.1; PID 0.17 → 1.63 → 3.68), because the held leader position makes the follower slow down and surge. F2's value is above F1's at every p for the learned policies (by 25–50% at p ≥ 0.5, by 5–10% at p ≤ 0.3) and equal to slightly above for PID.
- **Under loss PID is the roughest ride, not the smoothest**: at p ≤ 0.5 its followers accelerate about twice as hard as the PPO-Lag ones without any leader event (0.75 vs 0.39 at p = 0.5).
- At p = 0.1 the first stage amplifies for every controller in the collision-free episodes (1.35–1.43).

## What changes against the two-vehicle results

- F1 is unchanged; everything new is about F2 and about the platoon as a whole.
- The ranking of the controllers is the same: on collisions the retuned PID is best at p ≤ 0.3 and the learned controllers lose safety only at p ≤ 0.3 and almost only in hard-brake episodes; on tracking the learned controllers are 3–5 times tighter than PID in both positions.
- The platoon collision rate is higher than the two-vehicle rate only at p = 0.1 (+8 and +7 points for PPO-Lag and PPO).
- New: the learned policies have a stage gain of about 1 behind another follower, while PID has 0.74–0.88.

## Limits

- Two followers, not the paper's three; one evaluation seed; three training seeds per algorithm.
- Censoring: an episode ends at the first collision, so F2's rates at p ≤ 0.3 are measured mostly in the episodes where F1 did not collide.
- The amplification measures are this project's definition (not in the paper), computed over each follower's own time inside the tunnel.
- F1's emergency-brake counts are not comparable with the two-vehicle evaluation (longer episodes).
- PID gains were selected on evaluation seed 1000, the seed used here (confirmed on seed 2000 for two vehicles, not for the platoon).
