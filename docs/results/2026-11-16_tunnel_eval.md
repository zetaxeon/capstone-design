# Tunnel V2V experiment: PPO-Lag vs vanilla PPO vs PID (2026-11-16)

Question: when V2V packet success drops inside the tunnel, how do PPO-Lag, vanilla PPO (λ fixed at 0) and PID compare? Design and assumptions: `docs/implementation_notes.md` Sec. 6.

## Setup

- Training (6 runs): 3000 episodes each, seeds 42 / 43 / 44 per algorithm, p 0.9 outside the tunnel, in-tunnel mean p drawn per episode from U(0.1, 0.9), noise std 0.1 (τ 1 s), leader hard-brake probability 0.5. Code: ac05f91 for all six.
- Evaluation: 100 deterministic episodes per cell, evaluation seed 1000, so every controller meets the same leader speeds, packet draws and brake events. In-tunnel p 0.9 / 0.7 / 0.5 / 0.3 / 0.1, p 0.9 outside, noise std 0.1, brake probability 0.5. 35 cells, all 100 episodes, no traceback.
- Extra no-loss reference (4 cells): PID and PPO-Lag seeds 42 / 43 / 44 with p = 1.0 inside and outside the tunnel, noise 0, same seed and brake events. Measured loss 0 packets.
- The leader's hard brake happens in 49 of the 100 evaluation episodes. Leader target speed: 15 m/s in 34 episodes, 17.5 m/s in 38, 20 m/s in 28.
- Checkpoint md5 (first 8): PPO-Lag `9ff7f0a6` / `f19f34d8` / `4ca3fdca`, PPO `a6fce015` / `8c20aee6` / `31ab0ada`.
- Raw data: `outputs/tunnel/eval/` (local only). Tables: `outputs/tunnel/summary/` (`pooled.md`, `pooled_p1.md`, `zones_{all,brake_only,brake_none}_t<p>.{txt,csv}`). Reproduce with `analysis/tunnel_pooled.py` and `analysis/tunnel_summarize.py`.

PPO-Lag and PPO rows pool the three seeds (300 episodes). The three seeds face the same 100 scenarios, so the pooled 95% interval is somewhat too narrow; per-seed counts (s42 / s43 / s44) are given for that reason.

## Collisions

| In-tunnel p | PID (100) | PPO-Lag (300) | per seed | PPO (300) | per seed |
|---|---|---|---|---|---|
| 1.0 (no loss anywhere) | 28% [20, 37] | 0% [0, 1] | 0 / 0 / 0 | not run | |
| 0.9 | 28% [20, 37] | 0% [0, 1] | 0 / 0 / 0 | 0% [0, 1] | 0 / 0 / 0 |
| 0.7 | 28% [20, 37] | 0% [0, 1] | 0 / 0 / 0 | 0% [0, 1] | 0 / 0 / 0 |
| 0.5 | 28% [20, 37] | 0% [0, 1] | 0 / 0 / 0 | 0% [0, 2] | 0 / 0 / 1 |
| 0.3 | 31% [23, 41] | 3% [2, 6] | 2 / 0 / 8 | 7% [5, 11] | 2 / 3 / 17 |
| 0.1 | 71% [61, 79] | 36% [30, 41] | 35 / 23 / 49 | 44% [38, 49] | 39 / 41 / 51 |

Split by whether the leader braked hard in the episode:

| In-tunnel p | PID brake (49) | PID no brake (51) | PPO-Lag brake (147) | PPO-Lag no brake (153) | PPO brake (147) | PPO no brake (153) |
|---|---|---|---|---|---|---|
| 1.0 | 13 | 15 | 0 | 0 | not run | not run |
| 0.9 | 13 | 15 | 0 | 0 | 0 | 0 |
| 0.7 | 13 | 15 | 0 | 0 | 0 | 0 |
| 0.5 | 13 | 15 | 0 | 0 | 1 | 0 |
| 0.3 | 16 | 15 | 10 | 0 | 22 | 0 |
| 0.1 | 37 | 34 | 102 (69%) | 5 (3%) | 123 (84%) | 8 (5%) |

## Emergency-brake entries (distance < 5 m)

| In-tunnel p | PID | PPO-Lag s42 / s43 / s44 | PPO s42 / s43 / s44 |
|---|---|---|---|
| 0.9 | 28% | 0 / 0 / 0% | 0 / 16 / 98% |
| 0.5 | 29% | 0 / 0 / 0% | 0 / 13 / 99% |
| 0.3 | 33% | 2 / 0 / 9% | 3 / 20 / 99% |
| 0.1 | 70% | 37 / 23 / 49% | 39 / 50 / 96% |

At p 0.9 and 0.7 every PPO entry is in the final zone (the leader's stop at its destination), not in the tunnel and not in the 6 s after a hard brake; at p 0.5 all but one are. (p 0.7, not in the table: PPO 0 / 19 / 95%, PPO-Lag 0 / 0 / 0%.)

## Tracking inside the tunnel (mean |spacing error|, m)

| In-tunnel p | PID | PPO-Lag | PPO |
|---|---|---|---|
| 1.0 | 2.98 ± 0.07 | 0.73 ± 0.05 | not run |
| 0.9 | 3.07 ± 0.07 | 0.73 ± 0.05 | 0.53 ± 0.03 |
| 0.7 | 3.31 ± 0.07 | 0.71 ± 0.04 | 0.53 ± 0.03 |
| 0.5 | 3.80 ± 0.09 | 0.67 ± 0.04 | 0.55 ± 0.04 |
| 0.3 | 5.38 ± 0.31 | 0.89 ± 0.06 | 0.79 ± 0.07 |
| 0.1 | 22.75 ± 2.92 | 2.92 ± 0.20 | 2.51 ± 0.19 |

## Why PID collides: not communication

1. **PID collisions do not line up with the leader's hard brake.** At p = 0.9 the 28 collisions are 13 of the 49 brake episodes and 15 of the 51 episodes without one.
2. **They line up with the leader's target speed.** All 28 episodes at 20 m/s collide; none of the 72 at 15 or 17.5 m/s do. With a hard brake the follower crosses the 5 m emergency distance 3.2–3.5 s after the brake starts, inside the tunnel. Without one it comes at the leader's final stop, outside the tunnel, at t = 27.0 s in every case.
3. **With no packet loss at all (p = 1.0) PID collides in the same 28 episodes.** Collision rate 28%, the same episode numbers as at p = 0.9.

Conclusion: the PID collisions at p ≥ 0.5 are caused by the PID tuning against a leader decelerating from 20 m/s (final stop and hard brake), not by communication loss. Communication loss adds PID collisions only at p = 0.3 (+3) and p = 0.1 (+43). In the same no-loss condition PPO-Lag has 0 collisions and 0 emergency entries in all three seeds, which is the reference point for the rows above.

## Findings

1. **Down to p = 0.5 the tunnel loss changes safety for no controller.** PPO-Lag: 0 collisions, 0 emergency entries in 900 episodes (p 0.9 / 0.7 / 0.5), identical to its no-loss reference. PPO: 1 collision. PID: unchanged 28%.
2. **Loss becomes dangerous only together with the leader's hard brake.** At p = 0.3 every collision of the learned controllers is in a brake episode (PPO-Lag 10/147, PPO 22/147; 0/153 for both without a brake). At p = 0.1 brake episodes collide 69% (PPO-Lag) and 84% (PPO) of the time, against 3% and 5% without a brake.
3. **PPO-Lag has fewer collisions than PPO at p = 0.3 and 0.1, but three seeds do not settle it.** Pooled 3% vs 7% and 36% vs 44%; the per-seed ranges overlap (PPO-Lag 23–49, PPO 39–51 at p = 0.1), and seed 44 is the worst seed of both algorithms.
4. **The emergency-entry gap between PPO and PPO-Lag is one seed.** PPO seed 44 enters the emergency brake in 96–99% of episodes at every p, seed 43 in 13–20% (50% at p = 0.1), seed 42 in 0–3% (39% at p = 0.1). All three PPO-Lag seeds are at 0% down to p = 0.5. The pooled PPO figure (about 38%) should not be quoted without this.
5. **Both learned controllers keep the tunnel spacing error 4–8x below PID** at every p; PPO is slightly tighter than PPO-Lag (0.53 vs 0.73 m at p = 0.9).
6. **At p = 0.1 nothing is safe in a hard brake**: PID 76%, PPO-Lag 69%, PPO 84% collisions in brake episodes.

## Limits

- Three seeds per algorithm, one evaluation seed (the same 100 scenarios for everything).
- PID was not retuned; its result at 20 m/s is a tuning result, see above. Retuned on 2026-11-17 (kd 0.1 → 0.6): `2026-11-17_pid_retune.md`. The PID numbers in this document are the old gains.
- The tunnel model and the hard-brake event are assumptions of this project, not from the paper. The paper does compare against a standard PPO-based controller; how ours is built (λ fixed at 0, otherwise identical to PPO-Lag) is this project's choice. (Corrected 2026-11-17; see `2026-11-17_paper_crosscheck.md`.)
- The first evaluation cell (`lag_s42_t0.9`) failed twice at start-up with 0 episodes before succeeding on the third try; the cause was not captured (the retry overwrote the log). Its data is from the successful try.
