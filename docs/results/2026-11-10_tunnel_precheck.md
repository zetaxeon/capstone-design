# Tunnel communication zone: pre-check before training (2026-11-10)

Purpose: before spending training time, check whether the tunnel zone changes anything for the PID follower and for the existing PPO-Lag policy (`run_emcost_3000`, trained at fixed p = 0.9, no tunnel).

Setup: 30 deterministic episodes per cell, seed 42 (same leader speed sequence and same packet draws in every cell), p = 0.9 outside the tunnel. Code: working tree on top of 945a847 (uncommitted at the time). Raw data: `outputs/tunnel/precheck/` (local only), full table in `summary.txt` there. Reproduce the table with `analysis/tunnel_summarize.py`.

"b10" / "b20" = losses arrive in runs of mean length 10 / 20 ticks (0.5 s / 1 s), same average rate.

## Inside the tunnel

| In-tunnel p | PID mean \|spacing error\| | PPO-Lag mean \|spacing error\| | PID THW_min | PPO-Lag THW_min | Mean staleness of Δd |
|---|---|---|---|---|---|
| 0.9 (no drop) | 3.04 m | 0.28 m | 1.56 s | 1.39 s | 0.10 m |
| 0.5 | 3.94 m | 0.37 m | 1.58 s | 1.39 s | 0.86 m |
| 0.3 | 5.33 m | 0.92 m | 1.60 s | 1.39 s | 2.0 m |
| 0.1 | 16.3 m | 4.64 m | 1.72 s | 1.46 s | 7.6 m |
| 0.3, b10 | 14.0 m | 3.75 m | 1.62 s | 1.43 s | 5.9 m |
| 0.3, b20 | 25.6 m | 6.12 m | 1.72 s | 1.44 s | 10.5 m |
| 0.1, b20 | 29.7 m | 7.78 m | 1.90 s | 1.51 s | 13.7 m |

## Whole episode

| In-tunnel p | PID collisions | PPO-Lag collisions | PPO-Lag emergency-brake entries |
|---|---|---|---|
| 0.9 | 14/30 | 0/30 | 4/30 |
| 0.5 | 14/30 | 0/30 | 5/30 |
| 0.3 | 14/30 | 0/30 | 6/30 |
| 0.1 | 17/30 | 0/30 | 5/30 |
| 0.3, b10 | 16/30 | 0/30 | 5/30 |
| 0.3, b20 | 17/30 | 0/30 | 5/30 |
| 0.1, b20 | 21/30 | 0/30 | 5/30 |

## Findings

1. **Loss in the tunnel degrades tracking, not safety.** With Last Value Hold the held leader position is always behind the true one while the leader moves, so the follower believes the gap is smaller than it is and drops back. THW_min inside the tunnel never fell in any cell; it rose. No emergency-brake entry happened inside the tunnel in any cell.
2. **Mild loss barely matters.** Down to p = 0.5 with independent losses, the spacing error moves by under 1 m for PID and under 0.1 m for PPO-Lag. The effect becomes large only at p ≤ 0.3 with bursts, or p = 0.1.
3. **PID collides in 14 of 30 episodes with no tunnel effect at all.** Every collision is at the leader's final stop, outside the tunnel. Heavy tunnel loss raises this to 21/30 because the follower leaves the tunnel far behind and arrives at the stop while still catching up (worst THW 0.22 s).
4. **The existing PPO-Lag policy keeps a 3–4x smaller tunnel error than PID in every cell, without having been trained on the tunnel.**

## Consequence for the experiment

A "PPO-Lag is safer than PID/PPO in the tunnel" result cannot come from the tunnel zone itself under this communication model, because loss there pushes every controller in the safe direction. What the tunnel experiment can show is tracking error and jerk inside the tunnel, recovery after the exit, and the knock-on effect at the final stop.

## Addendum: leader hard brake inside the tunnel (same day)

Added because of the consequence above. 30 episodes per cell, seed 42, the leader brakes in every episode at a point 30–60% through the tunnel. "Brake window" = first 6 s from the start of the brake. Data: `outputs/tunnel/brake_check/` (local only).

| Condition | Controller | Emergency entry in brake window | Min THW in window (worst episode) | Min distance | Collisions (whole episode) |
|---|---|---|---|---|---|
| Brake to 5 m/s, tunnel p 0.7 ± 0.1 | PID | 0/30 | 1.02 s | 8.9 m | 14/30 (all at the final stop) |
| | PPO-Lag (emcost) | 0/30 | 1.30 s | 12.4 m | 0/30 |
| Full stop, perfect comm | PID | 13/30 | 0.79 s | 4.83 m | 14/30 |
| | PPO-Lag (emcost) | 0/25 | 1.26 s | 7.26 m | 0/25 (5 episodes lost to a client clash on the test server) |
| Full stop, tunnel p 0.7 ± 0.1 | PID | 13/30 | 0.75 s | 4.81 m | 14/30 |
| | PPO-Lag (emcost) | 0/30 | 1.25 s | 7.03 m | 0/30 |
| Full stop, tunnel p 0.3 in 1 s bursts | PID | 5/30 in window | 0.57 s | 4.82 m | 22/30 |
| | PPO-Lag (emcost) | 13/30 | 0.42 s | 4.79 m | 12/30 |

- Braking to 5 m/s is not a safety event for either controller. A full stop is.
- PID fails the full stop at 20 m/s leader speed with or without loss (13/30 in both), so its failure is not caused by the tunnel.
- The existing PPO-Lag handles the full stop at p 0.7 ± 0.1 with no emergency entry, and collides in 12/30 when the loss is heavy and bursty. This is the first setting where communication loss turns into a safety failure for the learned policy.
- These runs used the old 45 s episode limit; 8/30 (p 0.7), 5/30 (bursts) and 10/25 (perfect comm) PPO-Lag full-stop episodes ended by timeout during the final approach, not by a failure. The limit is now 60 s.
