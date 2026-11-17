# PID retune for the tunnel scenario (2026-11-17)

Goal: zero PID collisions with no packet loss (p = 1.0) and the leader's hard brake. The old gains collided in 28 of 100 episodes there, all at a leader speed of 20 m/s (`2026-11-16_tunnel_eval.md`).

**Result: kd 0.1 → 0.6, kp 0.4 and ki 0.01 unchanged.** `configs/scenario_town03.json` now holds the new value. h, d0, the brake mapping and all code are untouched. The PID numbers in the 11-16 document are the old gains.

## Cause

The control law is u = kp·Δd + ki·∫Δd + kd·Δv, brake = −u/4. When the leader brakes from 20 m/s at about 7 m/s², Δd is still positive (the follower runs about 3.5 m long at 20 m/s) and the only braking term is kd·Δv. With kd = 0.1 the follower's brake pedal stayed at 0 for the first 1.0 s after the leader started braking and reached 0.5 only after 1.8 s; it arrived at the 5 m line still moving at about 6 m/s. With kd = 0.6 braking starts 0.5 s after the leader's and is at full pedal by 2.0 s; in the same episode the follower stops 13 m behind the leader.

## Method

- No code change. `analysis/pid_retune_sweep.sh` writes a copy of the scenario file per gain set and passes it with `--scenario`; `analysis/pid_retune_summary.py` tabulates. Data: `outputs/tunnel/pid_retune/` (local only).
- Same evaluation as the tunnel experiment: 100 episodes, evaluation seed 1000 (same leader speeds, brake events and packet draws as every earlier cell), brake probability 0.5.
- Stage 1: kp {0.4, 0.8} × kd {0.3, 0.6, 1.0, 1.5} at p = 1.0. Stage 2: two candidates over in-tunnel p 0.9 / 0.7 / 0.5 / 0.3 / 0.1 (p 0.9 outside, noise std 0.1).
- Check: with the edited default scenario file, 5 episodes are byte-identical to the sweep cell of the chosen gains.

## Stage 1: no loss, with hard brake (100 episodes each)

| kp / kd | Collisions | Emergency entries | Min gap (m) | Mean \|spacing error\| (m) | Mean \|jerk\| (m/s³) | Min THW (s) |
|---|---|---|---|---|---|---|
| 0.4 / 0.1 (old) | 28 | 28 | 4.82 | 3.09 | 14.2 | 0.80 |
| 0.4 / 0.3 | 0 | 0 | 7.42 | 2.42 | 13.5 | 1.33 |
| **0.4 / 0.6 (chosen)** | **0** | **0** | **7.74** | **2.31** | **13.3** | **1.45** |
| 0.4 / 1.0 | 0 | 0 | 8.11 | 2.42 | 13.5 | 1.36 |
| 0.4 / 1.5 | 0 | 0 | 8.60 | 2.88 | 14.7 | 1.27 |
| 0.8 / 0.3 | 0 | 0 | 7.18 | 1.34 | 13.4 | 1.36 |
| 0.8 / 0.6 | 0 | 0 | 7.35 | 1.26 | 13.9 | 1.42 |
| 0.8 / 1.0 | 0 | 0 | 7.54 | 1.40 | 13.8 | 1.37 |
| 0.8 / 1.5 | 0 | 0 | 7.78 | 1.79 | 14.2 | 1.31 |

Every retuned set reaches zero collisions; the old kd was simply too small. With the chosen gains the worst THW at 20 m/s rises from 0.80 s to 1.45 s, above τ_safe = 1.2 s, and tracking and jerk improve at all three leader speeds.

## Stage 2: tunnel loss

Collisions out of 100 (PPO-Lag: 300, three seeds pooled, from the 11-16 evaluation):

| In-tunnel p | PID old | PID 0.4 / 0.6 (chosen) | PID 0.8 / 0.6 | PPO-Lag |
|---|---|---|---|---|
| 1.0 | 28 | 0 | 0 | 0% |
| 0.9 | 28 | 0 | 0 | 0% |
| 0.7 | 28 | 0 | 0 | 0% |
| 0.5 | 28 | 0 | 0 | 0% |
| 0.3 | 31 | 0 | 0 | 3.3% (2 / 0 / 8 per seed) |
| 0.1 | 71 | 2 | 27 | 35.7% (35 / 23 / 49 per seed) |

Mean |spacing error| inside the tunnel (m):

| In-tunnel p | PID old | PID 0.4 / 0.6 (chosen) | PID 0.8 / 0.6 | PPO-Lag |
|---|---|---|---|---|
| 1.0 | 2.98 | 2.54 | 1.32 | 0.73 |
| 0.9 | 3.07 | 2.64 | 1.41 | 0.73 |
| 0.7 | 3.31 | 2.86 | 1.66 | 0.71 |
| 0.5 | 3.80 | 3.31 | 2.20 | 0.67 |
| 0.3 | 5.38 | 4.71 | 4.13 | 0.89 |
| 0.1 | 22.75 | 16.15 | 20.28 | 2.92 |

Chosen gains, other safety metrics: no emergency-brake entry and no Caution or Danger step (THW < 1.2 s) at any p from 1.0 to 0.3; minimum THW 1.45–1.47 s. At p = 0.1: 2 collisions, 2 emergency entries, THW cost 0.24 per episode, 0.22 Danger steps per episode (PPO-Lag: 10.79 and 8.91).

- **Why 0.4 / 0.6 and not 0.8 / 0.6.** Both meet the goal. kp 0.8 tracks tighter without loss (1.26 vs 2.31 m over the episode) but collides 27 times at p = 0.1, all outside the tunnel and 20 of them in episodes without a hard brake: after falling far behind in the tunnel, the larger position gain closes the gap too fast. kp 0.4 has 2 collisions there (both in hard-brake episodes inside the tunnel, in episodes whose longest stretch without a packet was 4.0 s and 5.1 s). The chosen set is also a one-gain change from the old one.
- **What this changes in the comparison.** With a tuned PID, "the learned controllers are safer than PID" no longer holds. The retuned PID has no collision down to p = 0.3 and 2% at p = 0.1, below PPO-Lag (3.3% and 35.7%) and PPO (7.3% and 43.7%).
- **What still holds.** PID buys that safety by dropping back: under loss the held leader position lags the true one, so PID sees a smaller gap than the real one and slows down. Its in-tunnel spacing error is 3.5 times PPO-Lag's at p ≥ 0.9 (2.5–2.6 m vs 0.73 m), 5 times at p = 0.3 and 5.5 times at p = 0.1 (16 m vs 2.9 m). The learned controllers keep the gap tight under loss and pay for it in hard-brake episodes.

## Check on a fresh evaluation seed

Evaluation seed 2000 instead of 1000 (different leader speeds, brake events and packet draws; 36 of its 100 episodes are at 20 m/s), same settings otherwise. Chosen gains, 100 episodes per cell:

| In-tunnel p | Collisions | Emergency entries | Min gap (m) | Min THW (s) | \|err\| in tunnel (m) | THW cost / episode |
|---|---|---|---|---|---|---|
| 1.0, old gains (kd 0.1) | 36 | 36 | 4.82 | 0.80 | | |
| 1.0 | 0 | 0 | 7.75 | 1.45 | 2.51 | 0 |
| 0.9 | 0 | 0 | 7.76 | 1.46 | 2.60 | 0 |
| 0.7 | 0 | 0 | 7.75 | 1.46 | 2.82 | 0 |
| 0.5 | 0 | 0 | 7.76 | 1.46 | 3.25 | 0 |
| 0.3 | 0 | 0 | 7.74 | 1.46 | 4.49 | 0 |
| 0.1 | 2 | 2 | 4.83 | 0.29 | 14.12 | 0.55 |

The result carries over: zero collisions down to p = 0.3 and 2 of 100 at p = 0.1 (both in hard-brake episodes inside the tunnel), the same counts as on seed 1000. The old gains collide in all 36 of the 20 m/s episodes of this seed, so the new seed is not an easier one.

## Limits

- One 100-episode run per cell and two evaluation seeds; PID is deterministic given the seed, so there is no training-seed spread to report.
- `configs/scenario_town03_comm_50.json` and `_comm_70.json` still carry kd 0.1.
