# Implementation notes: paper vs. code

Reference: the team's ITSC 2026 manuscript (`3월_논문.pdf`, Algorithm 1, eq. (8)–(15), Table I).
Code: `episode_ppo_v2.py` (PPO-Lag follower) and `v2v_comm.py` (V2V channel).

- **[PAPER]**: the value or definition comes from the manuscript.
- **[ASSUMED]**: the manuscript does not specify it, so the code picked it. These items need to be either stated in the report or replaced with the paper's real setting.
- **[DEVIATION]**: the manuscript does specify it, but the code deliberately does it differently.

Last updated: 2026-11-03 (V2V communication model, 7-dim state, r_i changed to the planned-route reference; emergency-brake entry cost and override masking).

---

## 1. Communication model (Algorithm 1, lines 4 and 10)

| Item | Implementation | Source |
|---|---|---|
| Comm success probability | `--p-success`, default 0.9 | [PAPER] Table I |
| Failure process | One independent Bernoulli(p_success) draw per tick (T_s = 0.05 s) | [ASSUMED]: the paper says only "modeled probabilistically". No burst/Gilbert-Elliott model and no latency. |
| What is held on failure | The preceding vehicle's **position, speed, and acceleration** (last successfully received values) | [PAPER] Sec. IV-A: "the information of the preceding CAV was replaced with the most recently received value" |
| Δd, Δv on failure | Recomputed every tick from the held preceding-vehicle state and the **current** own-vehicle sensor values: Δd = ‖s_{i−1}^last − s_i(t)‖ − (d₀ + h·v_i(t)), Δv = v_{i−1}^last − v_i(t) | [ASSUMED] Option A. The ^last superscript in eq. (8) could also mean "freeze Δd and Δv themselves". We chose the reading that matches Sec. IV-A. |
| Communication memory reset | At the start of each episode, initialize to the preceding vehicle's actual state; t_i = 0, flag_i = 0 | [PAPER] Algorithm 1 line 4; the initial values are [ASSUMED] |
| Comm RNG | A `random.Random` per channel, reseeded with the string `(seed, channel index, episode number)` at the start of every episode | [ASSUMED]. Episodes within a run get different patterns; the same seed reproduces the same sequence. Separate from the policy-sampling RNG. |
| Channel structure | One `V2VChannel` per receiving vehicle, managed as `{receiver: channel}`. Currently only `("follower", "leader")` | For a future N-vehicle setup (Algorithm 1 line 21), extend the chain |
| PID controller | `--controller pid` in `episode_ppo_v2.py` still uses true values (no comm model) | For PID with comm degradation, see `episode_comm.py`. To be decided in P3. |

### Ground truth vs. observed values
- **Observed values (through comm)**: the policy input Δd, Δv, a_{i−1}.
- **True values (from the simulator)**: reward, THW cost, hard_violation check, 5 m emergency brake, and every performance metric in the CSV (spacing error, relative velocity, THW, jerk).
  - [ASSUMED] The paper does not say which values the reward and cost use. We use true values so the training signal and evaluation metric reflect actual safety.

## 2. State vector (eq. (8))

S_i = [Δd^last, Δv^last, a_{i−1}^last, v_i, r_i, flag_i, t_i]

| Index | Term | Definition | Normalization (÷) | Source |
|---|---|---|---|---|
| 0 | Δd^last | See above, [m]. Distance is Euclidean between vehicle centers | 20 | Definition [PAPER] eq. (4); **Euclidean distance is a [DEVIATION]** (paper uses path-wise distance) |
| 1 | Δv^last | v_{i−1}^last − v_i [m/s] | 10 | [PAPER] eq. (5) |
| 2 | a_{i−1}^last | Preceding vehicle's longitudinal acceleration [m/s²] (projection of `get_acceleration()` onto the heading) | 5 | [PAPER]; measurement method is [ASSUMED] |
| 3 | v_i | Own-vehicle speed [m/s] | 30 | [PAPER] |
| 4 | r_i | Signed lateral distance [m] from the follower's **planned route** (BasicAgent global plan, waypoints about every 2 m, treated as a polyline), positive to the right of the direction of travel | 2 | [ASSUMED]: the paper names it but does not define it |
| 5 | flag_i | 1 if this tick's comm failed, 0 if it succeeded | 1 | [ASSUMED] polarity: the paper only says "failure indicator" |
| 6 | t_i | Time since the last successful comm [s] (0 on success, +T_s per failure) | Clip to [0, 5] s, then ÷1 | Definition [PAPER]; clip and scale are [ASSUMED] |

- **Normalization**: the paper says nothing about it. We use fixed scales [ASSUMED] so every term is O(1). No running observation normalization.
- **r_i definition (changed 11/03)**
  - **Matching**: closest point on the route polyline, searched within 2 segments behind and 25 ahead of the previous match.
  - **Sign**: right of the segment direction is positive (CARLA left-handed coordinates).
  - **Ends**: the first and last segments are extended as straight lines. The follower starts behind the route start and stops about 20 m past the route end, because the leader brakes to a stop at its destination. Without the extension, longitudinal overshoot was counted as r_i (up to 20 m).
  - **Why the definition changed**: the original nearest-lane-centerline definition flipped from +1.7 to −1.7 m at a lane boundary, a 3.3 m jump in one tick (tick 257).
  - **11/03 eval result** (10/26 checkpoint, p=1.0, 668 ticks): max |r_i| = 1.67 m (tick 288, lag while following the lane-change segment), max tick-to-tick change = 0.24 m, |r_i| > 1 m for 36 ticks.
- **4-dim compatibility**: `--obs-dim 4` uses only indices 0–3. A checkpoint carries its `obs_dim` (pre-comm checkpoints are detected as 4 from the weight shape). The 10/26 checkpoint evaluated with `--p-success 1.0` is the ideal-communication baseline (confirmed on 11/03 that observed values equal true values exactly).

## 3. Reward and cost (eq. (10)–(15))

| Item | Implementation | Source |
|---|---|---|
| ω1, ω2, ω3 | 0.1, 0.25, 0.25 | [PAPER] |
| Q | diag(2, 1); Δq in eq. (10) is read as Δv | Values [PAPER], interpretation [ASSUMED] |
| a_i in r_traf, r_jerk | The follower's **measured longitudinal acceleration** [m/s²], not the action value | [ASSUMED] (the paper also uses a_i for the action) |
| d_far, d_close, α1, α2 | 8.0, −4.0, 20.0, 10.0 | [PAPER] |
| τ_safe, τ_danger | 1.2 s, 1.0 s | [PAPER] |
| THW computation | THW = ∞ (cost 0) when v_i ≤ 0.5 m/s | [ASSUMED] |
| Failure terminal penalty | hard_violation and collisions add −penalty/(1−γ) to the reward | [ASSUMED] |
| Collision terminal cost | Cost += 1/(1−γ) | [ASSUMED] |
| Emergency-brake entry cost | c_t = c_THW(THW_t) + K·1[first tick of the episode with D_t < d_em], d_em = 5 m (true center-to-center distance), K = d / r_target with r_target = 0.1 → K = 50. It is charged once per episode. It enters the cost critic (folded into the 10-tick window like any tick cost) and the λ update. `--emergency-target-rate 0` disables it and reproduces commit 819fd81. `episode_thw_cost` is logged separately so the paper's THW cost metric stays reportable. The episode CSV also carries `emergency_entered` and `emergency_ticks`, and the tick CSV carries `emergency_brake`. | **[DEVIATION]**: the paper's cost (eq. (14)–(15)) is THW-based only, and the 5 m emergency brake is our own safety layer. The override is silent and THW is undefined below 0.5 m/s, so the policy never saw creeping into the override zone as unsafe. It learned to rely on it at the final stop: 329/500 of run1026's last training episodes entered it, and 19/20 comm09 deterministic eval episodes at p=0.5. A per-tick indicator was rejected: one entry then costs about 20 (set by the 1 s stop-hold, not by design), and replaying the comm09 training logs showed λ would have stayed near 0 after episode 1000. With per-tick normalization, the effective target rate is d·T̄/(650·K) ≈ 10.9% at T̄ ≈ 707 ticks. r_target is [ASSUMED]. |
| Collision cost and λ (known asymmetry, not changed) | The collision terminal cost 1/(1−γ) is added to the last transition only, so it reaches the cost critic. λ uses `cost_sum / ticks` over the episode's tick costs, which does not include it. A collision therefore moves λ only through the THW ticks before the impact. Meanwhile an emergency-brake entry adds K = 50 to the λ input, so from λ's point of view an entry weighs more than a collision. | Recorded for the report; deliberately left as is |
| Time-limit endings | timeout and leader_stuck bootstrap with γ·V(s) instead of being treated as terminal | [ASSUMED] (standard handling) |
| hard_violation | Ends the episode when the spacing error exceeds 40 m | The ending condition is [PAPER] Algorithm 1 line 13; **the 40 m threshold is [ASSUMED]** |

## 4. PPO-Lag (Algorithm 1, lines 1, 18, 19)

| Item | Implementation | Source |
|---|---|---|
| Initial λ | 0.0 | [ASSUMED]: the paper only requires λ ≥ 0 |
| η (λ step size) | 0.01 per episode → 6.5 after per-tick scaling | [ASSUMED]: not in the paper |
| d (cost limit) | 5 (per-episode cumulative) → 5/650 = 0.00769 as a per-tick mean | Value [PAPER]; **per-tick unit is a [DEVIATION]** (removes the incentive to end episodes early) |
| Nominal episode length | 650 ticks (measured) | [ASSUMED] |
| Update period | Every 512 decisions in the rollout buffer, regardless of episode boundaries | **[DEVIATION]**: Algorithm 1 updates per episode |
| λ update period | Per PPO update, using the mean of episodes finished since the previous update | **[DEVIATION]** |
| Objective | Clipped surrogate (ε = 0.2); combined advantage (A_r − λA_c)/(1+λ) | Clip [ASSUMED]; the (1+λ) normalization follows OmniSafe PPOLag and is not in the paper |
| Override masking | Transitions whose whole 10-tick window was overridden by the emergency brake get actor_mask = 0. They are excluded from the clipped surrogate, the entropy term, and the advantage normalization statistics, but kept for both critics and GAE. Partially overridden windows are kept. On an episode's last window, only executed ticks count. `masked_transitions` per update is logged in `ppo_updates.csv`. Off when `--emergency-target-rate 0`. | **[DEVIATION]**: not in the paper. The executed control in those windows did not depend on the sampled action, so its log-prob ratio carried no information about the action's effect. |
| Networks | Separate actor, reward critic, and cost critic; each 2×128 tanh MLP | [ASSUMED] |
| Optimizer | Adam, lr 3e-4, 10 epochs, minibatch 64, separate grad clip 0.5 for actor and critics | [ASSUMED] |
| γ, GAE λ | 0.99 per tick (0.99¹⁰ per decision), 0.95 | [ASSUMED] |
| Entropy, value coefficients | 0.01, 0.5, 0.5 | [ASSUMED] |
| Reward-critic targets | Normalized with running mean/std | [ASSUMED] |
| Action | tanh-squashed Gaussian → [−1, 1] → throttle/brake (eq. (9)) | Mapping [PAPER]; squashing [ASSUMED] |
| Action repeat | One policy decision is held for 10 ticks (0.5 s) | **[DEVIATION]**: the paper decides every tick. Needed because this vehicle only moves from rest after throttle ≥ 0.2 is held. Comm runs every tick regardless. |

## 5. Environment and other

| Item | Implementation | Source |
|---|---|---|
| Map | Town03_Opt tunnel route, 2 vehicles (Leader + Follower 1) | **[DEVIATION]**: the paper uses Town04 highway with 3 followers |
| T_s, h, d₀ | 0.05 s, 1.0 s, 7.0 m | [PAPER] |
| Emergency brake | Brake = 1.0 when center-to-center distance < 5 m (overrides the policy). The first override tick of an episode is charged the entry cost K (Sec. 3), and fully overridden decision windows are masked out of the actor loss (Sec. 4) | [ASSUMED]: not in the paper. Roughly bumper contact. |
| Lateral control | BasicAgent (CARLA) steering | Paper uses a centerline-based PID; equivalent role |
| Policy deployment (line 21) | **Not implemented** | Planned for P4 |
| ROS2 | Training runs CARLA Python API directly, no ROS2 | **[DEVIATION]** from the paper's ROS2 co-simulation |
