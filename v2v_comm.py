"""Per-vehicle V2V communication channel with Last Value Hold (paper Algorithm 1, line 10).

Each follower owns one channel that receives the preceding vehicle's state. Every simulator
tick the channel draws one Bernoulli(p_success) packet:
  success -> store the sender's current location, speed, and acceleration; t_i = 0, flag_i = 0
  failure -> keep the last received values;                               t_i += dt, flag_i = 1

Only the preceding vehicle's raw information is held (paper Sec. IV-A: "the information of the
preceding CAV was replaced with the most recently received value"). The receiver recomputes
Delta d and Delta v from the held sender state and its own current sensor state.

Tunnel zone [ASSUMED, not in the paper]: while the caller reports the link as inside the tunnel,
the success probability is p_tunnel instead of p_success. p_tunnel may be a (low, high) range, in
which case one value is drawn per episode (training randomization). With tunnel_burst_ticks > 1
the in-tunnel losses follow a two-state Gilbert model with that mean loss-run length and the same
average success rate p_tunnel, so losses arrive in bursts instead of independently.
With p_tunnel_noise_std > 0 the in-tunnel probability itself fluctuates in time:
  p(t) = clip(p_tunnel + n(t), 0, 1),   n(t) Gaussian with that std and correlation time tau.
n(t) is a function of the clock only, so every controller run with the same seed sees the same
p(t). tau = 0 gives white noise, which leaves the packet statistics identical to a fixed
p_tunnel (a fresh zero-mean offset every tick averages out inside the Bernoulli draw); only
tau > 0 produces stretches where the link is noticeably better or worse than its mean.

Reproducibility: each channel has its own random.Random, reseeded at every episode from
(seed, channel_index, episode). Episodes inside one run therefore see different failure
patterns, the same seed replays the same sequence, and the draws never touch the policy's RNG.
Exactly one uniform number u_t is drawn per tick, whatever the probability, and the packet
succeeds when u_t is below the success probability of that tick. Controllers evaluated with the
same seed therefore share the same u_t sequence (common random numbers) even when their
positions, and so their probabilities, differ.
"""

import math
import random


class V2VChannel:
    def __init__(
        self, receiver_id, sender_id, channel_index, p_success, seed, p_tunnel=None, tunnel_burst_ticks=0.0,
        p_tunnel_noise_std=0.0, p_tunnel_noise_tau_s=0.0,
    ):
        if not 0.0 <= float(p_success) <= 1.0:
            raise ValueError(f"p_success must be in [0, 1], got {p_success}")
        if p_tunnel is not None:
            low, high = (float(p_tunnel), float(p_tunnel)) if isinstance(p_tunnel, (int, float)) else p_tunnel
            low, high = float(low), float(high)
            if not 0.0 <= low <= high <= 1.0:
                raise ValueError(f"p_tunnel must satisfy 0 <= low <= high <= 1, got {p_tunnel}")
            p_tunnel = (low, high)
        tunnel_burst_ticks = float(tunnel_burst_ticks)
        if tunnel_burst_ticks > 1.0 and p_tunnel is not None and 0.0 < p_tunnel[0] < 1.0:
            # Gilbert model: P(loss | previous success) = (1 - p) / (p * L) must stay <= 1.
            shortest = (1.0 - p_tunnel[0]) / p_tunnel[0]
            if tunnel_burst_ticks < shortest:
                raise ValueError(
                    f"tunnel_burst_ticks={tunnel_burst_ticks:g} is too short for p_tunnel={p_tunnel[0]:g}; "
                    f"need at least {shortest:.2f}"
                )
        self.p_tunnel_range = p_tunnel
        self.tunnel_burst_ticks = tunnel_burst_ticks
        self.p_tunnel = None
        self.p_now = float(p_success)
        self.noise_std = max(0.0, float(p_tunnel_noise_std))
        self.noise_tau_s = max(0.0, float(p_tunnel_noise_tau_s))
        self.noise = 0.0
        self.noise_rng = random.Random()
        self.receiver_id = receiver_id
        self.sender_id = sender_id
        self.channel_index = int(channel_index)
        self.p_success = float(p_success)
        self.seed = int(seed)
        self.rng = random.Random()
        self.last = None
        self.flag = 0
        self.t_since_success = 0.0
        self.last_success = True
        self.stats = {}

    def reset(self, episode, sender_state):
        """Algorithm 1 line 4: clear memory and start from the sender's actual state."""
        # A str seed is hashed deterministically (sha512), unlike tuple seeds.
        self.rng.seed(f"v2v:{self.seed}:{self.channel_index}:{int(episode)}")
        self.p_tunnel = None
        if self.p_tunnel_range is not None:
            low, high = self.p_tunnel_range
            # Own stream, so randomizing p_tunnel never shifts the per-tick u_t sequence.
            draw = random.Random(f"v2v-ptunnel:{self.seed}:{self.channel_index}:{int(episode)}").random()
            self.p_tunnel = low + (high - low) * draw
        self.p_now = self.p_success
        # Own stream again: the noise never shifts u_t, and u_t never shifts the noise.
        self.noise_rng.seed(f"v2v-noise:{self.seed}:{self.channel_index}:{int(episode)}")
        self.noise = self.noise_rng.gauss(0.0, self.noise_std) if self.noise_std > 0.0 else 0.0
        self.last = dict(sender_state)
        self.flag = 0
        self.t_since_success = 0.0
        self.last_success = True
        self.stats = {
            "ticks": 0,
            "failures": 0,
            "tunnel_ticks": 0,
            "tunnel_failures": 0,
            "current_streak": 0,
            "max_consecutive_failures": 0,
            "max_t_since_success_s": 0.0,
        }

    def _success_probability(self, in_tunnel):
        """Success probability of this tick's packet given the zone and the previous packet."""
        if not in_tunnel or self.p_tunnel is None:
            self.p_now = self.p_success
            return self.p_success
        p = min(1.0, max(0.0, self.p_tunnel + self.noise))
        self.p_now = p
        burst = self.tunnel_burst_ticks
        if burst <= 1.0 or p <= 0.0 or p >= 1.0:
            return p
        # Gilbert model: a loss run ends with probability 1/L per tick, and a success run breaks
        # with probability (1 - p) / (p * L), which keeps the long-run success rate at p.
        if self.last_success:
            return 1.0 - min(1.0, (1.0 - p) / (p * burst))
        return 1.0 / burst

    def step(self, sender_state, dt, in_tunnel=False):
        """Algorithm 1 line 10: one packet per tick. Returns the reading the receiver sees."""
        if self.last is None:
            raise RuntimeError("V2VChannel.step() called before reset().")
        # Always draw: the u_t sequence must not depend on the probability (see module docstring).
        u = self.rng.random()
        if self.noise_std > 0.0:
            # AR(1) step with stationary std noise_std; advanced every tick, inside the tunnel or not.
            keep = math.exp(-float(dt) / self.noise_tau_s) if self.noise_tau_s > 0.0 else 0.0
            self.noise = keep * self.noise + math.sqrt(1.0 - keep * keep) * self.noise_rng.gauss(0.0, self.noise_std)
        success = u < self._success_probability(bool(in_tunnel))
        stats = self.stats
        stats["ticks"] += 1
        in_zone = bool(in_tunnel) and self.p_tunnel is not None
        stats["tunnel_ticks"] += int(in_zone)
        stats["tunnel_failures"] += int(in_zone and not success)
        if success:
            self.last = dict(sender_state)
            self.t_since_success = 0.0
            self.flag = 0
            stats["current_streak"] = 0
        else:
            self.t_since_success += float(dt)
            self.flag = 1
            stats["failures"] += 1
            stats["current_streak"] += 1
            stats["max_consecutive_failures"] = max(
                stats["max_consecutive_failures"], stats["current_streak"]
            )
            stats["max_t_since_success_s"] = max(stats["max_t_since_success_s"], self.t_since_success)
        self.last_success = success
        return self.reading()

    def reading(self):
        return {
            "location": self.last["location"],
            "speed_mps": self.last["speed_mps"],
            "accel_mps2": self.last["accel_mps2"],
            "success": self.last_success,
            "flag": self.flag,
            "t_since_success_s": self.t_since_success,
            "p_now": self.p_now,
        }

    def summary(self):
        return {
            "comm_ticks": self.stats["ticks"],
            "comm_failures": self.stats["failures"],
            "comm_max_consecutive_failures": self.stats["max_consecutive_failures"],
            "comm_max_t_since_success_s": round(self.stats["max_t_since_success_s"], 3),
            "comm_tunnel_ticks": self.stats["tunnel_ticks"],
            "comm_tunnel_failures": self.stats["tunnel_failures"],
            "p_tunnel": "" if self.p_tunnel is None else round(self.p_tunnel, 4),
        }


def create_channels(
    chain, p_success, seed, p_tunnel=None, tunnel_burst_ticks=0.0, p_tunnel_noise_std=0.0, p_tunnel_noise_tau_s=0.0
):
    """chain: [(receiver_id, sender_id), ...] in platoon order, e.g. [("follower", "leader")].

    Returns {receiver_id: V2VChannel}. A future N-follower platoon only extends the chain:
    [("follower", "leader"), ("follower_2", "follower"), ...].
    """
    return {
        receiver_id: V2VChannel(
            receiver_id, sender_id, index, p_success, seed, p_tunnel, tunnel_burst_ticks,
            p_tunnel_noise_std, p_tunnel_noise_tau_s,
        )
        for index, (receiver_id, sender_id) in enumerate(chain)
    }
