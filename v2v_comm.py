"""Per-vehicle V2V communication channel with Last Value Hold (paper Algorithm 1, line 10).

Each follower owns one channel that receives the preceding vehicle's state. Every simulator
tick the channel draws one Bernoulli(p_success) packet:
  success -> store the sender's current location, speed, and acceleration; t_i = 0, flag_i = 0
  failure -> keep the last received values;                               t_i += dt, flag_i = 1

Only the preceding vehicle's raw information is held (paper Sec. IV-A: "the information of the
preceding CAV was replaced with the most recently received value"). The receiver recomputes
Delta d and Delta v from the held sender state and its own current sensor state.

Reproducibility: each channel has its own random.Random, reseeded at every episode from
(seed, channel_index, episode). Episodes inside one run therefore see different failure
patterns, the same seed replays the same sequence, and the draws never touch the policy's RNG.
"""

import random


class V2VChannel:
    def __init__(self, receiver_id, sender_id, channel_index, p_success, seed):
        if not 0.0 <= float(p_success) <= 1.0:
            raise ValueError(f"p_success must be in [0, 1], got {p_success}")
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
        self.last = dict(sender_state)
        self.flag = 0
        self.t_since_success = 0.0
        self.last_success = True
        self.stats = {
            "ticks": 0,
            "failures": 0,
            "current_streak": 0,
            "max_consecutive_failures": 0,
            "max_t_since_success_s": 0.0,
        }

    def step(self, sender_state, dt):
        """Algorithm 1 line 10: one packet per tick. Returns the reading the receiver sees."""
        if self.last is None:
            raise RuntimeError("V2VChannel.step() called before reset().")
        success = self.p_success >= 1.0 or self.rng.random() < self.p_success
        stats = self.stats
        stats["ticks"] += 1
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
        }

    def summary(self):
        return {
            "comm_ticks": self.stats["ticks"],
            "comm_failures": self.stats["failures"],
            "comm_max_consecutive_failures": self.stats["max_consecutive_failures"],
            "comm_max_t_since_success_s": round(self.stats["max_t_since_success_s"], 3),
        }


def create_channels(chain, p_success, seed):
    """chain: [(receiver_id, sender_id), ...] in platoon order, e.g. [("follower", "leader")].

    Returns {receiver_id: V2VChannel}. A future N-follower platoon only extends the chain:
    [("follower", "leader"), ("follower_2", "follower"), ...].
    """
    return {
        receiver_id: V2VChannel(receiver_id, sender_id, index, p_success, seed)
        for index, (receiver_id, sender_id) in enumerate(chain)
    }
