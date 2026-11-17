#!/usr/bin/env python3
"""One line per PID cell: collisions, emergency entries and tracking, split by leader target speed.

Usage: .venv/bin/python analysis/pid_retune_summary.py DIR [DIR ...]
Each DIR is an --output-dir of a PID run (contains pid/pid_episodes.csv).
"""
import sys
from pathlib import Path

import pandas as pd

rows = []
for arg in sys.argv[1:]:
    path = Path(arg) / "pid" / "pid_episodes.csv"
    if not path.exists():
        print(f"missing {path}", file=sys.stderr)
        continue
    d = pd.read_csv(path)
    fast = d[d.leader_target_speed_mps == 20.0]
    brake = d[d.leader_brake == 1]
    rows.append({
        "cell": Path(arg).name,
        "n": len(d),
        "collisions": int(d.collision.sum()),
        "coll_brake": int(brake.collision.sum()),
        "coll_20mps": int(fast.collision.sum()),
        "emerg_entries": int(d.emergency_entered.sum()),
        "not_completed": int((d.end_reason != "profile_completed").sum()),
        "min_dist_m": round(d.min_distance_m.min(), 2),
        "abs_err_m": round(d.mean_abs_spacing_error_m.mean(), 2),
        "jerk_mps3": round(d.mean_abs_jerk_mps3.mean(), 1),
        "thw_min_s": round(d.thw_min_s.min(), 2),
        "thw_min_mean_s": round(d.thw_min_s.mean(), 2),
    })
print(pd.DataFrame(rows).to_string(index=False))
