"""Zone-by-zone summary of tunnel-communication evaluations, for PID and PPO runs alike.

Usage (from anywhere):
  python3 analysis/tunnel_summarize.py LABEL=RUN_DIR [LABEL=RUN_DIR ...] [--csv out.csv]
                                       [--brake only|none]
--brake only keeps the episodes with a leader hard-brake event, --brake none the others.

RUN_DIR is the --output-dir of one episode_ppo_v2.py run; the tick CSVs and the per-episode
log are found under RUN_DIR/pid or RUN_DIR/ppo_eval (or RUN_DIR/ppo).

Each episode is cut into four zones by the follower's own position and the clock:
  pre       before the follower enters the tunnel
  tunnel    follower inside the tunnel
  recovery  first RECOVERY_SEC seconds after the follower leaves the tunnel
  final     the rest: the leader's stop at its destination
A fifth window, brake, covers BRAKE_WINDOW_SEC seconds from the start of the leader's hard-brake
event (episodes without the event are left out of it). It overlaps the zones above.
Per zone: mean and max |spacing error|, minimum THW, mean |jerk|, share of episodes whose first
emergency-brake tick falls in the zone, and the mean gap between the spacing error the
controller was given (held V2V value) and the true one. Values are means over episodes, with a
95% interval (normal approximation for means, Wilson for rates).
"""
import csv
import math
import sys
from pathlib import Path

RECOVERY_SEC = 2.0
THW_MIN_SPEED_MPS = 0.5
BRAKE_WINDOW_SEC = 6.0
ZONES = ("pre", "tunnel", "recovery", "final")
WINDOWS = ZONES + ("brake",)


def find_tick_dir(run_dir):
    for name in ("pid", "ppo_eval", "ppo"):
        folder = Path(run_dir) / name
        if list(folder.glob(f"{name}_episode_*.csv")):
            return folder, name
    raise SystemExit(f"No tick CSVs under {run_dir}")


def load(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def zone_labels(rows):
    inside = [i for i, r in enumerate(rows) if r["follower_in_tunnel"] == "1"]
    if not inside:
        return ["pre"] * len(rows)
    first, last = inside[0], inside[-1]
    exit_time = float(rows[last]["time_sec"])
    labels = []
    for i, r in enumerate(rows):
        if i < first:
            labels.append("pre")
        elif i <= last:
            labels.append("tunnel")
        elif float(r["time_sec"]) <= exit_time + RECOVERY_SEC:
            labels.append("recovery")
        else:
            labels.append("final")
    return labels


def episode_zone_metrics(rows):
    labels = zone_labels(rows)
    dt = float(rows[1]["time_sec"]) - float(rows[0]["time_sec"]) if len(rows) > 1 else 0.05
    entry_tick = next((i for i, r in enumerate(rows) if r["emergency_brake"] == "1"), None)
    out = {}
    brake_ticks = [i for i, r in enumerate(rows) if r.get("leader_brake_event") == "1"]
    brake_end = float(rows[brake_ticks[0]]["time_sec"]) + BRAKE_WINDOW_SEC if brake_ticks else None
    for zone in WINDOWS:
        if zone == "brake":
            idx = [i for i in range(brake_ticks[0], len(rows)) if float(rows[i]["time_sec"]) <= brake_end] if brake_ticks else []
        else:
            idx = [i for i, label in enumerate(labels) if label == zone]
        if not idx:
            continue
        err = [abs(float(rows[i]["spacing_error_m"])) for i in idx]
        thw = [
            float(rows[i]["distance_m"]) / float(rows[i]["follower_speed_mps"])
            for i in idx
            if float(rows[i]["follower_speed_mps"]) > THW_MIN_SPEED_MPS
        ]
        jerk = [
            abs(float(rows[i]["follower_accel_mps2"]) - float(rows[i - 1]["follower_accel_mps2"])) / dt
            for i in idx
            if i > 0
        ]
        info = [abs(float(rows[i]["obs_spacing_error_m"]) - float(rows[i]["spacing_error_m"])) for i in idx]
        out[zone] = {
            "err_mean": sum(err) / len(err),
            "err_max": max(err),
            "thw_min": min(thw) if thw else None,
            "jerk": sum(jerk) / len(jerk) if jerk else None,
            "info_err": sum(info) / len(info),
            "loss": sum(rows[i]["comm_success"] == "0" for i in idx) / len(idx),
            "entry": int(entry_tick is not None and entry_tick in set(idx)),
            "dist_min": min(float(rows[i]["distance_m"]) for i in idx),
        }
    return out


def mean_ci(values):
    values = [v for v in values if v is not None]
    if not values:
        return None, None
    m = sum(values) / len(values)
    if len(values) < 2:
        return m, 0.0
    sd = math.sqrt(sum((v - m) ** 2 for v in values) / (len(values) - 1))
    return m, 1.96 * sd / math.sqrt(len(values))


def wilson(k, n):
    if n == 0:
        return 0.0, 0.0, 0.0
    z = 1.96
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return p, max(0.0, centre - half), min(1.0, centre + half)


def summarize(label, run_dir, brake=None):
    folder, name = find_tick_dir(run_dir)
    episodes = load(folder / f"{name}_episodes.csv") if (folder / f"{name}_episodes.csv").exists() else []
    if not episodes and name != "pid":
        episodes = load(folder / "ppo_episodes.csv")
    per_episode = []
    keep = set()
    for path in sorted(folder.glob(f"{name}_episode_*.csv")):
        rows = load(path)
        if not rows or "follower_in_tunnel" not in rows[0]:
            continue
        had_brake = any(r.get("leader_brake_event") == "1" for r in rows)
        if brake is not None and had_brake != (brake == "only"):
            continue
        keep.add(rows[0]["episode"])
        per_episode.append(episode_zone_metrics(rows))
    if brake is not None:
        episodes = [e for e in episodes if e["episode"] in keep]
    n = len(episodes)
    result = {"label": label, "episodes": n, "zones": {}}
    for key, column, value in (
        ("completed", "end_reason", "profile_completed"),
        ("collision", "collision", "1"),
        ("emergency", "emergency_entered", "1"),
    ):
        result[key] = wilson(sum(e[column] == value for e in episodes), n)
    for zone in WINDOWS:
        present = [m[zone] for m in per_episode if zone in m]
        stats = {k: mean_ci([m[k] for m in present]) for k in ("err_mean", "err_max", "thw_min", "jerk", "info_err", "loss")}
        stats["thw_worst"] = min((m["thw_min"] for m in present if m["thw_min"] is not None), default=None)
        stats["dist_worst"] = min((m["dist_min"] for m in present), default=None)
        stats["n"] = len(present)
        stats["entry"] = wilson(sum(m["entry"] for m in present), len(present) if zone == "brake" else len(per_episode))
        result["zones"][zone] = stats
    return result


def fmt(pair, digits=2):
    m, h = pair
    return "   -   " if m is None else f"{m:.{digits}f}±{h:.{digits}f}"


def main(argv):
    csv_out = None
    if "--csv" in argv:
        at = argv.index("--csv")
        csv_out = argv[at + 1]
        argv = argv[:at] + argv[at + 2:]
    brake = None
    if "--brake" in argv:
        at = argv.index("--brake")
        brake = argv[at + 1]
        if brake not in ("only", "none"):
            raise SystemExit("--brake expects only or none")
        argv = argv[:at] + argv[at + 2:]
    if not argv:
        raise SystemExit(__doc__)
    results = [summarize(*arg.split("=", 1), brake=brake) for arg in argv]
    if brake:
        print(f"episodes: leader hard brake = {brake}\n")

    print(f"{'run':<22}{'eps':>4}  {'completed':>15}  {'collision':>15}  {'emergency entry':>15}")
    for r in results:
        cells = ["%3.0f%% [%2.0f,%3.0f]" % tuple(100 * v for v in r[k]) for k in ("completed", "collision", "emergency")]
        print(f"{r['label']:<22}{r['episodes']:>4}  {cells[0]:>15}  {cells[1]:>15}  {cells[2]:>15}")
    for zone in WINDOWS:
        if zone == "brake" and not any(r["zones"][zone]["n"] for r in results):
            continue
        print(f"\n[{zone}]" + (f"  first {BRAKE_WINDOW_SEC:g} s from the leader's hard brake" if zone == "brake" else ""))
        print(f"{'run':<22}{'eps':>4}{'|err| mean m':>13}{'|err| max m':>13}{'THW_min s':>12}{'worst THW':>10}"
              f"{'min dist':>9}{'jerk m/s3':>12}{'info err m':>12}{'loss':>7}{'entry':>7}")
        for r in results:
            z = r["zones"][zone]
            worst = "   -" if z["thw_worst"] is None else f"{z['thw_worst']:.2f}"
            dist = "   -" if z["dist_worst"] is None else f"{z['dist_worst']:.2f}"
            loss = "  -" if z["loss"][0] is None else f"{100 * z['loss'][0]:.0f}%"
            print(f"{r['label']:<22}{z['n']:>4}{fmt(z['err_mean']):>13}{fmt(z['err_max']):>13}{fmt(z['thw_min']):>12}{worst:>10}"
                  f"{dist:>9}{fmt(z['jerk'], 1):>12}{fmt(z['info_err']):>12}{loss:>7}{100 * z['entry'][0]:>6.0f}%")

    if csv_out:
        with open(csv_out, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["run", "episodes", "completed", "collision", "emergency_entry", "zone", "err_mean", "err_mean_ci",
                        "err_max", "err_max_ci", "thw_min", "thw_min_ci", "thw_worst", "jerk", "jerk_ci",
                        "info_err", "loss", "entry_in_zone"])
            for r in results:
                for zone in WINDOWS:
                    z = r["zones"][zone]
                    w.writerow([r["label"], r["episodes"], r["completed"][0], r["collision"][0], r["emergency"][0], zone,
                                *z["err_mean"], *z["err_max"], *z["thw_min"], z["thw_worst"], *z["jerk"],
                                z["info_err"][0], z["loss"][0], z["entry"][0]])
        print(f"\nwrote {csv_out}")


if __name__ == "__main__":
    main(sys.argv[1:])
