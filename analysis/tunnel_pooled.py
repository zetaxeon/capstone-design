"""Tunnel evaluation, seeds pooled: PID vs PPO-Lag (3 seeds) vs vanilla PPO (3 seeds) per in-tunnel p.

Usage (from anywhere):
  python3 analysis/tunnel_pooled.py [--eval-dir outputs/tunnel/eval] [--p "0.9 0.7 0.5 0.3 0.1"]
                                    [--seeds "42 43 44"] [--algos "pid lag ppo"]

Reads the cells written by analysis/tunnel_eval.sh (<eval-dir>/<pid|lag_sN|ppo_sN>_t<p>/) and prints
markdown tables for three episode subsets: all, leader hard brake only, no leader hard brake.
Rates carry a Wilson 95% interval. The pooled interval treats the 3 x 100 episodes of an algorithm
as independent, although the three seeds face the same 100 scenarios (evaluation seed 1000), so it
is somewhat too narrow; the per-seed counts are printed next to it for that reason.
Zone metrics come from analysis/tunnel_summarize.py (same definitions).
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tunnel_summarize import episode_zone_metrics, find_tick_dir, load, mean_ci, wilson  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
NAMES = {"pid": "PID", "lag": "PPO-Lag", "ppo": "PPO"}


def collect(run_dir):
    """One dict per episode: outcome flags, leader settings, zone metrics."""
    folder, name = find_tick_dir(run_dir)
    summary = folder / f"{name}_episodes.csv"
    if not summary.exists():
        summary = folder / "ppo_episodes.csv"
    by_episode = {e["episode"]: e for e in load(summary)}
    out = []
    for path in sorted(folder.glob(f"{name}_episode_*.csv")):
        rows = load(path)
        e = by_episode[rows[0]["episode"]]
        out.append({
            "episode": int(e["episode"]),
            "completed": e["end_reason"] == "profile_completed",
            "collision": e["collision"] == "1",
            "emergency": e["emergency_entered"] == "1",
            "brake": any(r.get("leader_brake_event") == "1" for r in rows),
            "speed": float(e["leader_target_speed_mps"]),
            "zones": episode_zone_metrics(rows),
        })
    return out


def rate(episodes, key):
    p, lo, hi = wilson(sum(e[key] for e in episodes), len(episodes))
    return f"{100 * p:.0f}% [{100 * lo:.0f}, {100 * hi:.0f}]"


def zone_mean(episodes, zone, key, digits=2):
    m, h = mean_ci([e["zones"][zone][key] for e in episodes if zone in e["zones"]])
    return "-" if m is None else f"{m:.{digits}f}±{h:.{digits}f}"


def zone_min(episodes, zone, key):
    values = [e["zones"][zone][key] for e in episodes if zone in e["zones"] and e["zones"][zone][key] is not None]
    return "-" if not values else f"{min(values):.2f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-dir", default=str(REPO / "outputs/tunnel/eval"))
    ap.add_argument("--p", default="0.9 0.7 0.5 0.3 0.1")
    ap.add_argument("--seeds", default="42 43 44")
    ap.add_argument("--algos", default="pid lag ppo")
    args = ap.parse_args()
    root = Path(args.eval_dir)
    seeds = args.seeds.split()
    data = {}
    for p in args.p.split():
        for algo in args.algos.split():
            cells = [f"pid_t{p}"] if algo == "pid" else [f"{algo}_s{s}_t{p}" for s in seeds]
            data[(p, algo)] = [collect(root / c) for c in cells]

    subsets = (
        ("all episodes", lambda e: True),
        ("leader hard brake only", lambda e: e["brake"]),
        ("no leader hard brake", lambda e: not e["brake"]),
    )
    for title, keep in subsets:
        print(f"\n### {title}\n")
        print("| tunnel p | controller | eps | completed | collision | per-seed collisions | emergency entry | "
              "tunnel mean abs err m | tunnel THW_min s | worst THW s | min dist m | tunnel jerk m/s3 |")
        print("|---|---|---|---|---|---|---|---|---|---|---|---|")
        for (p, algo), runs in data.items():
            runs = [[e for e in run if keep(e)] for run in runs]
            pooled = [e for run in runs for e in run]
            per_seed = " / ".join(f"{sum(e['collision'] for e in run)}" for run in runs)
            worst_thw = min((zone_min(pooled, z, "thw_min") for z in ("pre", "tunnel", "recovery", "final")),
                            key=lambda s: float(s) if s != "-" else 9e9)
            min_dist = min((zone_min(pooled, z, "dist_min") for z in ("pre", "tunnel", "recovery", "final")),
                           key=lambda s: float(s) if s != "-" else 9e9)
            print(f"| {p} | {NAMES[algo]} | {len(pooled)} | {rate(pooled, 'completed')} | {rate(pooled, 'collision')} | "
                  f"{per_seed} | {rate(pooled, 'emergency')} | {zone_mean(pooled, 'tunnel', 'err_mean')} | "
                  f"{zone_mean(pooled, 'tunnel', 'thw_min')} | {worst_thw} | {min_dist} | {zone_mean(pooled, 'tunnel', 'jerk', 1)} |")

    print("\n### collisions by leader target speed (all episodes, seeds pooled)\n")
    speeds = sorted({e["speed"] for runs in data.values() for run in runs for e in run})
    print("| tunnel p | controller | " + " | ".join(f"{s:g} m/s" for s in speeds) + " |")
    print("|---|---|" + "---|" * len(speeds))
    for (p, algo), runs in data.items():
        pooled = [e for run in runs for e in run]
        cells = []
        for s in speeds:
            sub = [e for e in pooled if e["speed"] == s]
            cells.append(f"{sum(e['collision'] for e in sub)}/{len(sub)}")
        print(f"| {p} | {NAMES[algo]} | " + " | ".join(cells) + " |")

    print("\n### collisions: leader hard brake x leader target speed (seeds pooled)\n")
    print("| tunnel p | controller | " + " | ".join(f"{s:g} brake | {s:g} no brake" for s in speeds) + " |")
    print("|---|---|" + "---|---|" * len(speeds))
    for (p, algo), runs in data.items():
        pooled = [e for run in runs for e in run]
        cells = []
        for s in speeds:
            for b in (True, False):
                sub = [e for e in pooled if e["speed"] == s and e["brake"] == b]
                cells.append(f"{sum(e['collision'] for e in sub)}/{len(sub)}")
        print(f"| {p} | {NAMES[algo]} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
