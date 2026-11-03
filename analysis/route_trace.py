"""Run episode_ppo_v2.py with r_i tracing; writes trace.json into the directory given as the last argument.
Run from anywhere: python3 analysis/route_trace.py"""
import os, sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]  # repo root; all paths below are relative to it
os.chdir(REPO); sys.path.insert(0, str(REPO))
import sys, atexit, json
path = str(REPO / "episode_ppo_v2.py")
src = open(path).read()
idx = src.index('if __name__ == "__main__":')
g = {"__name__": "epi", "__file__": path}
exec(compile(src[:idx], path, "exec"), g)
log = []
orig = g["_route_lateral_offset_m"]
orig_create = g["_create_route_tracker"]
info = {}
def create(plan):
    t = orig_create(plan); info["points"] = t["points"]; return t
def wrapped(tr, loc):
    r = orig(tr, loc); log.append((round(loc.x,2), round(loc.y,2), tr["segment"], round(r,3))); return r
g["_route_lateral_offset_m"] = wrapped
g["_create_route_tracker"] = create
atexit.register(lambda: json.dump({"log": log, "points": info.get("points")}, open(sys.argv[-1] + "/trace.json", "w")))
exec(compile(src[idx:].replace('if __name__ == "__main__":', 'if True:', 1), path, "exec"), g)
