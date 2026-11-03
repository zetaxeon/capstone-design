"""Dump the planned routes from CARLA (needs a running CARLA) for the r_i route-offset diagnosis.
Run from anywhere: python3 analysis/route_dump.py"""
import os, sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]  # repo root; all paths below are relative to it
os.chdir(REPO); sys.path.insert(0, str(REPO))
import sys, math
sys.argv = ["x"]
import carla, episode_ppo_v2 as e
from pathlib import Path
from ros2_native import load_config_file, spawn_actors_from_config, destroy_actors
sc = load_config_file(Path("configs/scenario_town03.json"))
st = load_config_file(Path("configs") / sc.get("stack_file", "stack.json"))
st = e._apply_scenario_to_stack(st, sc)
client = carla.Client("localhost", 2000); client.set_timeout(20)
world = client.get_world(); m = world.get_map()
st, routes = e._align_scenario_to_map(st, sc, m)
s = world.get_settings(); s.synchronous_mode = True; s.fixed_delta_seconds = 0.05; world.apply_settings(s)
vehicles, sensors, objs = spawn_actors_from_config(world, st)
try:
    world.tick()
    byid = {o["id"]: v for v, o in zip(vehicles, objs)}
    ag = e._build_basic_agent(byid["follower"], m, sc, routes["follower"])
    tr = e._create_route_tracker(ag.planned_route)
    pts = tr["points"]
    print("route points:", len(pts), "start", pts[0], "end", pts[-1])
    print("follower route targets:", [(round(l.x,1), round(l.y,1)) for l in routes["follower"]])
    seg = [math.hypot(b[0]-a[0], b[1]-a[1]) for a, b in zip(pts, pts[1:])]
    print("segment len min/max: %.2f / %.2f" % (min(seg), max(seg)))
    for k in range(len(pts)-12, len(pts)): print(k, [round(c,2) for c in pts[k]])
    big = [(k, round(x,2)) for k, x in enumerate(seg) if x > 3]
    print("segments > 3 m:", big[:20])
finally:
    destroy_actors(sensors, vehicles)
    s.synchronous_mode = False; s.fixed_delta_seconds = None; world.apply_settings(s)
