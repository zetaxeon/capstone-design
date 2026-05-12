#!/usr/bin/env python
"""
Town03 spawn point 좌표 확인용 헬퍼.
모든 spawn point를 출력하고, 특정 spawn point에 마커를 띄워 확인 가능.
"""

import argparse
import carla


def main(args):
    client = carla.Client(args.host, args.port)
    client.set_timeout(20.0)

    world = client.load_world("Town03") if args.load else client.get_world()
    map_ = world.get_map()
    spawn_points = map_.get_spawn_points()

    print(f"Total spawn points: {len(spawn_points)}\n")

    for i, sp in enumerate(spawn_points):
        loc = sp.location
        rot = sp.rotation
        print(
            f"[{i:3d}] x={loc.x:8.2f}  y={loc.y:8.2f}  z={loc.z:6.2f}  "
            f"yaw={rot.yaw:7.2f}"
        )

    if args.index is not None:
        sp = spawn_points[args.index]
        world.debug.draw_string(
            sp.location + carla.Location(z=2.0),
            f"SPAWN {args.index}",
            life_time=60.0,
            color=carla.Color(255, 0, 0),
        )
        print(f"\nMarked spawn point {args.index} for 60 seconds.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", default=2000, type=int)
    parser.add_argument("--load", action="store_true", help="Force load Town03")
    parser.add_argument("--index", type=int, default=None, help="Mark this spawn idx")

    main(parser.parse_args())