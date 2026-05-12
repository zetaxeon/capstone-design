#!/usr/bin/env python
"""
Town03 터널 경유 시나리오 (Phase 1).
- 1대 차량 (hero)
- BasicAgent 종/횡 제어
- 목적지 도달 또는 충돌 → reset → 무한반복
"""

import argparse
import logging
import math
import sys
from pathlib import Path

# scripts/에서 실행해도 src/carla_env.py를 import할 수 있게 경로 추가
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

import carla
from agents.navigation.basic_agent import BasicAgent

from carla_env import CarlaEnv


# Town03 spawn point 인덱스는 check_spawn_points.py로 확인 후 수정
SPAWN_INDEX = 0

# 터널 반대편 좌표로 나중에 교체
DESTINATION = carla.Location(x=0.0, y=0.0, z=0.0)

# 목적지 도달 판정 거리
GOAL_RADIUS = 5.0

# 에피소드 안전장치
MAX_EPISODE_TICKS = 4000  # 0.05s * 4000 = 200s


def distance(loc_a, loc_b):
    return math.sqrt((loc_a.x - loc_b.x) ** 2 + (loc_a.y - loc_b.y) ** 2)


def run_episode(env, agent, destination, ep_id):
    ticks = 0

    while True:
        env.tick()
        ticks += 1

        cur_loc = env.vehicle.get_transform().location

        if env.collision_flag:
            logging.info("[Ep %d] Terminated: COLLISION (%d ticks)", ep_id, ticks)
            return "collision"

        if distance(cur_loc, destination) < GOAL_RADIUS:
            logging.info("[Ep %d] Terminated: GOAL REACHED (%d ticks)", ep_id, ticks)
            return "goal"

        if agent.done():
            logging.info("[Ep %d] Terminated: AGENT DONE (%d ticks)", ep_id, ticks)
            return "agent_done"

        if ticks >= MAX_EPISODE_TICKS:
            logging.info("[Ep %d] Terminated: TIMEOUT (%d ticks)", ep_id, ticks)
            return "timeout"

        control = agent.run_step()
        env.vehicle.apply_control(control)


def main(args):
    logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.INFO)

    env = CarlaEnv(host=args.host, port=args.port, town="Town03")

    try:
        spawn_points = env.map.get_spawn_points()

        if args.spawn_index >= len(spawn_points):
            raise IndexError(f"spawn_index {args.spawn_index} >= {len(spawn_points)}")

        spawn_tf = spawn_points[args.spawn_index]

        # destination이 디폴트(0,0,0)이면 spawn으로부터 전방 100m 임시 목표
        dest = DESTINATION

        if dest.x == 0.0 and dest.y == 0.0:
            logging.warning(
                "Destination is default (0,0,0); using a placeholder 100m ahead."
            )
            forward = spawn_tf.get_forward_vector()
            dest = carla.Location(
                x=spawn_tf.location.x + forward.x * 100,
                y=spawn_tf.location.y + forward.y * 100,
                z=spawn_tf.location.z,
            )

        env.spawn_vehicle(spawn_tf)

        ep_id = 0

        while True:
            ep_id += 1

            # BasicAgent는 에피소드마다 새로 생성
            agent = BasicAgent(env.vehicle, target_speed=args.target_speed)
            agent.set_destination(dest)

            result = run_episode(env, agent, dest, ep_id)

            logging.info("[Ep %d] result=%s -> reset", ep_id, result)
            env.reset()

    except KeyboardInterrupt:
        print("\nCancelled by user. Bye!")

    finally:
        env.destroy()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", default=2000, type=int)
    parser.add_argument("--spawn-index", default=SPAWN_INDEX, type=int)
    parser.add_argument("--target-speed", default=30.0, type=float, help="km/h")

    main(parser.parse_args())