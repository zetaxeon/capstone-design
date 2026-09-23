#!/usr/bin/env python3.10

import argparse
import copy
import csv
import logging
import math
import os
import random
import sys
import time
from pathlib import Path

import carla

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.distributions import Normal
except ModuleNotFoundError as exc:
    raise SystemExit(
        "PyTorch is required for episode_ppo_v2.py. Install torch in the Python environment "
        "used to run this script."
    ) from exc

from ros2_native import (
    _make_transform,
    destroy_actors,
    load_config_file,
    spawn_actors_from_config,
)


DEFAULT_SCENARIO_FILE = Path(__file__).parent / "configs" / "scenario_town03.json"


def _load_basic_agent():
    try:
        from agents.navigation.basic_agent import BasicAgent
        return BasicAgent
    except ModuleNotFoundError:
        candidate_roots = []

        carla_root = os.environ.get("CARLA_ROOT")
        if carla_root:
            candidate_roots.append(Path(carla_root) / "PythonAPI" / "carla")

        candidate_roots.append(Path.home() / "carla_0.9.16" / "PythonAPI" / "carla")

        for candidate in candidate_roots:
            candidate_str = str(candidate)
            if candidate.exists() and candidate_str not in sys.path:
                sys.path.append(candidate_str)
                try:
                    from agents.navigation.basic_agent import BasicAgent
                    return BasicAgent
                except ModuleNotFoundError:
                    continue

    raise SystemExit(
        "Failed to import CARLA BasicAgent.\n"
        "Set CARLA_ROOT or install CARLA PythonAPI agents on the Python path."
    )


BasicAgent = _load_basic_agent()


def _distance_between(a, b):
    dx = a.x - b.x
    dy = a.y - b.y
    dz = a.z - b.z
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def _get_spawn_point_by_index(map_, index):
    spawn_points = map_.get_spawn_points()
    if not spawn_points:
        raise RuntimeError("No vehicle spawn points are available on this map.")

    if index < 0 or index >= len(spawn_points):
        raise IndexError(
            f"spawn_point_index={index} is out of range; available range is 0..{len(spawn_points) - 1}."
        )

    return spawn_points[index]


def _resolve_route_location(map_, point_config):
    if "location" in point_config:
        location = point_config["location"]
        return carla.Location(
            x=location["x"],
            y=location["y"],
            z=location.get("z", 0.0),
        )

    if "spawn_point" in point_config:
        return _make_transform(point_config["spawn_point"]).location

    if "spawn_point_index" in point_config:
        transform = _get_spawn_point_by_index(map_, int(point_config["spawn_point_index"]))
        return transform.location

    raise ValueError("Each route point must define location, spawn_point, or spawn_point_index.")


def _get_actor_scenario_config(scenario_config, actor_id):
    return scenario_config.get(actor_id, {})


def _find_object_config(stack_config, actor_id):
    for object_config in stack_config.get("objects", []):
        if object_config.get("id") == actor_id:
            return object_config
    raise ValueError(f'stack.json must define an object with id="{actor_id}".')


def _resolve_actor_route_locations(map_, actor_config):
    route_locations = [
        _resolve_route_location(map_, point_config)
        for point_config in actor_config.get("route_points", [])
    ]

    if "destination" in actor_config:
        route_locations.append(_resolve_route_location(map_, actor_config["destination"]))

    return route_locations


def _project_to_driving_location(map_, location):
    waypoint = map_.get_waypoint(
        location,
        project_to_road=True,
        lane_type=carla.LaneType.Driving,
    )
    return waypoint.transform.location, waypoint


def _project_spawn_transform(map_, spawn_config):
    requested_transform = _make_transform(spawn_config)
    projected_location, waypoint = _project_to_driving_location(map_, requested_transform.location)
    rotation = waypoint.transform.rotation
    rotation.roll = requested_transform.rotation.roll

    return carla.Transform(projected_location, rotation)


def _project_destination_location(map_, destination_config):
    destination_location = _resolve_route_location(map_, destination_config)
    projected_location, _ = _project_to_driving_location(map_, destination_location)
    return projected_location


def _apply_scenario_to_stack(stack_config, scenario_config):
    config = copy.deepcopy(stack_config)
    for actor_id in ("leader", "follower"):
        actor_stack_config = _find_object_config(config, actor_id)
        actor_scenario_config = _get_actor_scenario_config(scenario_config, actor_id)

        if "spawn_point" in actor_scenario_config:
            actor_stack_config["spawn_point"] = actor_scenario_config["spawn_point"]
            actor_stack_config.pop("spawn_point_index", None)
        elif "spawn_point_index" in actor_scenario_config:
            actor_stack_config["spawn_point_index"] = int(actor_scenario_config["spawn_point_index"])
            actor_stack_config.pop("spawn_point", None)

        actor_stack_config["autopilot"] = bool(actor_scenario_config.get("autopilot", False))

    return config


def _project_location_list(map_, route_locations):
    return [
        _project_destination_location(map_, {"location": {"x": location.x, "y": location.y, "z": location.z}})
        for location in route_locations
    ]


def _project_follower_spawn_from_leader(map_, leader_transform, follower_config):
    gap_m = float(follower_config.get("spawn_gap_m", 12.0))
    waypoint = map_.get_waypoint(
        leader_transform.location,
        project_to_road=True,
        lane_type=carla.LaneType.Driving,
    )
    if waypoint is None:
        raise RuntimeError("Could not project leader spawn to a driving waypoint for follower placement.")

    previous_waypoints = waypoint.previous(gap_m)
    follower_waypoint = previous_waypoints[0] if previous_waypoints else waypoint
    transform = follower_waypoint.transform
    transform.location.z += 0.05
    return transform


def _align_scenario_to_map(stack_config, scenario_config, map_):
    config = copy.deepcopy(stack_config)
    leader_stack_config = _find_object_config(config, "leader")
    follower_stack_config = _find_object_config(config, "follower")

    projected_leader_spawn = _project_spawn_transform(map_, leader_stack_config["spawn_point"])
    leader_stack_config["spawn_point"] = {
        "x": projected_leader_spawn.location.x,
        "y": projected_leader_spawn.location.y,
        "z": projected_leader_spawn.location.z + 0.05,
        "roll": projected_leader_spawn.rotation.roll,
        "pitch": projected_leader_spawn.rotation.pitch,
        "yaw": projected_leader_spawn.rotation.yaw,
    }

    follower_scenario_config = _get_actor_scenario_config(scenario_config, "follower")
    if "spawn_point" in follower_stack_config:
        projected_follower_spawn = _project_spawn_transform(map_, follower_stack_config["spawn_point"])
    else:
        projected_follower_spawn = _project_follower_spawn_from_leader(
            map_,
            projected_leader_spawn,
            follower_scenario_config,
        )
    follower_stack_config["spawn_point"] = {
        "x": projected_follower_spawn.location.x,
        "y": projected_follower_spawn.location.y,
        "z": projected_follower_spawn.location.z + 0.05,
        "roll": projected_follower_spawn.rotation.roll,
        "pitch": projected_follower_spawn.rotation.pitch,
        "yaw": projected_follower_spawn.rotation.yaw,
    }
    follower_stack_config.pop("spawn_point_index", None)

    leader_route_locations = _project_location_list(
        map_,
        _resolve_actor_route_locations(map_, _get_actor_scenario_config(scenario_config, "leader")),
    )
    follower_route_config = _get_actor_scenario_config(scenario_config, "follower")
    follower_route_locations = _resolve_actor_route_locations(map_, follower_route_config)
    if not follower_route_locations:
        follower_route_locations = list(leader_route_locations)
    else:
        follower_route_locations = _project_location_list(map_, follower_route_locations)

    return config, {"leader": leader_route_locations, "follower": follower_route_locations}


def _setup_world(client, map_name, tm_port, allow_map_load, map_load_requested):
    world = client.get_world()
    current_map = world.get_map().name.rsplit("/", 1)[-1]
    if current_map != map_name:
        if not allow_map_load:
            raise RuntimeError(
                f'Current CARLA map is "{current_map}", but scenario requires "{map_name}". '
                f"Launch CARLA directly with {map_name} before running episode.py."
            )
        if not map_load_requested:
            logging.info("Loading map %s", map_name)
            client.load_world(map_name)
            return None, None, None, True
        raise RuntimeError(
            f'Waiting for CARLA to finish loading "{map_name}". Current map is still "{current_map}".'
        )

    original_settings = world.get_settings()
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05
    world.apply_settings(settings)

    traffic_manager = client.get_trafficmanager(tm_port)
    traffic_manager.set_synchronous_mode(True)

    return world, original_settings, traffic_manager, map_load_requested


def _connect_world_with_retry(
    client,
    host,
    port,
    map_name,
    tm_port,
    retry_interval,
    max_wait_sec,
    allow_map_load,
):
    start_time = time.monotonic()
    map_load_requested = False

    while True:
        try:
            world, original_settings, traffic_manager, map_load_requested = _setup_world(
                client,
                map_name,
                tm_port,
                allow_map_load,
                map_load_requested,
            )
            if world is None:
                waited_sec = time.monotonic() - start_time
                logging.info(
                    "Map switch requested for %s. Waiting for CARLA to finish reloading (%.1fs elapsed).",
                    map_name,
                    waited_sec,
                )
                time.sleep(retry_interval)
                continue
            return world, original_settings, traffic_manager
        except RuntimeError as exc:
            message = str(exc)
            is_timeout = "time-out" in message and "simulator" in message
            is_map_reload = "Waiting for CARLA to finish loading" in message
            if not is_timeout and not is_map_reload:
                raise

            waited_sec = time.monotonic() - start_time
            if max_wait_sec > 0 and waited_sec >= max_wait_sec:
                raise SystemExit(
                    "Failed to connect to CARLA simulator.\n"
                    f"Checked {host}:{port} for {waited_sec:.1f} seconds.\n"
                    "Make sure CARLA is running and fully loaded before starting episode.py."
                ) from exc

            if is_map_reload:
                logging.info("%s", message)
            else:
                logging.warning(
                    "CARLA not ready at %s:%s yet; waited %.1fs so far. Retrying in %.1fs.",
                    host,
                    port,
                    waited_sec,
                    retry_interval,
                )
            time.sleep(retry_interval)


def _restore_world(world, original_settings, traffic_manager):
    if world is not None and original_settings is not None:
        world.apply_settings(original_settings)

    if traffic_manager is not None:
        traffic_manager.set_synchronous_mode(False)


def _attach_collision_sensor(world, vehicle):
    bp = world.get_blueprint_library().find("sensor.other.collision")
    sensor = world.spawn_actor(bp, carla.Transform(), attach_to=vehicle)
    state = {"collision": False}

    def _on_collision(_event):
        state["collision"] = True

    sensor.listen(_on_collision)
    return sensor, state


def _update_spectator(world, vehicles, camera_config):
    if not camera_config.get("enabled", False):
        return

    if not vehicles:
        return

    lead_vehicle = vehicles[0]
    lead_transform = lead_vehicle.get_transform()
    forward = lead_transform.get_forward_vector()
    up = lead_transform.get_up_vector()
    right = lead_transform.get_right_vector()

    distance = float(camera_config.get("distance", 8.0))
    height = float(camera_config.get("height", 3.5))
    lateral_offset = float(camera_config.get("lateral_offset", 0.0))
    pitch = float(camera_config.get("pitch", -15.0))
    yaw = lead_transform.rotation.yaw

    if len(vehicles) >= 2:
        trailing_transform = vehicles[1].get_transform()
        location = carla.Location(
            x=(lead_transform.location.x + trailing_transform.location.x) * 0.5,
            y=(lead_transform.location.y + trailing_transform.location.y) * 0.5,
            z=(lead_transform.location.z + trailing_transform.location.z) * 0.5,
        )
        spacing = _distance_between(lead_transform.location, trailing_transform.location)
        distance = max(distance, spacing * float(camera_config.get("distance_scale", 0.8)))
        height = max(height, 4.0 + spacing * float(camera_config.get("height_scale", 0.08)))
    else:
        location = lead_transform.location

    spectator_location = carla.Location(
        x=location.x - forward.x * distance + up.x * height + right.x * lateral_offset,
        y=location.y - forward.y * distance + up.y * height + right.y * lateral_offset,
        z=location.z - forward.z * distance + up.z * height + right.z * lateral_offset,
    )
    spectator_rotation = carla.Rotation(
        pitch=pitch,
        yaw=yaw,
        roll=0.0,
    )
    world.get_spectator().set_transform(carla.Transform(spectator_location, spectator_rotation))


def _tick_for_seconds(world, duration_sec):
    dt = float(world.get_settings().fixed_delta_seconds or 0.05)
    steps = max(1, int(round(duration_sec / dt)))
    for _ in range(steps):
        world.tick()


def _create_tick_guard(world):
    snapshot = world.get_snapshot()
    return {"last_frame": snapshot.frame if snapshot is not None else None}


def _tick_world_single_owner(world, tick_guard):
    frame = world.tick()
    last_frame = tick_guard["last_frame"]
    if last_frame is not None and frame != last_frame + 1:
        raise RuntimeError(
            "Detected external CARLA ticks while episode.py was running. "
            f"Expected frame {last_frame + 1}, but received {frame}. "
            "Another process is likely calling world.tick() on the same world."
        )
    tick_guard["last_frame"] = frame
    return frame


def _tick_for_seconds_single_owner(world, duration_sec, tick_guard):
    dt = float(world.get_settings().fixed_delta_seconds or 0.05)
    steps = max(1, int(round(duration_sec / dt)))
    for _ in range(steps):
        _tick_world_single_owner(world, tick_guard)


def _configure_traffic_lights(world, scenario_config):
    traffic_light_config = scenario_config.get("traffic_lights", {})
    if not bool(traffic_light_config.get("force_green", False)):
        return

    for traffic_light in world.get_actors().filter("*traffic_light*"):
        traffic_light.set_state(carla.TrafficLightState.Green)
        traffic_light.freeze(True)


def _get_speed_mps(vehicle):
    velocity = vehicle.get_velocity()
    return math.sqrt(velocity.x * velocity.x + velocity.y * velocity.y + velocity.z * velocity.z)


def _select_profile_target_speed_mps(scenario_config):
    speed_profile = scenario_config.get("speed_profile", {})
    target_speeds = [float(value) for value in speed_profile.get("target_speeds_mps", [15.0, 17.5, 20.0])]
    if not target_speeds:
        raise ValueError("speed_profile.target_speeds_mps must not be empty.")
    return random.choice(target_speeds)


def _clamp(value, minimum, maximum):
    return max(minimum, min(maximum, value))


def _create_speed_profile_state(scenario_config, initial_speed_mps):
    speed_profile = scenario_config.get("speed_profile", {})
    target_speed_mps = _select_profile_target_speed_mps(scenario_config)
    commanded_speed_mps = max(float(speed_profile.get("initial_speed_mps", initial_speed_mps)), initial_speed_mps)
    return {
        "phase": "accelerate",
        "target_speed_mps": target_speed_mps,
        "commanded_speed_mps": min(commanded_speed_mps, target_speed_mps),
        "accel_mps2": float(speed_profile.get("accel_mps2", 2.0)),
        "decel_mps2": float(speed_profile.get("decel_mps2", 3.5)),
        "cruise_time_sec": float(speed_profile.get("cruise_time_sec", 7.0)),
        "stop_speed_mps": float(speed_profile.get("stop_speed_mps", 0.4)),
        "stop_hold_sec": float(speed_profile.get("stop_hold_sec", 1.0)),
        "speed_tolerance_mps": float(speed_profile.get("speed_tolerance_mps", 0.6)),
        "cruise_elapsed_sec": 0.0,
        "stop_elapsed_sec": 0.0,
        "profile_completed": False,
    }


def _create_follower_pid_state(scenario_config):
    follower_control = scenario_config.get("follower_control", {})
    return {
        "h": float(follower_control.get("h", 1.0)),
        "d0": float(follower_control.get("d0", 7.0)),
        "kp": float(follower_control.get("kp", 0.4)),
        "ki": float(follower_control.get("ki", 0.01)),
        "kd": float(follower_control.get("kd", 0.1)),
        "integral_error": 0.0,
        "integral_limit": float(follower_control.get("integral_limit", 30.0)),
        "emergency_distance_m": float(follower_control.get("emergency_distance_m", 5.0)),
        "launch_speed_threshold_mps": float(follower_control.get("launch_speed_threshold_mps", 0.5)),
        "launch_lead_speed_threshold_mps": float(follower_control.get("launch_lead_speed_threshold_mps", 1.0)),
        "launch_spacing_error_threshold_m": float(follower_control.get("launch_spacing_error_threshold_m", 2.0)),
        "launch_min_throttle": float(follower_control.get("launch_min_throttle", 0.2)),
        "stop_speed_mps": float(follower_control.get("stop_speed_mps", 0.4)),
        "stop_hold_sec": float(follower_control.get("stop_hold_sec", 1.0)),
        "stop_elapsed_sec": 0.0,
        "log_interval_ticks": max(1, int(follower_control.get("log_interval_ticks", 10))),
    }


def _update_speed_profile(vehicle, agent, profile_state, dt):
    current_speed_mps = _get_speed_mps(vehicle)
    phase = profile_state["phase"]

    if phase == "accelerate":
        profile_state["commanded_speed_mps"] = min(
            profile_state["target_speed_mps"],
            profile_state["commanded_speed_mps"] + profile_state["accel_mps2"] * dt,
        )
        if current_speed_mps >= profile_state["target_speed_mps"] - profile_state["speed_tolerance_mps"]:
            profile_state["phase"] = "cruise"
            profile_state["cruise_elapsed_sec"] = 0.0

    elif phase == "cruise":
        profile_state["commanded_speed_mps"] = profile_state["target_speed_mps"]
        profile_state["cruise_elapsed_sec"] += dt
        if profile_state["cruise_elapsed_sec"] >= profile_state["cruise_time_sec"]:
            profile_state["phase"] = "decelerate"

    elif phase == "decelerate":
        profile_state["commanded_speed_mps"] = max(
            0.0,
            profile_state["commanded_speed_mps"] - profile_state["decel_mps2"] * dt,
        )
        if current_speed_mps <= profile_state["stop_speed_mps"]:
            profile_state["stop_elapsed_sec"] += dt
            if profile_state["stop_elapsed_sec"] >= profile_state["stop_hold_sec"]:
                profile_state["phase"] = "stopped"
                profile_state["profile_completed"] = True
        else:
            profile_state["stop_elapsed_sec"] = 0.0

    agent.set_target_speed(profile_state["commanded_speed_mps"] * 3.6)


def _apply_speed_profile_to_control(control, profile_state):
    if profile_state["phase"] != "decelerate":
        return control

    if profile_state["commanded_speed_mps"] > 1.5:
        return control

    control.throttle = 0.0
    control.brake = max(control.brake, 0.35)
    return control


def _build_basic_agent(vehicle, map_, scenario_config, route_locations, option_overrides=None):
    if not route_locations:
        raise ValueError("Scenario must define route_points and/or destination for BasicAgent.")

    agent_config = scenario_config.get("basic_agent", {})
    target_speed = float(agent_config.get("target_speed_kmh", 20.0))
    option_keys = (
        "ignore_traffic_lights",
        "ignore_stop_signs",
        "ignore_vehicles",
        "use_bbs_detection",
        "sampling_resolution",
        "base_tlight_threshold",
        "base_vehicle_threshold",
        "detection_speed_ratio",
        "max_brake",
        "offset",
    )
    opt_dict = {
        key: agent_config[key]
        for key in option_keys
        if key in agent_config
    }
    if option_overrides:
        opt_dict.update(option_overrides)

    agent = BasicAgent(vehicle, target_speed=target_speed, opt_dict=opt_dict, map_inst=map_)
    if bool(agent_config.get("follow_speed_limits", False)):
        agent.follow_speed_limits(True)

    route_plan = []
    start_location = vehicle.get_location()
    for target_location in route_locations:
        start_waypoint = map_.get_waypoint(start_location)
        end_waypoint = map_.get_waypoint(target_location)
        segment_plan = agent.trace_route(start_waypoint, end_waypoint)
        if not segment_plan:
            raise RuntimeError(
                f"BasicAgent could not build a route segment to x={target_location.x:.2f}, y={target_location.y:.2f}."
            )

        if route_plan and segment_plan:
            segment_plan = segment_plan[1:]
        route_plan.extend(segment_plan)
        start_location = target_location

    agent.set_global_plan(route_plan)
    return agent


def _advance_route_progress(vehicle, route_locations, next_route_index, goal_tolerance):
    if route_locations and next_route_index < len(route_locations):
        if _distance_between(vehicle.get_location(), route_locations[next_route_index]) <= goal_tolerance:
            next_route_index += 1
    return next_route_index


def _longitudinal_accel_mps2(vehicle):
    accel = vehicle.get_acceleration()
    forward = vehicle.get_transform().get_forward_vector()
    return accel.x * forward.x + accel.y * forward.y + accel.z * forward.z


def _compute_follower_control(leader_vehicle, follower_vehicle, follower_agent, follower_pid_state, dt):
    lead_speed_mps = _get_speed_mps(leader_vehicle)
    ego_speed_mps = _get_speed_mps(follower_vehicle)
    distance_m = _distance_between(leader_vehicle.get_location(), follower_vehicle.get_location())
    desired_distance_m = follower_pid_state["d0"] + follower_pid_state["h"] * ego_speed_mps
    spacing_error_m = distance_m - desired_distance_m
    relative_velocity_mps = lead_speed_mps - ego_speed_mps

    lead_accel_mps2 = _longitudinal_accel_mps2(leader_vehicle)
    # Paper eq. (8) with perfect communication: [Δd, Δv, a_(i-1), v_i].
    observation = [
        spacing_error_m,
        relative_velocity_mps,
        lead_accel_mps2,
        ego_speed_mps,
    ]
    if not all(math.isfinite(value) for value in observation):
        raise ValueError(f"Invalid PPO observation: {observation}")

    follower_pid_state["integral_error"] = _clamp(
        follower_pid_state["integral_error"] + spacing_error_m * dt,
        -follower_pid_state["integral_limit"],
        follower_pid_state["integral_limit"],
    )

    u = (
        follower_pid_state["kp"] * spacing_error_m
        + follower_pid_state["ki"] * follower_pid_state["integral_error"]
        + follower_pid_state["kd"] * relative_velocity_mps
    )

    throttle = 0.0
    brake = 0.0
    if distance_m < follower_pid_state["emergency_distance_m"]:
        brake = 1.0
    elif u >= 0.0:
        throttle = _clamp(u / 3.0, 0.0, 1.0)
    else:
        brake = _clamp((-u) / 4.0, 0.0, 1.0)

    if (
        ego_speed_mps <= follower_pid_state["launch_speed_threshold_mps"]
        and lead_speed_mps >= follower_pid_state["launch_lead_speed_threshold_mps"]
        and spacing_error_m >= follower_pid_state["launch_spacing_error_threshold_m"]
    ):
        throttle = max(throttle, follower_pid_state["launch_min_throttle"])
        brake = 0.0

    lateral_control = follower_agent.run_step()
    control = carla.VehicleControl()
    control.steer = lateral_control.steer
    control.throttle = throttle
    control.brake = brake
    control.hand_brake = False
    control.reverse = False
    control.manual_gear_shift = False

    if ego_speed_mps <= follower_pid_state["stop_speed_mps"]:
        follower_pid_state["stop_elapsed_sec"] += dt
    else:
        follower_pid_state["stop_elapsed_sec"] = 0.0

    metrics = {
        "distance_m": distance_m,
        "desired_distance_m": desired_distance_m,
        "spacing_error_m": spacing_error_m,
        "relative_velocity_mps": relative_velocity_mps,
        "lead_speed_mps": lead_speed_mps,
        "ego_speed_mps": ego_speed_mps,
        "lead_accel_mps2": lead_accel_mps2,
        "ego_accel_mps2": _longitudinal_accel_mps2(follower_vehicle),
        "observation": observation,
        "throttle": throttle,
        "brake": brake,
    }
    return control, metrics


def _log_follower_metrics(episode_count, tick_count, metrics):
    logging.info(
        (
            "Episode %d follower tick %d: d=%.2f d_des=%.2f err=%.2f "
            "rel_v=%.2f v_lead=%.2f v_follower=%.2f throttle=%.2f brake=%.2f"
        ),
        episode_count,
        tick_count,
        metrics["distance_m"],
        metrics["desired_distance_m"],
        metrics["spacing_error_m"],
        metrics["relative_velocity_mps"],
        metrics["lead_speed_mps"],
        metrics["ego_speed_mps"],
        metrics["throttle"],
        metrics["brake"],
    )

def _write_episode_csv(output_dir, episode_count, rows, controller="pid"):
    if not rows:
        return

    output_dir = Path(output_dir) / controller
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / f"{controller}_episode_{episode_count:03d}.csv"

    fieldnames = [
        "episode",
        "time_sec",
        "tick",
        "leader_phase",
        "leader_speed_mps",
        "follower_speed_mps",
        "distance_m",
        "desired_distance_m",
        "spacing_error_m",
        "relative_velocity_mps",
        "throttle",
        "brake",
    ]

    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    logging.info("Saved episode CSV: %s", csv_path)


def _leader_route_finished(leader_agent, route_state):
    """True once the leader has nothing left to drive to.

    BasicAgent.done() is the same condition that makes the local planner park the car, so it is
    what we trust first; the route index is a fallback for agents that do not expose done().
    """
    if leader_agent is not None and hasattr(leader_agent, "done"):
        if leader_agent.done():
            return True

    route_locations = route_state["leader_route_locations"]
    return bool(route_locations) and route_state["leader_index"] >= len(route_locations)


def _evaluate_episode(
    leader_vehicle,
    follower_vehicle,
    leader_agent,
    leader_profile_state,
    follower_pid_state,
    route_state,
    collision_states,
    timers,
    termination,
    dt,
):
    goal_tolerance = float(termination.get("goal_tolerance_m", 5.0))
    route_state["leader_index"] = _advance_route_progress(
        leader_vehicle,
        route_state["leader_route_locations"],
        route_state["leader_index"],
        goal_tolerance,
    )
    route_state["follower_index"] = _advance_route_progress(
        follower_vehicle,
        route_state["follower_route_locations"],
        route_state["follower_index"],
        goal_tolerance,
    )

    if leader_profile_state["profile_completed"] and (
        follower_pid_state["stop_elapsed_sec"] >= follower_pid_state["stop_hold_sec"]
    ):
        return "profile_completed"

    if collision_states["leader"]["collision"]:
        return "leader_collision"

    if collision_states["follower"]["collision"]:
        return "follower_collision"

    speed_mps = _get_speed_mps(leader_vehicle)
    if leader_profile_state["phase"] == "decelerate" and leader_profile_state["commanded_speed_mps"] <= 1.0:
        timers["stuck_sec"] = 0.0
        return None

    stuck_speed = float(termination.get("stuck_speed_mps", 0.5))
    stuck_time = float(termination.get("stuck_time_sec", 8.0))
    stuck_grace = float(termination.get("stuck_grace_sec", 0.0))

    # The leader reaches its destination and parks while the speed profile is often still in
    # "cruise", so the profile never completes on its own and the stuck timer used to end the
    # episode as "leader_stuck". Arriving is the end of the route, not a stall: finish the profile
    # and let the normal profile_completed ending wait for the follower to stop as well. This is
    # checked before the grace window, because arriving early is still arriving.
    if speed_mps < stuck_speed and _leader_route_finished(leader_agent, route_state):
        if not leader_profile_state["profile_completed"]:
            leader_profile_state["phase"] = "stopped"
            leader_profile_state["commanded_speed_mps"] = 0.0
            leader_profile_state["profile_completed"] = True
        timers["stuck_sec"] = 0.0
        return None

    if timers["elapsed_sec"] < stuck_grace:
        return None

    if speed_mps < stuck_speed:
        timers["stuck_sec"] += dt
        if timers["stuck_sec"] >= stuck_time:
            return "leader_stuck"
    else:
        timers["stuck_sec"] = 0.0

    return None


def _print_spawn_points(world):
    map_ = world.get_map()
    for index, transform in enumerate(map_.get_spawn_points()):
        location = transform.location
        rotation = transform.rotation
        print(
            f"[{index:03d}] "
            f"x={location.x:.2f} y={location.y:.2f} z={location.z:.2f} "
            f"yaw={rotation.yaw:.2f}"
        )



# ---------------------------------------------------------------------------
# PPO-Lagrangian longitudinal controller (baseline: team ITSC 2026 manuscript)
# Observation: [spacing_error Δd, relative_velocity Δv, leader_accel a_(i-1), follower_speed v_i]
#   Paper eq. (8) also has lateral offset r_i, comm-failure flag, and packet age t_i;
#   those are deferred until the V2V loss model (Last Value Hold) is added.
# Action: scalar in [-1, 1], + => throttle, - => brake (paper eq. 9)
#
# Values marked [PAPER] come from Table I. Values marked [ASSUMED] are not specified
# in the paper and were chosen here; change them in this block only.
# ---------------------------------------------------------------------------

REWARD_W_CONT = 0.1           # [PAPER] ω1
REWARD_W_TRAF = 0.25          # [PAPER] ω2
REWARD_W_JERK = 0.25          # [PAPER] ω3
REWARD_Q_SPACING = 2.0        # [PAPER] Q = diag(2, 1); eq. (10) writes Δq, read here as Δv [ASSUMED]
REWARD_Q_REL_SPEED = 1.0      # [PAPER]
REWARD_D_FAR = 8.0            # [PAPER] d_far
REWARD_D_CLOSE = -4.0         # [PAPER] d_close
REWARD_ALPHA_FAR = 20.0       # [PAPER] α1
REWARD_ALPHA_CLOSE = 10.0     # [PAPER] α2
THW_SAFE_SEC = 1.2            # [PAPER] τ_safe
THW_DANGER_SEC = 1.0          # [PAPER] τ_danger
THW_MIN_SPEED_MPS = 0.5       # [ASSUMED] below this speed THW is treated as infinite (cost 0)
# [ASSUMED] a_i in r_traf / r_jerk (eq. 11-12) is the follower's measured longitudinal
# acceleration [m/s^2], matching plot_pid_metrics.py; the paper also uses a_i for the action.

# [ASSUMED] Fixed scales so the policy sees roughly O(1) inputs; not given in the paper.
PPO_OBS_SCALES = (20.0, 10.0, 5.0, 30.0)

# [ASSUMED] Episode endings counted as follower failure; they receive a terminal penalty.
PPO_FAILURE_REASONS = ("hard_violation", "follower_collision", "leader_collision")
# A crash is the worst safety outcome, but it also ends the episode, so the cost critic would
# otherwise learn that the collision state costs one tick while sustained tailgating costs
# 1/(1-gamma). Charge the saturated THW cost as if it persisted, mirroring the reward penalty.
# hard_violation is deliberately excluded: falling far behind is unsafe for tracking, not for THW.
PPO_COLLISION_REASONS = ("follower_collision", "leader_collision")
PPO_MAX_TICK_COST = 1.0       # _thw_cost saturates at 1.0 per tick
# These endings cut the episode on a clock or on the leader, not because the follower finished
# its job, so the future is not worth zero. Fold gamma * V(s_final) into the last transition
# instead of letting done=True regress the critic toward 0. profile_completed stays terminal:
# there the leader has parked and the follower has stopped, so there is genuinely nothing left.
PPO_TRUNCATION_REASONS = ("timeout", "leader_stuck")

# The lambda constraint is measured per tick, not per episode.
#
# The paper states d = 5 as a cumulative per-episode THW cost. Taken literally that makes the
# constraint depend on how long an episode happens to last: in the 588-episode run, episodes ran
# 68 to 762 ticks, an 11.2x spread, so ending early was itself a way to lower J_c. Dividing the
# episode cost by its tick count removes that, and the cost critic already learns per-tick mean
# cost, so both sides of the Lagrangian finally speak the same unit.
#
# d and eta are rescaled by the nominal episode length so that a nominal-length episode
# reproduces the paper's lambda step exactly:
#   eta_tick * (S/N - d_tick) == eta * (S - d)   when N == PPO_NOMINAL_EPISODE_TICKS
PPO_PAPER_COST_LIMIT = 5.0          # [PAPER] Table I, as a per-episode tick sum
PPO_PAPER_LAMBDA_LR = 0.01          # [PAPER] eta, paired with the per-episode d
PPO_NOMINAL_EPISODE_TICKS = 650     # [ASSUMED] measured: 50 completed episodes ran 645 ticks
                                    # on average, median 652, at dt = 0.05 s
PPO_COST_LIMIT_PER_TICK = PPO_PAPER_COST_LIMIT / PPO_NOMINAL_EPISODE_TICKS
PPO_LAMBDA_LR_PER_TICK = PPO_PAPER_LAMBDA_LR * PPO_NOMINAL_EPISODE_TICKS


def _normalize_ppo_observation(observation):
    return [float(value) / scale for value, scale in zip(observation, PPO_OBS_SCALES)]


# The reward critic learns returns in normalized units (running mean/std of the GAE targets),
# so its loss and gradients stay O(1) instead of ~1e5. Rewards, costs, and lambda are unchanged;
# values are converted back to real units before GAE.
def _create_return_stats():
    return {"mean": 0.0, "var": 1.0, "count": 1e-4}


def _update_return_stats(stats, values):
    batch = torch.as_tensor(values, dtype=torch.float64)
    batch_count = batch.numel()
    batch_mean = float(batch.mean())
    batch_var = float(batch.var(unbiased=False))
    delta = batch_mean - stats["mean"]
    total = stats["count"] + batch_count
    stats["mean"] += delta * batch_count / total
    m2 = stats["var"] * stats["count"] + batch_var * batch_count + delta ** 2 * stats["count"] * batch_count / total
    stats["var"] = m2 / total
    stats["count"] = total


def _return_std(stats):
    return math.sqrt(stats["var"] + 1e-8)


def _denormalize_reward_value(stats, normalized_value):
    return normalized_value * _return_std(stats) + stats["mean"]


def _mlp(obs_dim, hidden_dim, out_dim):
    return nn.Sequential(
        nn.Linear(obs_dim, hidden_dim),
        nn.Tanh(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.Tanh(),
        nn.Linear(hidden_dim, out_dim),
    )


class _PPOLagPolicy(nn.Module):
    # Actor, reward critic, and cost critic are separate networks (OmniSafe-style) so the
    # large critic gradients cannot starve the actor through a shared backbone.
    def __init__(self, obs_dim=4, hidden_dim=128):
        super().__init__()
        self.actor_mean = _mlp(obs_dim, hidden_dim, 1)
        self.log_std = nn.Parameter(torch.tensor([-0.5], dtype=torch.float32))
        self.reward_critic = _mlp(obs_dim, hidden_dim, 1)
        self.cost_critic = _mlp(obs_dim, hidden_dim, 1)

    def actor_parameters(self):
        return list(self.actor_mean.parameters()) + [self.log_std]

    def critic_parameters(self):
        return list(self.reward_critic.parameters()) + list(self.cost_critic.parameters())

    def forward(self, obs):
        mean = self.actor_mean(obs)
        reward_value = self.reward_critic(obs).squeeze(-1)
        cost_value = self.cost_critic(obs).squeeze(-1)
        return mean, reward_value, cost_value

    def distribution(self, obs):
        mean, reward_value, cost_value = self.forward(obs)
        std = self.log_std.clamp(-5.0, 1.0).exp().expand_as(mean)
        return Normal(mean, std), reward_value, cost_value


def _atanh(value):
    value = value.clamp(-0.999999, 0.999999)
    return 0.5 * (torch.log1p(value) - torch.log1p(-value))


def _tanh_log_prob(distribution, raw_action, action):
    base_log_prob = distribution.log_prob(raw_action).sum(dim=-1)
    correction = torch.log(1.0 - action.pow(2) + 1e-6).sum(dim=-1)
    return base_log_prob - correction


def _create_ppo_state(args, scenario_config):
    ppo_config = scenario_config.get("ppo", {})
    device_name = str(ppo_config.get("device", args.ppo_device))
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)

    seed = int(ppo_config.get("seed", args.seed))
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    model = _PPOLagPolicy(
        obs_dim=4,
        hidden_dim=int(ppo_config.get("hidden_dim", args.ppo_hidden_dim)),
    ).to(device)

    lagrange_multiplier = max(0.0, float(ppo_config.get("initial_lambda", args.ppo_initial_lambda)))
    lr = float(ppo_config.get("lr", args.ppo_lr))
    actor_optimizer = optim.Adam(model.actor_parameters(), lr=lr)
    critic_optimizer = optim.Adam(model.critic_parameters(), lr=lr)
    update_count = 0
    return_stats = _create_return_stats()

    checkpoint_path = Path(args.ppo_checkpoint)
    if args.ppo_eval and not checkpoint_path.exists():
        raise SystemExit(f"--ppo-eval needs a trained checkpoint, but {checkpoint_path} does not exist.")

    if checkpoint_path.exists() and args.ppo_fresh and not args.ppo_eval:
        logging.warning("--ppo-fresh: ignoring %s; it will be overwritten at the first PPO update.", checkpoint_path)
    elif checkpoint_path.exists():
        payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
        try:
            model.load_state_dict(payload.get("model", payload))
        except RuntimeError as exc:
            raise SystemExit(
                f"{checkpoint_path} was saved with the old shared-backbone network and cannot be loaded "
                "into the separate actor/critic networks. Rerun with --ppo-fresh or pick another "
                "--ppo-checkpoint."
            ) from exc
        if isinstance(payload, dict):
            lagrange_multiplier = float(payload.get("lambda", lagrange_multiplier))
            update_count = int(payload.get("update_count", 0))
            return_stats = payload.get("return_stats", return_stats)
            if "actor_optimizer" in payload:
                actor_optimizer.load_state_dict(payload["actor_optimizer"])
            if "critic_optimizer" in payload:
                critic_optimizer.load_state_dict(payload["critic_optimizer"])
        logging.info(
            "Loaded PPO checkpoint: %s (update=%d lambda=%.4f)",
            checkpoint_path,
            update_count,
            lagrange_multiplier,
        )

    # Per-run metric logs start fresh, matching how episode CSVs are rewritten from 001 each run.
    metrics_dir = Path(args.output_dir) / _ppo_run_name(args)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    for name in ("ppo_episodes.csv", "ppo_updates.csv"):
        (metrics_dir / name).unlink(missing_ok=True)

    action_repeat = max(1, int(ppo_config.get("action_repeat", args.ppo_action_repeat)))
    # gamma is given per simulator tick; convert so the discount horizon in seconds is unchanged.
    gamma_per_tick = float(ppo_config.get("gamma", args.ppo_gamma))

    return {
        "device": device,
        "model": model,
        "actor_optimizer": actor_optimizer,
        "critic_optimizer": critic_optimizer,
        "lambda": lagrange_multiplier,
        "lambda_lr": float(ppo_config.get("lambda_lr", args.ppo_lambda_lr)),
        "episode_costs": [],
        "hard_violation_spacing_m": float(
            ppo_config.get("hard_violation_spacing_m", args.ppo_hard_violation_spacing_m)
        ),
        "action_repeat": action_repeat,
        "gamma": gamma_per_tick ** action_repeat,
        "gae_lambda": float(ppo_config.get("gae_lambda", args.ppo_gae_lambda)),
        "clip_ratio": float(ppo_config.get("clip_ratio", args.ppo_clip_ratio)),
        "entropy_coef": float(ppo_config.get("entropy_coef", args.ppo_entropy_coef)),
        "value_coef": float(ppo_config.get("value_coef", args.ppo_value_coef)),
        "cost_value_coef": float(ppo_config.get("cost_value_coef", args.ppo_cost_value_coef)),
        "cost_limit": float(ppo_config.get("cost_limit", args.ppo_cost_limit)),
        "update_epochs": int(ppo_config.get("update_epochs", args.ppo_update_epochs)),
        "minibatch_size": int(ppo_config.get("minibatch_size", args.ppo_minibatch_size)),
        "rollout_steps": int(ppo_config.get("rollout_steps", args.ppo_rollout_steps)),
        "checkpoint_path": checkpoint_path,
        "episodes_csv": metrics_dir / "ppo_episodes.csv",
        "updates_csv": metrics_dir / "ppo_updates.csv",
        "update_count": update_count,
        "return_stats": return_stats,
        "training": not args.ppo_eval,
        "buffer": [],
    }


def _ppo_run_name(args):
    return "ppo_eval" if args.ppo_eval else "ppo"


def _ppo_policy_step(ppo_state, observation, deterministic=False):
    device = ppo_state["device"]
    obs_norm = _normalize_ppo_observation(observation)
    obs_tensor = torch.tensor(obs_norm, dtype=torch.float32, device=device).unsqueeze(0)

    with torch.no_grad():
        distribution, reward_value, cost_value = ppo_state["model"].distribution(obs_tensor)
        if deterministic:
            raw_action = distribution.mean
        else:
            raw_action = distribution.sample()
        action = torch.tanh(raw_action)
        log_prob = _tanh_log_prob(distribution, raw_action, action)

    return {
        "obs": obs_norm,
        "action": float(action.squeeze(0).squeeze(-1).item()),
        "log_prob": float(log_prob.item()),
        "reward_value": _denormalize_reward_value(ppo_state["return_stats"], float(reward_value.item())),
        "cost_value": float(cost_value.item()),
    }


def _ppo_longitudinal_control(action, base_control, distance_m, emergency_distance_m):
    control = carla.VehicleControl()
    control.steer = base_control.steer
    control.hand_brake = False
    control.reverse = False
    control.manual_gear_shift = False

    if distance_m < emergency_distance_m:
        control.throttle = 0.0
        control.brake = 1.0
        return control

    action = _clamp(float(action), -1.0, 1.0)
    if action >= 0.0:
        control.throttle = action
        control.brake = 0.0
    else:
        control.throttle = 0.0
        control.brake = -action
    return control



def _spacing_penalty(spacing_error):
    if spacing_error > REWARD_D_FAR:
        return (spacing_error - REWARD_D_FAR) ** 2 / REWARD_ALPHA_FAR
    if spacing_error < REWARD_D_CLOSE:
        return (spacing_error - REWARD_D_CLOSE) ** 2 / REWARD_ALPHA_CLOSE
    return 0.0


def _thw_cost(thw):
    if thw >= THW_SAFE_SEC:
        return 0.0
    if thw <= THW_DANGER_SEC:
        return 1.0
    return ((THW_SAFE_SEC - thw) / (THW_SAFE_SEC - THW_DANGER_SEC)) ** 2


def _ppo_reward_and_cost(metrics, reward_state):
    """Paper eq. (10)-(15): r = exp(-(w1 r_cont + w2 r_traf + w3 r_jerk)) - r_penalty, THW 3-level cost."""
    spacing_error = float(metrics["spacing_error_m"])
    rel_speed = float(metrics["relative_velocity_mps"])
    speed = float(metrics["ego_speed_mps"])
    distance = float(metrics["distance_m"])
    accel = float(metrics["ego_accel_mps2"])

    prev_accel = reward_state["prev_accel_mps2"]
    reward_state["prev_accel_mps2"] = accel

    r_cont = REWARD_Q_SPACING * spacing_error ** 2 + REWARD_Q_REL_SPEED * rel_speed ** 2
    r_traf = accel ** 2
    r_jerk = 0.0 if prev_accel is None else (accel - prev_accel) ** 2
    penalty = _spacing_penalty(spacing_error)

    reward = math.exp(-(REWARD_W_CONT * r_cont + REWARD_W_TRAF * r_traf + REWARD_W_JERK * r_jerk)) - penalty

    thw = distance / speed if speed > THW_MIN_SPEED_MPS else float("inf")
    cost = _thw_cost(thw)

    return reward, cost, thw, penalty


def _ppo_value_estimate(ppo_state, observation):
    """V_r (in real reward units) and V_c for one observation, without sampling an action."""
    obs_norm = _normalize_ppo_observation(observation)
    obs_tensor = torch.tensor(obs_norm, dtype=torch.float32, device=ppo_state["device"]).unsqueeze(0)
    with torch.no_grad():
        _, reward_value, cost_value = ppo_state["model"](obs_tensor)
    return (
        _denormalize_reward_value(ppo_state["return_stats"], float(reward_value.item())),
        float(cost_value.item()),
    )


def _append_ppo_transition(ppo_state, policy_step, reward, cost, done):
    transition = dict(policy_step)
    transition["reward"] = float(reward)
    transition["cost"] = float(cost)
    transition["done"] = bool(done)
    ppo_state["buffer"].append(transition)


def _finish_ppo_episode(
    ppo_state,
    pending_step,
    reward_sum,
    cost_sum,
    window_ticks,
    terminal_reward=0.0,
    terminal_cost=0.0,
):
    if pending_step is not None and window_ticks > 0:
        _append_ppo_transition(
            ppo_state,
            pending_step,
            reward_sum / window_ticks + terminal_reward,
            cost_sum / window_ticks + terminal_cost,
            True,
        )
    elif ppo_state["buffer"]:
        ppo_state["buffer"][-1]["reward"] += terminal_reward
        ppo_state["buffer"][-1]["cost"] += terminal_cost
        ppo_state["buffer"][-1]["done"] = True


def _compute_gae(rewards, values, dones, gamma, gae_lambda, bootstrap_value):
    advantages = [0.0] * len(rewards)
    last_gae = 0.0
    next_value = float(bootstrap_value)

    for index in reversed(range(len(rewards))):
        nonterminal = 0.0 if dones[index] else 1.0
        delta = rewards[index] + gamma * next_value * nonterminal - values[index]
        last_gae = delta + gamma * gae_lambda * nonterminal * last_gae
        advantages[index] = last_gae
        next_value = values[index]

    returns = [adv + value for adv, value in zip(advantages, values)]
    return advantages, returns


def _ppo_update(ppo_state, bootstrap_observation=None):
    buffer = ppo_state["buffer"]
    if not buffer:
        return None

    device = ppo_state["device"]
    model = ppo_state["model"]

    bootstrap_reward_value = 0.0
    bootstrap_cost_value = 0.0
    if bootstrap_observation is not None and not buffer[-1]["done"]:
        bootstrap_reward_value, bootstrap_cost_value = _ppo_value_estimate(
            ppo_state, bootstrap_observation
        )

    reward_advantages, reward_returns = _compute_gae(
        [item["reward"] for item in buffer],
        [item["reward_value"] for item in buffer],
        [item["done"] for item in buffer],
        ppo_state["gamma"],
        ppo_state["gae_lambda"],
        bootstrap_reward_value,
    )
    cost_advantages, cost_returns = _compute_gae(
        [item["cost"] for item in buffer],
        [item["cost_value"] for item in buffer],
        [item["done"] for item in buffer],
        ppo_state["gamma"],
        ppo_state["gae_lambda"],
        bootstrap_cost_value,
    )

    obs = torch.tensor([item["obs"] for item in buffer], dtype=torch.float32, device=device)
    actions = torch.tensor([[item["action"]] for item in buffer], dtype=torch.float32, device=device)
    old_log_probs = torch.tensor([item["log_prob"] for item in buffer], dtype=torch.float32, device=device)
    reward_adv = torch.tensor(reward_advantages, dtype=torch.float32, device=device)
    cost_adv = torch.tensor(cost_advantages, dtype=torch.float32, device=device)
    return_stats = ppo_state["return_stats"]
    _update_return_stats(return_stats, reward_returns)
    reward_ret = torch.tensor(reward_returns, dtype=torch.float32, device=device)
    reward_ret = (reward_ret - return_stats["mean"]) / _return_std(return_stats)
    cost_ret = torch.tensor(cost_returns, dtype=torch.float32, device=device)

    reward_adv = (reward_adv - reward_adv.mean()) / (reward_adv.std(unbiased=False) + 1e-8)
    cost_adv = (cost_adv - cost_adv.mean()) / (cost_adv.std(unbiased=False) + 1e-8)

    sample_count = len(buffer)
    minibatch_size = max(1, min(ppo_state["minibatch_size"], sample_count))

    actor_loss_value = 0.0
    actor_grad_norm_value = 0.0
    critic_grad_norm_value = 0.0
    reward_value_loss_value = 0.0
    cost_value_loss_value = 0.0
    entropy_value = 0.0

    for _ in range(ppo_state["update_epochs"]):
        permutation = torch.randperm(sample_count, device=device)
        for start in range(0, sample_count, minibatch_size):
            indices = permutation[start:start + minibatch_size]
            batch_obs = obs[indices]
            batch_actions = actions[indices]
            batch_old_log_probs = old_log_probs[indices]
            batch_reward_adv = reward_adv[indices]
            batch_cost_adv = cost_adv[indices]
            batch_reward_ret = reward_ret[indices]
            batch_cost_ret = cost_ret[indices]

            distribution, reward_values, cost_values = model.distribution(batch_obs)
            raw_actions = _atanh(batch_actions)
            new_log_probs = _tanh_log_prob(distribution, raw_actions, batch_actions)
            ratio = torch.exp(new_log_probs - batch_old_log_probs)

            lagrange_multiplier = ppo_state["lambda"]
            # Dividing by (1 + lambda) keeps the actor step size bounded as lambda grows (OmniSafe PPOLag).
            combined_adv = (batch_reward_adv - lagrange_multiplier * batch_cost_adv) / (1.0 + lagrange_multiplier)

            surrogate_1 = ratio * combined_adv
            surrogate_2 = torch.clamp(
                ratio,
                1.0 - ppo_state["clip_ratio"],
                1.0 + ppo_state["clip_ratio"],
            ) * combined_adv
            actor_loss = -torch.min(surrogate_1, surrogate_2).mean()

            reward_value_loss = torch.nn.functional.mse_loss(reward_values, batch_reward_ret)
            cost_value_loss = torch.nn.functional.mse_loss(cost_values, batch_cost_ret)
            entropy = distribution.entropy().sum(dim=-1).mean()

            policy_loss = actor_loss - ppo_state["entropy_coef"] * entropy
            critic_loss = (
                ppo_state["value_coef"] * reward_value_loss
                + ppo_state["cost_value_coef"] * cost_value_loss
            )

            # The actor and critic graphs are disjoint, so one backward fills each group's
            # gradients independently; each group is then clipped and stepped on its own.
            ppo_state["actor_optimizer"].zero_grad()
            ppo_state["critic_optimizer"].zero_grad()
            (policy_loss + critic_loss).backward()
            actor_grad_norm = torch.nn.utils.clip_grad_norm_(model.actor_parameters(), 0.5)
            critic_grad_norm = torch.nn.utils.clip_grad_norm_(model.critic_parameters(), 0.5)
            ppo_state["actor_optimizer"].step()
            ppo_state["critic_optimizer"].step()

            actor_loss_value = float(actor_loss.item())
            reward_value_loss_value = float(reward_value_loss.item())
            cost_value_loss_value = float(cost_value_loss.item())
            entropy_value = float(entropy.item())
            actor_grad_norm_value = float(actor_grad_norm.item())
            critic_grad_norm_value = float(critic_grad_norm.item())

    # Paper eq.: lambda <- [lambda + eta * (E[C] - d)]+, over episodes finished since the last
    # update. C is the per-tick mean cost and d/eta are rescaled to match; see
    # PPO_NOMINAL_EPISODE_TICKS for why this is not the per-episode sum the paper writes.
    episode_costs = ppo_state["episode_costs"]
    mean_episode_cost = None
    if episode_costs:
        mean_episode_cost = sum(episode_costs) / len(episode_costs)
        ppo_state["lambda"] = max(
            0.0,
            ppo_state["lambda"] + ppo_state["lambda_lr"] * (mean_episode_cost - ppo_state["cost_limit"]),
        )
        episode_costs.clear()

    ppo_state["update_count"] += 1
    lagrange_value = ppo_state["lambda"]

    checkpoint_path = ppo_state["checkpoint_path"]
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "actor_optimizer": ppo_state["actor_optimizer"].state_dict(),
            "critic_optimizer": ppo_state["critic_optimizer"].state_dict(),
            "update_count": ppo_state["update_count"],
            "lambda": lagrange_value,
            "return_stats": ppo_state["return_stats"],
        },
        checkpoint_path,
    )

    result = {
        "samples": sample_count,
        "actor_loss": actor_loss_value,
        "reward_value_loss": reward_value_loss_value,
        "cost_value_loss": cost_value_loss_value,
        "entropy": entropy_value,
        "log_std": float(model.log_std.item()),
        "return_mean": return_stats["mean"],
        "return_std": _return_std(return_stats),
        "actor_grad_norm": actor_grad_norm_value,
        "critic_grad_norm": critic_grad_norm_value,
        "mean_reward": sum(item["reward"] for item in buffer) / sample_count,
        "mean_cost": sum(item["cost"] for item in buffer) / sample_count,
        "mean_episode_cost": mean_episode_cost,
        "lambda": lagrange_value,
        "update_count": ppo_state["update_count"],
    }
    ppo_state["buffer"].clear()
    return result


PPO_UPDATE_FIELDS = [
    "update",
    "episode",
    "samples",
    "mean_reward",
    "mean_cost",
    "mean_episode_cost",
    "lambda",
    "entropy",
    "actor_loss",
    "reward_value_loss",
    "cost_value_loss",
    "log_std",
    "actor_grad_norm",
    "critic_grad_norm",
    "return_mean",
    "return_std",
]

PPO_EPISODE_FIELDS = [
    "episode",
    "end_reason",
    "sim_time_sec",
    "updates_so_far",
    "lambda",
    "episode_reward",
    "mean_reward",
    "episode_cost",
    "mean_cost",
    "collision",
    "mean_action",
    "max_follower_speed_mps",
    "mean_follower_speed_mps",
    "mean_abs_spacing_error_m",
    "rmse_spacing_error_m",
    "mean_abs_relative_velocity_mps",
    "thw_mean_s",
    "thw_min_s",
    "mean_abs_jerk_mps3",
    "min_distance_m",
]


def _append_csv_row(csv_path, fieldnames, row):
    write_header = not csv_path.exists() or csv_path.stat().st_size == 0
    with csv_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def _flush_ppo_training(ppo_state, episode_count):
    """Run one last update on whatever is still buffered when training stops.

    _ppo_update is also what applies the lambda step and writes the checkpoint, so without this
    the final partial rollout, its episode costs, and up to one update of progress are dropped
    every time a run ends or is interrupted.
    """
    if ppo_state is None or not ppo_state["training"] or not ppo_state["buffer"]:
        return

    logging.info("Flushing %d buffered decisions before exit.", len(ppo_state["buffer"]))
    result = _ppo_update(ppo_state)
    if result is not None:
        _log_ppo_update(result)
        _record_ppo_update(ppo_state, result, episode_count)


def _record_ppo_update(ppo_state, result, episode_count):
    if result is None:
        return
    row = {
        "update": result["update_count"],
        "episode": episode_count,
        "samples": result["samples"],
        "mean_reward": round(result["mean_reward"], 5),
        "mean_cost": round(result["mean_cost"], 5),
        "mean_episode_cost": "" if result["mean_episode_cost"] is None else round(result["mean_episode_cost"], 5),
        "lambda": round(result["lambda"], 5),
        "entropy": round(result["entropy"], 5),
        "actor_loss": round(result["actor_loss"], 5),
        "reward_value_loss": round(result["reward_value_loss"], 5),
        "cost_value_loss": round(result["cost_value_loss"], 5),
        "log_std": round(result["log_std"], 5),
        "actor_grad_norm": round(result["actor_grad_norm"], 5),
        "critic_grad_norm": round(result["critic_grad_norm"], 5),
        "return_mean": round(result["return_mean"], 5),
        "return_std": round(result["return_std"], 5),
    }
    _append_csv_row(ppo_state["updates_csv"], PPO_UPDATE_FIELDS, row)


def _record_ppo_episode(ppo_state, episode_count, end_reason, elapsed_sec, rows, episode_stats, dt):
    if not rows or episode_stats["ticks"] == 0:
        return
    n = len(rows)
    ticks = episode_stats["ticks"]
    thw_values = episode_stats["finite_thw"]
    accels = episode_stats["accels"]
    jerks = [abs(b - a) / dt for a, b in zip(accels, accels[1:])]
    row = {
        "episode": episode_count,
        "end_reason": end_reason,
        "sim_time_sec": round(elapsed_sec, 3),
        "updates_so_far": ppo_state["update_count"],
        "lambda": round(ppo_state["lambda"], 5),
        "episode_reward": round(episode_stats["reward_sum"], 5),
        "mean_reward": round(episode_stats["reward_sum"] / ticks, 5),
        "episode_cost": round(episode_stats["cost_sum"], 5),
        "mean_cost": round(episode_stats["cost_sum"] / ticks, 5),
        "collision": int("collision" in end_reason),
        "mean_action": round(episode_stats["action_sum"] / ticks, 5),
        "max_follower_speed_mps": max(r["follower_speed_mps"] for r in rows),
        "mean_follower_speed_mps": round(sum(r["follower_speed_mps"] for r in rows) / n, 5),
        "mean_abs_spacing_error_m": round(sum(abs(r["spacing_error_m"]) for r in rows) / n, 5),
        "rmse_spacing_error_m": round(math.sqrt(sum(r["spacing_error_m"] ** 2 for r in rows) / n), 5),
        "mean_abs_relative_velocity_mps": round(sum(abs(r["relative_velocity_mps"]) for r in rows) / n, 5),
        # THW only exists while the follower is moving; empty when it never moved.
        "thw_mean_s": round(sum(thw_values) / len(thw_values), 5) if thw_values else "",
        "thw_min_s": round(min(thw_values), 5) if thw_values else "",
        "mean_abs_jerk_mps3": round(sum(jerks) / len(jerks), 5) if jerks else "",
        "min_distance_m": min(r["distance_m"] for r in rows),
    }
    _append_csv_row(ppo_state["episodes_csv"], PPO_EPISODE_FIELDS, row)


def _log_ppo_update(result):
    if result is None:
        return
    logging.info(
        (
            "PPO update %d: samples=%d actor_loss=%.4f reward_v_loss=%.4f "
            "cost_v_loss=%.4f entropy=%.4f log_std=%.4f actor_grad=%.4f critic_grad=%.3f "
            "return_mean=%.2f return_std=%.2f "
            "mean_reward=%.4f mean_cost=%.4f episode_cost=%s lambda=%.4f"
        ),
        result["update_count"],
        result["samples"],
        result["actor_loss"],
        result["reward_value_loss"],
        result["cost_value_loss"],
        result["entropy"],
        result["log_std"],
        result["actor_grad_norm"],
        result["critic_grad_norm"],
        result["return_mean"],
        result["return_std"],
        result["mean_reward"],
        result["mean_cost"],
        "n/a" if result["mean_episode_cost"] is None else f"{result['mean_episode_cost']:.3f}",
        result["lambda"],
    )


def main(args):
    scenario_path = Path(args.scenario).resolve()
    scenario_config = load_config_file(scenario_path)

    stack_path = Path(scenario_config.get("stack_file", "stack.json"))
    if not stack_path.is_absolute():
        stack_path = scenario_path.parent / stack_path
    stack_config = load_config_file(stack_path)
    stack_config = _apply_scenario_to_stack(stack_config, scenario_config)

    client = carla.Client(args.host, args.port)
    client.set_timeout(float(args.client_timeout))

    world = None
    original_settings = None
    traffic_manager = None
    ppo_state = None
    episode_count = 0

    try:
        logging.info(
            "Connecting to CARLA at %s:%s (timeout %.1fs)",
            args.host,
            args.port,
            float(args.client_timeout),
        )
        world, original_settings, traffic_manager = _connect_world_with_retry(
            client,
            args.host,
            args.port,
            scenario_config["map"],
            args.tm_port,
            float(args.retry_interval),
            float(args.max_connect_wait_sec),
            bool(scenario_config.get("allow_map_load", False)),
        )
        
        if args.no_rendering:
            settings = world.get_settings()
            settings.no_rendering_mode = True
            world.apply_settings(settings)
            logging.info("CARLA no-rendering mode enabled")

        map_ = world.get_map()
        _configure_traffic_lights(world, scenario_config)

        if args.print_spawn_points:
            _print_spawn_points(world)
            return

        stack_config, route_locations_by_actor = _align_scenario_to_map(stack_config, scenario_config, map_)
        max_time_sec = float(scenario_config.get("episode", {}).get("max_time_sec", 90.0))
        respawn_delay_sec = float(scenario_config.get("episode", {}).get("respawn_delay_sec", 2.0))
        dt = float(world.get_settings().fixed_delta_seconds or 0.05)
        tick_guard = _create_tick_guard(world)
        spectator_config = scenario_config.get("spectator", {})

        if args.controller == "ppo":
            ppo_state = _create_ppo_state(args, scenario_config)
            logging.info(
                "PPO-Lagrangian enabled on %s; checkpoint=%s action_repeat=%d gamma/decision=%.4f rollout=%d decisions",
                ppo_state["device"],
                ppo_state["checkpoint_path"],
                ppo_state["action_repeat"],
                ppo_state["gamma"],
                ppo_state["rollout_steps"],
            )

        while True:
            vehicles = []
            sensors = []
            collision_states = {}
            leader_agent = None
            follower_agent = None
            leader_profile_state = None
            follower_pid_state = None

            try:
                vehicles, sensors, objects = spawn_actors_from_config(world, stack_config)
                vehicles_by_id = {
                    object_config["id"]: vehicle
                    for vehicle, object_config in zip(vehicles, objects)
                }
                leader_vehicle = vehicles_by_id["leader"]
                follower_vehicle = vehicles_by_id["follower"]

                for actor_id, vehicle in vehicles_by_id.items():
                    collision_sensor, collision_state = _attach_collision_sensor(world, vehicle)
                    sensors.append(collision_sensor)
                    collision_states[actor_id] = collision_state

                _ = _tick_world_single_owner(world, tick_guard)

                _tick_for_seconds_single_owner(
                    world,
                    float(scenario_config.get("episode", {}).get("settle_time_sec", 1.0)),
                    tick_guard,
                )
                leader_agent = _build_basic_agent(
                    leader_vehicle,
                    map_,
                    scenario_config,
                    route_locations_by_actor["leader"],
                )
                follower_agent = _build_basic_agent(
                    follower_vehicle,
                    map_,
                    scenario_config,
                    route_locations_by_actor["follower"],
                    option_overrides={
                        "ignore_traffic_lights": True,
                        "ignore_stop_signs": True,
                        "ignore_vehicles": True,
                    },
                )
                leader_profile_state = _create_speed_profile_state(scenario_config, _get_speed_mps(leader_vehicle))
                follower_pid_state = _create_follower_pid_state(scenario_config)
                _update_spectator(world, [leader_vehicle, follower_vehicle], spectator_config)

                episode_count += 1
                rows = []  # ← 이 줄 추가
                logging.info(
                    "Episode %d started (leader target speed %.1f m/s)",
                    episode_count,
                    leader_profile_state["target_speed_mps"],
                )

                elapsed_sec = 0.0
                tick_count = 0
                pending_ppo_step = None
                window_reward_sum = 0.0
                window_cost_sum = 0.0
                window_ticks = 0
                episode_stats = {
                    "reward_sum": 0.0,
                    "cost_sum": 0.0,
                    "action_sum": 0.0,
                    "ticks": 0,
                    "accels": [],
                    "finite_thw": [],
                }
                reward_state = {"prev_accel_mps2": None}
                ppo_penalty = 0.0
                # Guards GAE against chaining into an episode that aborted with an exception.
                if ppo_state is not None and ppo_state["buffer"]:
                    ppo_state["buffer"][-1]["done"] = True
                route_state = {
                    "leader_index": 0,
                    "follower_index": 0,
                    "leader_route_locations": route_locations_by_actor["leader"],
                    "follower_route_locations": route_locations_by_actor["follower"],
                }
                timers = {"stuck_sec": 0.0, "elapsed_sec": 0.0}
                termination = scenario_config.get("termination", {})

                while True:
                    _ = _tick_world_single_owner(world, tick_guard)
                    elapsed_sec += dt
                    tick_count += 1
                    timers["elapsed_sec"] = elapsed_sec
                    _update_speed_profile(leader_vehicle, leader_agent, leader_profile_state, dt)
                    leader_control = leader_agent.run_step()
                    leader_control = _apply_speed_profile_to_control(leader_control, leader_profile_state)
                    leader_vehicle.apply_control(leader_control)

                    follower_control, follower_metrics = _compute_follower_control(
                        leader_vehicle,
                        follower_vehicle,
                        follower_agent,
                        follower_pid_state,
                        dt,
                    )

                    ppo_reward = None
                    ppo_cost = None
                    ppo_thw = None

                    if args.controller == "ppo":
                        # Current state is the outcome of the action chosen on the previous tick.
                        ppo_reward, ppo_cost, ppo_thw, ppo_penalty = _ppo_reward_and_cost(
                            follower_metrics,
                            reward_state,
                        )

                        if pending_ppo_step is not None:
                            window_reward_sum += ppo_reward
                            window_cost_sum += ppo_cost
                            window_ticks += 1

                        # Hold each action for action_repeat ticks so throttle lasts long enough to launch.
                        if pending_ppo_step is None or window_ticks >= ppo_state["action_repeat"]:
                            if pending_ppo_step is not None and ppo_state["training"]:
                                _append_ppo_transition(
                                    ppo_state,
                                    pending_ppo_step,
                                    window_reward_sum / window_ticks,
                                    window_cost_sum / window_ticks,
                                    False,
                                )

                            if ppo_state["training"] and len(ppo_state["buffer"]) >= ppo_state["rollout_steps"]:
                                update_result = _ppo_update(
                                    ppo_state,
                                    bootstrap_observation=follower_metrics["observation"],
                                )
                                _log_ppo_update(update_result)
                                _record_ppo_update(ppo_state, update_result, episode_count)

                            pending_ppo_step = _ppo_policy_step(
                                ppo_state,
                                follower_metrics["observation"],
                                deterministic=args.ppo_deterministic or args.ppo_eval,
                            )
                            window_reward_sum = 0.0
                            window_cost_sum = 0.0
                            window_ticks = 0
                        follower_control = _ppo_longitudinal_control(
                            pending_ppo_step["action"],
                            follower_control,
                            follower_metrics["distance_m"],
                            follower_pid_state["emergency_distance_m"],
                        )

                        follower_metrics["throttle"] = follower_control.throttle
                        follower_metrics["brake"] = follower_control.brake

                        episode_stats["reward_sum"] += ppo_reward
                        episode_stats["cost_sum"] += ppo_cost
                        episode_stats["action_sum"] += pending_ppo_step["action"]
                        episode_stats["ticks"] += 1
                        episode_stats["accels"].append(follower_metrics["ego_accel_mps2"])
                        if math.isfinite(ppo_thw):
                            episode_stats["finite_thw"].append(ppo_thw)

                        if tick_count % follower_pid_state["log_interval_ticks"] == 0:
                            logging.info(
                                "PPO obs=%s action=%.3f reward=%.3f cost=%.1f THW=%s",
                                [round(value, 3) for value in follower_metrics["observation"]],
                                pending_ppo_step["action"],
                                ppo_reward,
                                ppo_cost,
                                "inf" if math.isinf(ppo_thw) else f"{ppo_thw:.2f}",
                            )

                    follower_vehicle.apply_control(follower_control)
                    rows.append({
                        "episode": episode_count,
                        "time_sec": round(elapsed_sec, 3),
                        "tick": tick_count,
                        "leader_phase": leader_profile_state["phase"],
                        "leader_speed_mps": round(follower_metrics["lead_speed_mps"], 3),
                        "follower_speed_mps": round(follower_metrics["ego_speed_mps"], 3),
                        "distance_m": round(follower_metrics["distance_m"], 3),
                        "desired_distance_m": round(follower_metrics["desired_distance_m"], 3),
                        "spacing_error_m": round(follower_metrics["spacing_error_m"], 3),
                        "relative_velocity_mps": round(follower_metrics["relative_velocity_mps"], 3),
                        "throttle": round(follower_metrics["throttle"], 3),
                        "brake": round(follower_metrics["brake"], 3),
                    })
                   
                    if tick_count % follower_pid_state["log_interval_ticks"] == 0:
                        _log_follower_metrics(episode_count, tick_count, follower_metrics)

                    _update_spectator(world, [leader_vehicle, follower_vehicle], spectator_config)

                    reason = _evaluate_episode(
                        leader_vehicle,
                        follower_vehicle,
                        leader_agent,
                        leader_profile_state,
                        follower_pid_state,
                        route_state,
                        collision_states,
                        timers,
                        termination,
                        dt,
                    )
                    if (
                        reason is None
                        and args.controller == "ppo"
                        and follower_metrics["spacing_error_m"] > ppo_state["hard_violation_spacing_m"]
                    ):
                        reason = "hard_violation"
                    if reason is None and elapsed_sec >= max_time_sec:
                        reason = "timeout"

                    if reason is not None:
                        logging.info("Episode %d ended: %s", episode_count, reason)
                        csv_name = args.controller
                        if args.controller == "ppo":
                            csv_name = _ppo_run_name(args)
                            if ppo_state["training"]:
                                # Failure endings must not be an escape from ongoing penalty, so charge
                                # the final penalty as if it persisted forever: penalty / (1 - gamma).
                                terminal_reward = 0.0
                                terminal_cost = 0.0
                                if reason in PPO_FAILURE_REASONS:
                                    terminal_reward = -ppo_penalty / (1.0 - ppo_state["gamma"])
                                if reason in PPO_COLLISION_REASONS:
                                    terminal_cost = PPO_MAX_TICK_COST / (1.0 - ppo_state["gamma"])
                                if reason in PPO_TRUNCATION_REASONS:
                                    boot_r, boot_c = _ppo_value_estimate(
                                        ppo_state, follower_metrics["observation"]
                                    )
                                    terminal_reward = ppo_state["gamma"] * boot_r
                                    terminal_cost = ppo_state["gamma"] * boot_c
                                _finish_ppo_episode(
                                    ppo_state,
                                    pending_ppo_step,
                                    window_reward_sum,
                                    window_cost_sum,
                                    window_ticks,
                                    terminal_reward,
                                    terminal_cost,
                                )
                                # Per-tick mean, so a short episode is not cheaper than a long one.
                                episode_ticks = max(1, episode_stats["ticks"])
                                ppo_state["episode_costs"].append(
                                    episode_stats["cost_sum"] / episode_ticks
                                )
                            _record_ppo_episode(
                                ppo_state, episode_count, reason, elapsed_sec, rows, episode_stats, dt
                            )
                        _write_episode_csv(args.output_dir, episode_count, rows, controller=csv_name)
                        break
            except Exception as exc:
                logging.exception("Episode %d failed: %s", episode_count + 1, exc)

            finally:
                destroy_actors(sensors, vehicles)

            if args.once or (args.train_episodes > 0 and episode_count >= args.train_episodes):
                break

            _tick_for_seconds_single_owner(world, respawn_delay_sec, tick_guard)

        _flush_ppo_training(ppo_state, episode_count)

    except KeyboardInterrupt:
        print("\nCancelled by user. Bye!")
        _flush_ppo_training(ppo_state, episode_count)
    except RuntimeError as exc:
        message = str(exc)
        if "time-out" in message and "simulator" in message:
            raise SystemExit(
                "Failed to connect to CARLA simulator.\n"
                f"Checked {args.host}:{args.port} for {float(args.client_timeout):.1f} seconds.\n"
                "Make sure CARLA is running and fully loaded before starting episode.py."
            ) from exc
        raise SystemExit(f"CARLA runtime error: {message}") from exc
    except Exception as exc:
        raise SystemExit(f"Episode runner failed: {exc}") from exc
    finally:
        _restore_world(world, original_settings, traffic_manager)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="CARLA episode runner")
    argparser.add_argument(
        "--host",
        metavar="H",
        default="localhost",
        help="IP of the host CARLA Simulator (default: localhost)",
    )
    argparser.add_argument(
        "--port",
        metavar="P",
        default=2000,
        type=int,
        help="TCP port of CARLA Simulator (default: 2000)",
    )
    argparser.add_argument(
        "--scenario",
        default=str(DEFAULT_SCENARIO_FILE),
        help="Scenario file to execute (default: scenario_town03.json next to this script)",
    )
    argparser.add_argument(
        "--tm-port",
        default=8000,
        type=int,
        help="Traffic Manager port (default: 8000)",
    )
    argparser.add_argument(
        "--client-timeout",
        default=15.0,
        type=float,
        help="CARLA client timeout in seconds (default: 15.0)",
    )
    argparser.add_argument(
        "--retry-interval",
        default=2.0,
        type=float,
        help="Seconds to wait before retrying a CARLA connection (default: 2.0)",
    )
    argparser.add_argument(
        "--max-connect-wait-sec",
        default=0.0,
        type=float,
        help="Maximum seconds to wait for CARLA before exiting; 0 means wait forever (default: 0.0)",
    )
    argparser.add_argument(
        "--print-spawn-points",
        action="store_true",
        help="Print Town03 spawn point indexes and exit",
    )
    argparser.add_argument(
        "--once",
        action="store_true",
        help="Run a single episode and exit",
    )
    argparser.add_argument(
        "--train-episodes",
        type=int,
        default=0,
        help="Stop after this many episodes; 0 runs until Ctrl+C (default: 0)",
    )
    argparser.add_argument(
        "--no-rendering",
        action="store_true",
        help="Disable CARLA rendering for faster training",
    )
    argparser.add_argument(
        "--controller",
        choices=("pid", "ppo"),
        default="pid",
        help="Longitudinal controller to use: pid or ppo (default: pid)",
    )
    argparser.add_argument(
        "--output-dir",
        default="outputs",
        help="Root for episode CSVs and PPO metric logs; written under <dir>/<controller>/ (default: outputs)",
    )
    argparser.add_argument(
        "--ppo-checkpoint",
        default="outputs/ppo_lag.pt",
        help="PPO checkpoint path (default: outputs/ppo_lag.pt)",
    )
    argparser.add_argument(
        "--ppo-device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Torch device for PPO (default: auto)",
    )
    argparser.add_argument(
        "--ppo-deterministic",
        action="store_true",
        help="Use the PPO mean action instead of sampling while still training.",
    )
    argparser.add_argument(
        "--ppo-eval",
        action="store_true",
        help="Evaluate a trained checkpoint: deterministic actions, no updates, checkpoint untouched; "
        "results go to <output-dir>/ppo_eval/",
    )
    argparser.add_argument(
        "--ppo-fresh",
        action="store_true",
        help="Ignore an existing checkpoint and start training from scratch (the file is overwritten)",
    )
    argparser.add_argument("--seed", type=int, default=42)
    argparser.add_argument("--ppo-hidden-dim", type=int, default=128)
    argparser.add_argument("--ppo-lr", type=float, default=3e-4)
    argparser.add_argument(
        "--ppo-lambda-lr",
        type=float,
        default=PPO_LAMBDA_LR_PER_TICK,
        help=(
            "Step size eta in lambda <- [lambda + eta*(E[per-tick cost] - d)]+. "
            f"Paper eta={PPO_PAPER_LAMBDA_LR:g} paired with a per-episode d, rescaled here by the "
            f"nominal {PPO_NOMINAL_EPISODE_TICKS}-tick episode (default: {PPO_LAMBDA_LR_PER_TICK:g})"
        ),
    )
    argparser.add_argument("--ppo-initial-lambda", type=float, default=0.0)
    argparser.add_argument(
        "--ppo-hard-violation-spacing-m",
        type=float,
        default=40.0,
        help="End a PPO episode as a failure when spacing error exceeds this [m] (default: 40)",
    )
    argparser.add_argument(
        "--ppo-action-repeat",
        type=int,
        default=10,
        help="Simulator ticks to hold each PPO action (default: 10 = 0.5s at dt=0.05)",
    )
    argparser.add_argument(
        "--ppo-gamma",
        type=float,
        default=0.99,
        help="Discount per simulator tick; raised to the action-repeat power per decision (default: 0.99)",
    )
    argparser.add_argument("--ppo-gae-lambda", type=float, default=0.95)
    argparser.add_argument("--ppo-clip-ratio", type=float, default=0.2)
    argparser.add_argument("--ppo-entropy-coef", type=float, default=0.01)
    argparser.add_argument("--ppo-value-coef", type=float, default=0.5)
    argparser.add_argument("--ppo-cost-value-coef", type=float, default=0.5)
    argparser.add_argument(
        "--ppo-cost-limit",
        type=float,
        default=PPO_COST_LIMIT_PER_TICK,
        help=(
            "Per-tick mean THW cost limit d. "
            f"Paper Table I gives d={PPO_PAPER_COST_LIMIT:g} as a per-episode sum; divided by the "
            f"nominal {PPO_NOMINAL_EPISODE_TICKS}-tick episode (default: {PPO_COST_LIMIT_PER_TICK:.6f})"
        ),
    )
    argparser.add_argument("--ppo-update-epochs", type=int, default=10)
    argparser.add_argument("--ppo-minibatch-size", type=int, default=64)
    argparser.add_argument(
        "--ppo-rollout-steps",
        type=int,
        default=512,
        help="PPO decisions (not ticks) collected across episodes before each update (default: 512)",
    )
    argparser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        dest="debug",
        help="print debug information",
    )

    parsed_args = argparser.parse_args()

    log_level = logging.DEBUG if parsed_args.debug else logging.INFO
    logging.basicConfig(format="%(levelname)s: %(message)s", level=log_level)

    logging.info("Listening to server %s:%s", parsed_args.host, parsed_args.port)

    main(parsed_args)