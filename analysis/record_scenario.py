#!/usr/bin/env python3.10
"""Record one platoon evaluation episode from a chase camera and turn it into a slide video.

The episode is run by episode_ppo_v2.main() itself, unchanged. This script only wraps a few of
its module-level functions to observe the run:
  spawn_actors_from_config   -> vehicle handles; spawns the RGB camera (destroyed with the actors)
  create_channels            -> per-tick V2V reading of each follower (tick counter, packet state)
  _update_leader_brake_event -> leader hard-brake state
  _update_spectator          -> called once per tick after the controls: pull this frame's image,
                                log the HUD state, place the camera for the next frame
None of the wrappers touch the global `random`, torch, or any actor's control, so the episode is
the same as in the evaluation (checked afterwards by comparing the step CSVs).

The default argument values are taken from episode_ppo_v2.py's own argparse block, so this
script never carries a second copy of them.

  record   run the episode, write raw frames + frames.csv (+ the usual episode CSVs)
  compose  draw the HUD on the frames and encode the mp4 (no CARLA needed)

Usage (see outputs/video/README.md for the exact commands that made the video):
  .venv/bin/python analysis/record_scenario.py record --out-dir outputs/video/run_lag_s43_ep001
  .venv/bin/python analysis/record_scenario.py record --test-times 6,12,15,18,24,29 --out-dir ...
  .venv/bin/python analysis/record_scenario.py compose --run-dir ... --output outputs/video/x.mp4
"""

import argparse
import ast
import csv
import math
import queue
import shlex
import subprocess
import sys
import threading
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

FONT_PATH = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"
FONT_INDEX_KR = 1
DT = 0.05  # episode_ppo_v2._setup_world fixes fixed_delta_seconds to 0.05
FRAME_FIELDS = [
    "frame", "tick", "time_sec", "f1_comm_success", "f1_t_since_comm_s", "f1_comm_in_tunnel",
    "f1_comm_p", "f2_comm_success", "f2_t_since_comm_s", "brake_state", "brake_active",
    "brake_start_sec", "gap_f1_m", "gap_f2_m", "leader_speed_mps", "f1_speed_mps", "f2_speed_mps",
]


# ----------------------------------------------------------------------------------------- record

def _episode_argparser(ep):
    """Rebuild episode_ppo_v2's ArgumentParser from its __main__ block (statements before parse_args)."""
    tree = ast.parse(Path(ep.__file__).read_text(encoding="utf-8"))
    main_block = next(
        node for node in tree.body
        if isinstance(node, ast.If) and isinstance(node.test, ast.Compare)
        and getattr(node.test.left, "id", None) == "__name__"
    )
    body = []
    for stmt in main_block.body:
        if "parse_args" in ast.unparse(stmt):
            break
        body.append(stmt)
    namespace = dict(vars(ep))
    exec(compile(ast.Module(body=body, type_ignores=[]), ep.__file__, "exec"), namespace)
    return namespace["argparser"]


class Recorder:
    def __init__(self, ep, carla, opts):
        self.ep = ep
        self.carla = carla
        self.opts = opts
        self.frames_dir = Path(opts.out_dir) / "frames"
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        self.test_times = sorted(float(t) for t in opts.test_times.split(",")) if opts.test_times else []
        self.vehicles = {}
        self.camera = None
        self.images = queue.Queue()
        self.channels = {}
        self.tick = 0
        self.brake = None
        self.cam_pos = None
        self.cam_look = None
        self.frame_index = 0
        self.rows = []
        self.missing = 0
        self.writer_q = queue.Queue(maxsize=64)
        self.writer = threading.Thread(target=self._write_loop, daemon=True)
        self.writer.start()

    # -- wrappers -------------------------------------------------------------------------------
    def install(self):
        ep = self.ep
        orig_spawn = ep.spawn_actors_from_config
        orig_destroy = ep.destroy_actors
        orig_channels = ep.create_channels
        orig_brake = ep._update_leader_brake_event
        orig_spectator = ep._update_spectator

        def spawn(world, stack_config):
            vehicles, sensors, objects = orig_spawn(world, stack_config)
            self.vehicles = {obj["id"]: v for v, obj in zip(vehicles, objects)}
            self._spawn_camera(world)
            return vehicles, sensors, objects

        def destroy(sensors, vehicles):
            if self.camera is not None:
                self.camera.stop()
                self.camera.destroy()
                self.camera = None
            orig_destroy(sensors, vehicles)

        def channels(*args, **kwargs):
            result = orig_channels(*args, **kwargs)
            for receiver_id, channel in result.items():
                self._wrap_channel(receiver_id, channel)
            self.channels = result
            return result

        def brake(event, *args, **kwargs):
            orig_brake(event, *args, **kwargs)
            self.brake = event

        def spectator(world, vehicles, camera_config):
            orig_spectator(world, vehicles, camera_config)
            self._on_tick(world)

        ep.spawn_actors_from_config = spawn
        ep.destroy_actors = destroy
        ep.create_channels = channels
        ep._update_leader_brake_event = brake
        ep._update_spectator = spectator

    def _wrap_channel(self, receiver_id, channel):
        orig_step = channel.step
        orig_reset = channel.reset

        def step(sender_state, dt, in_tunnel=False):
            reading = orig_step(sender_state, dt, in_tunnel)
            channel.rec_in_tunnel = bool(in_tunnel)
            if receiver_id == "follower":
                self.tick += 1  # stepped exactly once per episode-loop tick, before the spectator call
            return reading

        def reset(episode, sender_state):
            orig_reset(episode, sender_state)
            channel.rec_in_tunnel = False
            if receiver_id == "follower":
                self.tick = 0

        channel.step = step
        channel.reset = reset

    # -- camera ---------------------------------------------------------------------------------
    def _spawn_camera(self, world):
        bp = world.get_blueprint_library().find("sensor.camera.rgb")
        bp.set_attribute("image_size_x", str(self.opts.width))
        bp.set_attribute("image_size_y", str(self.opts.height))
        bp.set_attribute("fov", str(self.opts.fov))
        bp.set_attribute("sensor_tick", "0.0")  # every simulation frame
        if bp.has_attribute("motion_blur_intensity"):
            bp.set_attribute("motion_blur_intensity", "0.0")
        self.camera = world.spawn_actor(bp, self.carla.Transform(self.carla.Location(z=200.0)))
        self.camera.listen(self.images.put)
        self.cam_pos = None
        self.cam_look = None

    def _camera_target(self):
        loc = {k: v.get_location() for k, v in self.vehicles.items()}
        f2 = self.vehicles.get(self.ep.PLATOON_FOLLOWER_2_ID) or self.vehicles["follower"]
        yaw = math.radians(f2.get_transform().rotation.yaw)
        rear = loc[self.ep.PLATOON_FOLLOWER_2_ID] if self.ep.PLATOON_FOLLOWER_2_ID in loc else loc["follower"]
        pos = (rear.x - math.cos(yaw) * self.opts.cam_back,
               rear.y - math.sin(yaw) * self.opts.cam_back,
               rear.z + self.opts.cam_height)
        w = self.opts.look_lead_weight  # look point: between F1 (0) and the leader (1)
        lead, f1 = loc["leader"], loc["follower"]
        look = (f1.x + w * (lead.x - f1.x), f1.y + w * (lead.y - f1.y), f1.z + w * (lead.z - f1.z))
        return pos, look

    def _place_camera(self, snap):
        pos, look = self._camera_target()
        if snap or self.cam_pos is None:
            self.cam_pos, self.cam_look = pos, look
        else:
            a, b = self.opts.pos_smooth, self.opts.look_smooth
            self.cam_pos = tuple(c + a * (t - c) for c, t in zip(self.cam_pos, pos))
            self.cam_look = tuple(c + b * (t - c) for c, t in zip(self.cam_look, look))
        dx, dy, dz = (l - c for l, c in zip(self.cam_look, self.cam_pos))
        yaw = math.degrees(math.atan2(dy, dx))
        pitch = math.degrees(math.atan2(dz, math.hypot(dx, dy))) + self.opts.pitch_bias
        self.camera.set_transform(self.carla.Transform(
            self.carla.Location(*self.cam_pos), self.carla.Rotation(pitch=pitch, yaw=yaw, roll=0.0)))

    # -- per tick -------------------------------------------------------------------------------
    def _on_tick(self, world):
        if self.camera is None or "follower" not in self.vehicles:
            return
        if self.tick == 0:
            # Before the first episode tick: snap the camera into place; the settle-tick images
            # are not part of the episode and get dropped below.
            self._place_camera(snap=True)
            return
        frame = world.get_snapshot().frame
        image = self._image_for(frame)
        time_sec = round(self.tick * DT, 3)
        self._log_state(frame, time_sec)
        if image is not None:
            if not self.test_times:
                self.writer_q.put((self.frames_dir / f"{self.frame_index:05d}.jpg", image, 95))
            elif any(abs(time_sec - t) < DT / 2 for t in self.test_times):
                self.writer_q.put((self.frames_dir / f"test_t{time_sec:05.2f}.png", image, None))
        self.frame_index += 1
        if self.test_times and time_sec >= self.test_times[-1]:
            raise StopRecording(f"test frames done at t={time_sec:.2f}s")
        self._place_camera(snap=False)

    def _image_for(self, frame):
        while True:
            try:
                image = self.images.get(timeout=10.0)
            except queue.Empty:
                self.missing += 1
                print(f"WARNING: no camera image for frame {frame}", file=sys.stderr)
                return None
            if image.frame == frame:
                return image
            if image.frame > frame:
                self.missing += 1
                print(f"WARNING: image frame {image.frame} ahead of world frame {frame}", file=sys.stderr)
                return None
            # older frame (settle ticks): drop

    def _log_state(self, frame, time_sec):
        ep = self.ep
        v = self.vehicles
        f2_id = ep.PLATOON_FOLLOWER_2_ID
        c1 = self.channels["follower"]
        c2 = self.channels.get(f2_id)
        loc = {k: x.get_location() for k, x in v.items()}
        b = self.brake or {"state": "off", "active": 0, "start_time_sec": ""}
        self.rows.append({
            "frame": self.frame_index, "tick": self.tick, "time_sec": time_sec,
            "f1_comm_success": int(c1.last_success),
            "f1_t_since_comm_s": round(c1.t_since_success, 3),
            "f1_comm_in_tunnel": int(c1.rec_in_tunnel),
            "f1_comm_p": round(c1.p_now, 4),
            "f2_comm_success": int(c2.last_success) if c2 else "",
            "f2_t_since_comm_s": round(c2.t_since_success, 3) if c2 else "",
            "brake_state": b["state"], "brake_active": int(b["active"]),
            "brake_start_sec": b["start_time_sec"],
            "gap_f1_m": round(ep._distance_between(loc["leader"], loc["follower"]), 3),
            "gap_f2_m": round(ep._distance_between(loc["follower"], loc[f2_id]), 3) if f2_id in loc else "",
            "leader_speed_mps": round(ep._get_speed_mps(v["leader"]), 3),
            "f1_speed_mps": round(ep._get_speed_mps(v["follower"]), 3),
            "f2_speed_mps": round(ep._get_speed_mps(v[f2_id]), 3) if f2_id in v else "",
        })

    def _write_loop(self):
        import numpy as np
        from PIL import Image
        while True:
            item = self.writer_q.get()
            if item is None:
                return
            path, image, quality = item
            bgra = np.frombuffer(image.raw_data, dtype=np.uint8).reshape(image.height, image.width, 4)
            rgb = Image.fromarray(bgra[:, :, 2::-1].copy())
            if quality is None:
                rgb.save(path)
            else:
                rgb.save(path, quality=quality, subsampling=0)

    def finish(self):
        self.writer_q.put(None)
        self.writer.join()
        with open(Path(self.opts.out_dir) / "frames.csv", "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=FRAME_FIELDS)
            writer.writeheader()
            writer.writerows(self.rows)
        print(f"frames logged={len(self.rows)} missing_images={self.missing}")


class StopRecording(Exception):
    pass


def cmd_record(opts):
    import carla
    import logging
    import episode_ppo_v2 as ep

    out_dir = Path(opts.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    episode_argv = [
        "--controller", "ppo", "--ppo-eval", "--ppo-checkpoint", opts.checkpoint,
        "--followers", "2", "--once",
        "--seed", str(opts.eval_seed), "--p-success", str(opts.p_out), "--p-tunnel", str(opts.p_tunnel),
        "--tunnel-burst-ticks", "0", "--p-tunnel-noise-std", str(opts.noise_std),
        "--p-tunnel-noise-tau-sec", "1.0", "--leader-brake-prob", str(opts.brake_prob),
        "--port", str(opts.port), "--tm-port", str(opts.port + 6000), "--max-connect-wait-sec", "60",
        "--output-dir", str(out_dir / "episode"),
    ]
    if opts.episode != 1:
        # Episode k would need episodes 1..k-1 run first (leader speeds come from one global
        # random stream), so only the first evaluation episode is supported.
        raise SystemExit("only --episode 1 is supported")
    args = _episode_argparser(ep).parse_args(episode_argv)
    (out_dir / "episode_args.txt").write_text(
        "episode_ppo_v2.py " + " ".join(shlex.quote(a) for a in episode_argv) + "\n")
    logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.INFO)

    rec = Recorder(ep, carla, opts)
    rec.install()
    try:
        ep.main(args)
    finally:
        rec.finish()


# ---------------------------------------------------------------------------------------- compose

def _display_comm_lost(rows, lost_after_s, min_hold_s):
    """Badge state per frame: lost once no packet for lost_after_s, each state held >= min_hold_s."""
    shown, last_change, out = False, -1e9, []
    for row in rows:
        t = float(row["time_sec"])
        raw = float(row["f1_t_since_comm_s"]) >= lost_after_s
        if raw != shown and t - last_change >= min_hold_s:
            shown, last_change = raw, t
        out.append(shown)
    return out


def _draw_box_text(draw, xy, text, font, fill_box, anchor_x="left", pad=(28, 14)):
    x, y = xy
    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
    w, h = right - left, bottom - top
    if anchor_x == "right":
        x -= w + 2 * pad[0]
    elif anchor_x == "center":
        x -= (w + 2 * pad[0]) // 2
    box = (x, y, x + w + 2 * pad[0], y + h + 2 * pad[1])
    draw.rounded_rectangle(box, radius=14, fill=fill_box)
    draw.text((x + pad[0] - left, y + pad[1] - top), text, font=font, fill=(255, 255, 255, 255))
    return box


def cmd_compose(opts):
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont

    run = Path(opts.run_dir)
    rows = list(csv.DictReader(open(run / "frames.csv")))
    tunnel = [i for i, r in enumerate(rows) if r["f1_comm_in_tunnel"] == "1"]
    if not tunnel:
        raise SystemExit("no tunnel frames in frames.csv")
    t_in, t_out = float(rows[tunnel[0]]["time_sec"]), float(rows[tunnel[-1]]["time_sec"])
    start = max(float(rows[0]["time_sec"]), t_in - opts.pre_sec)
    end = min(float(rows[-1]["time_sec"]), t_out + opts.post_sec)
    lost = _display_comm_lost(rows, opts.lost_after_s, opts.min_hold_s)
    brake_starts = [float(r["brake_start_sec"]) for r in rows if r["brake_start_sec"] not in ("", None)]
    brake_t = brake_starts[0] if brake_starts else None
    picked = [i for i, r in enumerate(rows) if start - 1e-6 <= float(r["time_sec"]) <= end + 1e-6]
    duration = len(picked) / opts.fps
    print(f"tunnel {t_in:.2f}-{t_out:.2f}s, clip {start:.2f}-{end:.2f}s = {len(picked)} frames = {duration:.2f}s, "
          f"brake at {brake_t}")

    H = None
    font_main = font_big = None
    output = Path(opts.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg = None
    preview_dir = Path(opts.preview_dir) if opts.preview_dir else None
    if preview_dir:
        preview_dir.mkdir(parents=True, exist_ok=True)
    preview_times = [float(t) for t in opts.preview_times.split(",")] if opts.preview_times else []

    for n, i in enumerate(picked):
        row = rows[i]
        t = float(row["time_sec"])
        frame_path = run / "frames" / f"{int(row['frame']):05d}.jpg"
        img = Image.open(frame_path).convert("RGB")
        if H is None:
            W, H = img.size
            font_main = ImageFont.truetype(FONT_PATH, int(H * opts.text_frac / 0.74), index=FONT_INDEX_KR)
            font_big = ImageFont.truetype(FONT_PATH, int(H * opts.banner_frac / 0.74), index=FONT_INDEX_KR)
            margin = int(H * 0.035)
            vf = f"fade=t=out:st={max(0.0, duration - opts.fade_sec):.3f}:d={opts.fade_sec:.3f}"
            cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                   "-s", f"{W}x{H}", "-r", str(opts.fps), "-i", "-", "-an", "-vf", vf,
                   "-c:v", "libx264", "-preset", "veryfast", "-crf", str(opts.crf), "-pix_fmt", "yuv420p",
                   "-r", str(opts.fps), "-movflags", "+faststart", str(output)]
            print("ffmpeg:", " ".join(shlex.quote(c) for c in cmd))
            ffmpeg = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        if lost[i]:
            _draw_box_text(draw, (margin, margin), "V2V 손실", font_main, (210, 35, 35, 225))
        else:
            _draw_box_text(draw, (margin, margin), "V2V 수신", font_main, (25, 150, 70, 225))
        if row["f1_comm_in_tunnel"] == "1":
            _draw_box_text(draw, (W - margin, margin), f"TUNNEL · p = {opts.p_label}", font_main,
                           (0, 0, 0, 160), anchor_x="right")
        if brake_t is not None and brake_t - 1e-6 <= t < brake_t + opts.banner_sec:
            # Centred horizontally but just under the top badges: lower down it covers the
            # leader and F1, which is exactly what the braking moment should show.
            _draw_box_text(draw, (W // 2, int(H * opts.banner_top_frac)), "LEADER HARD BRAKE", font_big,
                           (0, 0, 0, 170), anchor_x="center", pad=(40, 22))
        gap_text = f"추종차1 차간거리 {float(row['gap_f1_m']):.1f} m"
        l, tp, r, b = draw.textbbox((0, 0), gap_text, font=font_main)
        _draw_box_text(draw, (margin, H - margin - (b - tp) - 28), gap_text, font_main, (0, 0, 0, 160))
        out = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")
        ffmpeg.stdin.write(np.asarray(out).tobytes())
        if preview_dir and any(abs(t - pt) < DT / 2 for pt in preview_times):
            out.save(preview_dir / f"hud_t{t:05.2f}.png")
    ffmpeg.stdin.close()
    if ffmpeg.wait() != 0:
        raise SystemExit("ffmpeg failed")
    print(f"wrote {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("record")
    r.add_argument("--out-dir", required=True)
    r.add_argument("--checkpoint", default="outputs/tunnel/train_lag_s43/ppo_lag.pt")
    r.add_argument("--episode", type=int, default=1)
    r.add_argument("--eval-seed", type=int, default=1000)
    r.add_argument("--p-out", type=float, default=0.9)
    r.add_argument("--p-tunnel", type=float, default=0.3)
    r.add_argument("--noise-std", type=float, default=0.1)
    r.add_argument("--brake-prob", type=float, default=1.0)
    r.add_argument("--port", type=int, default=2000)
    r.add_argument("--width", type=int, default=1920)
    r.add_argument("--height", type=int, default=1080)
    r.add_argument("--fov", type=float, default=70.0)
    r.add_argument("--cam-back", type=float, default=10.0, help="camera distance behind F2 [m]")
    r.add_argument("--cam-height", type=float, default=5.0, help="camera height above F2 [m]")
    r.add_argument("--look-lead-weight", type=float, default=0.3, help="look point F1(0)..leader(1)")
    r.add_argument("--pitch-bias", type=float, default=0.0, help="extra pitch [deg]")
    r.add_argument("--pos-smooth", type=float, default=0.2)
    r.add_argument("--look-smooth", type=float, default=0.15)
    r.add_argument("--test-times", default="", help="comma list of episode times [s]: save PNGs and stop")
    c = sub.add_parser("compose")
    c.add_argument("--run-dir", required=True)
    c.add_argument("--output", required=True)
    c.add_argument("--fps", type=int, default=20)
    c.add_argument("--pre-sec", type=float, default=7.0)
    c.add_argument("--post-sec", type=float, default=2.0)
    c.add_argument("--fade-sec", type=float, default=0.5)
    c.add_argument("--lost-after-s", type=float, default=0.25)
    c.add_argument("--min-hold-s", type=float, default=0.5)
    c.add_argument("--banner-sec", type=float, default=2.0)
    c.add_argument("--text-frac", type=float, default=0.058, help="cap height / frame height (approx)")
    c.add_argument("--banner-frac", type=float, default=0.10)
    c.add_argument("--banner-top-frac", type=float, default=0.17, help="banner box top / frame height")
    c.add_argument("--p-label", default="0.3")
    c.add_argument("--crf", type=int, default=20)
    c.add_argument("--preview-dir", default="")
    c.add_argument("--preview-times", default="")
    opts = parser.parse_args()
    if opts.cmd == "record":
        try:
            cmd_record(opts)
        except StopRecording as exc:
            print(exc)
    else:
        cmd_compose(opts)


if __name__ == "__main__":
    main()
