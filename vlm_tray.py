"""Natural-language + visual tray manipulation powered by Gemini VLM.

Runs the 50 Hz friction-aware sliding controller continuously while accepting
natural-language instructions in the terminal (e.g.:
  - "move to the top-left corner"
  - "slide 30 mm to the right"
  - "trace a triangle around the tray"
  - "patrol back and forth between left and right"
  - "move to where my finger is pointing" / "slide next to the coin"
  - "center" (instant shortcut to (0, 0) mm)
  - "q" or Ctrl+C to stop
).

Each instruction grabs the live rectified top-down tray image from tray_vision.py,
overlays a millimeter coordinate grid so Gemini can ground both spatial commands
and visual targets (fingers, coins, coasters, obstacles) directly into tray
(x_mm, y_mm) coordinates, and updates the active waypoint queue without ever
pausing the 50 Hz servo stabilization loop.
"""

import argparse
import base64
import math
import os
import queue
import threading
import time
from typing import List

import cv2
import numpy as np
from google import genai
from pydantic import BaseModel, Field

import dashboard as db
import friction_control as fc
import tray_vision as tv
from iNemo_wrapper import iNemoEngine
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

SAFE_COORD_MM = 50.0        # keep targets clear of the corner ArUco stickers (~62 mm)
DEFAULT_MODEL = "gemini-4-hydrogen"
PLAN_PREVIEW_PATH = "/tmp/vlm_tray_plan.jpg"


class Waypoint(BaseModel):
    x_mm: float = Field(description="Target X coordinate in tray mm (-50 to +50; +X is right, -X is left).")
    y_mm: float = Field(description="Target Y coordinate in tray mm (-50 to +50; +Y is down/bottom, -Y is up/top).")


class TrayPlan(BaseModel):
    waypoints_mm: List[Waypoint] = Field(
        description="Ordered list of (x_mm, y_mm) waypoints in tray coordinates."
    )
    dwell_s: float = Field(
        default=1.0,
        description="Seconds to hold at each intermediate waypoint before advancing to the next."
    )
    loop: bool = Field(
        default=False,
        description="True if the user asked for a continuous/repeating motion (e.g. patrol, circle, loop)."
    )
    wait_before_start_s: float = Field(
        default=0.0,
        description="Seconds to wait at level before starting motion (set to 2.0 if a human hand/finger is visible inside the tray, else 0.0)."
    )
    explanation: str = Field(
        description="One concise sentence explaining what the plan does and why."
    )


SYSTEM_INSTRUCTION = """You are the spatial motion planner for a robotic tilting tray (150x150 mm).
You are given a live top-down rectified camera image of the tray with a millimeter coordinate grid overlaid, plus the user's natural-language command.

Coordinate System (Tray Frame, origin at centre (0, 0)):
- +X (mm) is RIGHT in the image; -X (mm) is LEFT in the image.
- +Y (mm) is DOWN (bottom) in the image; -Y (mm) is UP (top) in the image.
- Corner ArUco markers sit near (+/-62, +/-62) mm:
  - Top-Left (TL, marker 3): around (-42, -42) mm safe corner
  - Top-Right (TR, marker 0): around (+42, -42) mm safe corner
  - Bottom-Right (BR, marker 1): around (+42, +42) mm safe corner
  - Bottom-Left (BL, marker 2): around (-42, +42) mm safe corner
- Keep all waypoints strictly within [-50, +50] mm on both X and Y so the sliding weight never snags on the corner stickers or falls off the rim.

Guidelines:
- For single destinations ("go to top right", "move 25 mm left", "go to the coin/finger"), output 1 waypoint.
- For shapes or multi-step paths ("trace a square", "do a triangle", "visit all 4 corners", "draw a circle"), output 3 to 8 waypoints tracing that path.
- Set `loop=True` only when the user asks for continuous/repeating motion ("keep circling", "patrol left and right", "loop around a square").
- If you see a human hand or finger pointing inside the tray in the image, set the waypoint to the pointed spot and set `wait_before_start_s=2.0` so their hand can clear first.
"""


def render_vlm_frame(rect, cur_pos, cur_target, planned_wps=None):
    """Overlay a clear mm coordinate grid and markers onto the rectified tray image."""
    if rect is None:
        vis = np.zeros((tv.RECT_PX, tv.RECT_PX, 3), dtype=np.uint8)
    else:
        vis = rect.copy()

    c = tv.RECT_PX // 2
    # Grid lines every 20 mm
    for mm in (-60, -40, -20, 0, 20, 40, 60):
        px = int(c + mm * tv.PX_PER_MM)
        col = (0, 160, 0) if mm == 0 else (70, 70, 70)
        thick = 2 if mm == 0 else 1
        cv2.line(vis, (px, 0), (px, tv.RECT_PX), col, thick)
        cv2.line(vis, (0, px), (tv.RECT_PX, px), col, thick)
        if mm != 0 and abs(mm) <= 40:
            cv2.putText(vis, f"{mm:+d}", (px - 14, c - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 220, 180), 1)
            cv2.putText(vis, f"{mm:+d}", (c + 4, px + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 220, 180), 1)

    # Safe workspace box [-50, +50] mm
    s_px = int(SAFE_COORD_MM * tv.PX_PER_MM)
    cv2.rectangle(vis, (c - s_px, c - s_px), (c + s_px, c + s_px), (255, 180, 0), 1)

    # Axis & corner labels
    cv2.putText(vis, "TL(-42,-42)", (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1)
    cv2.putText(vis, "TR(+42,-42)", (tv.RECT_PX - 95, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1)
    cv2.putText(vis, "BL(-42,+42)", (8, tv.RECT_PX - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1)
    cv2.putText(vis, "BR(+42,+42)", (tv.RECT_PX - 95, tv.RECT_PX - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1)

    # Current target (magenta crosshair)
    tx, ty = int(c + cur_target[0] * tv.PX_PER_MM), int(c + cur_target[1] * tv.PX_PER_MM)
    cv2.drawMarker(vis, (tx, ty), (255, 0, 255), cv2.MARKER_CROSS, 14, 2)

    # Current weight position (green circle)
    if cur_pos is not None:
        wx, wy = int(c + cur_pos[0] * tv.PX_PER_MM), int(c + cur_pos[1] * tv.PX_PER_MM)
        cv2.circle(vis, (wx, wy), 11, (0, 255, 0), 2)
        cv2.putText(vis, f"weight({cur_pos[0]:+.0f},{cur_pos[1]:+.0f})",
                    (wx + 12, wy - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 0), 1)

    # Optional planned waypoints overlay
    if planned_wps:
        pts = [(int(c + x * tv.PX_PER_MM), int(c + y * tv.PX_PER_MM)) for x, y in planned_wps]
        if cur_pos is not None:
            cv2.arrowedLine(vis, (int(c + cur_pos[0] * tv.PX_PER_MM), int(c + cur_pos[1] * tv.PX_PER_MM)),
                            pts[0], (0, 140, 255), 2, tipLength=0.1)
        for idx, pt in enumerate(pts):
            cv2.circle(vis, pt, 7, (0, 140, 255), -1)
            cv2.putText(vis, f"#{idx + 1}", (pt[0] + 8, pt[1] + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 255), 1)
            if idx + 1 < len(pts):
                cv2.arrowedLine(vis, pt, pts[idx + 1], (0, 140, 255), 2, tipLength=0.1)

    return vis


def query_gemini_plan(client, model, instruction, rect, cur_pos, cur_target):
    """Send the annotated tray view + user command to Gemini and return a TrayPlan."""
    vis = render_vlm_frame(rect, cur_pos, cur_target)
    ok, jpg = cv2.imencode(".jpg", vis, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
    if not ok:
        raise RuntimeError("failed to JPEG-encode tray image")
    b64_img = base64.b64encode(jpg.tobytes()).decode("ascii")

    pos_str = f"({cur_pos[0]:+.1f}, {cur_pos[1]:+.1f}) mm" if cur_pos is not None else "unknown (occluded)"
    tgt_str = f"({cur_target[0]:+.1f}, {cur_target[1]:+.1f}) mm"
    prompt = (
        f"Current weight position: {pos_str}\n"
        f"Current active target: {tgt_str}\n"
        f"User instruction: {instruction}\n"
        "Return the waypoint plan in tray coordinates (mm)."
    )

    interaction = client.interactions.create(
        model=model,
        system_instruction=SYSTEM_INSTRUCTION,
        input=[
            {"type": "text", "text": prompt},
            {"type": "image", "data": b64_img, "mime_type": "image/jpeg"},
        ],
        response_format={
            "type": "text",
            "mime_type": "application/json",
            "schema": TrayPlan.model_json_schema(),
        },
    )
    plan = TrayPlan.model_validate_json(interaction.output_text)

    # Clamp all waypoints into safe tray bounds
    clamped = []
    for wp in plan.waypoints_mm:
        cx = float(np.clip(wp.x_mm, -SAFE_COORD_MM, SAFE_COORD_MM))
        cy = float(np.clip(wp.y_mm, -SAFE_COORD_MM, SAFE_COORD_MM))
        clamped.append((cx, cy))
    if not clamped:
        clamped = [(float(cur_target[0]), float(cur_target[1]))]

    preview = render_vlm_frame(rect, cur_pos, cur_target, planned_wps=clamped)
    cv2.imwrite(PLAN_PREVIEW_PATH, preview)
    return plan, clamped


def hold(tracker, level, seconds):
    for _ in range(int(seconds / db.DT)):
        tracker.step(*level)
        time.sleep(db.DT)


def wait_for_weight(tracker, vision, level, timeout_s=180.0):
    deadline = time.monotonic() + timeout_s
    next_report = 0.0
    while time.monotonic() < deadline:
        tracker.step(*level)
        xy = vision.read()
        if xy is not None:
            return xy
        if time.monotonic() >= next_report:
            print(f"    waiting — {vision.status()}")
            next_report = time.monotonic() + 2.0
        time.sleep(db.DT)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help=f"Gemini model name (default: {DEFAULT_MODEL})")
    ap.add_argument("--target", nargs=2, type=float, default=[0.0, 0.0],
                    metavar=("X_MM", "Y_MM"), help="initial target in tray mm (default centre)")
    args = ap.parse_args()

    if not (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")):
        raise SystemExit("GEMINI_API_KEY is not set. Run: export GEMINI_API_KEY='your_key'")

    client = genai.Client()
    calib = fc.FrictionCalib.load()
    J_pinv = np.load(db.J_PINV_PATH)

    db._inemo = iNemoEngine(dylib_path=db.DYLIB_PATH, kappa_acc=0.3, verbose=True)

    print(f"[Init] Connecting to robot on {db.ROBOT_PORT} ...")
    robot = SO101Follower(SO101FollowerConfig(port=db.ROBOT_PORT, id=db.ROBOT_ID,
                                              use_degrees=True))
    robot.connect(calibrate=False)

    db.start_imu_thread()
    if not db.wait_for_imu(timeout_s=5.0):
        robot.disconnect()
        raise SystemExit("no IMU data in 5s")
    print("[Init] IMU live.")

    vision = tv.VisionThread(debug_dropouts=True)
    vision.start()
    time.sleep(1.0)
    if not vision.markers_ok():
        vision.stop()
        robot.disconnect()
        raise SystemExit(f"fewer than {tv.MIN_MARKERS} of the 4 tray markers visible — check the camera")

    cmd_queue = queue.Queue()
    plan_queue = queue.Queue()
    stop_event = threading.Event()

    def stdin_reader():
        while not stop_event.is_set():
            try:
                line = input()
            except EOFError:
                break
            cmd_queue.put(line.strip())

    try:
        print("\n[Init] Running startup with an EMPTY tray ...")
        roll_offset, pitch_offset = db.startup(robot)
        tracker = fc.make_tilt_tracker(robot, J_pinv, roll_offset, pitch_offset)
        level = calib.level_rel(pitch_offset, roll_offset)
        print(f"[Init] true level is ({math.degrees(level[0]):+.2f}, "
              f"{math.degrees(level[1]):+.2f}) deg from the startup pose — moving there")
        hold(tracker, level, 1.5)

        print("\n>>> Place the weight on the tray.")
        print(">>> The arm is holding level — safe to reach in.")
        start = wait_for_weight(tracker, vision, level)
        if start is None:
            raise RuntimeError("timed out waiting for the weight")
        print(f">>> Weight at ({start[0]:+.0f}, {start[1]:+.0f}) mm. Holding level.")
        hold(tracker, level, 1.0)

        waypoints = [(float(start[0]), float(start[1]))]
        wp_idx = 0
        dwell_s = 1.0
        loop_path = False
        hold_since = None
        pause_until = 0.0
        planning_busy = False
        has_command = False
        announced_done = True

        ctrl = fc.FrictionController(calib, waypoints[0])
        est = fc.WeightEstimator()
        t0 = time.monotonic()
        next_print = 0.0

        print("\n======================================================================")
        print(f"[VLM Tray Ready] Model: {args.model}")
        print("Type a natural-language command below and press Enter:")
        print("  Examples:")
        print("    move to the top left corner")
        print("    slide 30 mm to the right")
        print("    trace a triangle around the tray")
        print("    move to where my finger is pointing")
        print("    center   (instant shortcut to (0, 0))")
        print("    q        (quit)")
        print("======================================================================")
        print("\nCommand > ", end="", flush=True)

        threading.Thread(target=stdin_reader, daemon=True).start()

        def run_vlm_worker(user_text, rect_snap, pos_snap, tgt_snap):
            nonlocal planning_busy
            try:
                t_call = time.monotonic()
                plan, wps = query_gemini_plan(
                    client, args.model, user_text, rect_snap, pos_snap, tgt_snap
                )
                dt_ms = (time.monotonic() - t_call) * 1000.0
                plan_queue.put((plan, wps, dt_ms, None))
            except Exception as exc:
                plan_queue.put((None, None, 0.0, exc))
            finally:
                planning_busy = False

        while not stop_event.is_set():
            now = time.monotonic()
            e = est.update(*vision.read_stamped(), now)
            cur_pos = e.pos.copy() if e is not None else None

            # Check for typed user commands
            while not cmd_queue.empty():
                user_cmd = cmd_queue.get_nowait()
                if not user_cmd:
                    if announced_done and not planning_busy:
                        print("Command > ", end="", flush=True)
                    continue
                if user_cmd.lower() in ("q", "quit", "exit"):
                    stop_event.set()
                    break
                if user_cmd.lower() in ("center", "centre", "0", "home"):
                    waypoints = [(0.0, 0.0)]
                    wp_idx = 0
                    loop_path = False
                    hold_since = None
                    has_command = True
                    announced_done = False
                    ctrl.target = np.array([0.0, 0.0])
                    ctrl._enter("break", now, u=ctrl.target - cur_pos if cur_pos is not None else None)
                    print("\n[Command] Moving to centre (0, 0) mm ...")
                    continue
                if planning_busy:
                    print("[VLM] Still planning previous command — please wait a moment ...")
                    continue
                planning_busy = True
                rect_snap = vision.read_rect()
                print(f"[VLM] Grounding '{user_cmd}' with {args.model} ...")
                threading.Thread(
                    target=run_vlm_worker,
                    args=(user_cmd, rect_snap, cur_pos, ctrl.target.copy()),
                    daemon=True,
                ).start()

            if stop_event.is_set():
                break

            # Check for completed VLM plans
            while not plan_queue.empty():
                plan, wps, dt_ms, err = plan_queue.get_nowait()
                if err is not None:
                    print(f"\n[VLM Error] {err}")
                    print("\nCommand > ", end="", flush=True)
                    continue
                waypoints = wps
                wp_idx = 0
                dwell_s = max(0.4, float(plan.dwell_s))
                loop_path = bool(plan.loop)
                hold_since = None
                has_command = True
                announced_done = False
                pause_until = now + max(0.0, float(plan.wait_before_start_s))
                ctrl.target = np.array(waypoints[0], float)
                if cur_pos is not None and np.linalg.norm(ctrl.target - cur_pos) > fc.TOL_MM:
                    ctrl._enter("break", now, u=ctrl.target - cur_pos)
                wp_str = " -> ".join(f"({x:+.0f},{y:+.0f})" for x, y in waypoints)
                print(f"\n[VLM Plan ({dt_ms:.0f} ms)] {plan.explanation}")
                print(f"  Waypoints : {wp_str} mm  (dwell={dwell_s:.1f}s, loop={loop_path})")
                if plan.wait_before_start_s > 0:
                    print(f"  Waiting {plan.wait_before_start_s:.1f}s at level before moving ...")
                print(f"  Preview   : {PLAN_PREVIEW_PATH}")

            # Before first command: hold level. Once a command is given: keep
            # FrictionController active continuously (just like center_weight.py)
            # so any overshoot or bump is automatically corrected back to ctrl.target.
            if not has_command:
                tilt, note = np.zeros(2), "idle - holding level"
            elif now < pause_until:
                tilt, note = np.zeros(2), f"WAITING {pause_until - now:3.1f}s (hands clear)"
            else:
                tilt, note = ctrl.update(e, now, db.DT)
                settled_on_target = (
                    e is not None
                    and ctrl.phase == "hold"
                    and float(np.linalg.norm(ctrl.target - e.pos)) <= fc.REENGAGE_MM
                    and e.speed < fc.V_STOP_MM_S
                )
                if settled_on_target:
                    if hold_since is None:
                        hold_since = now
                    req_dwell = dwell_s if len(waypoints) > 1 else 0.45
                    if now - hold_since >= req_dwell:
                        if wp_idx + 1 < len(waypoints):
                            wp_idx += 1
                            hold_since = None
                            announced_done = False
                            ctrl.target = np.array(waypoints[wp_idx], float)
                            print(f"  -> advancing to waypoint #{wp_idx + 1}/{len(waypoints)}: "
                                  f"({ctrl.target[0]:+.0f}, {ctrl.target[1]:+.0f}) mm")
                        elif loop_path and len(waypoints) > 1:
                            wp_idx = 0
                            hold_since = None
                            announced_done = False
                            ctrl.target = np.array(waypoints[0], float)
                            print(f"  -> looping back to waypoint #1/{len(waypoints)}: "
                                  f"({ctrl.target[0]:+.0f}, {ctrl.target[1]:+.0f}) mm")
                        elif not announced_done:
                            announced_done = True
                            err_mm = float(np.linalg.norm(ctrl.target - e.pos))
                            print(f"\n[Done] Settled at ({e.pos[0]:+.1f}, {e.pos[1]:+.1f}) mm "
                                  f"(err {err_mm:.1f} mm) — actively holding target.")
                            print("\nCommand > ", end="", flush=True)
                else:
                    hold_since = None
                    if ctrl.phase in ("break", "move"):
                        announced_done = False

            cmd = level + tilt
            tracker.step(float(cmd[0]), float(cmd[1]))

            if has_command and not announced_done and now >= next_print and not planning_busy:
                where = f"({cur_pos[0]:+6.1f},{cur_pos[1]:+6.1f})" if cur_pos is not None else "   lost   "
                tgt = f"({ctrl.target[0]:+4.0f},{ctrl.target[1]:+4.0f})"
                print(f"  t={now - t0:5.1f}s  pos {where} -> tgt {tgt} mm  "
                      f"tilt ({math.degrees(tilt[0]):+5.2f},{math.degrees(tilt[1]):+5.2f})deg  {note}")
                next_print = now + 1.0

            time.sleep(max(0.0, db.DT - (time.monotonic() - now)))

        print("\n[Run] done — levelling")
        hold(tracker, level, 1.5)

    except KeyboardInterrupt:
        print("\n[Abort] Ctrl+C")
    finally:
        stop_event.set()
        vision.stop()
        input("\n>>> TAKE THE WEIGHT OFF THE TRAY, then press Enter. "
              "Torque is about to be disabled and the arm will sag. ")
        db._park_and_disconnect(robot)


if __name__ == "__main__":
    main()
