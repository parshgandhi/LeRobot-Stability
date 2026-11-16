"""Slide the weight to a commanded spot on the tray (default: the centre).

Cascade:
    vision (30 Hz)   weight position -> WeightEstimator -> position + velocity
      -> FrictionController   BREAK / MOVE / HOLD, outputs a downhill slope
      -> calibration A        slope -> (pitch, roll) tilt about TRUE level
    IMU (50 Hz)      dashboard.py's PD (slew-limited target) holds that tilt

Needs friction_calib.json from calibrate_friction.py. See friction_control.py
for why a sliding weight needs this rather than a plain PD on position.

Run with dashboard.py STOPPED. Startup runs with an EMPTY tray; the script tells
you when to place the weight and when to remove it.
"""

import argparse
import csv
import math
import os
import time

import numpy as np

import dashboard as db
import friction_control as fc
import tray_vision as tv
from iNemo_wrapper import iNemoEngine
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig


def wait_for_weight(tracker, vision, level, timeout_s=180.0):
    """Hold the tray at true level until the weight is seen anywhere on it.

    Single threaded, since only one thread may drive the bus.
    """
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


def hold(tracker, level, seconds):
    for _ in range(int(seconds / db.DT)):
        tracker.step(*level)
        time.sleep(db.DT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", nargs=2, type=float, default=[0.0, 0.0],
                    metavar=("X_MM", "Y_MM"), help="target in tray mm (default centre)")
    ap.add_argument("--duration", type=float, default=0.0,
                    help="run time in seconds (default 0 = run forever until Ctrl+C)")
    ap.add_argument("--log", default=None,
                    help="CSV trace path (default: runs/run_<timestamp>.csv)")
    args = ap.parse_args()

    if args.log is None:
        os.makedirs("runs", exist_ok=True)
        args.log = time.strftime("runs/run_%Y%m%d_%H%M%S.csv")

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

    log_rows = []

    try:
        print("\n[Init] Running startup with an EMPTY tray ...")
        roll_offset, pitch_offset = db.startup(robot)
        tracker = fc.make_tilt_tracker(robot, J_pinv, roll_offset, pitch_offset)
        level = calib.level_rel(pitch_offset, roll_offset)
        print(f"[Init] true level is ({math.degrees(level[0]):+.2f}, "
              f"{math.degrees(level[1]):+.2f}) deg from the startup pose — moving there")
        hold(tracker, level, 1.5)

        print(f"\n>>> Place the weight on the tray, away from the target "
              f"({args.target[0]:+.0f}, {args.target[1]:+.0f}) mm.")
        print(">>> The arm is holding level — safe to reach in.")
        start = wait_for_weight(tracker, vision, level)
        if start is None:
            raise RuntimeError("timed out waiting for the weight")
        print(f">>> Weight at ({start[0]:+.0f}, {start[1]:+.0f}) mm. "
              f"Centering starts in 2s — hands clear.")
        hold(tracker, level, 2.0)

        ctrl = fc.FrictionController(calib, args.target)
        est = fc.WeightEstimator()
        t0 = time.monotonic()
        next_print = 0.0

        dur_str = f"for {args.duration:.0f}s" if args.duration > 0 else "forever"
        print(f"\n[Run] centering {dur_str} — Ctrl+C to stop")
        print(f"[Run] theta_s {math.degrees(calib.theta_s):.2f}deg, "
              f"theta_k {math.degrees(calib.theta_k):.2f}deg; logging to {args.log}\n")
        while args.duration <= 0 or (time.monotonic() - t0 < args.duration):
            now = time.monotonic()
            e = est.update(*vision.read_stamped(), now)
            tilt, note = ctrl.update(e, now, db.DT)
            cmd = level + tilt
            pitch_err, roll_err = tracker.step(float(cmd[0]), float(cmd[1]))

            pos = e.pos if e is not None else None
            vel = e.vel if e is not None else None
            log_rows.append([round(now - t0, 3),
                             *(np.round(pos, 2) if pos is not None else ("", "")),
                             *(np.round(vel, 1) if vel is not None else ("", "")),
                             ctrl.phase,
                             *np.round(np.degrees(ctrl.slope), 3),
                             *np.round(np.degrees(tilt), 3),
                             *np.round(np.degrees(tracker.meas - level), 3),
                             round(pitch_err, 3), round(roll_err, 3)])

            if now >= next_print:
                where = f"({pos[0]:+6.1f},{pos[1]:+6.1f})" if pos is not None else "   lost   "
                print(f"  t={now - t0:5.1f}s  pos {where} mm  "
                      f"tilt ({math.degrees(tilt[0]):+5.2f},{math.degrees(tilt[1]):+5.2f})deg  {note}")
                next_print = now + 0.5

            time.sleep(max(0.0, db.DT - (time.monotonic() - now)))

        print("\n[Run] done — levelling")
        hold(tracker, level, 2.0)

    except KeyboardInterrupt:
        print("\n[Abort] Ctrl+C")
    finally:
        vision.stop()
        if args.log and log_rows:
            with open(args.log, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["t_s", "x_mm", "y_mm", "vx_mm_s", "vy_mm_s", "phase",
                            "slope_x_deg", "slope_y_deg",
                            "tilt_cmd_pitch_deg", "tilt_cmd_roll_deg",
                            "tilt_meas_pitch_deg", "tilt_meas_roll_deg",
                            "pitch_err_deg", "roll_err_deg"])
                w.writerows(log_rows)
            print(f"[Log] wrote {args.log} ({len(log_rows)} rows)")
        input("\n>>> TAKE THE WEIGHT OFF THE TRAY, then press Enter. "
              "Torque is about to be disabled and the arm will sag. ")
        db._park_and_disconnect(robot)


if __name__ == "__main__":
    main()
