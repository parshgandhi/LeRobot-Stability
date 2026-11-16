"""Measure the tilt -> slide direction mapping properly.

The first calibration (calibrate_tilt.py) ramped each axis until the weight
twitched 8 mm and stopped. That turned out to be too little signal: in closed
loop the weight's motion was uncorrelated with the commanded direction
(mean cos = -0.14 over a 33 s run), so the matrix was mostly noise.

This version measures under the conditions the controller actually uses:
  * fixed tilt magnitude, held, with the same dither the controller applies
  * held until the weight travels far enough to give a clean direction
  * all four directions (+/-pitch, +/-roll), so opposite commands can be checked
    against each other - if +pitch and -pitch do not produce opposing motion,
    the weight is not behaving like a sliding body and no matrix will help

Opposite tests are run back to back so the weight returns toward where it
started, which avoids re-centring by hand between every test.

Run with dashboard.py STOPPED. Startup runs with an EMPTY tray.
"""

import json
import math
import time

import numpy as np

import dashboard as db
import tray_vision as tv
from calibrate_tilt import TiltTracker, hold_until_weight
from controller import PDController
from iNemo_wrapper import iNemoEngine
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

TILT_DEG = 5.0              # held magnitude, matches the controller's working range
DITHER_DEG = 0.7            # same agitation the controller uses to break stiction
DITHER_HZ = 2.0
TARGET_TRAVEL_MM = 20.0     # stop once the weight has moved this far
MAX_TEST_S = 12.0
RIM_STOP_MM = 58.0          # stop before the weight reaches the edge
SETTLE_S = 1.5

OUT_PATH = "tilt_calib.json"

DIRECTIONS = [
    ("+pitch", np.array([1.0, 0.0])),
    ("-pitch", np.array([-1.0, 0.0])),
    ("+roll", np.array([0.0, 1.0])),
    ("-roll", np.array([0.0, -1.0])),
]


def drive_test(tracker, vision, label, unit):
    """Hold a fixed dithered tilt along `unit` and record how the weight moves.

    Returns (displacement_xy_mm, seconds, stop_reason).
    """
    start = vision.read()
    if start is None:
        raise RuntimeError("no weight visible at test start")

    print(f"\n[{label}] holding {TILT_DEG:.1f}deg with dither ...")
    t0 = time.monotonic()
    last = start
    reason = "timeout"

    while True:
        now = time.monotonic()
        elapsed = now - t0
        if elapsed > MAX_TEST_S:
            break

        mag = TILT_DEG + DITHER_DEG * math.sin(2 * math.pi * DITHER_HZ * now)
        tilt = unit * math.radians(mag)
        tracker.step(pitch_target=float(tilt[0]), roll_target=float(tilt[1]))

        pos = vision.read()
        if pos is not None:
            last = pos
            if math.hypot(pos[0] - start[0], pos[1] - start[1]) >= TARGET_TRAVEL_MM:
                reason = "travelled"
                break
            if math.hypot(*pos) >= RIM_STOP_MM:
                reason = "near rim"
                break
        time.sleep(db.DT)

    elapsed = time.monotonic() - t0
    disp = np.array([last[0] - start[0], last[1] - start[1]])
    print(f"[{label}] {reason} after {elapsed:.1f}s, moved "
          f"({disp[0]:+.1f}, {disp[1]:+.1f}) mm = {np.linalg.norm(disp):.1f} mm")

    tracker.hold(SETTLE_S)
    return disp, elapsed, reason


def main():
    J_pinv = np.load(db.J_PINV_PATH)
    pd = PDController(Kp=db.Kp, Kd=db.Kd, limit=db.CMD_LIMIT)
    db._inemo = iNemoEngine(dylib_path=db.DYLIB_PATH, kappa_acc=0.3, verbose=True)

    print(f"[Init] Connecting to robot on {db.ROBOT_PORT} ...")
    robot = SO101Follower(SO101FollowerConfig(port=db.ROBOT_PORT, id=db.ROBOT_ID,
                                              use_degrees=True))
    robot.connect(calibrate=False)

    db.start_imu_thread()
    if not db.wait_for_imu(timeout_s=5.0):
        robot.disconnect()
        raise SystemExit("no IMU data in 5s")

    vision = tv.VisionThread()
    vision.start()
    time.sleep(1.0)
    if not vision.markers_ok():
        vision.stop()
        robot.disconnect()
        raise SystemExit(f"fewer than {tv.MIN_MARKERS} of the 4 tray markers visible — check the camera")

    results = {}
    try:
        print("\n[Init] Running startup with an EMPTY tray ...")
        roll_offset, pitch_offset = db.startup(robot)
        tracker = TiltTracker(robot, J_pinv, pd, roll_offset, pitch_offset)

        print("\n>>> Place the weight near the tray centre.")
        print(">>> The arm is holding level — safe to reach in.")
        if hold_until_weight(tracker, vision, timeout_s=180.0) is None:
            raise RuntimeError("timed out waiting for the weight")
        print(">>> Starting in 2s — hands clear.")
        tracker.hold(2.0)

        for label, unit in DIRECTIONS:
            pos = vision.read()
            if pos is None or math.hypot(*pos) > RIM_STOP_MM:
                where = "not visible" if pos is None else f"{math.hypot(*pos):.0f} mm out"
                print(f"\n>>> Weight is {where} — please place it near the centre "
                      f"for the {label} test.")
                if hold_until_weight(tracker, vision, timeout_s=180.0) is None:
                    raise RuntimeError("timed out waiting for re-centring")
                tracker.hold(2.0)

            disp, secs, reason = drive_test(tracker, vision, label, unit)
            results[label] = {"dx_mm": float(disp[0]), "dy_mm": float(disp[1]),
                              "seconds": secs, "stop": reason}

        print("\n" + "=" * 62)
        for label in ("pitch", "roll"):
            a = np.array([results[f"+{label}"]["dx_mm"], results[f"+{label}"]["dy_mm"]])
            b = np.array([results[f"-{label}"]["dx_mm"], results[f"-{label}"]["dy_mm"]])
            na, nb = np.linalg.norm(a), np.linalg.norm(b)
            if na < 3 or nb < 3:
                print(f"{label}: one direction barely moved — unusable")
                continue
            cos = float(a @ b / (na * nb))
            verdict = ("consistent" if cos < -0.7 else
                       "WEAK" if cos < -0.3 else "INCONSISTENT — not sliding cleanly")
            print(f"{label}: cos(+,-) = {cos:+.2f}  [{verdict}]")

        U, D = [], []
        for label, unit in DIRECTIONS:
            d = np.array([results[label]["dx_mm"], results[label]["dy_mm"]])
            if np.linalg.norm(d) < 3:
                continue
            U.append(unit)
            D.append(d / np.linalg.norm(d))
        if len(U) < 3:
            raise SystemExit("not enough usable tests to fit M")

        U = np.array(U).T
        D = np.array(D).T
        M = D @ np.linalg.pinv(U)
        print(f"\nM (cols = pitch, roll; rows = dx, dy):\n{np.round(M, 3)}")
        print(f"det(M) = {np.linalg.det(M):+.3f}")

        with open(OUT_PATH, "w") as f:
            json.dump({"M": M.tolist(), "tests": results,
                       "tilt_deg": TILT_DEG, "method": "fixed-tilt-with-dither"}, f, indent=2)
        print(f"saved {OUT_PATH}")

    except KeyboardInterrupt:
        print("\n[Abort] Ctrl+C")
    finally:
        vision.stop()
        input("\n>>> TAKE THE WEIGHT OFF THE TRAY, then press Enter. "
              "Torque is about to be disabled and the arm will sag. ")
        db._park_and_disconnect(robot)


if __name__ == "__main__":
    main()
