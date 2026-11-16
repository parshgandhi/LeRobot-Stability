"""Calibrate the mapping from commanded tray tilt to weight motion.

Determines, on hardware, the 2x2 matrix M such that

    [dx, dy]_tray  =  M @ [pitch_cmd, roll_cmd]

i.e. which tilt axis slides the weight along which tray axis, and with what
sign. The centering controller needs M^-1 to turn a desired weight motion into
a tilt command; guessing the signs would drive the weight outward instead.

Each axis is ramped slowly from level until the weight is seen to move, which
also measures the breakaway (static friction) angle for free.

Run with dashboard.py STOPPED — it holds the robot and IMU serial ports.
Keep the weight on the tray, near the centre. Ctrl+C aborts and parks the arm.
"""

import json
import math
import threading
import time

import cv2
import numpy as np

import dashboard as db
import tray_vision as tv
from controller import PDController
from iNemo_wrapper import iNemoEngine
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

# ── Test parameters ──────────────────────────────────────────────────────────
MAX_TEST_TILT_DEG = 7.0     # never command more than this during calibration
RAMP_S = 6.0                # seconds to go from level to MAX_TEST_TILT
SETTLE_S = 1.5              # hold at level between tests
MOVE_THRESH_MM = 8.0        # displacement that counts as "the weight broke loose"
VISION_TIMEOUT_S = 0.5      # abort if the tray estimate goes stale
# The tray overhangs the floor, so a weight that keeps sliding falls off. The
# ramp is abandoned and the tray levelled the moment either limit is hit; 8 mm
# of travel is already enough to read off the slide direction.
SAFE_RADIUS_MM = 55.0       # abort to level if the weight gets this far out

# Two different gates, deliberately. The weight creeps between being placed and
# the test starting (breakaway is only ~2 deg), so the start check has to be
# looser than the placement check or it rejects a perfectly good placement.
PLACE_RADIUS_MM = 30.0      # accept the weight as "placed" inside this
# START + MOVE_THRESH must stay below SAFE_RADIUS so a test can still complete.
START_RADIUS_MM = 42.0      # refuse to start a ramp outside this
SETTLE_BEFORE_TEST_S = 1.5
CALIB_PATH = "tilt_calib.json"


class PlacementError(RuntimeError):
    """The weight was not where the test needs it to be; retrying is fine."""


class TiltTracker:
    """Inner loop: hold the tray at a commanded tilt using the existing PD + Jacobian."""

    IMU_FILTER_TAU_S = 0.04     # pre-filter IMU before both P and D (preserves Kd/Kp phase lead)

    def __init__(self, robot, J_pinv, pd, roll_offset, pitch_offset, deadband_rad=None):
        self.robot = robot
        self.J_pinv = J_pinv
        self.pd = pd
        self.roll_offset = roll_offset
        self.pitch_offset = pitch_offset
        self.deadband_rad = db.DEADBAND_RAD if deadband_rad is None else float(deadband_rad)

        self.q = np.deg2rad([db.Q_NOMINAL_DEG[n] for n in db.JOINT_NAMES])
        self.lim_min = np.deg2rad([db.JOINT_LIMITS_DEG[n][0] for n in db.JOINT_NAMES])
        self.lim_max = np.deg2rad([db.JOINT_LIMITS_DEG[n][1] for n in db.JOINT_NAMES])

        self._pitch_f = None
        self._roll_f = None

    def step(self, pitch_target=0.0, roll_target=0.0):
        """One 50 Hz iteration tracking the given tilt setpoint (radians)."""
        if not db.check_imu_alive():
            raise RuntimeError("IMU went stale")

        roll, pitch = db.get_orientation()
        pitch_rel = db._angle_wrap(pitch - self.pitch_offset)
        roll_rel = db._angle_wrap(roll - self.roll_offset)

        if self._pitch_f is None:
            self._pitch_f, self._roll_f = pitch_rel, roll_rel
            pitch_rate, roll_rate = 0.0, 0.0
        else:
            alpha = db.DT / (self.IMU_FILTER_TAU_S + db.DT)
            new_pf = self._pitch_f + alpha * db._angle_wrap(pitch_rel - self._pitch_f)
            new_rf = self._roll_f + alpha * db._angle_wrap(roll_rel - self._roll_f)
            pitch_rate = (new_pf - self._pitch_f) / db.DT
            roll_rate = (new_rf - self._roll_f) / db.DT
            self._pitch_f, self._roll_f = new_pf, new_rf

        # error is measured against the commanded tilt rather than against flat
        pitch_err = db._angle_wrap(self._pitch_f - pitch_target)
        roll_err = db._angle_wrap(self._roll_f - roll_target)

        roll_db_ = roll_err if abs(roll_err) >= self.deadband_rad else 0.0
        pitch_db_ = pitch_err if abs(pitch_err) >= self.deadband_rad else 0.0

        cp, cr, _, _ = self.pd.compute(pitch_db_, roll_db_, pitch_rate, roll_rate)

        twist = np.array([0.0, 0.0, 0.0, -cr, -cp, 0.0])
        dq = np.clip(self.J_pinv @ twist, -db.MAX_JOINT_VEL, db.MAX_JOINT_VEL)

        self.q = np.clip(self.q + dq * db.DT, self.lim_min, self.lim_max)

        action = {f"{n}.pos": math.degrees(self.q[i])
                  for i, n in enumerate(db.JOINT_NAMES)}
        action["gripper.pos"] = db.GRIPPER_POS
        self.robot.send_action(action)

        return math.degrees(db._angle_wrap(pitch_rel - pitch_target)), math.degrees(db._angle_wrap(roll_rel - roll_target))

    def hold(self, seconds, pitch_target=0.0, roll_target=0.0):
        for _ in range(int(seconds / db.DT)):
            self.step(pitch_target, roll_target)
            time.sleep(db.DT)


def ramp_test(tracker, vision, axis):
    """Ramp one tilt axis until the weight slides.

    Returns (breakaway_deg, dx_mm, dy_mm) for a positive tilt on `axis`.
    The tray is levelled the instant motion is detected, so the weight travels
    only far enough to show which way it went.
    """
    assert axis in ("pitch", "roll")

    start = vision.read()
    if start is None:
        raise PlacementError("no tray reading at test start — check markers")
    if math.hypot(*start) > START_RADIUS_MM:
        raise PlacementError(f"weight drifted to {math.hypot(*start):.0f} mm from centre")

    print(f"\n[{axis}] ramping to +{MAX_TEST_TILT_DEG:.0f}deg ...")

    n_steps = int(RAMP_S / db.DT)
    max_rad = math.radians(MAX_TEST_TILT_DEG)

    breakaway, dx, dy = None, 0.0, 0.0
    for step in range(n_steps + 1):
        tilt = max_rad * step / n_steps
        tracker.step(pitch_target=tilt if axis == "pitch" else 0.0,
                     roll_target=tilt if axis == "roll" else 0.0)

        now = vision.read()
        if now is not None:
            dx, dy = now[0] - start[0], now[1] - start[1]
            if math.hypot(dx, dy) >= MOVE_THRESH_MM:
                breakaway = math.degrees(tilt)
                print(f"[{axis}] broke loose at {breakaway:.1f}deg "
                      f"(moved {math.hypot(dx, dy):.1f} mm, "
                      f"direction dx={dx:+.1f} dy={dy:+.1f}) — levelling")
                break
            if math.hypot(*now) >= SAFE_RADIUS_MM:
                print(f"[{axis}] SAFETY — weight reached "
                      f"{math.hypot(*now):.0f} mm from centre, levelling")
                break
        time.sleep(db.DT)

    print(f"[{axis}] returning to level ...")
    tracker.hold(SETTLE_S)

    settled = vision.read()
    if settled is not None:
        print(f"[{axis}] settled at ({settled[0]:+.1f}, {settled[1]:+.1f}) mm")

    if breakaway is None:
        print(f"[{axis}] WARNING — weight never moved within "
              f"{MAX_TEST_TILT_DEG:.0f}deg (displacement {math.hypot(dx, dy):.1f} mm)")

    return breakaway, dx, dy


def hold_until_weight(tracker, vision, timeout_s=120.0):
    """Keep the tray level and powered while waiting for the weight to be placed.

    Single threaded on purpose: only one thread may drive the servo bus.
    """
    deadline = time.monotonic() + timeout_s
    next_report = 0.0
    while time.monotonic() < deadline:
        tracker.step()
        xy = vision.read()
        if xy is not None and math.hypot(*xy) <= PLACE_RADIUS_MM:
            return xy

        now = time.monotonic()
        if now >= next_report:
            if xy is not None:
                print(f"    waiting — weight at ({xy[0]:+.0f}, {xy[1]:+.0f}) mm, "
                      f"r={math.hypot(*xy):.0f} mm, need r<={PLACE_RADIUS_MM:.0f}")
            else:
                print(f"    waiting — {vision.status()}")
            next_report = now + 2.0

        time.sleep(db.DT)
    return None


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
    print("[Init] IMU live.")

    vision = tv.VisionThread()
    vision.start()
    time.sleep(1.0)
    if not vision.markers_ok():
        vision.stop()
        robot.disconnect()
        raise SystemExit(f"fewer than {tv.MIN_MARKERS} of the 4 tray markers visible — check the camera")

    try:
        # The weight cannot sit on the tray yet: with torque off the tray hangs
        # tilted and anything on it slides straight off. Startup runs empty.
        print("\n[Init] Running startup with an EMPTY tray ...")
        roll_offset, pitch_offset = db.startup(robot)
        tracker = TiltTracker(robot, J_pinv, pd, roll_offset, pitch_offset)

        results = {}
        for axis in ("pitch", "roll"):
            for attempt in range(5):
                print(f"\n>>> Place the weight near the tray centre "
                      f"(within {PLACE_RADIUS_MM:.0f} mm) for the {axis} test.")
                print(">>> The arm is holding position — it is safe to reach in.")

                if hold_until_weight(tracker, vision) is None:
                    raise RuntimeError("timed out waiting for the weight near the centre")

                print(f">>> Weight detected. Starting in "
                      f"{SETTLE_BEFORE_TEST_S:.1f}s — hands clear.")
                tracker.hold(SETTLE_BEFORE_TEST_S)

                try:
                    brk, dx, dy = ramp_test(tracker, vision, axis)
                    break
                except PlacementError as e:
                    print(f"[{axis}] {e} — re-centre it and we'll try again")
            else:
                raise RuntimeError(f"{axis} test failed after 5 placement attempts")

            results[axis] = {"breakaway_deg": brk, "dx_mm": dx, "dy_mm": dy}

        # columns: effect of +pitch and +roll on tray (x, y)
        M = np.array([[results["pitch"]["dx_mm"], results["roll"]["dx_mm"]],
                      [results["pitch"]["dy_mm"], results["roll"]["dy_mm"]]])

        print("\n" + "=" * 60)
        print("tilt -> weight motion matrix M (mm per test, columns = pitch, roll)")
        print(M)
        det = np.linalg.det(M)
        print(f"det(M) = {det:+.1f}")
        if abs(det) < 1e-3:
            print("WARNING — M is near singular; the two axes moved the weight "
                  "the same way. Re-run with a larger tilt or check the tests.")
        else:
            Minv = np.linalg.inv(M)
            print("\nM^-1 (desired weight motion -> tilt command):")
            print(Minv)

        with open(CALIB_PATH, "w") as f:
            json.dump({"M": M.tolist(), "tests": results,
                       "max_test_tilt_deg": MAX_TEST_TILT_DEG}, f, indent=2)
        print(f"\nsaved {CALIB_PATH}")

    except KeyboardInterrupt:
        print("\n[Abort] Ctrl+C")
    finally:
        vision.stop()
        input("\n>>> TAKE THE WEIGHT OFF THE TRAY, then press Enter. "
              "Torque is about to be disabled and the arm will sag. ")
        db._park_and_disconnect(robot)


if __name__ == "__main__":
    main()
