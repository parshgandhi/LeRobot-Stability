"""Offline test bench: simulated tray, sliding weight, tilt loop and camera.

Runs the real calibrate_friction routines and FrictionController against a
Coulomb-friction model, so changes can be checked without the robot:

    python sim_centering.py                 # calibrate, then centre from 24 starts
    python sim_centering.py --seed 3 --theta-s 3 --theta-k 2 --servo-tau 0.1

The model: dashboard.py's PD (deadband, clamps) drives the joints; the joints
reach the tray through an unknown gain, gear backlash and a servo lag, and the
servos jitter; the fused IMU reading lags and carries coloured noise; the IMU is
mounted slightly off true level; the camera runs at 30 Hz with latency, jitter
and a little tilt-dependent parallax; the weight sticks until the slope exceeds
theta_s and then slides against kinetic friction theta_k.
"""

import argparse
import math
from collections import deque

import numpy as np

import calibrate_friction as cf
import friction_control as fc

DT = cf.DT
R = math.radians
DEADBAND_DEG = 0.5          # dashboard.py feedback settings, mirrored by the tracker
CMD_LIMIT = 0.5
MAX_JOINT_VEL = 0.3


def rot(deg):
    c, s = math.cos(R(deg)), math.sin(R(deg))
    return np.array([[c, -s], [s, c]])


class World:
    PHYS_DT = 0.001
    TRAY_HALF_MM = 75.0

    def __init__(self, rng, theta_s_deg=2.0, theta_k_deg=1.4, startup_offset_deg=(1.73, 1.9),
                 mount_err_deg=(0.4, -0.3), joint_gain=(0.75, 1.25), axes_rot_deg=200.0,
                 servo_tau=0.06, backlash_deg=0.6, vib_deg=0.15, vib_hz=9.0,
                 cam_hz=30.0, cam_latency=0.06, cam_noise_mm=0.3,
                 parallax_mm_per_rad=15.0, imu_noise_deg=0.5, imu_noise_tau=0.05, imu_tau=0.15):
        self.rng = rng
        self.t = 0.0
        self.theta_s, self.theta_k = R(theta_s_deg), R(theta_k_deg)
        self.offset = np.radians(startup_offset_deg)        # [pitch, roll] IMU reading at startup
        # tracker-frame tilt at which the tray is truly level
        self.level_true = fc.gravity_zero_rel(*self.offset) + np.radians(mount_err_deg)
        self.A_true = rot(axes_rot_deg) @ np.diag([1.0, -1.0])
        # tilt the joints produce per unit of commanded tilt - unknown to the PD
        self.joint_gain = np.array(joint_gain)
        self.servo_tau = servo_tau
        self.joint = np.zeros(2)        # tilt the joints are commanding
        self.mech = np.zeros(2)         # after gear backlash
        self.backlash = R(backlash_deg)
        self.vib, self.vib_w = R(vib_deg), 2 * math.pi * vib_hz     # servo jitter, shakes the tray
        self.tilt = np.zeros(2)         # actual tray tilt (tracker frame)
        self.imu_noise, self.imu_noise_tau = R(imu_noise_deg), imu_noise_tau
        self.imu_err = np.zeros(2)
        # the fused IMU angle lags the real tilt (filter + accelerometer smoothing)
        self.imu_tau = imu_tau
        self.imu = np.zeros(2)

        self.weight_on = False
        self.pos = np.zeros(2)
        self.vel = np.zeros(2)
        self.fell = False

        self.cam_period, self.cam_latency = 1.0 / cam_hz, cam_latency
        self.cam_noise, self.parallax = cam_noise_mm, parallax_mm_per_rad
        self.next_capture = 0.0
        self.pending = deque()
        self.vis = None
        self.vis_stamp = 0.0

    def slope(self):
        return self.A_true @ (self.tilt - self.level_true)

    def place(self, xy):
        self.weight_on, self.fell = True, False
        self.pos, self.vel = np.array(xy, float), np.zeros(2)
        self.pending.clear()
        self.vis = None

    def advance(self, seconds):
        n = max(1, int(round(seconds / self.PHYS_DT)))
        for _ in range(n):
            self._phys(self.PHYS_DT)

    def _phys(self, dt):
        self.t += dt
        gap = self.joint * self.joint_gain - self.mech
        half = 0.5 * self.backlash
        self.mech += np.where(gap > half, gap - half, np.where(gap < -half, gap + half, 0.0))
        shake = self.vib * np.array([math.sin(self.vib_w * self.t), math.cos(1.3 * self.vib_w * self.t)])
        self.tilt += (self.mech + shake - self.tilt) * dt / self.servo_tau
        self.imu += (self.tilt - self.imu) * dt / self.imu_tau if self.imu_tau > 0 else self.tilt - self.imu
        # coloured IMU noise (vibration pickup, fusion error)
        a = dt / self.imu_noise_tau
        self.imu_err += -a * self.imu_err + self.imu_noise * math.sqrt(2 * a) * self.rng.normal(size=2)

        if self.weight_on and not self.fell:
            s = self.slope()
            G = fc.G_MM_S2
            speed = np.linalg.norm(self.vel)
            if speed > 0:
                a = G * s - G * self.theta_k * self.vel / speed
                v_new = self.vel + a * dt
                if v_new @ self.vel <= 0 and np.linalg.norm(s) <= self.theta_s:
                    v_new = np.zeros(2)
                self.vel = v_new
            elif np.linalg.norm(s) > self.theta_s:
                self.vel = (G * s - G * self.theta_k * s / np.linalg.norm(s)) * dt
            self.pos = self.pos + self.vel * dt
            if np.max(np.abs(self.pos)) > self.TRAY_HALF_MM + 8:
                self.fell = True

        if self.t >= self.next_capture:
            self.next_capture += self.cam_period
            if self.weight_on and not self.fell:
                obs = (self.pos + self.rng.normal(0, self.cam_noise, 2)
                       + self.parallax * (self.A_true @ self.tilt))
                self.pending.append((self.t + self.cam_latency, obs))
        while self.pending and self.pending[0][0] <= self.t:
            _, self.vis = self.pending.popleft()
            self.vis_stamp = self.t


class SimClock:
    def __init__(self, world):
        self.w = world

    def now(self):
        return self.w.t

    def sleep(self, seconds):
        self.w.advance(seconds)


class SimVision:
    def __init__(self, world):
        self.w = world

    def read_stamped(self):
        if self.w.vis is None or self.w.t - self.w.vis_stamp > 0.5:
            return None, 0.0
        return self.w.vis.copy(), self.w.vis_stamp

    def read(self):
        return self.read_stamped()[0]


class SimTracker:
    """Mirrors make_tilt_tracker: dashboard.py's PD (deadband, clamps) on a slewed target."""

    KP, KD = 3.0, 0.5
    IMU_FILTER_TAU_S = 0.04

    def __init__(self, world):
        self.w = world
        self.target = np.zeros(2)
        self.meas = np.zeros(2)
        self.meas_filt = np.zeros(2)
        self._meas_pd = np.zeros(2)
        self._has_meas = False
        self.prev = None

    def step(self, pitch_target=0.0, roll_target=0.0):
        w = self.w
        goal = np.array([pitch_target, roll_target], float)
        max_step = R(fc.TILT_SLEW_DEG_S) * DT
        delta = goal - self.target
        n = np.linalg.norm(delta)
        if n > max_step:
            delta *= max_step / n
        self.target = self.target + delta

        self.meas = w.imu + w.imu_err
        if not self._has_meas:
            self.meas_filt = self.meas.copy()
            self._meas_pd = self.meas.copy()
            self._has_meas = True
        else:
            alpha_m = DT / (fc.MEAS_FILTER_TAU_S + DT)
            self.meas_filt += alpha_m * (self.meas - self.meas_filt)
            alpha_pd = DT / (self.IMU_FILTER_TAU_S + DT)
            self._meas_pd += alpha_pd * (self.meas - self._meas_pd)

        rate = np.zeros(2) if self.prev is None else (self._meas_pd - self.prev) / DT
        self.prev = self._meas_pd.copy()

        err = self._meas_pd - self.target
        err[np.abs(err) < R(fc.TRACKER_DEADBAND_DEG)] = 0.0
        cmd = np.clip(-(self.KP * err + self.KD * rate), -CMD_LIMIT, CMD_LIMIT)
        w.joint = w.joint + np.clip(cmd, -MAX_JOINT_VEL, MAX_JOINT_VEL) * DT
        return tuple(np.degrees(self.meas - self.target))


def run_centering(world, tracker, calib, start_xy, target=(0.0, 0.0), seconds=20.0):
    """Returns (arrival_s or None, final_err_mm, fell, n_breaks)."""
    clock = SimClock(world)
    vision = SimVision(world)
    level = calib.level_rel(*world.offset)
    cf.hold(tracker, clock, 0.6, level)
    world.place(start_xy)
    cf.hold(tracker, clock, 0.4, level)

    ctrl = fc.FrictionController(calib, target)
    est = fc.WeightEstimator()
    t0, arrival, errs = clock.now(), None, []
    while clock.now() - t0 < seconds and not world.fell:
        now = clock.now()
        e = est.update(*vision.read_stamped(), now)
        tilt, _ = ctrl.update(e, now, DT)
        cmd = level + tilt
        tracker.step(float(cmd[0]), float(cmd[1]))
        err = float(np.linalg.norm(world.pos - np.array(target)))
        if (arrival is None and err <= fc.TOL_MM + 1.0
                and np.linalg.norm(world.vel) < 1e-6 and ctrl.phase == "hold"):
            arrival = now - t0
        errs.append(err)
        clock.sleep(DT)
    final = max(errs[-int(2.0 / DT):]) if errs else float("nan")
    return arrival, final, world.fell, ctrl.n_breaks


def starts(n=24, rng=None):
    pts = []
    for i in range(n):
        r = 20 + 50 * ((i * 7) % n) / (n - 1)
        a = 2 * math.pi * i / n
        pts.append((r * math.cos(a), r * math.sin(a)))
    return pts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--theta-s", type=float, default=2.0)
    ap.add_argument("--theta-k", type=float, default=1.4)
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--servo-tau", type=float, default=0.06, help="tray tilt lag behind the joints, s")
    ap.add_argument("--cam-latency", type=float, default=0.06, help="camera capture-to-stamp delay, s")
    ap.add_argument("--imu-tau", type=float, default=0.15, help="fused IMU angle lag, s")
    ap.add_argument("--imu-noise", type=float, default=0.5, help="IMU tilt noise, deg rms")
    ap.add_argument("--backlash", type=float, default=0.6, help="gear backlash, deg")
    ap.add_argument("--vib", type=float, default=0.15, help="servo vibration amplitude, deg")
    ap.add_argument("--startup-offset", nargs=2, type=float, default=[1.73, 1.9],
                    metavar=("PITCH", "ROLL"), help="IMU reading at the startup pose, deg")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    w = World(rng, theta_s_deg=args.theta_s, theta_k_deg=args.theta_k,
              servo_tau=args.servo_tau, cam_latency=args.cam_latency,
              startup_offset_deg=args.startup_offset, imu_tau=args.imu_tau,
              imu_noise_deg=args.imu_noise, backlash_deg=args.backlash, vib_deg=args.vib)
    clock, vision = SimClock(w), SimVision(w)
    tracker = SimTracker(w)

    print("[sim] Part 1: inner loop (empty tray)")
    cf.inner_loop_check(tracker, clock)

    print("\n[sim] Part 2: friction and true level")
    level0 = fc.gravity_zero_rel(*w.offset)

    def recenter(level):
        cf.hold(tracker, clock, 1.0, level)
        w.place((rng.uniform(-15, 15), rng.uniform(-15, 15)))
        cf.hold(tracker, clock, 1.0, level)

    recenter(level0)
    level, theta_s, theta_k, A, _ = cf.calibrate_friction(
        tracker, vision, clock, recenter, level0=level0)
    lvl_err = np.degrees(level - w.level_true)
    A_err = math.degrees(math.acos(np.clip(np.mean([A[:, i] @ w.A_true[:, i] for i in (0, 1)]), -1, 1)))
    print(f"\n  level error  pitch {lvl_err[0]:+.2f}deg  roll {lvl_err[1]:+.2f}deg")
    print(f"  theta_s {math.degrees(theta_s):.2f}deg (true {args.theta_s:.2f}),  "
          f"theta_k {math.degrees(theta_k):.2f}deg (true {args.theta_k:.2f}),  A off by ~{A_err:.1f}deg")

    calib = fc.FrictionCalib(w.offset + level, theta_s, theta_k, A)

    def suite(label, c):
        res = [run_centering(w, tracker, c, s) for s in starts(args.n)]
        ok = [r for r in res if r[0] is not None and r[1] <= fc.REENGAGE_MM and not r[2]]
        fell = sum(r[2] for r in res)
        t = [r[0] for r in ok]
        print(f"  {label:34s} centred {len(ok):2d}/{len(res)}  fell {fell}  "
              + (f"time median {np.median(t):4.1f}s max {max(t):4.1f}s  " if t else "")
              + f"final err max {max(r[1] for r in res):5.1f} mm  "
              f"breaks/run {np.mean([r[3] for r in res]):.1f}")
        return len(ok), fell

    print(f"\n[sim] centering from {args.n} starts, r = 20-70 mm")
    results = [suite("calibrated", calib)]
    mk = lambda **kw: fc.FrictionCalib(**{**dict(level_abs=calib.level_abs, theta_s=calib.theta_s,
                                                theta_k=calib.theta_k, A=calib.A,
                                                ), **kw})
    results.append(suite("theta_k 30% high", mk(theta_k=min(calib.theta_k * 1.3, calib.theta_s))))
    results.append(suite("theta_k 30% low", mk(theta_k=calib.theta_k * 0.7)))
    results.append(suite("A rotated 15deg", mk(A=rot(15) @ calib.A)))
    results.append(suite("level off by 0.4deg", mk(level_abs=calib.level_abs + R(0.4) * np.array([1, -1]) / math.sqrt(2))))
    total = sum(r[0] for r in results)
    print(f"\n[sim] {total}/{len(results) * args.n} runs centred, "
          f"{sum(r[1] for r in results)} fell off")


if __name__ == "__main__":
    main()
