"""Friction-aware weight centering: shared pieces for calibration, control and simulation.

Why not a plain PD on position (ball-on-plate): the weight SLIDES. Below the
breakaway angle theta_s nothing moves at all, and once it breaks loose the tilt
sets its acceleration, not its speed. So the controller is a small state machine:

    BREAK  weight stuck  -> ramp the slope toward the target until it moves
    MOVE   weight moving -> velocity servo along a braking profile, with the
                            kinetic friction cancelled by feed-forward:
                                slope = a_cmd / g + theta_k * v_hat
    HOLD   on target     -> tray at TRUE level; kinetic friction stops the weight

Every angle here is relative to TRUE (gravity) level, measured by
calibrate_friction.py, not to the startup pose, which is ~2 deg off - about the
breakaway angle, enough to push the weight on its own.

Slopes are 2-vectors in tray mm axes (radians, pointing downhill). The tilt
command (pitch, roll) that produces a slope comes from A, the measured map from
tilt to downhill direction.
"""

import json
import math
import time
from collections import deque

import numpy as np

G_MM_S2 = 9810.0
FRICTION_CALIB_PATH = "friction_calib.json"



class RealClock:
    def now(self):
        return time.monotonic()

    def sleep(self, seconds):
        time.sleep(seconds)


def _wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def gravity_zero_rel(pitch_offset, roll_offset):
    """Tilt target (tracker frame) at which the IMU reads level.

    Level is 0 deg OR +-180 deg depending on how the IMU is mounted - this one
    reads roll ~ -178 deg on a flat tray, i.e. it is upside down. Taking 0 as
    level there commands a ~178 deg roll.
    """
    return np.array([_wrap(round(o / math.pi) * math.pi - o) for o in (pitch_offset, roll_offset)])


# ── Calibration data ─────────────────────────────────────────────────────────

class FrictionCalib:
    """Everything the controller needs, measured by calibrate_friction.py.

    level_abs  [pitch, roll] IMU reading (rad) at which the tray is truly level
    theta_s    breakaway (static friction) angle, rad
    theta_k    kinetic friction angle (slope at which a moving weight coasts at
               constant speed), rad
    A          2x2, columns = downhill direction in tray mm axes for +pitch, +roll
    """

    def __init__(self, level_abs, theta_s, theta_k, A, extra=None):
        self.level_abs = np.array(level_abs, float)
        self.theta_s = float(theta_s)
        self.theta_k = float(theta_k)
        self.A = np.array(A, float)
        self.extra = extra or {}
        if not 0.0 < self.theta_k <= self.theta_s:
            raise ValueError(f"implausible friction: theta_s {math.degrees(self.theta_s):.2f}deg, "
                             f"theta_k {math.degrees(self.theta_k):.2f}deg - re-calibrate")
        if abs(np.linalg.det(self.A)) < 0.2:
            raise ValueError(f"A is near singular (det {np.linalg.det(self.A):+.2f}) - "
                             "the two tilt axes moved the weight the same way; re-calibrate")
        self.A_inv = np.linalg.inv(self.A)
        self.theta_s_axis = np.array([self.theta_s, self.theta_s], float)
        self.theta_k_axis = np.array([self.theta_k, self.theta_k], float)
        tests = self.extra.get("tests") if isinstance(self.extra, dict) else None
        if tests:
            for idx, name in enumerate(("pitch", "roll")):
                pos = [t["tilt_at_break_deg"] for t in tests if t.get("axis") == name and t.get("sign") == 1]
                neg = [t["tilt_at_break_deg"] for t in tests if t.get("axis") == name and t.get("sign") == -1]
                if pos and neg:
                    ts_ax = math.radians(0.5 * (float(np.mean(pos)) - float(np.mean(neg))))
                    if ts_ax > math.radians(0.3):
                        self.theta_s_axis[idx] = ts_ax
                tks = [t["theta_k_deg"] for t in tests
                       if t.get("axis") == name and t.get("theta_k_deg") is not None]
                if tks:
                    tk_ax = math.radians(float(np.mean(tks)))
                    if 0.0 < tk_ax:
                        self.theta_k_axis[idx] = min(tk_ax, 0.95 * self.theta_s_axis[idx])
                else:
                    self.theta_k_axis[idx] = min(self.theta_k, 0.95 * self.theta_s_axis[idx])

    def theta_s_dir(self, u):
        """Directional static breakaway angle (rad) for tray unit vector `u`."""
        u = np.asarray(u, float)
        n = float(np.linalg.norm(u))
        if n < 1e-9:
            return self.theta_s
        t = self.A_inv @ (u / n)
        tn = float(np.linalg.norm(t))
        if tn < 1e-9:
            return self.theta_s
        w = (t / tn) ** 2
        return float(np.dot(w, self.theta_s_axis))

    def theta_k_dir(self, u):
        """Directional kinetic friction angle (rad) for tray unit vector `u`."""
        u = np.asarray(u, float)
        n = float(np.linalg.norm(u))
        if n < 1e-9:
            return self.theta_k
        t = self.A_inv @ (u / n)
        tn = float(np.linalg.norm(t))
        if tn < 1e-9:
            return self.theta_k
        w = (t / tn) ** 2
        return float(np.dot(w, self.theta_k_axis))

    def level_rel(self, pitch_offset, roll_offset):
        """True level as a tilt target for a tracker zeroed at the given startup offsets."""
        return np.array([_wrap(self.level_abs[0] - pitch_offset),
                         _wrap(self.level_abs[1] - roll_offset)])

    def tilt_for_slope(self, slope):
        """Tilt (pitch, roll) rad, relative to true level, giving `slope` (tray axes, rad)."""
        mag = float(np.linalg.norm(slope))
        if mag < 1e-9:
            return np.zeros(2)
        t = self.A_inv @ slope
        return t * (mag / np.linalg.norm(t))

    def to_json(self):
        d = {"level_pitch_abs_deg": math.degrees(self.level_abs[0]),
             "level_roll_abs_deg": math.degrees(self.level_abs[1]),
             "theta_s_deg": math.degrees(self.theta_s),
             "theta_k_deg": math.degrees(self.theta_k),
             "A": self.A.tolist()}
        d.update(self.extra)
        return d

    def save(self, path=FRICTION_CALIB_PATH):
        with open(path, "w") as f:
            json.dump(self.to_json(), f, indent=2)

    @classmethod
    def load(cls, path=FRICTION_CALIB_PATH):
        try:
            with open(path) as f:
                d = json.load(f)
        except FileNotFoundError:
            raise SystemExit(f"no {path} - run: python calibrate_friction.py")
        return cls([math.radians(d["level_pitch_abs_deg"]), math.radians(d["level_roll_abs_deg"])],
                   math.radians(d["theta_s_deg"]), math.radians(d["theta_k_deg"]),
                   d["A"], extra={"tests": d.get("tests")})


# ── Weight state estimate ────────────────────────────────────────────────────

class Estimate:
    __slots__ = ("pos", "vel", "speed", "raw_pos")

    def __init__(self, pos, vel, raw_pos=None):
        self.pos = pos
        self.vel = vel
        self.speed = float(np.linalg.norm(vel))
        self.raw_pos = pos.copy() if raw_pos is None else raw_pos


class WeightEstimator:
    """Position and velocity from the 30 Hz vision stream.

    A line fit over the last WINDOW_S of fresh frames: differentiating single
    frames would turn ~0.3 mm of marker jitter into ~10 mm/s of speed noise. The
    position is projected forward over the camera's capture-to-stamp lag only
    once the weight is genuinely moving.
    """

    WINDOW_S = 0.20
    MIN_SAMPLES = 3
    STALE_S = 0.30
    CAPTURE_LAG_S = 0.05
    MAX_LEAD_S = 0.08

    def __init__(self):
        self.buf = deque()
        self.last_stamp = None

    def reset(self):
        self.buf.clear()
        self.last_stamp = None

    def update(self, pos, stamp, now):
        """Feed the latest vision reading; returns an Estimate or None if lost."""
        if pos is not None and stamp != self.last_stamp:
            self.buf.append((stamp, float(pos[0]), float(pos[1])))
            self.last_stamp = stamp
        while self.buf and self.buf[-1][0] - self.buf[0][0] > self.WINDOW_S:
            self.buf.popleft()
        if not self.buf or now - self.buf[-1][0] > self.STALE_S:
            return None

        data = np.array(self.buf)
        t_last = data[-1, 0]
        raw_latest = data[-1, 1:].copy()
        if len(data) < self.MIN_SAMPLES:
            return Estimate(raw_latest, np.zeros(2), raw_pos=raw_latest)

        t = data[:, 0] - t_last
        X = np.column_stack([np.ones_like(t), t])
        coef, *_ = np.linalg.lstsq(X, data[:, 1:], rcond=None)
        p, v = coef[0], coef[1]
        lead = min(max(0.0, now - t_last) + self.CAPTURE_LAG_S, self.MAX_LEAD_S)
        return Estimate(p + v * lead, v, raw_pos=p.copy())


# ── Inner loop: tilt tracking ────────────────────────────────────────────────
# Exactly dashboard.py's PD + Jacobian loop (via TiltTracker), the one that holds
# the tray steady on this arm. Additions for active tilt tracking:
#   - a slew limit on the target so steps don't saturate the PD and lurch the gears
#   - dashboard.py's 0.5 deg deadband so IMU noise and gear backlash don't cause
#     limit-cycle hunting on the stationary axis.

TILT_SLEW_DEG_S = 30.0
TRACKER_DEADBAND_DEG = 0.5
MEAS_FILTER_TAU_S = 0.10    # smoothed IMU tilt (rejects servo vibration & IMU noise)


def make_tilt_tracker(robot, J_pinv, roll_offset, pitch_offset):
    import dashboard as db
    from calibrate_tilt import TiltTracker
    from controller import PDController

    class SlewTiltTracker(TiltTracker):
        def __init__(self):
            pd = PDController(Kp=db.Kp, Kd=db.Kd, limit=db.CMD_LIMIT)
            super().__init__(robot, J_pinv, pd, roll_offset, pitch_offset,
                             deadband_rad=math.radians(TRACKER_DEADBAND_DEG))
            self.target = np.zeros(2)       # slewed setpoint actually handed to the PD
            self.meas = np.zeros(2)         # [pitch, roll] rad, raw IMU, relative to the startup offsets
            self.meas_filt = np.zeros(2)
            self._has_meas = False

        def step(self, pitch_target=0.0, roll_target=0.0):
            goal = np.array([pitch_target, roll_target], float)
            max_step = math.radians(TILT_SLEW_DEG_S) * db.DT
            delta = goal - self.target
            n = np.linalg.norm(delta)
            if n > max_step:
                delta *= max_step / n
            self.target = self.target + delta
            pitch_err, roll_err = super().step(float(self.target[0]), float(self.target[1]))
            self.meas = self.target + np.radians([pitch_err, roll_err])
            if not self._has_meas:
                self.meas_filt = self.meas.copy()
                self._has_meas = True
            else:
                alpha = db.DT / (MEAS_FILTER_TAU_S + db.DT)
                self.meas_filt += alpha * (self.meas - self.meas_filt)
            return pitch_err, roll_err

    return SlewTiltTracker()


# ── Outer loop: the centering controller ─────────────────────────────────────

TOL_MM = 5.0                # on target
REENGAGE_MM = 10.0          # hysteresis before driving again
V_STOP_MM_S = 4.0           # slower than this counts as stuck
V_MOVING_MM_S = 8.0         # faster than this counts as broken loose
# ...or moved this far from where it rested (while above V_STOP_MM_S so tilt
# parallax alone never triggers a false breakaway).
BREAK_DISP_MM = 2.8
V_MAX_MM_S = 50.0
BRAKE_FRAC = 0.75           # plan to stop using only this share of the friction decel
K_V = 7.0                   # 1/s, velocity-error -> commanded acceleration
K_POS = 1.3                 # 1/s, caps the approach speed near the target (short hops stay slow)
# Cancel only part of the measured kinetic friction: over-cancelling acts as
# negative damping (the weight hops back and forth across the target), while
# under-cancelling just adds damping that the velocity loop absorbs.
FRICTION_FF_FRAC = 0.85
BREAK_START_FRAC = 1.0      # start the breakaway ramp at this share of theta_s
BREAK_RAMP_DEG_S = 2.2      # gentle enough not to wind up ahead of the inner tilt loop
SLOPE_MARGIN_DEG = 4.5      # never command more than theta_s_dir(u) + this
SLOPE_HARD_MAX_DEG = 8.5
SLOPE_SLEW_DEG_S = 40.0
MIN_MOVE_S = 0.25           # dwell in MOVE before a slow reading can mean "stuck"
DITHER_DEG = 0.25           # only while breaking loose
DITHER_HZ = 3.0
RIM_MM = 73.0               # per axis (square tray)


class FrictionController:
    """Break / move / hold controller for a sliding weight. Angles in radians."""

    def __init__(self, calib, target_mm=(0.0, 0.0)):
        self.c = calib
        self.target = np.array(target_mm, float)
        self.phase = "hold"
        self.phase_since = 0.0
        self.slope = np.zeros(2)
        self.break_theta = 0.0
        self.last_break_theta = 0.0
        self.break_from = None
        self.max_slope = math.radians(min(math.degrees(float(np.max(calib.theta_s_axis))) + SLOPE_MARGIN_DEG,
                                          SLOPE_HARD_MAX_DEG))
        a_brake = G_MM_S2 * math.tan(calib.theta_k)
        self.a_plan = BRAKE_FRAC * a_brake
        self.n_breaks = 0

    def _enter(self, phase, now, u=None, restuck=False):
        self.phase, self.phase_since = phase, now
        if phase == "break":
            ts_u = self.c.theta_s_dir(u) if u is not None else self.c.theta_s
            base = BREAK_START_FRAC * ts_u
            if restuck and self.last_break_theta > 0.0:
                self.break_theta = max(base, 0.90 * self.last_break_theta)
            else:
                self.break_theta = base
                self.last_break_theta = 0.0
            self.break_from = None
            self.n_breaks += 1
        elif phase == "move":
            self.last_break_theta = max(self.last_break_theta, self.break_theta)
            n = float(np.linalg.norm(self.slope))
            cap = max(self.c.theta_s_dir(u) if u is not None else self.c.theta_s,
                      self.break_theta)
            if n > cap:
                self.slope = self.slope * (cap / n)

    def update(self, est, now, dt):
        """est: Estimate or None. Returns (tilt [pitch, roll] rad relative to true level, note)."""
        if est is None:
            goal, note = np.zeros(2), "vision lost - level"
            if self.phase != "hold":
                self._enter("hold", now)
            return self._out(goal, dt, note)

        e = self.target - est.pos
        dist = float(np.linalg.norm(e))
        u = e / dist if dist > 1e-9 else np.zeros(2)
        v, speed = est.vel, est.speed
        max_slope_u = math.radians(min(math.degrees(self.c.theta_s_dir(u)) + SLOPE_MARGIN_DEG,
                                       SLOPE_HARD_MAX_DEG))

        # Near the rim and still heading out: level, let friction stop it.
        edge = np.abs(est.pos) > RIM_MM
        if np.any(edge & (np.sign(est.pos) * v > V_STOP_MM_S)):
            if self.phase != "hold":
                self._enter("hold", now)
            return self._out(np.zeros(2), dt, f"at rim, moving out ({speed:.0f} mm/s) - level")

        if self.phase == "hold":
            if dist > REENGAGE_MM:
                self._enter("move" if speed > V_MOVING_MM_S else "break", now, u=u)
            else:
                return self._out(np.zeros(2), dt, f"ON TARGET  err {dist:4.1f} mm")

        if self.phase == "break":
            if self.break_from is None:
                self.break_from = est.pos.copy()
            if (speed > V_MOVING_MM_S
                    or (np.linalg.norm(est.pos - self.break_from) > BREAK_DISP_MM
                        and speed > V_STOP_MM_S)):
                self._enter("move", now, u=u)
            elif dist <= TOL_MM:
                self._enter("hold", now)
                return self._out(np.zeros(2), dt, "on target")
            else:
                self.break_theta = min(self.break_theta + math.radians(BREAK_RAMP_DEG_S) * dt,
                                       max_slope_u)
                mag = self.break_theta + math.radians(DITHER_DEG) * math.sin(2 * math.pi * DITHER_HZ * now)
                return self._out(u * mag, dt,
                                 f"BREAK {math.degrees(self.break_theta):4.1f}deg  err {dist:5.1f} mm")

        # move: stay here until the weight has actually stopped, braking to zero
        # speed AT the target - leaving at the tolerance edge while still sliding
        # is what overshoots.
        if speed < V_STOP_MM_S and now - self.phase_since > MIN_MOVE_S:
            if dist <= TOL_MM:
                self._enter("hold", now)
                return self._out(np.zeros(2), dt, "arrived - level")
            self._enter("break", now, u=u, restuck=True)
            return self._out(u * self.break_theta, dt, "stuck - breaking loose")

        tk_u = self.c.theta_k_dir(u)
        if self.last_break_theta > 0.0:
            tk_u = max(tk_u, 0.80 * self.last_break_theta)
        a_plan_u = BRAKE_FRAC * G_MM_S2 * math.tan(tk_u)
        v_des = min(V_MAX_MM_S, math.sqrt(2.0 * a_plan_u * dist), K_POS * dist)
        a_cmd = K_V * (v_des * u - v)
        # friction feed-forward, faded out at crawl speeds where v_hat is mostly noise
        fade = min(1.0, speed / V_MOVING_MM_S)
        v_hat = v / speed if speed > 1e-6 else u
        goal = a_cmd / G_MM_S2 + fade * FRICTION_FF_FRAC * tk_u * v_hat
        n = np.linalg.norm(goal)
        if n > max_slope_u:
            goal *= max_slope_u / n
        return self._out(goal, dt, f"MOVE  v {speed:4.0f}/{v_des:4.0f} mm/s  err {dist:5.1f} mm")

    def _out(self, goal, dt, note):
        step = math.radians(SLOPE_SLEW_DEG_S) * dt
        delta = goal - self.slope
        n = np.linalg.norm(delta)
        if n > step:
            delta *= step / n
        self.slope = self.slope + delta
        return self.c.tilt_for_slope(self.slope), note
