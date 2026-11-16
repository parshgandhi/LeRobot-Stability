"""Measure what the friction-aware centering controller needs.

Part 1 - EMPTY tray: check the inner tilt loop (dashboard.py's PD, unchanged)
holds +/-4 deg steps - steady error, IMU noise and settle time are reported.

Part 2 - weight on the tray (friction and true level), per axis: find a tilt
where the weight rests ("home"), ramp one way until it breaks loose, jump back
home and let it rest, then ramp the other way until it breaks loose again.
  * The IMU is too noisy (servo vibration, backlash) to read a ~1-2 deg
    breakaway from single samples, so the tilt at breakaway is the COMMANDED
    tilt (noise-free) minus the PD's tracking lag, averaged over the second
    before the slide. When the weight moved comes from the camera (~0.3 mm).
  * sliding starts where |tilt - true_level| = theta_s, so
        true_level = (break+ + break-) / 2      theta_s = (break+ - break-) / 2
    and this midpoint stays exact even if the other axis is still off level.
    Each break is measured from rest, so it is fine if the weight was already
    sliding when the sweep began (the startup pose is ~2 deg off level).
  * the +/- slide directions give the tilt -> downhill map A
  * a slide starts and ends at rest, so the slope it saw averages exactly to the
    kinetic friction angle theta_k (net impulse zero) - no speed fit needed.
  Two rounds: the second starts from the level found by the first and sweeps
  the opposite way first, which cancels most of the remaining bias.
  The first round starts from the IMU's own gravity zero (0 or 180 deg,
  whichever the mounting gives; right if the IMU is mounted flat on the tray);
  --from-startup-pose starts from the startup pose.

Run with dashboard.py STOPPED. Writes friction_calib.json.
"""

import argparse
import json
import math
import time
from collections import deque

import numpy as np

import friction_control as fc

DT = 0.02

# Part 1
STEP_TEST_DEG = 4.0
STEP_HOLD_S = 3.0
MAX_STEADY_ERR_DEG = 1.0
REJECTED_PATH = "friction_calib_rejected.json"

# Part 2
RAMP_DEG_S = 0.5            # slow: little extra slope builds up before a break is seen
BRAKE_DEG = 1.4             # uphill past home while forward speed > BRAKE_DONE_MM_S
BRAKE_MAX_S = 0.55
BRAKE_DONE_MM_S = 5.0
MAX_RAMP_DEG = 8.0          # furthest a sweep goes past where the weight last rested
ARM_RAMP_MAX_DEG = 4.0      # how far to ramp looking for rest before trying the other way
BREAK_MM = 2.2              # displacement + speed that count as broken loose
BREAK_SPEED_MM_S = 4.5
STILL_MM = 1.4              # resting = all positions over STILL_S within this of their mean
STILL_S = 0.35
REST_TIMEOUT_S = 4.5        # after returning home, the weight must rest within this
ONSET_MM = 1.0              # a slide starts/ends where the weight is this close to rest
ONSET_MAX_LOOKBACK_S = 0.35 # at 0.5 deg/s, breakaway occurs ~0.15-0.30 s before BREAK_MM
IMU_LAG_S = 0.14            # shift slide_tilts window to match fused IMU filter lag
MIN_SLIDE_SAMPLES = 8       # IMU samples needed to average theta_k over a slide
LAG_WINDOW_S = 1.0          # average the PD's tracking lag over this much ramp before a break
SETTLE_S = 1.0
START_RADIUS_MM = 25.0
SAFE_RADIUS_MM = 62.0
ROUNDS = 2
AXIS_TRIES = 3              # repeat an axis whose result is physically implausible
MIN_THETA_S_DEG = 0.3
MAX_DIR_COS = -0.15         # + and - slides must oppose (allowing orthogonal bias in round 1)
MAX_GRAVITY_ZERO_DEG = 6.0  # trust the IMU's gravity zero only if it is this close


def hold(tracker, clock, seconds, tilt, record=None):
    """Hold a tilt target; optionally record the measured tilt each tick."""
    end = clock.now() + seconds
    while clock.now() < end:
        tracker.step(float(tilt[0]), float(tilt[1]))
        if record is not None:
            record.append((clock.now(), tracker.meas.copy()))
        clock.sleep(DT)


# ── Part 1: inner loop ───────────────────────────────────────────────────────

def inner_loop_check(tracker, clock, step_deg=STEP_TEST_DEG):
    """Hold +/-step_deg on each axis. Returns {axis: (steady_err_deg, imu_noise_deg, t90_s)}.

    steady_err is signed toward the step (+ = beyond the target), averaged over
    the last second so IMU noise drops out; t90 uses a 5-sample moving average.
    """
    out = {}
    for idx, name in enumerate(("pitch", "roll")):
        errs, noises, t90s = [], [], []
        for sign in (1, -1):
            hold(tracker, clock, 2.0, np.zeros(2))
            tgt = np.zeros(2)
            tgt[idx] = sign * math.radians(step_deg)
            rec = []
            t0 = clock.now()
            hold(tracker, clock, STEP_HOLD_S, tgt, record=rec)
            ts = np.array([t for t, m in rec]) - t0
            y = np.array([m[idx] for t, m in rec]) * sign
            late = y[ts >= STEP_HOLD_S - 1.0]
            errs.append(math.degrees(late.mean() - math.radians(step_deg)))
            noises.append(math.degrees(late.std()))
            smooth = np.convolve(y, np.ones(5) / 5, mode="same")
            hit = np.nonzero(smooth >= 0.9 * math.radians(step_deg))[0]
            t90s.append(ts[hit[0]] if len(hit) else float("inf"))
        hold(tracker, clock, 2.0, np.zeros(2))
        out[name] = (float(np.mean(errs)), float(np.mean(noises)), float(np.mean(t90s)))
        print(f"  [{name}] {step_deg:.0f}deg step: steady error {out[name][0]:+.2f}deg, "
              f"IMU noise {out[name][1]:.2f}deg rms, 90% in {out[name][2] * 1000:.0f} ms")
    return out


# ── Part 2: friction and true level ──────────────────────────────────────────

class SweepFailed(RuntimeError):
    def __init__(self, msg, arming=False):
        super().__init__(msg)
        self.arming = arming


def axis_sweep(tracker, vision, clock, idx, level, first_sign):
    """Breakaway test of one tilt axis in both directions, from `level` (rad, tracker frame).

    Find a resting tilt (ramping first_sign if the weight is sliding), call it
    home. Ramp first_sign until the weight breaks loose, return to home and wait
    for both the weight and the tray to settle; then the same toward -first_sign.
    Every break is measured from a settled rest state. Returns the logged ticks
    and the two segments (start_idx, break_idx, rest_idx, start_pos, rest_pos, sign).
    """
    est = fc.WeightEstimator()
    cmd = np.array(level, float)
    rate = math.radians(RAMP_DEG_S) * DT
    log = []        # (t, measured tilt, raw camera pos or None, camera stamp, commanded tilt)
    segs = []
    is_arming = True

    def tick():
        tracker.step(float(cmd[0]), float(cmd[1]))
        now = clock.now()
        raw, stamp = vision.read_stamped()
        e = est.update(raw, stamp, now)
        fresh = raw is not None and (not log or stamp != log[-1][3])
        log.append((now, float(tracker.meas[idx]),
                    np.array(raw, float) if fresh else None, stamp, float(tracker.target[idx])))
        if e is not None and np.max(np.abs(e.raw_pos)) > SAFE_RADIUS_MM:
            raise SweepFailed("weight reached the safety radius", arming=is_arming)
        clock.sleep(DT)
        return e

    def wait_until_still(timeout_s, arming=False, sign=0, u_fwd=None):
        """Hold `cmd` fixed (or nudge against persistent forward creep) until stationary over STILL_S."""
        recent, t_start = deque(), clock.now()
        while True:
            e = tick()
            now = log[-1][0]
            if e is not None:
                recent.append((now, e.raw_pos.copy()))
            while recent and now - recent[0][0] > STILL_S:
                recent.popleft()
            if recent and now - recent[0][0] >= STILL_S - 2 * DT and len(recent) >= 5:
                pts = np.array([p for _, p in recent])
                if np.max(np.linalg.norm(pts - pts.mean(axis=0), axis=1)) < STILL_MM:
                    return len(log) - 1, pts.mean(axis=0)
            if sign != 0 and u_fwd is not None and e is not None and now - t_start > 0.25:
                if float(e.vel @ u_fwd) > 3.0:
                    cmd[idx] -= 1.5 * sign * rate
            if now - t_start > timeout_s:
                raise SweepFailed("weight never came to rest", arming=arming)

    def arm_to_rest(sign):
        """Ensure the weight rests at `home` and the tray is settled before ramping."""
        try:
            _, p0 = wait_until_still(STILL_S + 0.25, arming=True)
            return len(log) - 1, p0
        except SweepFailed:
            pass
        # Weight is creeping at `level`; ramp in `sign` to find a resting tilt,
        # then hold fixed at that tilt so the tray settles before the break test.
        recent, ramped = deque(), 0.0
        while True:
            e = tick()
            now = log[-1][0]
            if e is not None:
                recent.append((now, e.raw_pos.copy()))
            while recent and now - recent[0][0] > STILL_S:
                recent.popleft()
            if recent and now - recent[0][0] >= STILL_S - 2 * DT and len(recent) >= 5:
                pts = np.array([p for _, p in recent])
                if np.max(np.linalg.norm(pts - pts.mean(axis=0), axis=1)) < STILL_MM:
                    break
            cmd[idx] += sign * rate
            ramped += rate
            if ramped > math.radians(ARM_RAMP_MAX_DEG):
                raise SweepFailed("weight never came to rest", arming=True)
        for _ in range(int(0.6 * SETTLE_S / DT)):
            tick()
        return wait_until_still(REST_TIMEOUT_S, arming=True)

    def ramp_until_break(sign, start_pos):
        ramped = 0.0
        while True:
            cmd[idx] += sign * rate
            ramped += rate
            if ramped > math.radians(MAX_RAMP_DEG):
                raise SweepFailed(f"weight never slid within {MAX_RAMP_DEG:.0f}deg")
            e = tick()
            if (e is not None and np.linalg.norm(e.pos - start_pos) >= BREAK_MM
                    and (e.speed >= BREAK_SPEED_MM_S
                         or np.linalg.norm(e.pos - start_pos) >= 3.5)):
                d = e.pos - start_pos
                u_fwd = d / max(float(np.linalg.norm(d)), 1e-9)
                return len(log) - 1, u_fwd, float(cmd[idx])

    def brake_and_return(sign, u_fwd, rest_cmd):
        # Briefly aim BRAKE_DEG uphill of `rest_cmd` while the weight is still
        # moving forward in `u_fwd`, then hold `rest_cmd`.
        cmd[idx] = rest_cmd - sign * math.radians(BRAKE_DEG)
        t0 = clock.now()
        while clock.now() - t0 < BRAKE_MAX_S:
            e = tick()
            if e is not None and clock.now() - t0 > 0.12 and float(e.vel @ u_fwd) < BRAKE_DONE_MM_S:
                break
        cmd[idx] = rest_cmd

    for _ in range(int(SETTLE_S / DT)):
        tick()
    i0, p0 = arm_to_rest(first_sign)
    home = float(cmd[idx])
    if np.linalg.norm(p0) > START_RADIUS_MM + 10.0:
        raise SweepFailed("weight drifted out of start radius while settling", arming=True)
    is_arming = False
    b1, u1, cmd_b1 = ramp_until_break(first_sign, p0)
    if first_sign * (cmd_b1 - home) < math.radians(0.75):
        home = cmd_b1 - first_sign * math.radians(0.85)
    brake_and_return(first_sign, u1, home)
    i1, p1 = wait_until_still(REST_TIMEOUT_S, sign=first_sign, u_fwd=u1)
    segs.append((i0, b1, i1, p0, p1, first_sign))

    # Let the physical tray settle completely at the resting tilt before starting
    # the opposite ramp so segment 1's recovery never leaks into segment 2.
    home = float(cmd[idx])
    for _ in range(int(SETTLE_S / DT)):
        tick()
    i1_start, p1_start = wait_until_still(REST_TIMEOUT_S, sign=first_sign, u_fwd=u1)
    b2, u2, cmd_b2 = ramp_until_break(-first_sign, p1_start)
    home2 = 0.5 * (cmd_b1 + cmd_b2) if (-first_sign) * (cmd_b2 - home) < math.radians(0.75) else home
    brake_and_return(-first_sign, u2, home2)
    i2, p2 = wait_until_still(REST_TIMEOUT_S, sign=-first_sign, u_fwd=u2)
    segs.append((i1_start, b2, i2, p1_start, p2, -first_sign))
    return log, segs


def _segment_stats(log, seg):
    """Breakaway tilt, slide interval and direction of one segment.

    Slide start/stop are timed from the raw camera frames, shifted back by the
    capture lag. Walking backward from `t_break` within ONSET_MAX_LOOKBACK_S
    makes `t_onset` immune to slow tilt-parallax drift during the ramp.

    The tilt at breakaway is the commanded tilt at that moment minus the PD's
    tracking lag (commanded - measured, averaged over the preceding ramp), so
    single noisy IMU samples never decide it.
    """
    i_start, i_break, i_rest, start, rest, sign = seg
    lag = fc.WeightEstimator.CAPTURE_LAG_S
    frames = [(log[k][3] - lag, log[k][2]) for k in range(i_start, i_rest + 1) if log[k][2] is not None]
    t_break = log[i_break][0]

    pre = [(t, p) for t, p in frames if t <= t_break]
    t_min_onset = max(pre[0][0], t_break - ONSET_MAX_LOOKBACK_S)
    t_onset = t_min_onset
    for idx_f in range(len(pre) - 1, -1, -1):
        t_cur, p_cur = pre[idx_f]
        if t_cur < t_min_onset:
            break
        if np.linalg.norm(p_cur - start) < ONSET_MM:
            t_onset = t_cur
            break

    t_stop = frames[-1][0]
    for t, p in reversed(frames):
        if t <= t_onset or np.linalg.norm(p - rest) >= ONSET_MM:
            break
        t_stop = t

    ts = np.array([r[0] for r in log])
    meas = np.array([r[1] for r in log])
    cmds = np.array([r[4] for r in log])
    win = (ts >= max(t_onset - LAG_WINDOW_S, log[i_start][0] + 0.15)) & (ts <= t_onset)
    if win.sum() >= 10:
        track_lag = float(np.mean(cmds[win] - meas[win]))
    else:
        near = (ts >= max(log[i_start][0], t_onset - 0.30)) & (ts <= t_onset + 0.05)
        track_lag = float(np.mean(cmds[near] - meas[near])) if near.any() else 0.0
    full_d = rest - start
    moved_mm = float(np.linalg.norm(full_d))
    dir_vec = full_d / max(moved_mm, 1e-9)
    target_early = min(12.0, 0.7 * moved_mm)
    for _, p in frames:
        d_early = p - start
        n_early = float(np.linalg.norm(d_early))
        if n_early >= target_early and n_early > 1e-6:
            dir_vec = d_early / n_early
            break
    return {"sign": sign, "tilt_at_break": float(np.interp(t_onset, ts, cmds)) - track_lag,
            "track_lag": track_lag, "t_onset": t_onset, "t_stop": t_stop,
            "slide_tilts": meas[(ts >= t_onset + IMU_LAG_S) & (ts <= t_stop + IMU_LAG_S)],
            "dir": dir_vec, "moved_mm": moved_mm}


def _stabilize_axis(tracker, vision, clock, idx, level):
    """If the weight is sliding because `level[idx]` is beyond breakaway, ramp `idx` to rest."""
    rate = math.radians(RAMP_DEG_S) * DT
    for sign in (1, -1):
        cmd = np.array(level, float)
        est = fc.WeightEstimator()
        recent, ramped = deque(), 0.0
        ok = False
        while ramped <= math.radians(ARM_RAMP_MAX_DEG):
            tracker.step(float(cmd[0]), float(cmd[1]))
            now = clock.now()
            e = est.update(*vision.read_stamped(), now)
            if e is not None:
                if np.max(np.abs(e.raw_pos)) > SAFE_RADIUS_MM:
                    break
                recent.append((now, e.raw_pos.copy()))
            while recent and now - recent[0][0] > STILL_S:
                recent.popleft()
            if recent and now - recent[0][0] >= STILL_S - 2 * DT and len(recent) >= 5:
                pts = np.array([p for _, p in recent])
                if np.max(np.linalg.norm(pts - pts.mean(axis=0), axis=1)) < STILL_MM:
                    ok = True
                    break
            cmd[idx] += sign * rate
            ramped += rate
            clock.sleep(DT)
        if ok:
            level[idx] = cmd[idx]
            hold(tracker, clock, SETTLE_S, level)
            return True
        hold(tracker, clock, 0.5, level)
    return False


def calibrate_friction(tracker, vision, clock, recenter, level0=np.zeros(2), rounds=ROUNDS):
    """Fit true level, theta_s, theta_k and A from +/- breakaway sweeps of both axes.

    `recenter(level)` must return once the weight is within START_RADIUS_MM of
    the centre (asks the user on hardware). Returns (level_rel, theta_s,
    theta_k, A, tests).
    """
    level = np.array(level0, float)
    tests = []
    theta_s_axes, A, theta_k_segs = [None, None], np.eye(2), []

    def weight_pos():
        est = fc.WeightEstimator()
        e = None
        for _ in range(10):
            tracker.step(float(level[0]), float(level[1]))
            e = est.update(*vision.read_stamped(), clock.now()) or e
            clock.sleep(DT)
        return None if e is None else e.raw_pos

    for rnd in range(rounds):
        for idx, name in enumerate(("pitch", "roll")):
            for tries in range(AXIS_TRIES):
                first = 1 if (rnd + tries) % 2 == 0 else -1
                for attempt in range(3):
                    pos = weight_pos()
                    if pos is None or np.linalg.norm(pos) > START_RADIUS_MM:
                        recenter(level)
                        pos = weight_pos()
                    if rnd > 0 and tries == 0 and attempt == 0 and pos is not None:
                        proj = float(pos @ A[:, idx])
                        if abs(proj) > 8.0:
                            first = -1 if proj > 0 else 1
                    print(f"  [round {rnd + 1} {name}] sweeping {'+-'[first < 0]} first from "
                          f"({math.degrees(level[0]):+.2f}, {math.degrees(level[1]):+.2f})deg ...")
                    try:
                        log, segs = axis_sweep(tracker, vision, clock, idx, level, first)
                        break
                    except SweepFailed as err:
                        hold(tracker, clock, 1.0, level)
                        if attempt == 2:
                            raise RuntimeError(f"{name} sweep failed: {err}. If the weight slides "
                                               "whatever the tilt, try --from-startup-pose") from err
                        print(f"    {err} — retrying in opposite direction after recentering")
                        first = -first
                        recenter(level)

                st = {s["sign"]: s for s in (_segment_stats(log, g) for g in segs)}
                for s in st.values():
                    print(f"    {'+' if s['sign'] > 0 else '-'}{name}: broke loose at "
                          f"{math.degrees(s['tilt_at_break']):+.2f}deg "
                          f"(tracking lag {math.degrees(s['track_lag']):+.2f}deg), slid "
                          f"({s['dir'][0]:+.2f}, {s['dir'][1]:+.2f}) {s['moved_mm']:.0f} mm")
                bp, bm = st[1]["tilt_at_break"], st[-1]["tilt_at_break"]
                problem = None
                if 0.5 * (bp - bm) < math.radians(MIN_THETA_S_DEG):
                    problem = (f"+{name} broke loose below -{name} "
                               f"({math.degrees(bp):+.2f} vs {math.degrees(bm):+.2f}deg)")
                elif st[1]["dir"] @ st[-1]["dir"] > MAX_DIR_COS:
                    problem = "the + and - slides did not go opposite ways"
                elif min(st[1]["moved_mm"], st[-1]["moved_mm"]) < BREAK_MM:
                    problem = "a slide was too short to be real"
                if problem is None:
                    break
                print(f"    [reject] {problem} — the tray was probably not holding the "
                      f"commanded tilt; repeating ({tries + 1}/{AXIS_TRIES})")
                hold(tracker, clock, 1.5, level)
            else:
                raise RuntimeError(f"{name}: no consistent breakaway after {AXIS_TRIES} tries. Check "
                                   "the Part 1 step test: large overshoot means the tray does not "
                                   "hold the commanded tilt")

            level[idx] = 0.5 * (bp + bm)
            theta_s_axes[idx] = 0.5 * (bp - bm)
            col = st[1]["dir"] - st[-1]["dir"]
            A[:, idx] = col / np.linalg.norm(col)

            # A slide starts and ends at rest, so the slope it saw averages to theta_k.
            for s in st.values():
                tilts = s["slide_tilts"]
                raw_tk = (float(np.mean(s["sign"] * (tilts - level[idx])))
                          if len(tilts) >= MIN_SLIDE_SAMPLES else None)
                s["theta_k"] = raw_tk if (raw_tk is not None and raw_tk > 0.0) else None
                tests.append(dict(round=rnd + 1, axis=name, sign=s["sign"],
                                  tilt_at_break_deg=math.degrees(s["tilt_at_break"]),
                                  dir=s["dir"].tolist(), moved_mm=s["moved_mm"],
                                  slide_s=s["t_stop"] - s["t_onset"],
                                  theta_k_deg=None if s["theta_k"] is None else math.degrees(s["theta_k"])))
            if rnd == rounds - 1:
                theta_k_segs += [s["theta_k"] for s in st.values() if s["theta_k"] is not None]
            print(f"  -> {name}: true level {math.degrees(level[idx]):+.2f}deg, "
                  f"breakaway {math.degrees(theta_s_axes[idx]):.2f}deg")
            hold(tracker, clock, 1.0, level)

    theta_s = float(np.mean(theta_s_axes))
    if theta_k_segs:
        theta_k = float(np.clip(np.mean(theta_k_segs), 0.3 * theta_s, 0.95 * theta_s))
    else:
        print("  [warn] no usable slides for theta_k - assuming 0.7 * theta_s")
        theta_k = 0.7 * theta_s
    return level, theta_s, theta_k, A, tests


# ── Hardware entry point ─────────────────────────────────────────────────────

def main():
    import dashboard as db
    import tray_vision as tv
    from iNemo_wrapper import iNemoEngine
    from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-inner", action="store_true",
                    help="skip the Part 1 inner-loop check")
    ap.add_argument("--from-startup-pose", action="store_true",
                    help="start the first sweep from the startup pose instead of the IMU's gravity zero")
    args = ap.parse_args()

    J_pinv = np.load(db.J_PINV_PATH)
    db._inemo = iNemoEngine(dylib_path=db.DYLIB_PATH, kappa_acc=0.3, verbose=True)
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

    clock = fc.RealClock()
    try:
        print("\n[Init] Running startup with an EMPTY tray ...")
        roll_offset, pitch_offset = db.startup(robot)

        tracker = fc.make_tilt_tracker(robot, J_pinv, roll_offset, pitch_offset)

        inner = None
        if not args.skip_inner:
            print("\n[Part 1] inner tilt loop — keep the tray EMPTY")
            inner = inner_loop_check(tracker, clock)
            if max(abs(e) for e, _, _ in inner.values()) > MAX_STEADY_ERR_DEG:
                print(f"  [warn] the tray settles more than {MAX_STEADY_ERR_DEG:.0f}deg off the commanded "
                      "tilt; breakaway angles will be less accurate")

        def recenter(level):
            print(f"\n>>> Place the weight near the tray centre (within {START_RADIUS_MM:.0f} mm). "
                  "The arm is holding — safe to reach in.")
            deadline = time.monotonic() + 180.0
            next_report = 0.0
            while True:
                if time.monotonic() > deadline:
                    raise RuntimeError("timed out waiting for the weight")
                tracker.step(float(level[0]), float(level[1]))
                xy = vision.read()
                if xy is not None and math.hypot(*xy) <= START_RADIUS_MM:
                    break
                if time.monotonic() >= next_report:
                    where = (f"weight at ({xy[0]:+.0f}, {xy[1]:+.0f}) mm" if xy is not None
                             else vision.status())
                    print(f"    waiting — {where}")
                    next_report = time.monotonic() + 2.0
                time.sleep(DT)
            print(">>> Starting in 2s — hands clear.")
            hold(tracker, clock, 2.0, level)

        print("\n[Part 2] friction and true level")
        level0 = np.zeros(2)
        if not args.from_startup_pose:
            level0 = fc.gravity_zero_rel(pitch_offset, roll_offset)
            print(f"  IMU says the startup pose is ({math.degrees(-level0[0]):+.2f}, "
                  f"{math.degrees(-level0[1]):+.2f})deg off IMU zero (pitch, roll)")
            if np.max(np.abs(level0)) > math.radians(MAX_GRAVITY_ZERO_DEG):
                print(f"  [warn] that is more than {MAX_GRAVITY_ZERO_DEG:.0f}deg - the IMU is not "
                      "mounted flat on the tray; starting from the startup pose instead")
                level0 = np.zeros(2)
        else:
            print("  Starting from the levelled startup pose (0.0, 0.0)deg")
        recenter(level0)
        level, theta_s, theta_k, A, tests = calibrate_friction(tracker, vision, clock, recenter,
                                                               level0=level0)

        level_abs = [fc._wrap(pitch_offset + level[0]), fc._wrap(roll_offset + level[1])]
        extra = {"tests": tests, "inner_loop": inner,
                 "startup_offset_deg": [math.degrees(pitch_offset), math.degrees(roll_offset)]}
        try:
            calib = fc.FrictionCalib(level_abs, theta_s, theta_k, A, extra=extra)
        except ValueError as err:
            with open(REJECTED_PATH, "w") as f:
                json.dump({"error": str(err), "theta_s_deg": math.degrees(theta_s),
                           "theta_k_deg": math.degrees(theta_k), "A": A.tolist(), **extra}, f, indent=2)
            raise RuntimeError(f"calibration NOT saved: {err} (details in {REJECTED_PATH})") from err
        calib.save()
        print("\n" + "=" * 62)
        print(f"true level vs startup pose: pitch {math.degrees(level[0]):+.2f}deg, "
              f"roll {math.degrees(level[1]):+.2f}deg")
        print(f"breakaway angle theta_s = {math.degrees(theta_s):.2f}deg, "
              f"kinetic angle theta_k = {math.degrees(theta_k):.2f}deg")
        print(f"A (cols = +pitch, +roll downhill in tray axes):\n{np.round(A, 2)}")
        print(f"saved {fc.FRICTION_CALIB_PATH}")
        hold(tracker, clock, 1.0, level)

    except KeyboardInterrupt:
        print("\n[Abort] Ctrl+C")
    finally:
        vision.stop()
        input("\n>>> TAKE THE WEIGHT OFF THE TRAY, then press Enter. "
              "Torque is about to be disabled and the arm will sag. ")
        db._park_and_disconnect(robot)


if __name__ == "__main__":
    main()
