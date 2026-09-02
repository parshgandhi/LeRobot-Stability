"""
real_controller.py
==================
Ball-balancing controller for the SO-101 arm on real hardware.

Pipeline (50Hz):
    IMU serial → iNemo sensor fusion → PD controller
    → Cartesian twist → J_pinv → joint commands → SO-101

Built in parts:
    Part 1 — Imports and configuration          (this section)
    Part 2 — IMU serial reader thread
    Part 3 — Complementary filter wrapper
    Part 4 — Startup sequence
    Part 5 — Control loop
    Part 6 — Kill switch
    Part 7 — Logging
"""

# ── Standard library ──────────────────────────────────────────────────────────
import json
import math
import os
import threading
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path

# ── Third party ───────────────────────────────────────────────────────────────
import numpy as np
import serial

# ── Local ────────────────────────────────────────────────────────────────────
from iNemo_wrapper import iNemoEngine
from controller    import PDController

# ── LeRobot ───────────────────────────────────────────────────────────────────
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

# ═════════════════════════════════════════════════════════════════════════════
# HARDWARE PORTS
# ═════════════════════════════════════════════════════════════════════════════

ROBOT_PORT = "/dev/tty.usbmodem5B3D0470181"   # SO-101 follower arm
ROBOT_ID   = "follower"                        # must match --robot.id from calibration
IMU_PORT   = "/dev/tty.usbmodem205C336757521"  # SensorTile.box ASM330LHH
IMU_BAUD   = 20000000

# ═════════════════════════════════════════════════════════════════════════════
# JOINT NAMES — order must match q_nominal, limits, and J_pinv columns
# ═════════════════════════════════════════════════════════════════════════════

JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
]

# gripper is held fixed — excluded from Jacobian and control
GRIPPER_POS = 3.0   # degrees, held constant throughout

# ═════════════════════════════════════════════════════════════════════════════
# HOME POSE — locked in from IMU flatness tuning
# degrees, LeRobot convention (0° = calibration midpoint)
# ═════════════════════════════════════════════════════════════════════════════

Q_NOMINAL_DEG = {
    "shoulder_pan":   0.0,
    "shoulder_lift": -100.0,
    "elbow_flex":     25.0,
    "wrist_flex":    -90.0,
    "wrist_roll":     0.0,
}

# ═════════════════════════════════════════════════════════════════════════════
# JOINT LIMITS — safety envelope around q_nominal (degrees)
# These are conservative software limits inside the calibrated hardware limits.
# Arm will never be commanded outside these regardless of what the controller asks.
# ═════════════════════════════════════════════════════════════════════════════

JOINT_LIMITS_DEG = {
    "shoulder_pan":  (-20.0,  20.0),
    "shoulder_lift": (-130.0, -70.0),
    "elbow_flex":    ( 10.0,  60.0),
    "wrist_flex":    (-120.0, -60.0),
    "wrist_roll":    ( -30.0,  30.0),
}

# ═════════════════════════════════════════════════════════════════════════════
# TIMING
# ═════════════════════════════════════════════════════════════════════════════

DT               = 0.02    # seconds — 50Hz control loop (LeRobot limitation)
MOVE_DURATION_S  = 2.0     # seconds — startup interpolation to q_nominal
STARTUP_HOLD_S   = 15.0    # seconds — hold at q_nominal while iNemo bias converges

# ═════════════════════════════════════════════════════════════════════════════
# iNEMO SENSOR FUSION
# ═════════════════════════════════════════════════════════════════════════════

DYLIB_PATH = 'libiNemoEnginePlus.dylib'

# ═════════════════════════════════════════════════════════════════════════════
# PD CONTROLLER GAINS
# Starting conservative — tune Kp up first until response visible,
# then add Kd to damp oscillation.
# These are much lower than Franka (Kp=8, Kd=3) because:
#   - We command position increments not velocity directly
#   - SO-101 servos are slower and more compliant than Franka
#   - 50Hz loop vs 1kHz means each step has 20× more impact
# ═════════════════════════════════════════════════════════════════════════════

Kp        = 3.0   # proportional gain — raised from 0.3, servo resolution needs this
Kd        = 0.5    # derivative gain
CMD_LIMIT = 0.5    # rad/s — EE angular velocity clamp inside PDController

# ═════════════════════════════════════════════════════════════════════════════
# SAFETY LIMITS
# ═════════════════════════════════════════════════════════════════════════════

MAX_JOINT_VEL = 0.3          # rad/s — clip dq before integration
                               # prevents Jacobian amplification from jerking arm
DEADBAND_DEG  = 0.5          # degrees — ignore tilt smaller than this
                               # prevents chasing sensor noise when ball is centred
DEADBAND_RAD  = math.radians(DEADBAND_DEG)

# ═════════════════════════════════════════════════════════════════════════════
# JACOBIAN
# Precomputed at q_nominal using compute_jacobian.py
# Shape: (5, 6) — maps 6D Cartesian twist → 5 joint velocities
# ═════════════════════════════════════════════════════════════════════════════

J_PINV_PATH = "assets/J_pinv_so101.npy"

# ═════════════════════════════════════════════════════════════════════════════
# DASHBOARD
# ═════════════════════════════════════════════════════════════════════════════

DASHBOARD_PORT = 8080
DASHBOARD_HTML = Path(__file__).parent / "dashboard.html"



# ═════════════════════════════════════════════════════════════════════════════
# PART 8 — WEB DASHBOARD SERVER
# ═════════════════════════════════════════════════════════════════════════════

# Shared state dict — control loop writes, web server reads for /data endpoint.
dashboard_state = {
    "roll_deg": 0.0, "pitch_deg": 0.0,
    "roll_offset_deg": 0.0, "pitch_offset_deg": 0.0,
    "roll_err_deg": 0.0, "pitch_err_deg": 0.0,
    "in_deadband": True,
    "cp": 0.0, "cr": 0.0,
    "q_target_deg": [0.0] * 5,
    "q_nominal_deg": list(Q_NOMINAL_DEG.values()),
    "joint_limits_deg": [list(JOINT_LIMITS_DEG[n]) for n in JOINT_NAMES],
    "dq": [0.0] * 5,
    "loop_hz": 0.0,
    "imu_alive": False,
    "elapsed": 0.0,
    "kp": Kp, "kd": Kd,
    "deadband_deg": DEADBAND_DEG,
    "ax_mg": 0.0, "ay_mg": 0.0, "az_mg": 0.0,
    "gx_mdps": 0.0, "gy_mdps": 0.0, "gz_mdps": 0.0,
    "fire_log": [],
}

_dashboard_html_cache = None

def _load_dashboard_html():
    global _dashboard_html_cache
    if _dashboard_html_cache is None:
        with open(DASHBOARD_HTML, "rb") as f:
            _dashboard_html_cache = f.read()
    return _dashboard_html_cache


class _DashboardHandler(BaseHTTPRequestHandler):
    """Serves dashboard.html at / and JSON state at /data."""

    def do_GET(self):
        if self.path == "/data":
            payload = json.dumps(dashboard_state, default=lambda o: o.tolist() if hasattr(o, 'tolist') else float(o))
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(payload.encode())
        elif self.path == "/" or self.path == "/index.html":
            try:
                html = _load_dashboard_html()
                self.send_response(200)
                self.send_header("Content-Type", "text/html")
                self.end_headers()
                self.wfile.write(html)
            except FileNotFoundError:
                self.send_response(404)
                self.end_headers()
                self.wfile.write(b"dashboard.html not found")
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        pass  # suppress request logs


def start_dashboard_server(port: int = DASHBOARD_PORT):
    """Start the web dashboard as a background daemon thread."""
    server = HTTPServer(("0.0.0.0", port), _DashboardHandler)
    t = threading.Thread(target=server.serve_forever, daemon=True, name="dashboard")
    t.start()
    print(f"[Dashboard] http://localhost:{port}")
    return server


# ═════════════════════════════════════════════════════════════════════════════
# PART 2 — IMU SERIAL READER THREAD
# ═════════════════════════════════════════════════════════════════════════════

# Shared state dict — thread writes, main loop reads.
# Stores raw sensor units (mg and mdps) — conversion done by caller.
imu_state = {
    "ax_mg":     0.0,    # mg   — accelerometer X (raw)
    "ay_mg":     0.0,    # mg   — accelerometer Y (raw)
    "az_mg":     0.0,    # mg   — accelerometer Z (raw)
    "gx_mdps":   0.0,    # mdps — gyroscope X (raw)
    "gy_mdps":   0.0,    # mdps — gyroscope Y (raw)
    "gz_mdps":   0.0,    # mdps — gyroscope Z (raw)
    "valid":     False,  # True once first successful parse completes
    "timestamp": 0.0,    # time.monotonic() of last successful update
}


def _parse_imu_line(line: str) -> tuple | None:
    """Parse SensorTile line: ax,ay,az; gx,gy,gz in mg and mdps. No conversion."""
    try:
        parts = line.strip().split(";")
        if len(parts) != 2:
            return None
        a = [float(x) for x in parts[0].strip().split(",")]
        g = [float(x) for x in parts[1].strip().split(",")]
        if len(a) != 3 or len(g) != 3:
            return None
        return a[0], a[1], a[2], g[0], g[1], g[2]
    except (ValueError, IndexError):
        return None


def imu_reader_thread(port: str, baud: int) -> None:
    """
    Daemon thread: reads IMU serial continuously and updates imu_state.

    Runs until the main process exits (daemon=True ensures this).
    All exceptions caught internally — never crashes the main loop.

    Args:
        port: serial port string e.g. "/dev/tty.usbmodemXXXX"
        baud: baud rate (115200 for SensorTile)
    """
    print(f"[IMU] Opening {port} at {baud} baud ...")
    try:
        ser = serial.Serial(port, baud, timeout=0.1)
    except serial.SerialException as e:
        print(f"[IMU] ERROR — could not open port: {e}")
        return

    print("[IMU] Serial port open. Reading ...")

    while True:
        try:
            raw = ser.readline()
            if not raw:
                continue   # timeout — no data, loop again

            line = raw.decode("utf-8", errors="ignore")
            result = _parse_imu_line(line)

            if result is None:
                continue   # malformed line — skip silently

            ax_mg, ay_mg, az_mg, gx_mdps, gy_mdps, gz_mdps = result

            # Write raw values to shared state
            imu_state["ax_mg"]     = ax_mg
            imu_state["ay_mg"]     = ay_mg
            imu_state["az_mg"]     = az_mg
            imu_state["gx_mdps"]   = gx_mdps
            imu_state["gy_mdps"]   = gy_mdps
            imu_state["gz_mdps"]   = gz_mdps
            imu_state["valid"]     = True
            imu_state["timestamp"] = time.monotonic()

        except serial.SerialException as e:
            print(f"[IMU] Serial error: {e} — retrying ...")
            time.sleep(0.1)
        except Exception as e:
            print(f"[IMU] Unexpected error: {e}")
            time.sleep(0.1)


def start_imu_thread(port: str = IMU_PORT, baud: int = IMU_BAUD) -> threading.Thread:
    """
    Spawn and start the IMU reader as a background daemon thread.

    Returns the thread object (rarely needed but useful for debugging).
    """
    t = threading.Thread(
        target=imu_reader_thread,
        args=(port, baud),
        daemon=True,          # dies automatically when main process exits
        name="imu-reader",
    )
    t.start()
    return t


def wait_for_imu(timeout_s: float = 5.0) -> bool:
    """
    Block until the IMU thread produces its first valid reading.

    Args:
        timeout_s: how long to wait before giving up

    Returns:
        True  — IMU is live and producing data
        False — timed out, IMU not responding
    """
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if imu_state["valid"]:
            return True
        time.sleep(0.05)
    return False


def check_imu_alive(stale_threshold_s: float = 0.2) -> bool:
    """
    Returns True if the IMU produced a reading within the last stale_threshold_s.
    Call this inside the control loop to detect silent serial failures.
    """
    return (time.monotonic() - imu_state["timestamp"]) < stale_threshold_s


# ═════════════════════════════════════════════════════════════════════════════
# PART 3 — iNEMO SENSOR FUSION
# ═════════════════════════════════════════════════════════════════════════════
#
# Single responsibility: imu_state{} → (roll, pitch) in radians.
# Converts raw mg/mdps → g/dps before calling iNemo wrapper.
# iNemo outputs degrees — converted to radians here for the PD controller.
# Axis convention handled by iNemo's acc_ref setting — no manual fix needed.
# ═════════════════════════════════════════════════════════════════════════════

_inemo = None   # initialised in main() after dylib path is confirmed


def get_orientation() -> tuple[float, float]:
    """
    Read raw IMU state, convert units, run iNemo fusion.
    Returns (roll, pitch) in RADIANS for the PD controller.
    """
    ax_g   = imu_state["ax_mg"]   / 1000.0   # mg   → g
    ay_g   = imu_state["ay_mg"]   / 1000.0
    az_g   = imu_state["az_mg"]   / 1000.0
    gx_dps = imu_state["gx_mdps"] / 1000.0   # mdps → dps
    gy_dps = imu_state["gy_mdps"] / 1000.0
    gz_dps = imu_state["gz_mdps"] / 1000.0

    roll_deg, pitch_deg, _ = _inemo.update(ax_g, ay_g, az_g,
                                           gx_dps, gy_dps, gz_dps, DT)

    return math.radians(roll_deg), math.radians(pitch_deg)


# ═════════════════════════════════════════════════════════════════════════════
# PART 4 — STARTUP SEQUENCE
# ═════════════════════════════════════════════════════════════════════════════
#
# Three phases run once before the control loop activates:
#
#   Phase A — move arm smoothly from current position → q_nominal
#   Phase B — hold q_nominal while filter converges (STARTUP_HOLD_S)
#   Phase C — sample filter output to record home offset (roll/pitch at flat)
#
# Returns (roll_offset, pitch_offset) in radians.
# These are passed into the control loop and subtracted every iteration
# so the controller corrects relative to "home flat" not absolute 0°.
# ═════════════════════════════════════════════════════════════════════════════

N_OFFSET_SAMPLES = 20   # number of filter readings to average for Phase C


def _move_to_nominal(robot: SO101Follower) -> None:
    """
    Phase A — interpolate smoothly from current position to q_nominal.

    Uses sinusoidal ease-in/ease-out so the arm accelerates and
    decelerates rather than jumping at constant speed.
    Calls get_orientation() each step so the filter warms up during the move.
    """
    print("[Startup] Phase A — moving to q_nominal ...")

    obs     = robot.get_observation()
    current = {k.replace(".pos", ""): v for k, v in obs.items()
               if k.endswith(".pos")}

    n_steps = int(MOVE_DURATION_S / DT)

    for step in range(n_steps + 1):
        # Check IMU is still live during move
        if not check_imu_alive():
            raise RuntimeError("[Startup] IMU went stale during Phase A — check serial connection.")

        # Sinusoidal ease-in/ease-out: 0 → 1
        t      = step / n_steps
        alpha  = 0.5 * (1 - math.cos(math.pi * t))

        action = {}
        for name in JOINT_NAMES:
            start = current.get(name, Q_NOMINAL_DEG[name])
            end   = Q_NOMINAL_DEG[name]
            action[f"{name}.pos"] = start * (1 - alpha) + end * alpha

        # Gripper held fixed throughout
        action["gripper.pos"] = GRIPPER_POS

        robot.send_action(action)
        get_orientation()   # filter accumulates — output ignored here
        time.sleep(DT)

    print("[Startup] Phase A complete — arm at q_nominal.")


def _hold_and_converge(robot: SO101Follower) -> None:
    """
    Phase B — hold q_nominal while the complementary filter converges.

    The filter needs time to wash out initial gyro drift via the
    accelerometer correction term. STARTUP_HOLD_S = 2.0s gives
    comfortable margin at alpha=0.85, DT=0.02s.

    With iNemo this can be shortened to 1.0s — change STARTUP_HOLD_S
    in Part 1 config, nothing else changes.
    """
    print(f"[Startup] Phase B — holding for {STARTUP_HOLD_S:.0f}s while filter converges ...")

    action = {f"{name}.pos": Q_NOMINAL_DEG[name] for name in JOINT_NAMES}
    action["gripper.pos"] = GRIPPER_POS

    n_steps      = int(STARTUP_HOLD_S / DT)
    last_print_s = -1

    for step in range(n_steps):
        if not check_imu_alive():
            raise RuntimeError("[Startup] IMU went stale during Phase B — check serial connection.")

        robot.send_action(action)
        get_orientation()   # filter accumulates — output ignored here

        # Print countdown once per second
        elapsed_s = int(step * DT)
        if elapsed_s != last_print_s:
            remaining = int(STARTUP_HOLD_S) - elapsed_s
            print(f"[Startup]   converging ... {remaining}s remaining")
            last_print_s = elapsed_s

        time.sleep(DT)

    print("[Startup] Phase B complete — filter converged.")


def _record_home_offset(robot: SO101Follower) -> tuple[float, float]:
    """
    Phase C — sample filter output at q_nominal to get the home baseline.

    The tray is not perfectly flat at q_nominal (roll≈1.9°, pitch≈1.73°
    confirmed from IMU tuning). Records this as the zero-error reference
    so the controller corrects toward physical-flat, not mathematical 0°.

    Averages N_OFFSET_SAMPLES readings to reduce noise in the offset.
    """
    print(f"[Startup] Phase C — recording home offset ({N_OFFSET_SAMPLES} samples) ...")

    action = {f"{name}.pos": Q_NOMINAL_DEG[name] for name in JOINT_NAMES}
    action["gripper.pos"] = GRIPPER_POS

    roll_samples  = []
    pitch_samples = []

    for _ in range(N_OFFSET_SAMPLES):
        if not check_imu_alive():
            raise RuntimeError("[Startup] IMU went stale during Phase C — check serial connection.")

        robot.send_action(action)
        roll, pitch = get_orientation()
        roll_samples.append(roll)
        pitch_samples.append(pitch)
        time.sleep(DT)

    roll_offset  = sum(roll_samples)  / len(roll_samples)
    pitch_offset = sum(pitch_samples) / len(pitch_samples)

    print(f"[Startup] Phase C complete — home offset recorded:")
    print(f"[Startup]   roll_offset  = {math.degrees(roll_offset):+.2f}°")
    print(f"[Startup]   pitch_offset = {math.degrees(pitch_offset):+.2f}°")
    print(f"[Startup]   (expected roll≈+1.9°, pitch≈+1.7° from IMU tuning)")

    return roll_offset, pitch_offset


def startup(robot: SO101Follower) -> tuple[float, float]:
    """
    Run the full startup sequence and return home offsets.

    Call once from main() before starting the control loop.

    Args:
        robot: connected SO101Follower instance

    Returns:
        (roll_offset, pitch_offset) in radians — pass to control_loop()
    """
    print("\n[Startup] ══════════════════════════════════════════")
    print("[Startup] Starting up — 3 phases before control loop")
    print("[Startup] ══════════════════════════════════════════\n")

    # Phase A — move to home pose
    _move_to_nominal(robot)

    # Phase B — hold and let filter converge
    _hold_and_converge(robot)

    # Phase C — record flat-tray baseline
    roll_offset, pitch_offset = _record_home_offset(robot)

    print("\n[Startup] ══════════════════════════════════════════")
    print("[Startup] Startup complete — control loop starting")
    print("[Startup] ══════════════════════════════════════════\n")

    return roll_offset, pitch_offset


# ═════════════════════════════════════════════════════════════════════════════
# PART 7 — LOGGING
# ═════════════════════════════════════════════════════════════════════════════

_fire_log     = []
_MAX_FIRE_LOG = 6
_t_start      = None   # set in main() — all timestamps relative to this


def _log_fire(roll_err, pitch_err, cp, cr, dq):
    """Record a controller fire event into the ring buffer."""
    elapsed = time.monotonic() - _t_start
    mm = int(elapsed // 60)
    ss = elapsed % 60
    dominant_idx  = int(np.argmax(np.abs(dq)))
    entry = {
        "ts":        f"{mm:02d}:{ss:05.2f}",
        "roll_err":  roll_err,
        "pitch_err": pitch_err,
        "cp":        cp,
        "cr":        cr,
        "dq":        dq.copy(),
        "dominant":  JOINT_NAMES[dominant_idx],
        "dom_val":   dq[dominant_idx],
    }
    _fire_log.append(entry)
    if len(_fire_log) > _MAX_FIRE_LOG:
        _fire_log.pop(0)


def _render(roll, pitch, roll_offset, pitch_offset,
            roll_err, pitch_err, in_deadband,
            cp, cr, dq, q_target_deg, loop_hz):
    """Refresh the terminal display."""

    def deadband_bar(err_rad):
        frac   = min(abs(err_rad) / DEADBAND_RAD, 2.0)
        filled = int(frac * 10)
        bar    = "█" * min(filled, 10) + "░" * max(0, 10 - filled)
        side   = "▶ OUTSIDE" if abs(err_rad) >= DEADBAND_RAD else "  inside "
        return f"{bar} {side} deadband"

    imu_ok       = "✅" if check_imu_alive() else "⚠️  STALE"
    ctrl_status  = "FIRING 🔴" if not in_deadband else "IDLE   ⬜"
    elapsed      = time.monotonic() - _t_start
    mm, ss       = int(elapsed // 60), elapsed % 60

    os.system("clear" if os.name == "posix" else "cls")
    print(f"── SO-101 Ball Balance Controller  [{mm:02d}:{ss:05.2f}] ─────────────")
    print(f"  IMU: {imu_ok}   Loop: {loop_hz:.1f} Hz")
    print()
    print(f"  {'':14s}  {'Roll':>10}  {'Pitch':>10}")
    print(f"  {'─'*14}  {'─'*10}  {'─'*10}")
    print(f"  {'Filtered':14s}  {math.degrees(roll):>+10.2f}°  {math.degrees(pitch):>+10.2f}°")
    print(f"  {'Offset':14s}  {math.degrees(roll_offset):>+10.2f}°  {math.degrees(pitch_offset):>+10.2f}°")
    print(f"  {'Error':14s}  {math.degrees(roll_err):>+10.2f}°  {math.degrees(pitch_err):>+10.2f}°")
    print()
    print(f"  Deadband ({math.degrees(DEADBAND_RAD):.1f}°)")
    print(f"    roll  : {deadband_bar(roll_err)}")
    print(f"    pitch : {deadband_bar(pitch_err)}")
    print()
    print(f"  Controller : {ctrl_status}")
    if not in_deadband:
        print(f"    cp={cp:>+.4f}  cr={cr:>+.4f}  rad/s")
        print(f"    dq per joint (rad/s):")
        for name, val in zip(JOINT_NAMES, dq):
            bar = "█" * int(min(abs(val) / MAX_JOINT_VEL * 15, 15))
            print(f"      {name:<16}: {val:>+.5f}  {bar}")
    print()
    print(f"  Joint targets (°)")
    for name, val in zip(JOINT_NAMES, q_target_deg):
        nominal = Q_NOMINAL_DEG[name]
        delta   = val - nominal
        print(f"    {name:<16}: {val:>+8.2f}°   (Δ={delta:>+.2f}° from nominal)")
    print()
    print("── Fire Log ──────────────────────────────────────────────────────")
    if not _fire_log:
        print("  (no events yet)")
    else:
        for e in reversed(_fire_log):
            dq_str = "  ".join(f"{v:>+.3f}" for v in e["dq"])
            print(f"  [{e['ts']}] roll={math.degrees(e['roll_err']):>+5.1f}°  "
                  f"pitch={math.degrees(e['pitch_err']):>+5.1f}°  "
                  f"cp={e['cp']:>+.3f}  cr={e['cr']:>+.3f}")
            print(f"             dq=[{dq_str}]")
            print(f"             dominant: {e['dominant']} ({e['dom_val']:>+.4f} rad/s)")
    print("──────────────────────────────────────────────────────────────────")
    print("  Ctrl+C to stop safely.")


# ═════════════════════════════════════════════════════════════════════════════
# PART 5 — CONTROL LOOP
# ═════════════════════════════════════════════════════════════════════════════

def _angle_wrap(angle_rad: float) -> float:
    """Wrap angle to [-π, +π] so ±180° crossings give correct small differences."""
    return (angle_rad + math.pi) % (2 * math.pi) - math.pi


def control_loop(
    robot:        SO101Follower,
    J_pinv:       np.ndarray,
    pd:           PDController,
    roll_offset:  float,
    pitch_offset: float,
) -> None:
    """
    Main 50Hz control loop — runs until Ctrl+C or IMU failure.

    Every iteration (20ms):
        1. Check IMU alive
        2. get_orientation() → roll, pitch
        3. subtract home offset → roll_err, pitch_err
        4. deadband
        5. PDController.compute() → cp, cr
        6. build twist [0,0,0,cr,-cp,0]
        7. dq = J_pinv @ twist
        8. velocity clip
        9. q_target = q_nominal_rad + dq*DT
       10. joint limit clip
       11. send_action()
       12. log + display
       13. sleep to 50Hz

    Args:
        robot:        connected SO101Follower
        J_pinv:       (5,6) pseudoinverse loaded from assets/
        pd:           PDController instance
        roll_offset:  radians — recorded in startup Phase C
        pitch_offset: radians — recorded in startup Phase C
    """

    # Pre-compute q_nominal in radians and joint limits in radians
    q_nom_rad   = np.array([math.radians(Q_NOMINAL_DEG[n]) for n in JOINT_NAMES])
    lim_min_rad = np.array([math.radians(JOINT_LIMITS_DEG[n][0]) for n in JOINT_NAMES])
    lim_max_rad = np.array([math.radians(JOINT_LIMITS_DEG[n][1]) for n in JOINT_NAMES])

    # Display timing
    DISPLAY_HZ   = 10
    dt_display   = 1.0 / DISPLAY_HZ
    last_display = time.monotonic()

    # Loop Hz tracking
    loop_count = 0
    loop_hz    = 0.0
    hz_t0      = time.monotonic()

    # Current state for display (updated every iteration)
    q_target_deg = list(Q_NOMINAL_DEG.values())
    dq_display   = np.zeros(5)
    cp_display   = 0.0
    cr_display   = 0.0

    # Accumulated joint position — starts at q_nominal, integrates over time
    q_current_rad = q_nom_rad.copy()

    # Previous orientation for numerical derivative (D term)
    roll_prev  = 0.0
    pitch_prev = 0.0

    # Auto-recalibration: track how long we've been in the deadband
    idle_time = 0.0

    print("[Control] Loop running at 50Hz. Ctrl+C to stop.\n")

    while True:
        t0 = time.monotonic()

        # ── 1. IMU liveness ───────────────────────────────────────────
        if not check_imu_alive():
            print("\n[Control] WARNING — IMU stale. Stopping loop.")
            break

        # ── 2. Orientation ────────────────────────────────────────────
        roll, pitch = get_orientation()

        # ── 3. Subtract home offset (with angle wrapping) ─────────────
        # iNemo roll is near ±180° at home pose — crossing that boundary
        # without wrapping creates 360° jumps in the error.
        roll_err  = _angle_wrap(roll  - roll_offset)
        pitch_err = _angle_wrap(pitch - pitch_offset)

        # ── 4. Deadband ───────────────────────────────────────────────
        roll_db  = roll_err  if abs(roll_err)  >= DEADBAND_RAD else 0.0
        pitch_db = pitch_err if abs(pitch_err) >= DEADBAND_RAD else 0.0
        in_deadband = (roll_db == 0.0 and pitch_db == 0.0)

        # ── 5. PD controller ──────────────────────────────────────────
        # D term: numerical derivative of iNemo output, not raw gyro.
        # This eliminates all gyro bias issues — both P and D terms
        # come entirely from iNemo fused, bias-free angles.
        pitch_rate = (pitch - pitch_prev) / DT   # rad/s
        roll_rate  = (roll  - roll_prev)  / DT
        pitch_prev = pitch
        roll_prev  = roll
        cp, cr, pp, dp = pd.compute(pitch_db, roll_db, pitch_rate, roll_rate)

        # ── 6. Cartesian twist ────────────────────────────────────────
        twist = np.array([0.0, 0.0, 0.0, -cr, -cp, 0.0])  # negate both: servo direction

        # ── 7. Jacobian → joint velocities ───────────────────────────
        dq = J_pinv @ twist

        # ── 8. Velocity clip ──────────────────────────────────────────
        dq = np.clip(dq, -MAX_JOINT_VEL, MAX_JOINT_VEL)

        # ── 9. Integrate — accumulate position over time ─────────────
        q_current_rad = q_current_rad + dq * DT
        q_target_rad  = q_current_rad

        # ── 10. Joint limit clip ──────────────────────────────────────
        q_target_rad  = np.clip(q_target_rad, lim_min_rad, lim_max_rad)
        q_current_rad = q_target_rad.copy()

        # ── 11. Send action ───────────────────────────────────────────
        action = {f"{name}.pos": math.degrees(q_target_rad[i])
                  for i, name in enumerate(JOINT_NAMES)}
        action["gripper.pos"] = GRIPPER_POS
        robot.send_action(action)

        # ── 12. Log + display ─────────────────────────────────────────
        if not in_deadband:
            _log_fire(roll_err, pitch_err, cp, cr, dq)

        # Update display cache
        q_target_deg = [math.degrees(v) for v in q_target_rad]
        dq_display   = dq
        cp_display   = cp
        cr_display   = cr

        # Compute loop Hz
        loop_count += 1
        now = time.monotonic()
        if now - hz_t0 >= 1.0:
            loop_hz    = loop_count / (now - hz_t0)
            loop_count = 0
            hz_t0      = now

        # Refresh terminal + dashboard at DISPLAY_HZ
        if now - last_display >= dt_display:
            _render(
                roll, pitch, roll_offset, pitch_offset,
                roll_err, pitch_err, in_deadband,
                cp_display, cr_display, dq_display,
                q_target_deg, loop_hz,
            )

            # Update web dashboard state
            dashboard_state.update({
                "roll_deg": math.degrees(roll),
                "pitch_deg": math.degrees(pitch),
                "roll_offset_deg": math.degrees(roll_offset),
                "pitch_offset_deg": math.degrees(pitch_offset),
                "roll_err_deg": math.degrees(roll_err),
                "pitch_err_deg": math.degrees(pitch_err),
                "in_deadband": in_deadband,
                "cp": float(cp_display),
                "cr": float(cr_display),
                "q_target_deg": q_target_deg,
                "dq": [float(v) for v in dq_display],
                "loop_hz": loop_hz,
                "imu_alive": check_imu_alive(),
                "elapsed": now - _t_start if _t_start else 0,
                "ax_mg": imu_state["ax_mg"],
                "ay_mg": imu_state["ay_mg"],
                "az_mg": imu_state["az_mg"],
                "gx_mdps": imu_state["gx_mdps"],
                "gy_mdps": imu_state["gy_mdps"],
                "gz_mdps": imu_state["gz_mdps"],
                "fire_log": [
                    {
                        "ts": e["ts"],
                        "roll_err_deg": math.degrees(e["roll_err"]),
                        "pitch_err_deg": math.degrees(e["pitch_err"]),
                        "dominant": e["dominant"],
                        "dom_val": float(e["dom_val"]),
                        "dq": [float(v) for v in e["dq"]],
                    }
                    for e in _fire_log
                ],
            })

            last_display = now

        # ── 13. Sleep to 50Hz ─────────────────────────────────────────
        elapsed = time.monotonic() - t0
        time.sleep(max(0.0, DT - elapsed))


# ═════════════════════════════════════════════════════════════════════════════
# PART 6 — KILL SWITCH + MAIN
# ═════════════════════════════════════════════════════════════════════════════

def _park_and_disconnect(robot: SO101Follower) -> None:
    """
    Safe shutdown — move back to q_nominal, disable torque, disconnect.
    Always runs via finally block regardless of how the loop exits.
    """
    try:
        print("\n[Shutdown] Returning to home pose ...")
        action = {f"{name}.pos": Q_NOMINAL_DEG[name] for name in JOINT_NAMES}
        action["gripper.pos"] = GRIPPER_POS
        n_steps = int(MOVE_DURATION_S / DT)
        obs     = robot.get_observation()
        current = {k.replace(".pos", ""): v for k, v in obs.items()
                   if k.endswith(".pos")}
        for step in range(n_steps + 1):
            alpha  = 0.5 * (1 - math.cos(math.pi * step / n_steps))
            interp = {}
            for name in JOINT_NAMES:
                start = current.get(name, Q_NOMINAL_DEG[name])
                interp[f"{name}.pos"] = start * (1 - alpha) + Q_NOMINAL_DEG[name] * alpha
            interp["gripper.pos"] = GRIPPER_POS
            robot.send_action(interp)
            time.sleep(DT)
        print("[Shutdown] At home pose.")
        robot.bus.sync_write("Torque_Enable",
                             {name: 0 for name in JOINT_NAMES + ["gripper"]})
        print("[Shutdown] Torque disabled.")
    except Exception as e:
        print(f"[Shutdown] Warning during shutdown: {e}")
    finally:
        robot.disconnect()
        print("[Shutdown] Disconnected.")
        if _inemo:
            _inemo.close()


def main():
    global _t_start

    # ── Load Jacobian ─────────────────────────────────────────────────
    if not os.path.exists(J_PINV_PATH):
        print(f"ERROR — {J_PINV_PATH} not found. Run compute_jacobian.py first.")
        return
    J_pinv = np.load(J_PINV_PATH)
    pd     = PDController(Kp=Kp, Kd=Kd, limit=CMD_LIMIT)
    print(f"[Init] Loaded J_pinv {J_pinv.shape} from {J_PINV_PATH}")

    # ── Load iNemo ────────────────────────────────────────────────────
    global _inemo
    _inemo = iNemoEngine(dylib_path=DYLIB_PATH, kappa_acc=0.3, verbose=True)

    # ── Connect robot ─────────────────────────────────────────────────
    print(f"[Init] Connecting to robot on {ROBOT_PORT} ...")
    config = SO101FollowerConfig(port=ROBOT_PORT, id=ROBOT_ID, use_degrees=True)
    robot  = SO101Follower(config)
    robot.connect(calibrate=False)
    print("[Init] Robot connected.")

    # ── Start IMU thread ──────────────────────────────────────────────
    start_imu_thread()
    print(f"[Init] Waiting for IMU on {IMU_PORT} ...")
    if not wait_for_imu(timeout_s=5.0):
        print("ERROR — no IMU data in 5s. Check port and connection.")
        robot.disconnect()
        return
    print("[Init] IMU live.")

    # ── Start dashboard server ────────────────────────────────────
    start_dashboard_server()

    # ── Run startup + control loop with kill switch ───────────────────
    try:
        roll_offset, pitch_offset = startup(robot)
        _t_start = time.monotonic()
        control_loop(robot, J_pinv, pd, roll_offset, pitch_offset)

    except KeyboardInterrupt:
        print("\n[Control] Ctrl+C received.")

    except Exception as e:
        print(f"\n[Control] Unexpected error: {e}")
        raise

    finally:
        _park_and_disconnect(robot)


if __name__ == "__main__":
    main()