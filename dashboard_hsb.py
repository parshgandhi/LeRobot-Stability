"""
dashboard_hsb.py
================
Ball-balancing controller for the SO-101 arm on real hardware.
IMU data received over HSB (UDP) instead of serial.

Pipeline (50Hz):
    IMU via HSB UDP → iNemo sensor fusion → PD controller
    → Cartesian twist → J_pinv → joint commands → SO-101

Based on dashboard.py — only the IMU source changed from serial to UDP.
The bridge script (asm_imu_dashboard_bridge.py --raw-udp) sends raw
(un-remapped) sensor data matching the SensorTile convention that
iNemo with ENU acc_ref expects.
"""

# ── Standard library ──────────────────────────────────────────────────────────
import json
import math
import os
import socket
import struct
import threading
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path

# ── Third party ───────────────────────────────────────────────────────────────
import numpy as np

# ── Local ────────────────────────────────────────────────────────────────────
from iNemo_wrapper import iNemoEngine
from controller    import PDController

# ── LeRobot ───────────────────────────────────────────────────────────────────
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

# ═════════════════════════════════════════════════════════════════════════════
# HARDWARE PORTS
# ═════════════════════════════════════════════════════════════════════════════

ROBOT_PORT = "/dev/ttyACM0"
ROBOT_ID   = "follower"

# HSB raw sensor UDP — replaces serial IMU
IMU_UDP_PORT = 12289

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

GRIPPER_POS = 3.0

# ═════════════════════════════════════════════════════════════════════════════
# HOME POSE
# ═════════════════════════════════════════════════════════════════════════════

Q_NOMINAL_DEG = {
    "shoulder_pan":   0.0,
    "shoulder_lift": -100.0,
    "elbow_flex":     30.0,
    "wrist_flex":    -90.0,
    "wrist_roll":     0.0,
}

# ═════════════════════════════════════════════════════════════════════════════
# JOINT LIMITS
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

DT               = 0.02
MOVE_DURATION_S  = 2.0
STARTUP_HOLD_S   = 15.0

# ═════════════════════════════════════════════════════════════════════════════
# iNEMO SENSOR FUSION
# ═════════════════════════════════════════════════════════════════════════════

DYLIB_PATH = 'sensorFusion/libinemo.so'

# ═════════════════════════════════════════════════════════════════════════════
# PD CONTROLLER GAINS
# ═════════════════════════════════════════════════════════════════════════════

Kp        = 3.0
Kd        = 0.5
CMD_LIMIT = 0.5

# ═════════════════════════════════════════════════════════════════════════════
# SAFETY LIMITS
# ═════════════════════════════════════════════════════════════════════════════

MAX_JOINT_VEL = 0.3
DEADBAND_DEG  = 0.5
DEADBAND_RAD  = math.radians(DEADBAND_DEG)

# ═════════════════════════════════════════════════════════════════════════════
# JACOBIAN
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
        pass


def start_dashboard_server(port: int = DASHBOARD_PORT):
    server = HTTPServer(("0.0.0.0", port), _DashboardHandler)
    t = threading.Thread(target=server.serve_forever, daemon=True, name="dashboard")
    t.start()
    print(f"[Dashboard] http://localhost:{port}")
    return server


# ═════════════════════════════════════════════════════════════════════════════
# PART 2 — IMU UDP RECEIVER THREAD (replaces serial reader)
# ═════════════════════════════════════════════════════════════════════════════

imu_state = {
    "ax_mg":     0.0,
    "ay_mg":     0.0,
    "az_mg":     0.0,
    "gx_mdps":   0.0,
    "gy_mdps":   0.0,
    "gz_mdps":   0.0,
    "valid":     False,
    "timestamp": 0.0,
}

_RAW_SENSOR_PACKET_SIZE = 24  # 6 floats


def imu_udp_receiver_thread(udp_port: int) -> None:
    """
    Daemon thread: receives raw sensor UDP packets from the HSB bridge
    and updates imu_state.

    The bridge sends un-remapped raw sensor values in the same convention
    as the SensorTile (az ≈ -981mg at rest). No axis manipulation needed
    here — iNemo with ENU acc_ref handles it.

    Packet: struct.pack("<6f", ax_mg, ay_mg, az_mg, gx_mdps, gy_mdps, gz_mdps)
    """
    print(f"[IMU-HSB] Listening for raw sensor UDP on port {udp_port} ...")

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
    except AttributeError:
        pass
    sock.bind(("0.0.0.0", udp_port))
    sock.settimeout(1.0)

    print("[IMU-HSB] Socket bound. Waiting for data from HSB bridge ...")

    while True:
        try:
            data, addr = sock.recvfrom(256)
            if len(data) != _RAW_SENSOR_PACKET_SIZE:
                continue

            ax_mg, ay_mg, az_mg, gx_mdps, gy_mdps, gz_mdps = struct.unpack("<6f", data)

            imu_state["ax_mg"]     = ax_mg
            imu_state["ay_mg"]     = ay_mg
            imu_state["az_mg"]     = az_mg
            imu_state["gx_mdps"]   = gx_mdps
            imu_state["gy_mdps"]   = gy_mdps
            imu_state["gz_mdps"]   = gz_mdps
            imu_state["valid"]     = True
            imu_state["timestamp"] = time.monotonic()

        except socket.timeout:
            continue
        except Exception as e:
            print(f"[IMU-HSB] Error: {e}")
            time.sleep(0.1)


def start_imu_thread(udp_port: int = IMU_UDP_PORT) -> threading.Thread:
    t = threading.Thread(
        target=imu_udp_receiver_thread,
        args=(udp_port,),
        daemon=True,
        name="imu-hsb-reader",
    )
    t.start()
    return t


def wait_for_imu(timeout_s: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if imu_state["valid"]:
            return True
        time.sleep(0.05)
    return False


def check_imu_alive(stale_threshold_s: float = 0.2) -> bool:
    return (time.monotonic() - imu_state["timestamp"]) < stale_threshold_s


# ═════════════════════════════════════════════════════════════════════════════
# PART 3 — iNEMO SENSOR FUSION
# ═════════════════════════════════════════════════════════════════════════════

_inemo = None


def get_orientation() -> tuple[float, float]:
    """
    Read raw IMU state, convert units, run iNemo fusion.
    Returns (roll, pitch) in RADIANS for the PD controller.
    """
    ax_g   = imu_state["ax_mg"]   / 1000.0
    ay_g   = imu_state["ay_mg"]   / 1000.0
    az_g   = imu_state["az_mg"]   / 1000.0
    gx_dps = imu_state["gx_mdps"] / 1000.0
    gy_dps = imu_state["gy_mdps"] / 1000.0
    gz_dps = imu_state["gz_mdps"] / 1000.0

    roll_deg, pitch_deg, _ = _inemo.update(ax_g, ay_g, az_g,
                                           gx_dps, gy_dps, gz_dps, DT)

    return math.radians(roll_deg), math.radians(pitch_deg)


# ═════════════════════════════════════════════════════════════════════════════
# PART 4 — STARTUP SEQUENCE
# ═════════════════════════════════════════════════════════════════════════════

N_OFFSET_SAMPLES = 20


def _move_to_nominal(robot: SO101Follower) -> None:
    print("[Startup] Phase A — moving to q_nominal ...")

    obs     = robot.get_observation()
    current = {k.replace(".pos", ""): v for k, v in obs.items()
               if k.endswith(".pos")}

    n_steps = int(MOVE_DURATION_S / DT)

    for step in range(n_steps + 1):
        if not check_imu_alive():
            raise RuntimeError("[Startup] IMU went stale during Phase A — check HSB bridge.")

        t      = step / n_steps
        alpha  = 0.5 * (1 - math.cos(math.pi * t))

        action = {}
        for name in JOINT_NAMES:
            start = current.get(name, Q_NOMINAL_DEG[name])
            end   = Q_NOMINAL_DEG[name]
            action[f"{name}.pos"] = start * (1 - alpha) + end * alpha

        action["gripper.pos"] = GRIPPER_POS

        robot.send_action(action)
        get_orientation()
        time.sleep(DT)

    print("[Startup] Phase A complete — arm at q_nominal.")


def _hold_and_converge(robot: SO101Follower) -> None:
    print(f"[Startup] Phase B — holding for {STARTUP_HOLD_S:.0f}s while filter converges ...")

    action = {f"{name}.pos": Q_NOMINAL_DEG[name] for name in JOINT_NAMES}
    action["gripper.pos"] = GRIPPER_POS

    n_steps      = int(STARTUP_HOLD_S / DT)
    last_print_s = -1

    for step in range(n_steps):
        if not check_imu_alive():
            raise RuntimeError("[Startup] IMU went stale during Phase B — check HSB bridge.")

        robot.send_action(action)
        get_orientation()

        elapsed_s = int(step * DT)
        if elapsed_s != last_print_s:
            remaining = int(STARTUP_HOLD_S) - elapsed_s
            print(f"[Startup]   converging ... {remaining}s remaining")
            last_print_s = elapsed_s

        time.sleep(DT)

    print("[Startup] Phase B complete — filter converged.")


def _record_home_offset(robot: SO101Follower) -> tuple[float, float]:
    print(f"[Startup] Phase C — recording home offset ({N_OFFSET_SAMPLES} samples) ...")

    action = {f"{name}.pos": Q_NOMINAL_DEG[name] for name in JOINT_NAMES}
    action["gripper.pos"] = GRIPPER_POS

    roll_samples  = []
    pitch_samples = []

    for _ in range(N_OFFSET_SAMPLES):
        if not check_imu_alive():
            raise RuntimeError("[Startup] IMU went stale during Phase C — check HSB bridge.")

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

    return roll_offset, pitch_offset


def startup(robot: SO101Follower) -> tuple[float, float]:
    print("\n[Startup] ══════════════════════════════════════════")
    print("[Startup] Starting up — 3 phases before control loop")
    print("[Startup] ══════════════════════════════════════════\n")

    _move_to_nominal(robot)
    _hold_and_converge(robot)
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
_t_start      = None


def _log_fire(roll_err, pitch_err, cp, cr, dq):
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
    print(f"── SO-101 Ball Balance Controller (HSB)  [{mm:02d}:{ss:05.2f}] ────────")
    print(f"  IMU: {imu_ok}   Loop: {loop_hz:.1f} Hz   Source: UDP:{IMU_UDP_PORT}")
    print()
    print(f"  {'':14s}  {'Roll':>10}  {'Pitch':>10}")
    print(f"  {'─'*14}  {'─'*10}  {'─'*10}")
    print(f"  {'Filtered':14s}  {math.degrees(roll):>+10.2f}°  {math.degrees(pitch):>+10.2f}°")
    print(f"  {'Offset':14s}  {math.degrees(roll_offset):>+10.2f}°  {math.degrees(pitch_offset):>+10.2f}°")
    print(f"  {'Error':14s}  {math.degrees(roll_err):>+10.2f}°  {math.degrees(pitch_err):>+10.2f}°")
    print()
    print(f"  Raw IMU:  ax={imu_state['ax_mg']:+.0f}  ay={imu_state['ay_mg']:+.0f}  az={imu_state['az_mg']:+.0f} mg")
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
# PART 5 — CONTROL LOOP (identical to dashboard.py)
# ═════════════════════════════════════════════════════════════════════════════

def _angle_wrap(angle_rad: float) -> float:
    return (angle_rad + math.pi) % (2 * math.pi) - math.pi


def control_loop(
    robot:        SO101Follower,
    J_pinv:       np.ndarray,
    pd:           PDController,
    roll_offset:  float,
    pitch_offset: float,
) -> None:

    q_nom_rad   = np.array([math.radians(Q_NOMINAL_DEG[n]) for n in JOINT_NAMES])
    lim_min_rad = np.array([math.radians(JOINT_LIMITS_DEG[n][0]) for n in JOINT_NAMES])
    lim_max_rad = np.array([math.radians(JOINT_LIMITS_DEG[n][1]) for n in JOINT_NAMES])

    DISPLAY_HZ   = 10
    dt_display   = 1.0 / DISPLAY_HZ
    last_display = time.monotonic()

    loop_count = 0
    loop_hz    = 0.0
    hz_t0      = time.monotonic()

    q_target_deg = list(Q_NOMINAL_DEG.values())
    dq_display   = np.zeros(5)
    cp_display   = 0.0
    cr_display   = 0.0

    q_current_rad = q_nom_rad.copy()

    roll_prev  = 0.0
    pitch_prev = 0.0

    print("[Control] Loop running at 50Hz. Ctrl+C to stop.\n")

    while True:
        t0 = time.monotonic()

        # 1. IMU liveness
        if not check_imu_alive():
            print("\n[Control] WARNING — IMU stale. Stopping loop.")
            break

        # 2. Orientation
        roll, pitch = get_orientation()

        # 3. Subtract home offset
        roll_err  = _angle_wrap(roll  - roll_offset)
        pitch_err = _angle_wrap(pitch - pitch_offset)

        # 4. Deadband
        roll_db  = roll_err  if abs(roll_err)  >= DEADBAND_RAD else 0.0
        pitch_db = pitch_err if abs(pitch_err) >= DEADBAND_RAD else 0.0
        in_deadband = (roll_db == 0.0 and pitch_db == 0.0)

        # 5. PD controller
        pitch_rate = (pitch - pitch_prev) / DT
        roll_rate  = (roll  - roll_prev)  / DT
        pitch_prev = pitch
        roll_prev  = roll
        cp, cr, pp, dp = pd.compute(pitch_db, roll_db, pitch_rate, roll_rate)

        # 6. Cartesian twist
        twist = np.array([0.0, 0.0, 0.0, -cr, -cp, 0.0])

        # 7. Jacobian → joint velocities
        dq = J_pinv @ twist

        # 8. Velocity clip
        dq = np.clip(dq, -MAX_JOINT_VEL, MAX_JOINT_VEL)

        # 9. Integrate
        q_current_rad = q_current_rad + dq * DT
        q_target_rad  = q_current_rad

        # 10. Joint limit clip
        q_target_rad  = np.clip(q_target_rad, lim_min_rad, lim_max_rad)
        q_current_rad = q_target_rad.copy()

        # 11. Send action
        action = {f"{name}.pos": math.degrees(q_target_rad[i])
                  for i, name in enumerate(JOINT_NAMES)}
        action["gripper.pos"] = GRIPPER_POS
        robot.send_action(action)

        # 12. Log + display
        if not in_deadband:
            _log_fire(roll_err, pitch_err, cp, cr, dq)

        q_target_deg = [math.degrees(v) for v in q_target_rad]
        dq_display   = dq
        cp_display   = cp
        cr_display   = cr

        loop_count += 1
        now = time.monotonic()
        if now - hz_t0 >= 1.0:
            loop_hz    = loop_count / (now - hz_t0)
            loop_count = 0
            hz_t0      = now

        if now - last_display >= dt_display:
            _render(
                roll, pitch, roll_offset, pitch_offset,
                roll_err, pitch_err, in_deadband,
                cp_display, cr_display, dq_display,
                q_target_deg, loop_hz,
            )

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

        # 13. Sleep to 50Hz
        elapsed = time.monotonic() - t0
        time.sleep(max(0.0, DT - elapsed))


# ═════════════════════════════════════════════════════════════════════════════
# PART 6 — KILL SWITCH + MAIN
# ═════════════════════════════════════════════════════════════════════════════

def _park_and_disconnect(robot: SO101Follower) -> None:
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

    # ── Resolve robot port ────────────────────────────────────────────
    robot_port = ROBOT_PORT
    if os.path.islink(robot_port) or os.path.exists(robot_port):
        robot_port = os.path.realpath(robot_port)
    print(f"[Init] Robot port: {robot_port}")
    print(f"[Init] IMU source: HSB UDP port {IMU_UDP_PORT}")

    # ── Load Jacobian ─────────────────────────────────────────────────
    if not os.path.exists(J_PINV_PATH):
        print(f"ERROR — {J_PINV_PATH} not found. Run jacobian.py first.")
        return
    J_pinv = np.load(J_PINV_PATH)
    pd     = PDController(Kp=Kp, Kd=Kd, limit=CMD_LIMIT)
    print(f"[Init] Loaded J_pinv {J_pinv.shape} from {J_PINV_PATH}")

    # ── Load iNemo ────────────────────────────────────────────────────
    global _inemo
    _inemo = iNemoEngine(dylib_path=DYLIB_PATH, kappa_acc=0.3, verbose=True)

    # ── Connect robot ─────────────────────────────────────────────────
    print(f"[Init] Connecting to robot on {robot_port} ...")
    config = SO101FollowerConfig(port=robot_port, id=ROBOT_ID, use_degrees=True)
    robot  = SO101Follower(config)
    robot.connect(calibrate=False)
    print("[Init] Robot connected.")

    # ── Start IMU UDP receiver thread ─────────────────────────────────
    start_imu_thread()
    print(f"[Init] Waiting for IMU data on UDP port {IMU_UDP_PORT} ...")
    if not wait_for_imu(timeout_s=10.0):
        print("ERROR — no IMU data in 10s. Is asm_imu_dashboard_bridge.py running with --raw-udp?")
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
