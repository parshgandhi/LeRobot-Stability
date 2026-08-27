"""
controller.py
=============
PD controller for tray stabilization.

Extracted from traySimv3.py — no changes to the logic.
Kept as a separate file so it can be imported by both:
  - traySimv3.py  (Python simulation)
  - isaac_controller.py  (Isaac Sim deployment)

Inputs every call:
    fp  — filtered pitch from ComplementaryFilter (radians)
    fr  — filtered roll  from ComplementaryFilter (radians)
    gp  — raw gyro pitch rate (rad/s) — used directly as D term, no lag
    gr  — raw gyro roll  rate (rad/s)

Outputs every call:
    cp  — EE pitch velocity command (rad/s) → ωy in Cartesian twist
    cr  — EE roll  velocity command (rad/s) → ωx in Cartesian twist
    pp  — P term value (for logging/display only)
    dp  — D term value (for logging/display only)

Control law:
    pp = Kp × fp          (P term: proportional to current tilt)
    dp = Kd × gp          (D term: proportional to tilt rate from gyro)
    cp = clamp(-(pp+dp), ±limit)   (negative = command opposes tilt)

Same formula for roll using fr and gr.

Real robot equivalent (libfranka):
    return franka::CartesianVelocities({0, 0, 0, cr, cp, 0});
"""


class PDController:
    """
    PD controller that reads filtered IMU orientation and outputs
    EE Cartesian angular velocity commands to level the tray.

    Parameters:
        Kp    — proportional gain (default 8.0)
                Higher = faster response, risks oscillation
        Kd    — derivative gain (default 3.0)
                Higher = more damping, slower but no overshoot
        limit — EE angular velocity limit in rad/s (default 1.5)
                Matches Franka's safe EE angular velocity range
    """

    def __init__(self, Kp=8.0, Kd=3.0, limit=1.5):
        self.Kp    = Kp
        self.Kd    = Kd
        self.limit = limit

    def compute(self, fp, fr, gp, gr):
        """
        Compute EE velocity commands from filtered IMU state.

        Args:
            fp  — filtered pitch (rad) from complementary filter
            fr  — filtered roll  (rad) from complementary filter
            gp  — raw gyro pitch rate (rad/s) — D term, no filter lag
            gr  — raw gyro roll  rate (rad/s)

        Returns:
            (cp, cr, pp, dp)
            cp  — EE pitch cmd (rad/s), clamped to ±limit
            cr  — EE roll  cmd (rad/s), clamped to ±limit
            pp  — P term for pitch (diagnostic)
            dp  — D term for pitch (diagnostic)
        """
        pp = self.Kp * fp    # P term pitch: how tilted are we?
        dp = self.Kd * gp    # D term pitch: how fast is it tilting?
        pr = self.Kp * fr    # P term roll
        dr = self.Kd * gr    # D term roll

        # Negate: tilt is positive → command must be negative to correct
        cp = max(-self.limit, min(self.limit, -(pp + dp)))
        cr = max(-self.limit, min(self.limit, -(pr + dr)))

        return cp, cr, pp, dp