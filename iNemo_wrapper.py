"""
inemo_wrapper.py
================
Python wrapper for the iNemo Engine Plus sensor fusion library.

Loads libiNemoEnginePlus.dylib via ctypes and exposes a single clean class
that all other scripts import. No ctypes boilerplate appears anywhere else.

Usage:
    from inemo_wrapper import iNemoEngine

    inemo = iNemoEngine()           # loads dylib, inits instance, configures
    roll, pitch = inemo.update(     # call every loop at your control rate
        ax, ay, az,                 # accelerometer in m/s²  (thread output)
        gx, gy, gz,                 # gyroscope    in rad/s  (thread output)
        dt                          # timestep in seconds
    )
    bias = inemo.get_bias()         # (gx, gy, gz) in dps — for diagnostics
    inemo.close()                   # call in kill switch / finally block

Unit conversion:
    The IMU thread produces m/s² and rad/s (SI units).
    iNemo internally needs g and dps.
    This wrapper converts transparently — callers always use SI.

Axis convention:
    SensorTile mounted with -Z pointing up (az ≈ -9.81 m/s² at rest).
    Handled via inemo_set_acc_ref() — no manual axis negation needed.

Output convention:
    roll  — positive = tray tilting right   (degrees)
    pitch — positive = tray tilting forward (degrees)
    yaw   — rotation around vertical axis   (degrees, not used for balancing)

iNemo euler output order is [yaw, pitch, roll] — this wrapper unpacks correctly.
"""

import ctypes
import math
import os


# ─────────────────────────────────────────────────────────────────────────────
# Constants matching inemo.h enums
# ─────────────────────────────────────────────────────────────────────────────
INEMO_DISABLE = 0
INEMO_ENABLE  = 1
INEMO_6X      = 0   # accel + gyro only (no magnetometer)
INEMO_9X      = 1   # accel + gyro + mag (not used)


# ─────────────────────────────────────────────────────────────────────────────
# ctypes structs — must match inemo.h byte-for-byte
# ─────────────────────────────────────────────────────────────────────────────

class _InemoInput(ctypes.Structure):
    """
    inemo_input from inemo.h:
        float acc[3]   accelerometer [g]
        float mag[3]   magnetometer  [uT/50]  — zeroed in 6X mode
        float gyr[3]   gyroscope     [dps]
        float dtime    timestep      [s]
    """
    _fields_ = [
        ("acc",   ctypes.c_float * 3),
        ("mag",   ctypes.c_float * 3),
        ("gyr",   ctypes.c_float * 3),
        ("dtime", ctypes.c_float),
    ]


class _InemoOutput(ctypes.Structure):
    """
    inemo_output from inemo.h:
        float quaternion[4]   [qx, qy, qz, qw]
        float gravity[3]      gravity vector [g]
        float linear[3]       linear acceleration [g]
        float euler[3]        [yaw, pitch, roll] degrees
    """
    _fields_ = [
        ("quaternion", ctypes.c_float * 4),
        ("gravity",    ctypes.c_float * 3),
        ("linear",     ctypes.c_float * 3),
        ("euler",      ctypes.c_float * 3),
    ]


class _InemoConf(ctypes.Structure):
    """
    inemo_conf from inemo.h — all 17 fields in order.
    """
    _fields_ = [
        ("alpha_acc",             ctypes.c_float),
        ("alpha_gyr",             ctypes.c_float),
        ("alpha_mag",             ctypes.c_float),
        ("kappa_acc",             ctypes.c_float),
        ("kappa_mag",             ctypes.c_float),
        ("acc_max",               ctypes.c_float),
        ("acc_var_max",           ctypes.c_float),
        ("mag_var_max",           ctypes.c_float),
        ("gb_alpha",              ctypes.c_float),
        ("gb_acc_max",            ctypes.c_float),
        ("gb_gyr_max",            ctypes.c_float),
        ("gb_acc_var_max",        ctypes.c_float),
        ("gb_gyr_var_max",        ctypes.c_float),
        ("gb_gyr_var_max_greedy", ctypes.c_float),
        ("gb_time",               ctypes.c_float),
        ("adapt_k_gain",          ctypes.c_float),
        ("adapt_alpha_min",       ctypes.c_float),
    ]


# No unit conversion constants — the wrapper accepts iNemo's native units
# directly (g and dps). All conversion happens in the caller before update().


# ─────────────────────────────────────────────────────────────────────────────
# Main wrapper class
# ─────────────────────────────────────────────────────────────────────────────

class iNemoEngine:
    """
    iNemo Engine Plus sensor fusion, wrapped for Python.

    Manages the dylib lifecycle, ctypes structs, and unit conversion.
    All inputs and outputs use SI / degree units — no ctypes in calling code.

    Parameters:
        dylib_path  path to libiNemoEnginePlus.dylib
        kappa_acc   accelerometer trust weight [0-1]
                    0.02 = default (conservative), 0.3 = recommended for arm
                    higher = faster convergence but noisier output
        acc_ref     accelerometer axis reference frame string
                    "ENU" = East-North-Up (try first)
                    "NED" = North-East-Down (try if roll/pitch inverted)
        verbose     print configuration details on init
    """

    def __init__(
        self,
        dylib_path: str = "libiNemoEnginePlus.dylib",
        kappa_acc:  float = 0.3,
        acc_ref:    str   = "ENU",
        verbose:    bool  = True,
    ):
        self._lib      = None
        self._instance = None
        self._inp      = _InemoInput()
        self._out      = _InemoOutput()
        self._gbias    = (ctypes.c_float * 3)(0.0, 0.0, 0.0)

        # Load library
        self._lib = self._load_lib(dylib_path)

        # Create instance
        self._instance = self._lib.inemo_init(INEMO_6X)
        if not self._instance:
            raise RuntimeError("inemo_init() returned NULL — library may be corrupt or wrong architecture")

        # Print version
        if verbose:
            ver_buf = ctypes.create_string_buffer(35)
            self._lib.inemo_get_version(ver_buf)
            print(f"[iNemo] Version   : {ver_buf.value.decode('utf-8', errors='ignore')}")

        # Enable euler output (disabled by default)
        self._lib.inemo_set_enable_euler(self._instance, INEMO_ENABLE)

        # Enable gyro bias calibration (removes your gy≈-840mdps drift automatically)
        self._lib.inemo_set_enable_gbias(self._instance, INEMO_ENABLE)

        # Tune kappa_acc
        conf = _InemoConf()
        self._lib.inemo_get_conf(self._instance, ctypes.byref(conf))
        if verbose:
            print(f"[iNemo] kappa_acc : {conf.kappa_acc:.4f} → {kappa_acc:.4f}")
        conf.kappa_acc = kappa_acc
        self._lib.inemo_set_conf(self._instance, ctypes.byref(conf))

        # Set axis reference frame
        self._lib.inemo_set_acc_ref(self._instance, acc_ref.encode())
        self._lib.inemo_set_gyr_ref(self._instance, acc_ref.encode())
        if verbose:
            print(f"[iNemo] Axis ref  : {acc_ref}")
            print(f"[iNemo] Mode      : 6X (accel + gyro, no magnetometer)")
            print(f"[iNemo] Ready.")

        # Zero magnetometer — not used in 6X mode, must be set once
        self._inp.mag[0] = 0.0
        self._inp.mag[1] = 0.0
        self._inp.mag[2] = 0.0

        # FILE* for inemo_run debug logging.
        # Passing Python None can segfault on ARM64 Mac — the library may
        # dereference the pointer without checking NULL.
        # Open /dev/null as a safe sink, or use explicit c_void_p(0) as fallback.
        self._fp = ctypes.c_void_p(0)   # explicit NULL
        try:
            # Try opening /dev/null via libc — safest option
            self._libc = ctypes.CDLL(None)  # loads default libc
            self._libc.fopen.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
            self._libc.fopen.restype  = ctypes.c_void_p
            self._devnull = self._libc.fopen(b"/dev/null", b"w")
            if self._devnull:
                self._fp = self._devnull
                if verbose:
                    print("[iNemo] FILE* → /dev/null (safe sink)")
            else:
                if verbose:
                    print("[iNemo] FILE* → NULL (fopen failed, using fallback)")
        except Exception:
            if verbose:
                print("[iNemo] FILE* → NULL (libc not available, using fallback)")


    def update(
        self,
        ax: float, ay: float, az: float,   # g    (iNemo native units)
        gx: float, gy: float, gz: float,   # dps  (iNemo native units)
        dt: float,                          # seconds
    ) -> tuple[float, float, float]:
        """
        Run one step of the iNemo fusion algorithm.

        Accepts iNemo's native units directly — no conversion performed here.
        The caller is responsible for converting before calling:
            SensorTile mg  → g    (÷1000)
            SensorTile mdps → dps (÷1000)

        Args:
            ax, ay, az  accelerometer in g    (NOT mg, NOT m/s²)
            gx, gy, gz  gyroscope    in dps   (NOT mdps, NOT rad/s)
            dt          timestep     in seconds

        Returns:
            (roll, pitch, yaw) in degrees
            roll  — positive = tray tilting right
            pitch — positive = tray tilting forward
            yaw   — rotation around vertical (not used for balancing)
        """
        self._inp.acc[0] = ax
        self._inp.acc[1] = ay
        self._inp.acc[2] = az

        self._inp.gyr[0] = gx
        self._inp.gyr[1] = gy
        self._inp.gyr[2] = gz

        self._inp.dtime = dt

        self._lib.inemo_run(
            self._instance,
            ctypes.byref(self._out),
            ctypes.byref(self._inp),
            self._fp,   # FILE* — /dev/null or explicit NULL
        )

        # inemo_output.euler = [yaw, pitch, roll] per header
        # BUT with ENU axis ref and our IMU mounting, euler[1] maps to
        # physical roll and euler[2] maps to physical pitch.
        # Confirmed by tilting tray forward → euler[2] changes, not euler[1].
        yaw   = self._out.euler[0]
        roll  = self._out.euler[1]   # physical roll  (left/right tilt)
        pitch = self._out.euler[2]   # physical pitch (forward/back tilt)

        return roll, pitch, yaw


    def get_bias(self) -> tuple[float, float, float]:
        """
        Return current gyro bias estimates in dps.
        Watch these converge toward 0 during the startup hold phase.
        When |bias| < 0.5 dps the filter is considered settled.
        """
        self._lib.inemo_get_gbias(self._instance, self._gbias)
        return float(self._gbias[0]), float(self._gbias[1]), float(self._gbias[2])


    def get_gravity(self) -> tuple[float, float, float]:
        """
        Return the estimated gravity vector in g.
        Should be approximately (0, 0, -1) when flat.
        Useful for verifying axis convention is correct.
        """
        return (
            float(self._out.gravity[0]),
            float(self._out.gravity[1]),
            float(self._out.gravity[2]),
        )


    def get_quaternion(self) -> tuple[float, float, float, float]:
        """Return orientation quaternion [qx, qy, qz, qw]."""
        return tuple(float(self._out.quaternion[i]) for i in range(4))


    def is_bias_converged(self, threshold_dps: float = 0.5) -> bool:
        """
        Return True when gyro bias magnitude is below threshold.
        Use this in the startup sequence to know when the filter has settled.
        """
        gx, gy, gz = self.get_bias()
        return math.sqrt(gx**2 + gy**2 + gz**2) < threshold_dps


    def close(self) -> None:
        """
        Mark instance as closed. Safe to call multiple times.

        Does NOT call inemo_deinit() — that function has a malloc bug on
        ARM64 Mac that causes "pointer being freed was not allocated" abort.
        Since the iNemo instance lives for the entire process lifetime,
        skipping deinit is safe — the OS reclaims all memory on exit.
        """
        self._instance = None


    # No __del__ — calling close() during Python shutdown causes malloc errors
    # because the ctypes library may already be unloaded. Call close() explicitly
    # in your finally block instead.


    @staticmethod
    def _load_lib(dylib_path: str) -> ctypes.CDLL:
        """Load the dylib and set function signatures for all used functions."""
        if not os.path.exists(dylib_path):
            raise FileNotFoundError(
                f"dylib not found: {dylib_path}\n"
                f"Copy libiNemoEnginePlus.dylib to the same directory as your script."
            )

        lib = ctypes.CDLL(dylib_path)

        # inemo *inemo_init(inemo_mode)
        lib.inemo_init.argtypes = [ctypes.c_int]
        lib.inemo_init.restype  = ctypes.c_void_p

        # void inemo_deinit(inemo*)
        lib.inemo_deinit.argtypes = [ctypes.c_void_p]
        lib.inemo_deinit.restype  = None

        # void inemo_run(inemo*, inemo_output*, const inemo_input*, FILE*)
        lib.inemo_run.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(_InemoOutput),
            ctypes.POINTER(_InemoInput),
            ctypes.c_void_p,
        ]
        lib.inemo_run.restype = None

        # void inemo_get_conf(inemo*, inemo_conf*)
        lib.inemo_get_conf.argtypes = [ctypes.c_void_p, ctypes.POINTER(_InemoConf)]
        lib.inemo_get_conf.restype  = None

        # void inemo_set_conf(inemo*, inemo_conf*)
        lib.inemo_set_conf.argtypes = [ctypes.c_void_p, ctypes.POINTER(_InemoConf)]
        lib.inemo_set_conf.restype  = None

        # void inemo_set_enable_gbias(inemo*, inemo_enable)
        lib.inemo_set_enable_gbias.argtypes = [ctypes.c_void_p, ctypes.c_int]
        lib.inemo_set_enable_gbias.restype  = None

        # void inemo_get_gbias(inemo*, float*)
        lib.inemo_get_gbias.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float)]
        lib.inemo_get_gbias.restype  = None

        # void inemo_set_enable_euler(inemo*, inemo_enable)
        lib.inemo_set_enable_euler.argtypes = [ctypes.c_void_p, ctypes.c_int]
        lib.inemo_set_enable_euler.restype  = None

        # uint8_t inemo_get_version(char*)
        lib.inemo_get_version.argtypes = [ctypes.c_char_p]
        lib.inemo_get_version.restype  = ctypes.c_uint8

        # void inemo_set_acc_ref(inemo*, const char[4])
        lib.inemo_set_acc_ref.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        lib.inemo_set_acc_ref.restype  = None

        # void inemo_set_gyr_ref(inemo*, const char[4])
        lib.inemo_set_gyr_ref.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        lib.inemo_set_gyr_ref.restype  = None

        return lib