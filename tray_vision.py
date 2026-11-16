"""Tray-frame vision: locate the weight in millimetres relative to the tray centre.

Pipeline:
    full-res frame
      -> detect the ArUco corner markers (any one may be hidden under the weight)
      -> homography to tray coordinates (mm, origin = tray centre): from the 4
         marker centres when all are visible, otherwise fit to the corners of the
         3 visible markers using the layout learned from 4-marker frames
      -> rectified top-down view of the tray
      -> weight = ArUco id 4 stuck on its top face, mapped through the homography
         (legacy fallback: background-difference blob, WEIGHT_DETECT = "blob")

Tray frame convention (as seen in the rectified view):
    +x to the right, +y downward, origin at the tray centre, units mm.

The marker-quad side is self-calibrated from the known 25 mm marker size, so the
markers did not need to be placed at precise positions.
"""

import cv2
import cv2.aruco as aruco
import math
import numpy as np
import os
import threading
import time

# ── Geometry ─────────────────────────────────────────────────────────────────
MARKER_MM = 25.0
TRAY_MM = 150.0

# Corner -> marker id, matching how the markers were actually stuck on the tray
# (clockwise from top-left: 3, 0, 1, 2).
CORNER_IDS = {"TL": 3, "TR": 0, "BR": 1, "BL": 2}
CORNER_ORDER = ["TL", "TR", "BR", "BL"]

# ── Rectified output ─────────────────────────────────────────────────────────
# The view deliberately extends past the tray. Rectifying exactly 150 mm meant a
# weight resting at the edge was half outside the image and simply vanished -
# 44% of frames were lost in one run with the weight sitting at r = 75-82 mm.
PX_PER_MM = 3
VIEW_MM = 190.0
RECT_PX = int(VIEW_MM * PX_PER_MM)
# The homography origin/scale comes from the marker CENTRES, but the tray itself
# extends to the markers' outer edges and beyond. The acceptance region is
# therefore the full tray (outer marker edge or TRAY_MM, whichever is larger)
# plus an overhang allowance, not the marker-centre quad.
WEIGHT_OVERHANG_MM = 20.0
VALID_HALF_MM = TRAY_MM / 2.0 + WEIGHT_OVERHANG_MM   # default before the quad is measured


def valid_half_mm(quad_mm):
    """Half-width (mm, from tray centre) inside which a weight detection is accepted."""
    if quad_mm is None:
        return VALID_HALF_MM
    tray_half = max(TRAY_MM / 2.0, quad_mm / 2.0 + MARKER_MM / 2.0)
    return tray_half + WEIGHT_OVERHANG_MM

# ── Weight marker ────────────────────────────────────────────────────────────
# The blob detector below locked onto edges, shading and glare even with an empty
# tray. A marker on the weight is unambiguous: either id 4 decodes or the weight
# is reported missing. Made with generate_weight_aruco.py.
WEIGHT_DETECT = "aruco"       # "aruco" or "blob" (legacy background differencing)
WEIGHT_ID = 4
WEIGHT_MARKER_MM = 14.0
# Reject decodes whose size in the tray plane is implausible (the marker sits on
# top of the weight, so parallax makes it read slightly larger than 14 mm).
WEIGHT_MARKER_MM_RANGE = (8.0, 24.0)

# ── Weight detection (legacy blob) ───────────────────────────────────────────
# The tray is rigidly mounted to the wrist, so the gripper, screws and any
# reflections are static in the tray frame and are removed with a one-time
# empty-tray reference.
#
# The weight is polished metal: sometimes it catches a highlight and reads
# bright, sometimes it reflects something dark. Testing only for "brighter"
# loses it whenever the highlight moves off, so the test is two-sided and the
# size gate does the rejecting instead — pose-dependent shading produces blobs
# of >1000 mm2, while the weight is ~45-90 mm2.
BG_PATH = "tray_bg.png"
DIFF_THRESH = 40              # grey levels, either direction
# The lit area varies enormously with pose: ~44 mm2 when only a specular streak
# catches the light, ~550 mm2 when the whole top face is lit. Area alone is
# therefore a weak discriminator, so the real gate is physical size - the weight
# is a ~20-25 mm disc, while the shading artefacts that caused false positives
# measured 64 x 37 mm.
MIN_BLOB_MM2 = 10.0
MAX_BLOB_MM2 = 900.0
MAX_BLOB_DIM_MM = 35.0        # longest bounding-box side
EDGE_MARGIN_MM = 3.0

# ── Tracking ───────────────────────────────────────────────────────────────────
# Tilt-dependent rim shadows occasionally pass the size gate and appear as a
# second "weight" on the far side of the tray. Position alone cannot tell them
# apart, but motion can: the real weight cannot teleport across the tray between
# frames, so candidates implying an impossible speed are rejected.
MAX_SPEED_MM_S = 350.0
MIN_GATE_MM = 12.0            # always allow at least this much motion per frame
REACQUIRE_S = 1.0             # after this long with no match, accept any blob again

_detector = aruco.ArucoDetector(
    aruco.getPredefinedDictionary(aruco.DICT_4X4_50),
    aruco.DetectorParameters(),
)


def _detect_by_id(frame):
    """Return {marker_id: (4,2) corner points} for every decoded marker."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = _detector.detectMarkers(gray)
    if ids is None:
        return {}
    return {int(i): c[0] for c, i in zip(corners, ids.flatten())}


def _corners_from(by_id):
    return {name: by_id[mid] for name, mid in CORNER_IDS.items() if mid in by_id}


def detect_markers(frame):
    """Return {corner_name: (4,2) marker corner points} for markers that were found."""
    return _corners_from(_detect_by_id(frame))


def find_weight_marker(by_id, H, half_mm=VALID_HALF_MM):
    """Locate the weight's ArUco marker in tray mm.

    The marker is decoded on the full camera frame, so the only position gate is
    the tray extent `half_mm`; the decoded id plus the size check reject junk.

    Returns (x_mm, y_mm, candidates); candidates holds at most one
    (area_mm2, x_mm, y_mm, ok) entry so callers can display it like a blob.
    """
    pts = by_id.get(WEIGHT_ID)
    if pts is None:
        return None, None, []

    q = cv2.perspectiveTransform(pts.reshape(-1, 1, 2).astype(np.float32), H).reshape(-1, 2)
    x, y = (float(v) for v in q.mean(axis=0))
    side = float(np.mean([np.linalg.norm(q[k] - q[(k + 1) % 4]) for k in range(4)]))
    lo, hi = WEIGHT_MARKER_MM_RANGE
    ok = (lo <= side <= hi
          and abs(x) <= half_mm and abs(y) <= half_mm)
    cand = [(side * side, x, y, ok)]
    return (x, y, cand) if ok else (None, None, cand)


# The weight can slide over a corner marker, but it is too small to cover two at
# once, so the tray frame must survive any single marker being hidden.
MIN_MARKERS = 3

# Every marker corner's position in tray mm, learned from frames where all four
# markers are visible. The tray is rigid, so this layout does not change with
# pose and lets the homography be solved from whichever markers remain visible.
_LAYOUT_ALPHA = 0.2
_layout = {}
_layout_quad_mm = None


def _homography_from_centres(found):
    """Square-quad homography from marker centres (3 or 4 markers).

    With 3 markers the hidden centre is completed as a parallelogram, which is
    only approximate under perspective - used until a layout has been learned.
    """
    names = [n for n in CORNER_ORDER if n in found]
    centres = {n: found[n].mean(axis=0) for n in names}
    if len(names) == 3:
        i = next(k for k, n in enumerate(CORNER_ORDER) if n not in found)
        prev_, next_, opp = (CORNER_ORDER[(i + d) % 4] for d in (-1, 1, 2))
        centres[CORNER_ORDER[i]] = centres[prev_] + centres[next_] - centres[opp]
    centres = np.array([centres[n] for n in CORNER_ORDER], np.float32)

    marker_side_px = np.mean([
        np.linalg.norm(found[n][k] - found[n][(k + 1) % 4])
        for n in names for k in range(4)
    ])
    px_per_mm = marker_side_px / MARKER_MM

    quad_side_px = np.mean([
        np.linalg.norm(centres[i] - centres[(i + 1) % 4]) for i in range(4)
    ])
    half = quad_side_px / px_per_mm / 2.0

    dst_mm = np.array([[-half, -half], [half, -half],
                       [half, half], [-half, half]], np.float32)

    H = cv2.getPerspectiveTransform(centres, dst_mm)
    return H, half * 2.0


def _learn_layout(found, H, quad_mm):
    global _layout_quad_mm
    for n, pts in found.items():
        mm = cv2.perspectiveTransform(pts.reshape(-1, 1, 2).astype(np.float32), H).reshape(-1, 2)
        _layout[n] = mm if n not in _layout else (1 - _LAYOUT_ALPHA) * _layout[n] + _LAYOUT_ALPHA * mm
    _layout_quad_mm = (quad_mm if _layout_quad_mm is None
                       else (1 - _LAYOUT_ALPHA) * _layout_quad_mm + _LAYOUT_ALPHA * quad_mm)


def tray_homography(found):
    """Homography mapping image pixels -> tray millimetres, or (None, None).

    Needs at least MIN_MARKERS markers, so one may be hidden under the weight.
    With all four, the quad's physical side is derived from the known marker
    size (no hand measurement of the marker inset needed) and the layout is
    learned. With three, the homography is fit to every visible marker corner
    against that learned layout.
    """
    if len(found) < MIN_MARKERS:
        return None, None

    if len(found) == 4:
        H, quad_mm = _homography_from_centres(found)
        _learn_layout(found, H, quad_mm)
        return H, quad_mm

    if len(_layout) == 4:
        names = [n for n in CORNER_ORDER if n in found]
        src = np.concatenate([found[n] for n in names]).astype(np.float32)
        dst = np.concatenate([_layout[n] for n in names]).astype(np.float32)
        H, _ = cv2.findHomography(src, dst, 0)
        if H is not None:
            return H, float(_layout_quad_mm)

    return _homography_from_centres(found)


def _mm_to_px():
    return np.array([[PX_PER_MM, 0, RECT_PX / 2],
                     [0, PX_PER_MM, RECT_PX / 2],
                     [0, 0, 1]], np.float32)


def rectify(frame, H):
    """Warp the tray into a square top-down view, VIEW_MM across."""
    return cv2.warpPerspective(frame, _mm_to_px() @ H, (RECT_PX, RECT_PX))


def marker_polys_px(found, H, grow=1.45):
    """Detected marker quads in rectified pixels, enlarged to cover the quiet zone.

    Background subtraction does not cancel the markers: a sub-pixel homography
    error shifts their black/white edges and leaves small high-contrast blobs
    that pass the weight size gate. They sit at r = 59-83 mm, which is exactly
    where phantom detections kept appearing, so they are masked out explicitly.
    """
    A = _mm_to_px() @ H
    polys = []
    for pts in found.values():
        q = cv2.perspectiveTransform(pts.reshape(-1, 1, 2).astype(np.float32), A)
        q = q.reshape(-1, 2)
        centre = q.mean(axis=0)
        polys.append(np.round(centre + (q - centre) * grow).astype(np.int32))
    return polys


def find_weight(rect, background, exclude_polys=None, half_mm=VALID_HALF_MM):
    """Locate the weight in the rectified tray view by differencing against
    the empty-tray reference.

    Returns (x_mm, y_mm, mask, candidates) with the position relative to the
    tray centre, or (None, None, mask, candidates) if nothing plausible was found.
    """
    gray = cv2.cvtColor(rect, cv2.COLOR_BGR2GRAY).astype(np.int16)

    valid = np.zeros(gray.shape, np.uint8)
    c = RECT_PX // 2
    half = min(int(half_mm * PX_PER_MM), c)
    cv2.rectangle(valid, (c - half, c - half), (c + half, c + half), 255, -1)
    if exclude_polys:
        cv2.fillPoly(valid, exclude_polys, 0)

    brighter = cv2.GaussianBlur(
        np.abs(gray - background.astype(np.int16)).astype(np.float32), (5, 5), 0)
    mask = ((brighter >= DIFF_THRESH) & (valid > 0)).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))

    n, _, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
    best, best_area = None, 0.0
    candidates = []
    for i in range(1, n):
        area_mm2 = stats[i, cv2.CC_STAT_AREA] / (PX_PER_MM ** 2)
        if area_mm2 < 5.0:
            continue
        w_mm = stats[i, cv2.CC_STAT_WIDTH] / PX_PER_MM
        h_mm = stats[i, cv2.CC_STAT_HEIGHT] / PX_PER_MM
        cx, cy = centroids[i]
        ok = (MIN_BLOB_MM2 <= area_mm2 <= MAX_BLOB_MM2
              and max(w_mm, h_mm) <= MAX_BLOB_DIM_MM)
        candidates.append((area_mm2,
                           (cx - RECT_PX / 2) / PX_PER_MM,
                           (cy - RECT_PX / 2) / PX_PER_MM,
                           ok))
        if ok and area_mm2 > best_area:
            best, best_area = i, area_mm2

    if best is None:
        return None, None, mask, candidates

    cx, cy = centroids[best]
    return ((cx - RECT_PX / 2) / PX_PER_MM,
            (cy - RECT_PX / 2) / PX_PER_MM, mask, candidates)


def _marker_centre_px(name):
    """Nominal marker centre in the rectified view (markers sit on a centred square)."""
    sign = {"TL": (-1, -1), "TR": (1, -1), "BR": (1, 1), "BL": (-1, 1)}[name]
    off = _marker_centre_px.half_mm * PX_PER_MM
    return int(RECT_PX / 2 + sign[0] * off), int(RECT_PX / 2 + sign[1] * off)


_marker_centre_px.half_mm = 50.0   # updated per-frame by process()


def load_background(path=BG_PATH):
    bg = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if bg is None:
        return None
    if bg.shape[0] != RECT_PX:
        # Resizing would stretch a different field of view onto this one and
        # silently misalign every pixel, so refuse instead.
        raise SystemExit(
            f"{path} was captured at {bg.shape[0]}px but the view is now {RECT_PX}px.\n"
            f"Re-run:  python tray_vision.py --calibrate   (with an empty tray)")
    return bg


def save_background(frame, path=BG_PATH):
    """Rectify an empty-tray frame and store it as the reference. Returns True on success."""
    found = detect_markers(frame)
    if MIN_MARKERS <= len(found) < 4:
        print(f"    [warn] {len(found)}/4 tray markers visible — capturing the reference "
              f"from {len(found)}; an empty tray with all 4 visible is more accurate")
    H, quad_mm = tray_homography(found)
    if H is None:
        return False
    rect = rectify(frame, H)
    cv2.imwrite(path, cv2.cvtColor(rect, cv2.COLOR_BGR2GRAY))
    return True


def process(frame, background=None):
    """Full pipeline on one frame. Returns a result dict."""
    by_id = _detect_by_id(frame)
    found = _corners_from(by_id)
    out = {"n_markers": len(found), "found": found, "candidates": [],
           "marker_polys": [], "mask": None,
           "x_mm": None, "y_mm": None, "rect": None, "quad_mm": None,
           "valid_half_mm": VALID_HALF_MM}
    if len(found) < MIN_MARKERS:
        return out

    H, quad_mm = tray_homography(found)
    if H is None:
        return out
    _marker_centre_px.half_mm = quad_mm / 2.0
    half_mm = valid_half_mm(quad_mm)

    rect = rectify(frame, H)
    polys = marker_polys_px(found, H)
    out.update(rect=rect, quad_mm=quad_mm, marker_polys=polys, valid_half_mm=half_mm)

    if WEIGHT_DETECT == "aruco":
        x_mm, y_mm, candidates = find_weight_marker(by_id, H, half_mm)
        out.update(x_mm=x_mm, y_mm=y_mm, candidates=candidates)
        return out

    if background is None:
        background = load_background()
    if background is None:
        return out

    x_mm, y_mm, mask, candidates = find_weight(rect, background, polys, half_mm)
    out.update(mask=mask, x_mm=x_mm, y_mm=y_mm, candidates=candidates)
    return out


def annotate(rect, x_mm, y_mm):
    """Draw tray axes and the detected weight onto a copy of the rectified view."""
    vis = rect.copy()
    c = RECT_PX // 2
    cv2.line(vis, (c, 0), (c, RECT_PX), (0, 180, 0), 1)
    cv2.line(vis, (0, c), (RECT_PX, c), (0, 180, 0), 1)
    cv2.circle(vis, (c, c), 4, (0, 255, 0), -1)

    if x_mm is not None:
        px = int(c + x_mm * PX_PER_MM), int(c + y_mm * PX_PER_MM)
        cv2.circle(vis, px, 10, (0, 0, 255), 2)
        cv2.arrowedLine(vis, px, (c, c), (0, 0, 255), 2, tipLength=0.08)
        cv2.putText(vis, f"({x_mm:+.1f}, {y_mm:+.1f}) mm  r={np.hypot(x_mm, y_mm):.1f}",
                    (8, RECT_PX - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    else:
        cv2.putText(vis, "weight not found", (8, RECT_PX - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    return vis


def _grab(index=0):
    """Single full-resolution frame from the fixed camera."""
    cap = cv2.VideoCapture(index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    for _ in range(15):
        cap.read()
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"could not read from camera {index}")
    return frame


class VisionThread(threading.Thread):
    """Continuous tray detection, so a 50 Hz servo loop never blocks on a frame."""

    STALE_S = 0.5
    DEBUG_DIR = "/tmp/tray_dropouts"
    MAX_DEBUG_FRAMES = 8

    def __init__(self, cam_index=0, debug_dropouts=False):
        super().__init__(daemon=True, name="tray-vision")
        self.cap = cv2.VideoCapture(cam_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        if not self.cap.isOpened():
            raise RuntimeError(f"could not open camera {cam_index}")

        self.background = load_background() if WEIGHT_DETECT == "blob" else None
        if WEIGHT_DETECT == "blob" and self.background is None:
            raise RuntimeError(f"no {BG_PATH} — run: python tray_vision.py --calibrate")

        self.lock = threading.Lock()
        self.xy = None
        self.stamp = 0.0
        self.rect = None
        self.n_markers = 0
        self.marker_stamp = 0.0
        self.candidates = []
        self.running = True
        self.debug_dropouts = debug_dropouts
        self._debug_saved = 0
        self._lost_since = None
        self._track = None          # last accepted (x_mm, y_mm)
        self._track_stamp = 0.0
        self.rejected = 0
        if debug_dropouts:
            os.makedirs(self.DEBUG_DIR, exist_ok=True)

    def _select(self, candidates, now):
        """Pick the candidate consistent with where the weight was last seen."""
        usable = [(a, x, y) for a, x, y, ok in candidates if ok]
        if not usable:
            return None

        # A decoded marker id is unambiguous, so the anti-teleport gate is only
        # needed for blob detection.
        if WEIGHT_DETECT == "aruco":
            _, x, y = usable[0]
            return x, y

        if self._track is None or now - self._track_stamp > REACQUIRE_S:
            a, x, y = max(usable, key=lambda c: c[0])
            return x, y

        gate = max(MIN_GATE_MM, MAX_SPEED_MM_S * (now - self._track_stamp))
        px, py = self._track
        near = [(math.hypot(x - px, y - py), x, y) for _, x, y in usable]
        near.sort()
        if near[0][0] > gate:
            self.rejected += 1
            return None
        return near[0][1], near[0][2]

    def run(self):
        while self.running:
            ok, frame = self.cap.read()
            if not ok:
                continue
            r = process(frame, self.background)
            now = time.monotonic()
            sel = self._select(r["candidates"], now) if r["rect"] is not None else None
            if sel is not None:
                self._track, self._track_stamp = sel, now
            r = dict(r, x_mm=sel[0] if sel else None, y_mm=sel[1] if sel else None)
            with self.lock:
                self.n_markers = r["n_markers"]
                self.candidates = r["candidates"]
                self.marker_stamp = now
                if r["rect"] is not None:
                    self.rect = r["rect"]
                if sel is not None:
                    self.xy = sel
                    self.stamp = now
            self._maybe_save_dropout(r)

    def _maybe_save_dropout(self, r):
        """Dump the rectified view when the weight has been missing for a while.

        Detection failures are otherwise invisible after the fact, and the frames
        where it fails are exactly the ones not sitting on disk.
        """
        if not self.debug_dropouts or self._debug_saved >= self.MAX_DEBUG_FRAMES:
            return
        now = time.monotonic()
        if r["x_mm"] is not None:
            self._lost_since = None
            return
        if self._lost_since is None:
            self._lost_since = now
            return
        if now - self._lost_since < 1.0:
            return

        self._debug_saved += 1
        self._lost_since = None
        tag = f"{self.DEBUG_DIR}/drop{self._debug_saved:02d}"
        if r["rect"] is not None:
            cv2.imwrite(f"{tag}_rect.jpg", r["rect"])
        if r.get("mask") is not None:
            cv2.imwrite(f"{tag}_mask.png", r["mask"])
        print(f"    [debug] saved {tag}_* ({self.status()})")

    def read(self):
        """Latest (x_mm, y_mm), or None if stale."""
        xy, _ = self.read_stamped()
        return xy

    def read_stamped(self):
        """Latest ((x_mm, y_mm), timestamp), or (None, 0.0) if stale."""
        with self.lock:
            if self.xy is None or time.monotonic() - self.stamp > self.STALE_S:
                return None, 0.0
            return self.xy, self.stamp

    def read_rect(self):
        """Latest rectified top-down tray image (BGR numpy array) or None."""
        with self.lock:
            return None if self.rect is None else self.rect.copy()

    def markers_ok(self):
        """True if enough tray markers are visible to locate the tray (one may be hidden)."""
        with self.lock:
            return (self.n_markers >= MIN_MARKERS
                    and time.monotonic() - self.marker_stamp < self.STALE_S)

    def status(self):
        """Human-readable detector state, for diagnosing a missing detection."""
        with self.lock:
            n, cands = self.n_markers, list(self.candidates)
        if n < MIN_MARKERS:
            return f"markers {n}/4 visible (need {MIN_MARKERS})"
        m = f"markers {n}/4"
        if WEIGHT_DETECT == "aruco":
            if not cands:
                return f"{m}, weight marker id {WEIGHT_ID} not decoded"
            a, x, y, ok = cands[0]
            return (f"{m}, weight marker {math.sqrt(a):.1f}mm side "
                    f"@({x:+.0f},{y:+.0f}){'' if ok else ' REJECTED'}")
        if not cands:
            return f"{m}, no blob at all (weight not seen)"
        blobs = ", ".join(f"{a:.0f}mm2@({x:+.0f},{y:+.0f}){'' if ok else ' REJECTED'}"
                          for a, x, y, ok in cands[:4])
        return f"{m}, blobs: {blobs}"

    def stop(self):
        self.running = False
        time.sleep(0.1)
        self.cap.release()


if __name__ == "__main__":
    import sys

    args = sys.argv[1:]

    if args and args[0] == "--calibrate":
        src = args[1] if len(args) > 1 else None
        frame = cv2.imread(src) if src else _grab()
        ok = save_background(frame)
        print(f"background saved to {BG_PATH}" if ok else
              f"calibration failed - at least {MIN_MARKERS} of the 4 markers must be visible")
        raise SystemExit(0 if ok else 1)

    src = args[0] if args else None
    frame = cv2.imread(src) if src else _grab()
    r = process(frame)
    print(f"markers found : {r['n_markers']}")
    if r["rect"] is None:
        raise SystemExit(f"need at least {MIN_MARKERS} of the 4 tray markers")
    print(f"marker quad   : {r['quad_mm']:.1f} mm")
    print(f"weight        : ({r['x_mm']:+.1f}, {r['y_mm']:+.1f}) mm"
          if r["x_mm"] is not None else "weight        : not found")
    cv2.imwrite("/tmp/tray_rect.jpg", annotate(r["rect"], r["x_mm"], r["y_mm"]))
    print("wrote /tmp/tray_rect.jpg")
