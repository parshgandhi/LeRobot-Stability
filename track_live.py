"""Live tray tracker — slide the weight by hand and watch the tray-frame estimate.

Shows the rectified tray with the detected weight, plus a running trail, so you
can confirm the estimate stays smooth and accurate across the whole tray and
while the arm moves.

Keys:
    q  quit
    b  re-capture the empty-tray background (blob mode only; take the weight off first)
    c  clear the trail
"""

import collections
import time

import cv2
import numpy as np

import tray_vision as tv

CAM_INDEX = 0
TRAIL_LEN = 60


def main():
    cap = cv2.VideoCapture(CAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    cap.set(cv2.CAP_PROP_FPS, 30)
    if not cap.isOpened():
        raise SystemExit(f"could not open camera {CAM_INDEX}")

    background = tv.load_background() if tv.WEIGHT_DETECT == "blob" else None
    if tv.WEIGHT_DETECT == "blob" and background is None:
        raise SystemExit(f"no {tv.BG_PATH} — run: python tray_vision.py --calibrate")

    trail = collections.deque(maxlen=TRAIL_LEN)
    last, fps = time.time(), 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        r = tv.process(frame, background)

        now = time.time()
        fps = 0.9 * fps + 0.1 / max(now - last, 1e-6)
        last = now

        if r["rect"] is None:
            vis = np.zeros((tv.RECT_PX, tv.RECT_PX, 3), np.uint8)
            cv2.putText(vis, f"markers: {r['n_markers']}/4 (need {tv.MIN_MARKERS})", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            if r["x_mm"] is not None:
                trail.append((r["x_mm"], r["y_mm"]))
            vis = tv.annotate(r["rect"], r["x_mm"], r["y_mm"])

            c = tv.RECT_PX // 2

            # show exactly what the detector considers, and why it rejects things
            half = int(r["valid_half_mm"] * tv.PX_PER_MM)
            cv2.rectangle(vis, (c - half, c - half), (c + half, c + half),
                          (255, 255, 0), 1)
            cv2.polylines(vis, r["marker_polys"], True, (0, 0, 255), 1)

            for area, x, y, ok in r["candidates"]:
                col = (0, 255, 0) if ok else (0, 0, 255)
                px = (int(c + x * tv.PX_PER_MM), int(c + y * tv.PX_PER_MM))
                cv2.circle(vis, px, 6, col, 1)
                cv2.putText(vis, f"{area:.0f}", (px[0] + 8, px[1] - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1)

            for i in range(1, len(trail)):
                p0 = (int(c + trail[i - 1][0] * tv.PX_PER_MM),
                      int(c + trail[i - 1][1] * tv.PX_PER_MM))
                p1 = (int(c + trail[i][0] * tv.PX_PER_MM),
                      int(c + trail[i][1] * tv.PX_PER_MM))
                cv2.line(vis, p0, p1, (255, 200, 0), 1)

            mode = (f"weight marker id {tv.WEIGHT_ID}" if tv.WEIGHT_DETECT == "aruco" else
                    f"accept {tv.MIN_BLOB_MM2:.0f}-{tv.MAX_BLOB_MM2:.0f} mm2")
            cv2.putText(vis, f"{fps:4.1f} fps   quad {r['quad_mm']:.0f} mm   {mode}",
                        (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

        cv2.imshow("tray frame", cv2.resize(vis, (tv.RECT_PX * 2, tv.RECT_PX * 2)))

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("c"):
            trail.clear()
        if key == ord("b"):
            if tv.save_background(frame):
                background = tv.load_background()
                print("background re-captured")
            else:
                print(f"re-capture failed — at least {tv.MIN_MARKERS} of the 4 markers must be visible")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
