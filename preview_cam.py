"""Live webcam preview for aiming the fixed camera. Press q to quit, s to save a snapshot."""

import cv2

CAM_INDEX = 0
CAPTURE_W, CAPTURE_H, FPS = 1920, 1080, 30

# Crop region in full-res capture coords, then resized to OUT_W x OUT_H.
# Tuned with the arm in its dashboard.py running pose, not the home pose.
# Must stay identical between recording and inference.
CROP = (200, 230, 1520, 1020)   # x0, y0, x1, y1
OUT_W, OUT_H = 640, 480

cap = cv2.VideoCapture(CAM_INDEX)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAPTURE_W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_H)
cap.set(cv2.CAP_PROP_FPS, FPS)

if not cap.isOpened():
    raise SystemExit(f"could not open camera index {CAM_INDEX}")

print(f"camera {CAM_INDEX}: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x"
      f"{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))} @ {cap.get(cv2.CAP_PROP_FPS)}fps")
print("q = quit, s = save snapshot, arrows/WASD = nudge crop, [ ] = shrink/grow crop")

x0, y0, x1, y1 = CROP

while True:
    ok, frame = cap.read()
    if not ok:
        break

    h, w = frame.shape[:2]
    x0, x1 = max(0, x0), min(w, x1)
    y0, y1 = max(0, y0), min(h, y1)

    crop = cv2.resize(frame[y0:y1, x0:x1], (OUT_W, OUT_H))

    # thirds grid to help center the arm in frame
    for x in (OUT_W // 3, 2 * OUT_W // 3):
        cv2.line(crop, (x, 0), (x, OUT_H), (0, 255, 0), 1)
    for y in (OUT_H // 3, 2 * OUT_H // 3):
        cv2.line(crop, (0, y), (OUT_W, y), (0, 255, 0), 1)
    cv2.putText(crop, f"CROP = ({x0}, {y0}, {x1}, {y1})", (8, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    full = frame.copy()
    cv2.rectangle(full, (x0, y0), (x1, y1), (0, 255, 255), 3)
    cv2.imshow("full frame", cv2.resize(full, (960, 540)))
    cv2.imshow("policy view (cropped)", crop)

    key = cv2.waitKey(1) & 0xFF
    step = 20
    if key == ord("q"):
        break
    elif key in (ord("a"), 2):      # left
        x0 -= step; x1 -= step
    elif key in (ord("d"), 3):      # right
        x0 += step; x1 += step
    elif key in (ord("w"), 0):      # up
        y0 -= step; y1 -= step
    elif key in (ord("s"), 1):      # down
        y0 += step; y1 += step
    elif key == ord("["):           # zoom in
        x0 += step; x1 -= step; y0 += int(step * 0.75); y1 -= int(step * 0.75)
    elif key == ord("]"):           # zoom out
        x0 -= step; x1 += step; y0 -= int(step * 0.75); y1 += int(step * 0.75)
    elif key == ord(" "):
        cv2.imwrite("/tmp/cam_snapshot.jpg", crop)
        print(f"saved /tmp/cam_snapshot.jpg   CROP = ({x0}, {y0}, {x1}, {y1})")

cap.release()
cv2.destroyAllWindows()
