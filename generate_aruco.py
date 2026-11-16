"""Generate a printable ArUco marker sheet for the tray corners.

Markers are DICT_4X4_50, ids 0-3, laid out so id order is TL, TR, BR, BL
when stuck on the tray. Three candidate sizes are printed so you can pick
whichever fits the tray rim; use the same size for all four corners.

Output: aruco_sheet.png (US Letter @ 300 DPI, print at 100% / "actual size").
"""

import cv2
import cv2.aruco as aruco
import numpy as np

DPI = 300
PAGE_W_IN, PAGE_H_IN = 8.5, 11.0
DICT = aruco.DICT_4X4_50
IDS = [0, 1, 2, 3]
CORNER_NAMES = ["TL", "TR", "BR", "BL"]
SIZES_MM = [20, 25, 30]
QUIET_ZONE_CELLS = 1      # white border required for reliable detection

MM_PER_IN = 25.4


def mm_to_px(mm):
    return int(round(mm / MM_PER_IN * DPI))


def make_marker(dictionary, marker_id, side_px):
    """Marker bitmap plus the white quiet zone ArUco detection requires."""
    img = aruco.generateImageMarker(dictionary, marker_id, side_px)
    cell = side_px // 6          # 4x4 marker + 1 cell black border each side
    pad = cell * QUIET_ZONE_CELLS
    return cv2.copyMakeBorder(img, pad, pad, pad, pad,
                              cv2.BORDER_CONSTANT, value=255)


def main():
    dictionary = aruco.getPredefinedDictionary(DICT)
    page = np.full((mm_to_px(PAGE_H_IN * MM_PER_IN),
                    mm_to_px(PAGE_W_IN * MM_PER_IN)), 255, np.uint8)

    y = mm_to_px(15)
    for size_mm in SIZES_MM:
        side_px = mm_to_px(size_mm)
        cv2.putText(page, f"{size_mm} mm", (mm_to_px(12), y - mm_to_px(4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, 0, 2)

        x = mm_to_px(10)
        for marker_id, name in zip(IDS, CORNER_NAMES):
            tile = make_marker(dictionary, marker_id, side_px)
            h, w = tile.shape
            page[y:y + h, x:x + w] = tile
            cv2.rectangle(page, (x, y), (x + w, y + h), 200, 1)   # cut guide
            cv2.putText(page, f"id{marker_id} {name}", (x, y + h + mm_to_px(6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, 0, 1)
            x += w + mm_to_px(10)

        y += mm_to_px(size_mm + QUIET_ZONE_CELLS * size_mm / 3) + mm_to_px(26)

    # 50 mm reference bar to verify the printer didn't rescale the page
    bar_y = y + mm_to_px(10)
    cv2.rectangle(page, (mm_to_px(12), bar_y),
                  (mm_to_px(62), bar_y + mm_to_px(4)), 0, -1)
    cv2.putText(page, "this bar must measure exactly 50 mm",
                (mm_to_px(66), bar_y + mm_to_px(4)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, 0, 1)

    cv2.imwrite("aruco_sheet.png", page)

    detector = aruco.ArucoDetector(dictionary, aruco.DetectorParameters())
    found, _, _ = detector.detectMarkers(page)
    print(f"self-test: decoded {len(found)} markers on the sheet (expect 12)")
    print("wrote aruco_sheet.png — print at 100% scale, then measure the 50 mm bar")


if __name__ == "__main__":
    main()
