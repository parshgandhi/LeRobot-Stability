"""Generate a printable ArUco marker sheet for the sliding weight.

Marker is DICT_4X4_50 id 4 (the tray corners use ids 0-3). The weight's top
face is a ~20-25 mm disc, so several small sizes are printed; pick the largest
one whose square (including the white border) fits inside the disc. The dashed
circles show 20 mm and 25 mm discs for reference.

Output: weight_aruco_sheet.png (US Letter @ 300 DPI, print at 100% / "actual size").
"""

import cv2
import cv2.aruco as aruco
import numpy as np

from generate_aruco import DICT, DPI, MM_PER_IN, PAGE_H_IN, PAGE_W_IN, make_marker, mm_to_px

WEIGHT_ID = 4
SIZES_MM = [10, 12, 14, 16]   # black-square side; with border the tile is 4/3 larger
COPIES = 4
DISC_GUIDES_MM = [20, 25]


def main():
    dictionary = aruco.getPredefinedDictionary(DICT)
    page = np.full((mm_to_px(PAGE_H_IN * MM_PER_IN),
                    mm_to_px(PAGE_W_IN * MM_PER_IN)), 255, np.uint8)

    cv2.putText(page, f"Weight marker: DICT_4X4_50 id {WEIGHT_ID}",
                (mm_to_px(12), mm_to_px(12)), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 0, 2)

    y = mm_to_px(30)
    for size_mm in SIZES_MM:
        side_px = mm_to_px(size_mm)
        tile = make_marker(dictionary, WEIGHT_ID, side_px)
        h, w = tile.shape
        tile_mm = h / DPI * MM_PER_IN
        cv2.putText(page, f"{size_mm} mm marker ({tile_mm:.1f} mm with border)",
                    (mm_to_px(12), y - mm_to_px(4)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 0, 2)

        x = mm_to_px(12)
        cy = y + h // 2
        for _ in range(COPIES):
            page[y:y + h, x:x + w] = tile
            cv2.rectangle(page, (x, y), (x + w, y + h), 200, 1)   # cut guide
            cx = x + w // 2
            for d in DISC_GUIDES_MM:
                cv2.circle(page, (cx, cy), mm_to_px(d / 2), 170, 1, cv2.LINE_AA)
            x += mm_to_px(max(DISC_GUIDES_MM) + 15)

        y += max(h, mm_to_px(max(DISC_GUIDES_MM))) + mm_to_px(22)

    # 50 mm reference bar to verify the printer didn't rescale the page
    bar_y = y + mm_to_px(5)
    cv2.rectangle(page, (mm_to_px(12), bar_y),
                  (mm_to_px(62), bar_y + mm_to_px(4)), 0, -1)
    cv2.putText(page, "this bar must measure exactly 50 mm",
                (mm_to_px(66), bar_y + mm_to_px(4)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, 0, 1)
    cv2.putText(page, "circles = 20 mm / 25 mm disc outlines (fit check only)",
                (mm_to_px(12), bar_y + mm_to_px(14)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, 0, 1)

    cv2.imwrite("weight_aruco_sheet.png", page)

    detector = aruco.ArucoDetector(dictionary, aruco.DetectorParameters())
    _, ids, _ = detector.detectMarkers(page)
    n = 0 if ids is None else int((ids.flatten() == WEIGHT_ID).sum())
    print(f"self-test: decoded {n} id-{WEIGHT_ID} markers (expect {len(SIZES_MM) * COPIES})")
    print("wrote weight_aruco_sheet.png — print at 100% scale, then measure the 50 mm bar")


if __name__ == "__main__":
    main()
