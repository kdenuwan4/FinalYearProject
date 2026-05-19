"""
=============================================================================
Project: Multi-Robot Communication System with Vision-Based Tracking
Module : Central Intelligence & Mapping
Author : Amunugama H.M.K.D. (E/21/029)
Version: 2.0 (Always-On Detection with Graceful Fallback)
Date   : May 3, 2026

Description:
Upgrades the perception pipeline from a one-shot boundary lock to a
continuous re-detection model. The system re-detects the chalk boundary
every frame so the grid follows the camera if it moves. When the boundary
is temporarily lost, the warped window holds the last good frame (never
closes) and the camera view shows the raw feed until detection recovers.
Robot positions are only plotted when a fresh boundary exists that frame
to avoid projecting stale/misleading coordinates.

Changes from v1:
- [Core] Detection runs every frame instead of locking permanently.
- [Fallback] last_warped persists on detection loss — warped window never closes.
- [Controls] L now toggles a manual freeze (pause re-detection) instead of
  one-shot force-lock. R key removed (no longer needed).
- [UI] HUD shows TRACKING / LOST / MANUAL LOCK states distinctly.
=============================================================================
"""

import cv2
import cv2.aruco as aruco
import numpy as np
import time

# ================================================================
#  CONFIG
# ================================================================
CAMERA_SOURCE  = 0
GRID_ROWS      = 3
GRID_COLS      = 3
WARP_W         = 600
WARP_H         = 600
CANNY_LOW      = 30
CANNY_HIGH     = 120
MIN_ARENA_AREA = 8000

# ================================================================
#  ARUCO — auto-selects old (4.7) or new (4.8+) API
# ================================================================
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
parameters = aruco.DetectorParameters()

try:
    _detector   = aruco.ArucoDetector(aruco_dict, parameters)
    USE_NEW_API = True
except AttributeError:
    USE_NEW_API = False

def detect_aruco(gray):
    if USE_NEW_API:
        corners, ids, _ = _detector.detectMarkers(gray)
    else:
        corners, ids, _ = aruco.detectMarkers(
            gray, aruco_dict, parameters=parameters)
    return corners, ids

# ================================================================
#  GEOMETRY HELPERS
# ================================================================
DST_PTS = np.array(
    [[0, 0], [WARP_W, 0], [WARP_W, WARP_H], [0, WARP_H]],
    dtype="float32"
)

def order_corners(pts):
    """Return corners in TL, TR, BR, BL order regardless of detection order."""
    pts = pts.reshape(4, 2).astype("float32")
    s   = pts.sum(axis=1)
    d   = np.diff(pts, axis=1).flatten()
    return np.array([
        pts[np.argmin(s)],   # TL
        pts[np.argmin(d)],   # TR
        pts[np.argmax(s)],   # BR
        pts[np.argmax(d)],   # BL
    ], dtype="float32")

def find_rect(gray):
    """Find the largest convex quadrilateral — the arena boundary."""
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    edges   = cv2.Canny(blurred, CANNY_LOW, CANNY_HIGH)
    kernel  = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    edges   = cv2.dilate(edges, kernel, iterations=2)

    contours, _ = cv2.findContours(
        edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best, best_area = None, 0
    for c in contours:
        area = cv2.contourArea(c)
        if area < MIN_ARENA_AREA:
            continue
        peri   = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.03 * peri, True)
        if len(approx) == 4 and cv2.isContourConvex(approx) and area > best_area:
            best, best_area = approx, area

    return best

# ================================================================
#  DRAWING HELPERS
# ================================================================
def draw_grid(img, rows, cols, color=(0, 255, 0), thickness=2):
    h, w   = img.shape[:2]
    cell_w = w // cols
    cell_h = h // rows
    for r in range(rows + 1):
        cv2.line(img, (0, r * cell_h), (w, r * cell_h), color, thickness)
    for c in range(cols + 1):
        cv2.line(img, (c * cell_w, 0), (c * cell_w, h), color, thickness)
    for r in range(rows):
        for c in range(cols):
            cx = c * cell_w + cell_w // 2
            cy = r * cell_h + cell_h // 2
            cv2.putText(img, f"{r},{c}", (cx - 18, cy + 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)

def highlight_cell(img, row, col, rows, cols,
                   color=(0, 80, 255), alpha=0.35):
    h, w   = img.shape[:2]
    cell_w = w // cols
    cell_h = h // rows
    x1     = max(0, col * cell_w)
    y1     = max(0, row * cell_h)
    x2     = min(w, x1 + cell_w)
    y2     = min(h, y1 + cell_h)
    overlay = img.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

def project_grid_back(frame, ordered, rows, cols, warp_w, warp_h, M_inv):
    """Re-project flat grid lines onto the camera frame via inverse homography."""
    cell_w = warp_w // cols
    cell_h = warp_h // rows

    def wp(px, py):
        pt = np.array([[[float(px), float(py)]]], dtype="float32")
        return tuple(cv2.perspectiveTransform(pt, M_inv)[0][0].astype(int))

    for r in range(1, rows):
        cv2.line(frame, wp(0, r * cell_h), wp(warp_w, r * cell_h), (0, 255, 0), 2)
    for c in range(1, cols):
        cv2.line(frame, wp(c * cell_w, 0), wp(c * cell_w, warp_h), (0, 255, 0), 2)
    for i in range(4):
        p1 = tuple(ordered[i].astype(int))
        p2 = tuple(ordered[(i + 1) % 4].astype(int))
        cv2.line(frame, p1, p2, (0, 140, 255), 3)

def draw_hud(frame, rect_found, manual_lock, fps):
    h = frame.shape[0]
    if manual_lock:
        label = "Boundary: MANUAL LOCK"
        color = (0, 200, 255)
    elif rect_found:
        label = "Boundary: TRACKING"
        color = (0, 220, 80)
    else:
        label = "Boundary: LOST — searching..."
        color = (0, 80, 255)

    cv2.putText(frame, label,             (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.72, color, 2)
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 58),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 2)
    cv2.putText(frame, "Q/ESC=quit  L=toggle manual lock  D=edge debug",
                (10, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1)

# ================================================================
#  MAIN
# ================================================================
def main():
    cap = cv2.VideoCapture(CAMERA_SOURCE)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        print(f"ERROR: Cannot open camera source: {CAMERA_SOURCE}")
        return

    # ---- Persistent boundary state ----
    # Updated every frame detection succeeds.
    # Kept unchanged when detection fails — acts as fallback.
    last_ordered = None
    last_M       = None
    last_M_inv   = None
    last_warped  = None   # last good warped frame shown in the warped window

    manual_lock = False   # L key: freeze re-detection, reuse last boundary
    show_debug  = False
    prev_time   = time.time()

    print("\n=== Arena Tracker (v2) ===")
    print("  Q / ESC — quit")
    print("  L       — toggle manual lock (freeze / unfreeze detection)")
    print("  D       — toggle edge debug view")
    print("==========================\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Cannot read frame — check CAMERA_SOURCE")
            break

        display = frame.copy()
        gray    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        now       = time.time()
        fps       = 1.0 / max(now - prev_time, 1e-9)
        prev_time = now

        # ============================================================
        #  STEP 1 — BOUNDARY DETECTION
        #
        #  Runs every frame unless manual_lock is on.
        #  Success → update last_ordered / last_M / last_M_inv.
        #  Failure → leave them unchanged (automatic fallback).
        # ============================================================
        rect_found = False

        if not manual_lock:
            rect = find_rect(gray)
            if rect is not None:
                last_ordered = order_corners(rect)
                last_M       = cv2.getPerspectiveTransform(last_ordered, DST_PTS)
                last_M_inv   = cv2.getPerspectiveTransform(DST_PTS, last_ordered)
                rect_found   = True
        else:
            # Manual lock — treat cached boundary as current
            rect_found = last_ordered is not None

        # ============================================================
        #  STEP 2 — WARP + ARUCO
        #  Runs whenever a boundary exists (current or cached).
        # ============================================================
        if last_ordered is not None:

            cell_w = WARP_W // GRID_COLS
            cell_h = WARP_H // GRID_ROWS

            if rect_found:
                # Fresh detection this frame — warp current frame
                warped      = cv2.warpPerspective(frame, last_M, (WARP_W, WARP_H))
                last_warped = warped
            # If rect_found is False, last_warped stays frozen from previous frame

            # ArUco — only plot when boundary is fresh to avoid stale projections
            if rect_found:
                corners, ids = detect_aruco(gray)

                if ids is not None:
                    warped_draw = last_warped.copy()

                    for i in range(len(ids)):
                        c        = corners[i][0]
                        robot_id = int(ids[i][0])

                        cam_cx = float(c[:, 0].mean())
                        cam_cy = float(c[:, 1].mean())

                        pt        = np.array([[[cam_cx, cam_cy]]], dtype="float32")
                        warped_pt = cv2.perspectiveTransform(pt, last_M)[0][0]
                        wx = max(0, min(WARP_W - 1, int(warped_pt[0])))
                        wy = max(0, min(WARP_H - 1, int(warped_pt[1])))

                        grid_col = min(wx // cell_w, GRID_COLS - 1)
                        grid_row = min(wy // cell_h, GRID_ROWS - 1)

                        top_mid    = c[0:2].mean(axis=0)
                        bottom_mid = c[2:4].mean(axis=0)
                        angle = np.degrees(np.arctan2(
                            -(top_mid[1] - bottom_mid[1]),
                             (top_mid[0] - bottom_mid[0])
                        ))

                        fwd_cam = np.array([[[
                            cam_cx + (top_mid[0] - cam_cx) * 2.0,
                            cam_cy + (top_mid[1] - cam_cy) * 2.0,
                        ]]], dtype="float32")
                        fwd_w = cv2.perspectiveTransform(
                            fwd_cam, last_M)[0][0].astype(int)

                        # Warped view
                        highlight_cell(warped_draw, grid_row, grid_col,
                                       GRID_ROWS, GRID_COLS)
                        cv2.circle(warped_draw, (wx, wy), 7, (0, 0, 255), -1)
                        cv2.arrowedLine(warped_draw, (wx, wy), tuple(fwd_w),
                                        (0, 200, 255), 2, tipLength=0.3)
                        cv2.putText(warped_draw,
                                    f"ID{robot_id}  ({grid_row},{grid_col})  {angle:.0f}deg",
                                    (wx + 10, wy - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)

                        # Camera view
                        cv2.circle(display, (int(cam_cx), int(cam_cy)),
                                   5, (255, 80, 0), -1)
                        cv2.putText(display, f"ID{robot_id}",
                                    (int(cam_cx) + 8, int(cam_cy) - 8),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 80, 0), 2)

                    draw_grid(warped_draw, GRID_ROWS, GRID_COLS)
                    last_warped = warped_draw

                else:
                    # No robots this frame — just draw grid on fresh warp
                    draw_grid(last_warped, GRID_ROWS, GRID_COLS)

                # Grid projected back onto camera view
                project_grid_back(display, last_ordered,
                                  GRID_ROWS, GRID_COLS,
                                  WARP_W, WARP_H, last_M_inv)

        # ============================================================
        #  WARPED WINDOW — always open, never closes
        # ============================================================
        if last_warped is not None:
            show_warped = last_warped.copy()
            if not rect_found:
                # Stale frame — show subtle banner
                cv2.rectangle(show_warped, (0, 0), (WARP_W, 36), (0, 0, 0), -1)
                cv2.putText(show_warped, "Searching... (last known view)",
                            (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0, 140, 255), 2)
            cv2.imshow("Warped Grid", show_warped)

        # ============================================================
        #  CAMERA WINDOW
        # ============================================================
        draw_hud(display, rect_found, manual_lock, fps)
        cv2.imshow("Camera", display)

        # Debug edge view
        if show_debug:
            blurred = cv2.GaussianBlur(gray, (7, 7), 0)
            edges   = cv2.Canny(blurred, CANNY_LOW, CANNY_HIGH)
            cv2.imshow("Edges (debug)", edges)

        # ============================================================
        #  KEY HANDLING
        # ============================================================
        key = cv2.waitKey(1) & 0xFF

        if key in (ord('q'), 27):
            break

        elif key == ord('l'):
            manual_lock = not manual_lock
            state = "ON (boundary frozen)" if manual_lock else "OFF (live tracking)"
            print(f"Manual lock: {state}")

        elif key == ord('d'):
            show_debug = not show_debug
            if not show_debug:
                cv2.destroyWindow("Edges (debug)")

    cap.release()
    cv2.destroyAllWindows()
    print("Exited cleanly.")


if __name__ == "__main__":
    main()