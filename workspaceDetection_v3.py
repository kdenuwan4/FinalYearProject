"""
=============================================================================
Project: Multi-Robot Communication System with Vision-Based Tracking (VLC)
Module : Central Intelligence & Mapping
Author : Amunugama H.M.K.D. (E/21/029)
Version: 3.0 (Always-On Detection with Graceful Fallback)
Date   : May 3, 2026

Description:
Unlike v2 which permanently locked the boundary after first detection,
v3 re-detects the chalk rectangle every frame. This means the grid follows
the camera if it moves. When the rectangle is temporarily lost (hand in the
way, bad lighting etc.) the warped window holds the last good frame instead
of closing, and the camera window simply shows the raw feed until detection
recovers.

Behaviour summary:
  Camera window  — live always. Grid drawn on it whenever rectangle is found.
  Warped window  — shows last good warp. Never closes. Updates when rect found.

Key changes from v2:
- Removed permanent lock. Detection runs every frame.
- last_warped persists across detection failures (warped window never closes).
- last_M / last_M_inv / last_ordered persist too so camera grid stays drawn
  for one extra frame on borderline detections (optional, can remove).
- Added 'L' key to temporarily pause re-detection (manual lock) if needed.
=============================================================================
"""

import cv2
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
#  GEOMETRY
# ================================================================
DST_PTS = np.array(
    [[0, 0], [WARP_W, 0], [WARP_W, WARP_H], [0, WARP_H]],
    dtype="float32"
)

def order_corners(pts):
    """Sort 4 points into TL, TR, BR, BL order."""
    pts = pts.reshape(4, 2).astype("float32")
    s   = pts.sum(axis=1)
    d   = np.diff(pts, axis=1).flatten()
    return np.array([
        pts[np.argmin(s)],   # TL
        pts[np.argmin(d)],   # TR
        pts[np.argmax(s)],   # BR
        pts[np.argmax(d)],   # BL
    ], dtype="float32")

def find_chalk_rect(gray):
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
#  DRAWING
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

def project_grid_back(frame, ordered, rows, cols, warp_w, warp_h, M_inv):
    """Draw grid lines on the camera frame using inverse homography."""
    cell_w = warp_w // cols
    cell_h = warp_h // rows

    def wp(px, py):
        pt = np.array([[[float(px), float(py)]]], dtype="float32")
        return tuple(cv2.perspectiveTransform(pt, M_inv)[0][0].astype(int))

    for r in range(1, rows):
        cv2.line(frame, wp(0, r * cell_h), wp(warp_w, r * cell_h),
                 (0, 255, 0), 2)
    for c in range(1, cols):
        cv2.line(frame, wp(c * cell_w, 0), wp(c * cell_w, warp_h),
                 (0, 255, 0), 2)
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
    cv2.putText(frame,
                "Q/ESC=quit  L=toggle manual lock  D=edge debug",
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
        print(f"ERROR: Cannot open camera: {CAMERA_SOURCE}")
        return

    last_ordered = None
    last_M       = None
    last_M_inv   = None
    last_warped  = None

    manual_lock = False
    show_debug  = False
    prev_time   = time.time()

    print("\n=== Workspace Tracker v3 ===")
    print("  Q / ESC — quit")
    print("  L       — toggle manual lock (freeze detection on current boundary)")
    print("  D       — toggle edge debug view")
    print("============================\n")

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
        #  STEP 1 — DETECT BOUNDARY
        #  Runs every frame unless the user has toggled manual lock.
        #  On success  → update last_ordered / last_M / last_M_inv.
        #  On failure  → keep whatever was there before (do nothing).
        # ============================================================
        rect_found = False

        if not manual_lock:
            rect = find_chalk_rect(gray)
            if rect is not None:
                last_ordered = order_corners(rect)
                last_M       = cv2.getPerspectiveTransform(last_ordered, DST_PTS)
                last_M_inv   = cv2.getPerspectiveTransform(DST_PTS, last_ordered)
                rect_found   = True
        else:
            rect_found = last_ordered is not None

        # ============================================================
        #  STEP 2 — WARP
        #  Only possible when we have a boundary (current or cached).
        # ============================================================
        if last_ordered is not None:

            if rect_found:
                warped      = cv2.warpPerspective(frame, last_M, (WARP_W, WARP_H))
                draw_grid(warped, GRID_ROWS, GRID_COLS)
                last_warped = warped

            if rect_found:
                project_grid_back(display, last_ordered,
                                  GRID_ROWS, GRID_COLS,
                                  WARP_W, WARP_H, last_M_inv)

        # ============================================================
        #  WARPED WINDOW — always show last good frame, never close
        # ============================================================
        if last_warped is not None:
            show_warped = last_warped.copy()
            if not rect_found:
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
            state = "ON (boundary frozen)" if manual_lock else "OFF (tracking)"
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