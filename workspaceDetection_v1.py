"""
=============================================================================
Project: Multi-Robot Communication System with Vision-Based Tracking (VLC)
Module : Central Intelligence & Mapping
Author : Amunugama H.M.K.D. (E/21/029)
Version: 1.0 (Environment Discretization & Homography)
Date   : April 30, 2026

Description:
This script establishes the foundational environment mapping system. It uses 
computer vision to detect the physical boundaries of the robot operating area. 
By calculating the homography matrix, it performs a perspective warp to 
correct camera tilt, generating a standardized top-down 2D grid. This grid 
forms the discrete coordinate system required for the A* path-planning algorithm.

Core Features Implemented:
- [Perception] Implemented Canny edge detection and contour approximation.
- [Mathematics] Applied perspective transformation to create a flat, top-down view.
- [Mapping] Discretized the physical workspace into a logical, uniform grid.
- [Visualization] Created an inverse projection to render the logical grid 
  back onto the live, angled camera feed for real-time debugging.
=============================================================================
"""

import cv2
import numpy as np
import time

# ---------------- CONFIG ----------------
CAMERA_SOURCE = 0
GRID_ROWS = 3
GRID_COLS = 3
WARP_W, WARP_H = 600, 600

# ---------------- CORNER ORDERING ----------------
def order_corners(pts):
    pts = pts.reshape(4, 2).astype("float32")
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).flatten()
    return np.array([
        pts[np.argmin(s)],   # TL
        pts[np.argmin(d)],   # TR
        pts[np.argmax(s)],   # BR
        pts[np.argmax(d)],   # BL
    ], dtype="float32")

# ---------------- CHALK RECT DETECTION ----------------
def find_chalk_rect(gray):
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    edges = cv2.Canny(blurred, 30, 120)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    edges = cv2.dilate(edges, kernel, iterations=2)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best, best_area = None, 0
    for c in contours:
        area = cv2.contourArea(c)
        if area < 8000:
            continue
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.03 * peri, True)
        if len(approx) == 4 and cv2.isContourConvex(approx) and area > best_area:
            best, best_area = approx, area

    return best

# ---------------- GRID HELPERS ----------------
def draw_grid(img, rows, cols, color=(0, 255, 0), thickness=2):
    h, w = img.shape[:2]
    cell_w, cell_h = w // cols, h // rows
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
    """Draw grid lines onto the original camera frame using inverse homography."""
    cell_w, cell_h = warp_w // cols, warp_h // rows

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

# ---------------- MAIN ----------------
cap = cv2.VideoCapture(CAMERA_SOURCE)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

dst_pts = np.array([[0, 0], [WARP_W, 0], [WARP_W, WARP_H], [0, WARP_H]], dtype="float32")

show_debug = False
prev_time = time.time()

print("Controls: Q / ESC = quit  |  D = toggle edge debug view")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Cannot read frame — check CAMERA_SOURCE")
        break

    display = frame.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # FPS
    now = time.time()
    fps = 1.0 / (now - prev_time + 1e-9)
    prev_time = now
    cv2.putText(display, f"FPS: {fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

    # ---- 1. DETECT CHALK RECTANGLE ----
    rect = find_chalk_rect(gray)

    if rect is not None:
        ordered = order_corners(rect)
        M     = cv2.getPerspectiveTransform(ordered, dst_pts)
        M_inv = cv2.getPerspectiveTransform(dst_pts, ordered)
        warped = cv2.warpPerspective(frame, M, (WARP_W, WARP_H))

        draw_grid(warped, GRID_ROWS, GRID_COLS)
        project_grid_back(display, ordered, GRID_ROWS, GRID_COLS, WARP_W, WARP_H, M_inv)

        cv2.imshow("Warped Grid", warped)

    else:
        cv2.putText(display, "No boundary detected", (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        try:
            cv2.destroyWindow("Warped Grid")
        except:
            pass

    if show_debug:
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)
        edges = cv2.Canny(blurred, 30, 120)
        cv2.imshow("Edges", edges)

    cv2.imshow("Camera", display)

    key = cv2.waitKey(1) & 0xFF
    if key in (ord('q'), 27):
        break
    elif key == ord('d'):
        show_debug = not show_debug
        if not show_debug:
            cv2.destroyWindow("Edges")

cap.release()
cv2.destroyAllWindows()