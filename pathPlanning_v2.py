"""
=============================================================================
Project: Multi-Robot Communication System with Vision-Based Tracking
Module : Central Intelligence & Mapping + A* Path Planning
Author : Amunugama H.M.K.D. (E/21/029)
Version: 2.0 (8-Direction A* with Corner-Cut Prevention)
Date   : May 13, 2026

Description:
Upgrades the A* planner from 4-direction to 8-direction movement.
Diagonal moves cost 1.414 (√2) instead of 1. Corner-cutting is prevented —
a diagonal step is only allowed if both adjacent cardinal cells are free,
ensuring robots never clip through obstacle corners.

Changes from v1:
- [Planner] 8-direction neighbours replacing 4-direction.
- [Planner] Diagonal cost set to 1.414 (√2 approximation).
- [Planner] Corner-cut prevention: both side cells checked before
  allowing a diagonal step.
- [Planner] Heuristic upgraded to Chebyshev distance to stay
  admissible for 8-direction grids.
=============================================================================
"""

import cv2
import cv2.aruco as aruco
import numpy as np
import time
import heapq

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

DIAGONAL_COST  = 1.414   # ≈ √2

# ================================================================
#  STATIC A* CONFIG
#  0 = free cell, 1 = obstacle
# ================================================================
occupancy_grid = [
    [0, 1, 0],
    [0, 0, 0],
    [0, 0, 0],
]

GOAL_CELL = (2, 2)

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
#  GEOMETRY
# ================================================================
DST_PTS = np.array(
    [[0, 0], [WARP_W, 0], [WARP_W, WARP_H], [0, WARP_H]],
    dtype="float32"
)

def order_corners(pts):
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
#  A* PATH PLANNING — 8-DIRECTION
# ================================================================

# All 8 neighbours: (row_delta, col_delta, is_diagonal)
NEIGHBOURS = [
    (-1,  0, False),   # N
    ( 1,  0, False),   # S
    ( 0, -1, False),   # W
    ( 0,  1, False),   # E
    (-1, -1, True),    # NW
    (-1,  1, True),    # NE
    ( 1, -1, True),    # SW
    ( 1,  1, True),    # SE
]

def heuristic(a, b):
    """
    Chebyshev distance — admissible heuristic for 8-direction grids.
    h = max(|dr|, |dc|)  (diagonal moves cost 1, here scaled to DIAGONAL_COST)
    Using: h = max(dr, dc) + (√2 - 1) * min(dr, dc)
    which is the exact optimal cost when there are no obstacles.
    """
    dr = abs(a[0] - b[0])
    dc = abs(a[1] - b[1])
    return max(dr, dc) + (DIAGONAL_COST - 1) * min(dr, dc)

def astar(grid, start, goal):
    rows = len(grid)
    cols = len(grid[0])

    # Guard: start or goal is an obstacle
    if grid[start[0]][start[1]] == 1 or grid[goal[0]][goal[1]] == 1:
        return None

    open_set = []
    heapq.heappush(open_set, (0.0, start))

    came_from = {}
    g_score   = {(r, c): float("inf") for r in range(rows) for c in range(cols)}
    g_score[start] = 0.0

    while open_set:
        _, current = heapq.heappop(open_set)

        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            path.reverse()
            return path

        r, c = current

        for dr, dc, is_diagonal in NEIGHBOURS:
            nr, nc = r + dr, c + dc

            # Bounds check
            if nr < 0 or nr >= rows or nc < 0 or nc >= cols:
                continue

            # Target cell must be free
            if grid[nr][nc] == 1:
                continue

            # ---- CORNER-CUT PREVENTION ----
            # For a diagonal step, both side cells must be free.
            # Example: moving NE (dr=-1, dc=+1) requires
            #   cell (r-1, c) and cell (r, c+1) to both be free.
            if is_diagonal:
                if grid[r + dr][c] == 1 or grid[r][c + dc] == 1:
                    continue

            move_cost      = DIAGONAL_COST if is_diagonal else 1.0
            tentative_g    = g_score[current] + move_cost

            if tentative_g < g_score[(nr, nc)]:
                came_from[(nr, nc)] = current
                g_score[(nr, nc)]   = tentative_g
                f                   = tentative_g + heuristic((nr, nc), goal)
                heapq.heappush(open_set, (f, (nr, nc)))

    return None   # no path found

# ================================================================
#  DRAWING HELPERS
# ================================================================
def draw_obstacles(img, grid, rows, cols):
    h, w   = img.shape[:2]
    cell_w = w // cols
    cell_h = h // rows
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 1:
                x1, y1 = c * cell_w, r * cell_h
                x2, y2 = x1 + cell_w, y1 + cell_h
                cv2.rectangle(img, (x1, y1), (x2, y2), (60, 60, 60), -1)
                cv2.putText(img, "X",
                            (x1 + cell_w // 2 - 10, y1 + cell_h // 2 + 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

def draw_path(img, path, rows, cols):
    """Magenta path line with cyan goal circle."""
    if not path:
        return
    h, w   = img.shape[:2]
    cell_w = w // cols
    cell_h = h // rows
    for i in range(len(path) - 1):
        r1, c1 = path[i]
        r2, c2 = path[i + 1]
        x1 = c1 * cell_w + cell_w // 2
        y1 = r1 * cell_h + cell_h // 2
        x2 = c2 * cell_w + cell_w // 2
        y2 = r2 * cell_h + cell_h // 2
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 4)
    gr, gc = path[-1]
    cv2.circle(img,
               (gc * cell_w + cell_w // 2, gr * cell_h + cell_h // 2),
               12, (0, 255, 255), -1)

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
        label, color = "Boundary: MANUAL LOCK",        (0, 200, 255)
    elif rect_found:
        label, color = "Boundary: TRACKING",           (0, 220, 80)
    else:
        label, color = "Boundary: LOST — searching...", (0, 80, 255)

    cv2.putText(frame, label,             (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.72, color, 2)
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 58),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 2)
    cv2.putText(frame, "Q/ESC=quit  L=manual lock  D=edge debug",
                (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1)

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
    last_warped  = None   # clean warp — no drawings

    manual_lock = False
    show_debug  = False
    prev_time   = time.time()

    print("\n=== A* Arena Tracker (8-direction) ===")
    print("  Q / ESC -> quit")
    print("  L       -> toggle manual lock")
    print("  D       -> toggle debug edges")
    print("=======================================\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Cannot read frame")
            break

        display = frame.copy()
        gray    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        now       = time.time()
        fps       = 1.0 / max(now - prev_time, 1e-9)
        prev_time = now

        # ============================================================
        #  STEP 1 — BOUNDARY DETECTION
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
            rect_found = last_ordered is not None

        # ============================================================
        #  STEP 2 — WARP + ARUCO + A*
        # ============================================================
        if last_ordered is not None:

            cell_w = WARP_W // GRID_COLS
            cell_h = WARP_H // GRID_ROWS

            if rect_found:
                # Save clean warp — drawings always go on a copy
                last_warped = cv2.warpPerspective(frame, last_M, (WARP_W, WARP_H))

            warped_draw = last_warped.copy()

            if rect_found:
                corners, ids = detect_aruco(gray)

                if ids is not None:
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

                        path = astar(occupancy_grid, (grid_row, grid_col), GOAL_CELL)

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

                        # Draw order: obstacles → path → highlight → robot
                        draw_obstacles(warped_draw, occupancy_grid, GRID_ROWS, GRID_COLS)
                        draw_path(warped_draw, path, GRID_ROWS, GRID_COLS)
                        highlight_cell(warped_draw, grid_row, grid_col, GRID_ROWS, GRID_COLS)

                        cv2.circle(warped_draw, (wx, wy), 7, (0, 0, 255), -1)
                        cv2.arrowedLine(warped_draw, (wx, wy), tuple(fwd_w),
                                        (0, 200, 255), 2, tipLength=0.3)
                        cv2.putText(warped_draw,
                                    f"ID{robot_id} ({grid_row},{grid_col}) {angle:.0f}deg",
                                    (wx + 10, wy - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)

                        cv2.circle(display, (int(cam_cx), int(cam_cy)), 5, (255, 80, 0), -1)
                        cv2.putText(display, f"ID{robot_id}",
                                    (int(cam_cx) + 8, int(cam_cy) - 8),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 80, 0), 2)

                else:
                    # No robots — still show obstacles
                    draw_obstacles(warped_draw, occupancy_grid, GRID_ROWS, GRID_COLS)

                # Grid lines always last — sit on top of everything
                draw_grid(warped_draw, GRID_ROWS, GRID_COLS)
                project_grid_back(display, last_ordered,
                                  GRID_ROWS, GRID_COLS,
                                  WARP_W, WARP_H, last_M_inv)

            # --------------------------------------------------------
            #  WARPED WINDOW — always open, never closes
            # --------------------------------------------------------
            show_warped = warped_draw.copy()
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