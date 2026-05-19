"""
=============================================================================
Project: Multi-Robot Communication System with Vision-Based Tracking
Module : Central Intelligence & Mapping + Hybrid Dynamic A* Path Planning
Author : Amunugama H.M.K.D. (E/21/029)
Version: 3.0 (Hybrid Dynamic Occupancy + Adaptive Decay A*)
Date   : May 16, 2026

Description:
Upgrades the A* navigation system into a hybrid dynamic occupancy mapping
architecture with two independent layers fused at planning time.

Layer Architecture:
  STATIC LAYER   — Permanent constraints (arena walls, predefined obstacles).
                   Hardcoded at startup. Never modified at runtime.
                   Always treated as absolute truth (always blocked).

  DYNAMIC LAYER  — Real-time robot occupancy derived from ArUco detections.
                   Each cell holds a continuous confidence value [0.0, 1.0].
                   Confidence rises on detection (reinforcement) and decays
                   gradually when a robot is absent (temporal smoothing).
                   Decay rate is adaptive: faster when detection is noisy
                   or inconsistent, slower when observations are stable.

  FUSED GRID     — Used by A* at planning time.
                   Cell is BLOCKED if: static[r][c] == 1
                                    OR dynamic_confidence[r][c] >= THRESHOLD
                   Otherwise FREE.

Path Recalculation:
  A* is not re-run every frame. The fused grid is hashed each frame and
  compared to the hash at last planning time. Replanning only occurs when
  the fused grid has meaningfully changed, avoiding unnecessary computation.

Changes from v2:
- [Architecture] Two-layer occupancy model (static + dynamic).
- [Occupancy]    Confidence-based dynamic layer with [0,1] float values.
- [Decay]        Adaptive decay: stability tracker per cell drives decay rate.
- [Planning]     Fused grid passed to A* instead of raw static grid.
- [Planning]     Hash-based change detection — replanning on meaningful change only.
- [Modularity]   OccupancyMap class encapsulates all grid logic for future
                 extension (multi-robot, VLC command dispatch, execution layer).
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

DIAGONAL_COST  = 1.414   # √2

# ================================================================
#  OCCUPANCY CONFIG
# ================================================================

# Static layer — 0 = free, 1 = permanent obstacle. Never modified at runtime.
STATIC_GRID = [
    [0, 0, 0],
    [0, 1, 0],
    [0, 0, 0],
]

GOAL_CELL = (2, 2)

# Dynamic layer tuning
CONFIDENCE_REINFORCE  = 0.35   # added per frame robot IS detected in cell
CONFIDENCE_DECAY_SLOW = 0.04   # decay/frame when observations are stable
CONFIDENCE_DECAY_FAST = 0.12   # decay/frame when observations are noisy
CONFIDENCE_THRESHOLD  = 0.55   # above this → cell treated as dynamically blocked
STABILITY_WINDOW      = 6      # frames used to judge detection stability


# ================================================================
#  OCCUPANCY MAP
# ================================================================
class OccupancyMap:
    """
    Two-layer occupancy map — the single source of truth for all planning.

    Static layer  : permanent constraints, set at init, never changes.
    Dynamic layer : per-cell confidence float [0,1] updated from detections.
    Fused grid    : what A* sees. Recomputed lazily, cached until dirty.

    Designed to be pluggable into future modules:
      - Multi-robot coordinator: call update() with each robot's cell set.
      - VLC command dispatch: read robot_cells / fused_grid() directly.
      - Execution layer: subscribe to fused_changed() for trigger events.
    """

    def __init__(self, static_grid):
        self.rows = len(static_grid)
        self.cols = len(static_grid[0])

        # Immutable static layer
        self._static = [row[:] for row in static_grid]

        # Dynamic layer
        self.confidence = np.zeros((self.rows, self.cols), dtype=np.float32)

        # Ring buffer: recent detection booleans per cell for stability tracking
        self._history     = np.zeros(
            (self.rows, self.cols, STABILITY_WINDOW), dtype=bool)
        self._history_idx = 0

        # Fused cache
        self._fused      = None
        self._fused_hash = None
        self._prev_hash  = None   # hash at last fused_changed() call
        self._dirty      = True

    # ------------------------------------------------------------------
    def update(self, occupied_cells: set):
        """
        Call once per frame with the set of (row, col) cells
        where a robot was detected. Pass empty set when none detected.
        """
        slot = self._history_idx % STABILITY_WINDOW

        for r in range(self.rows):
            for c in range(self.cols):
                detected = (r, c) in occupied_cells
                self._history[r, c, slot] = detected

                if detected:
                    self.confidence[r, c] = min(
                        1.0,
                        self.confidence[r, c] + CONFIDENCE_REINFORCE
                    )
                else:
                    # Adaptive decay rate driven by recent detection stability
                    detection_rate = self._history[r, c].sum() / STABILITY_WINDOW
                    decay = (
                        CONFIDENCE_DECAY_SLOW +
                        (CONFIDENCE_DECAY_FAST - CONFIDENCE_DECAY_SLOW)
                        * (1.0 - detection_rate)
                    )
                    self.confidence[r, c] = max(0.0, self.confidence[r, c] - decay)

        self._history_idx += 1
        self._dirty = True

    # ------------------------------------------------------------------
    def fused_grid(self):
        """
        Returns 2D list: 1 = blocked, 0 = free.
        Static obstacle OR dynamic confidence >= threshold → blocked.
        Cached — only recomputed when dirty (i.e. after update()).
        """
        if not self._dirty and self._fused is not None:
            return self._fused

        fused = []
        for r in range(self.rows):
            row = []
            for c in range(self.cols):
                blocked = (
                    self._static[r][c] == 1 or
                    float(self.confidence[r, c]) >= CONFIDENCE_THRESHOLD
                )
                row.append(1 if blocked else 0)
            fused.append(row)

        self._fused      = fused
        self._fused_hash = self._hash(fused)
        self._dirty      = False
        return fused

    def fused_changed(self):
        """
        Returns True if the fused grid changed since the last call.
        Used to gate A* replanning — avoids running A* every frame.
        """
        self.fused_grid()   # recompute if dirty
        changed          = self._fused_hash != self._prev_hash
        self._prev_hash  = self._fused_hash
        return changed

    # ------------------------------------------------------------------
    def static_blocked(self, r, c):
        return self._static[r][c] == 1

    def dynamic_confidence(self, r, c):
        return float(self.confidence[r, c])

    def is_dynamic_occupied(self, r, c):
        return float(self.confidence[r, c]) >= CONFIDENCE_THRESHOLD

    @staticmethod
    def _hash(grid):
        return tuple(cell for row in grid for cell in row)


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
#  A* — 8-DIRECTION WITH CORNER-CUT PREVENTION
# ================================================================
NEIGHBOURS = [
    (-1,  0, False),  # N
    ( 1,  0, False),  # S
    ( 0, -1, False),  # W
    ( 0,  1, False),  # E
    (-1, -1, True),   # NW
    (-1,  1, True),   # NE
    ( 1, -1, True),   # SW
    ( 1,  1, True),   # SE
]

def heuristic(a, b):
    """Chebyshev distance — admissible for 8-direction grids."""
    dr = abs(a[0] - b[0])
    dc = abs(a[1] - b[1])
    return max(dr, dc) + (DIAGONAL_COST - 1) * min(dr, dc)

def astar(grid, start, goal):
    """
    8-direction A* on a 2D list (0=free, 1=blocked).
    Diagonal cost = DIAGONAL_COST. Corner cutting is prevented.
    Returns list of (row, col) or None.
    """
    rows = len(grid)
    cols = len(grid[0])

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
            if nr < 0 or nr >= rows or nc < 0 or nc >= cols:
                continue
            if grid[nr][nc] == 1:
                continue
            # Corner-cut prevention
            if is_diagonal and (grid[r + dr][c] == 1 or grid[r][c + dc] == 1):
                continue

            move_cost   = DIAGONAL_COST if is_diagonal else 1.0
            tentative_g = g_score[current] + move_cost

            if tentative_g < g_score[(nr, nc)]:
                came_from[(nr, nc)] = current
                g_score[(nr, nc)]   = tentative_g
                f = tentative_g + heuristic((nr, nc), goal)
                heapq.heappush(open_set, (f, (nr, nc)))

    return None


# ================================================================
#  DRAWING HELPERS
# ================================================================
def draw_occupancy_layer(img, occ_map, rows, cols):
    """
    Visualise both layers on the warped view:
      Static obstacles      → dark grey fill + red X
      Dynamic occupied      → red tint scaled by confidence + progress bar
      Dynamic low-conf      → faint yellow hint
    """
    h, w   = img.shape[:2]
    cell_w = w // cols
    cell_h = h // rows

    for r in range(rows):
        for c in range(cols):
            x1, y1 = c * cell_w, r * cell_h
            x2, y2 = x1 + cell_w, y1 + cell_h

            if occ_map.static_blocked(r, c):
                cv2.rectangle(img, (x1, y1), (x2, y2), (60, 60, 60), -1)
                cv2.putText(img, "X",
                            (x1 + cell_w // 2 - 10, y1 + cell_h // 2 + 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            else:
                conf = occ_map.dynamic_confidence(r, c)
                if conf >= CONFIDENCE_THRESHOLD:
                    # Occupied — red tint proportional to confidence
                    alpha   = min(0.55, conf * 0.6)
                    overlay = img.copy()
                    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 200), -1)
                    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
                    # Confidence bar at bottom of cell
                    bar_w = int((x2 - x1) * conf)
                    cv2.rectangle(img, (x1, y2 - 6), (x1 + bar_w, y2),
                                  (0, 0, 255), -1)
                elif conf > 0.05:
                    # Low confidence — faint yellow
                    alpha   = conf * 0.4
                    overlay = img.copy()
                    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 200, 255), -1)
                    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

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

def draw_hud(frame, rect_found, manual_lock, fps, replan_count):
    h = frame.shape[0]
    if manual_lock:
        label, color = "Boundary: MANUAL LOCK",         (0, 200, 255)
    elif rect_found:
        label, color = "Boundary: TRACKING",            (0, 220, 80)
    else:
        label, color = "Boundary: LOST — searching...", (0, 80, 255)

    cv2.putText(frame, label,                      (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.72, color, 2)
    cv2.putText(frame, f"FPS: {fps:.1f}",          (10, 58),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 2)
    cv2.putText(frame, f"Replans: {replan_count}", (10, 84),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)
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

    # Vision state
    last_ordered = None
    last_M       = None
    last_M_inv   = None
    last_warped  = None

    # Occupancy map — single source of truth for planning
    occ_map = OccupancyMap(STATIC_GRID)

    # Planning state — cached per robot, only updated when fused grid changes
    robot_paths  = {}   # robot_id -> path list or None
    replan_count = 0

    manual_lock = False
    show_debug  = False
    prev_time   = time.time()

    print("\n=== Hybrid Dynamic Occupancy A* Tracker ===")
    print("  Q / ESC -> quit")
    print("  L       -> toggle manual lock")
    print("  D       -> toggle debug edges")
    print("===========================================")
    print(f"  Confidence threshold : {CONFIDENCE_THRESHOLD}")
    print(f"  Decay slow / fast    : {CONFIDENCE_DECAY_SLOW} / {CONFIDENCE_DECAY_FAST}")
    print(f"  Reinforce rate       : {CONFIDENCE_REINFORCE}")
    print(f"  Stability window     : {STABILITY_WINDOW} frames\n")

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
        #  STEP 2 — ARUCO DETECTION + OCCUPANCY UPDATE
        # ============================================================
        occupied_this_frame = set()
        detected_robots     = {}
        # detected_robots: robot_id -> (wx, wy, grid_row, grid_col,
        #                               angle, fwd_w, cam_cx, cam_cy)

        if last_ordered is not None and rect_found:
            cell_w = WARP_W // GRID_COLS
            cell_h = WARP_H // GRID_ROWS

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

                    occupied_this_frame.add((grid_row, grid_col))
                    detected_robots[robot_id] = (
                        wx, wy, grid_row, grid_col,
                        angle, fwd_w, cam_cx, cam_cy
                    )

        # Always call update — drives decay even when no robots detected
        if rect_found:
            occ_map.update(occupied_this_frame)

        # ============================================================
        #  STEP 3 — REPLANNING (only when fused grid meaningfully changes)
        # ============================================================
        if rect_found and occ_map.fused_changed():
            fused        = occ_map.fused_grid()
            replan_count += 1

            for robot_id, (_, _, grid_row, grid_col, _, _, _, _) in detected_robots.items():
                # Temporarily free the robot's own cell so it isn't
                # blocked by its own dynamic confidence during planning
                saved = fused[grid_row][grid_col]
                fused[grid_row][grid_col] = 0
                robot_paths[robot_id] = astar(fused, (grid_row, grid_col), GOAL_CELL)
                fused[grid_row][grid_col] = saved

            # Remove paths for robots no longer visible
            for rid in list(robot_paths):
                if rid not in detected_robots:
                    del robot_paths[rid]

        # ============================================================
        #  STEP 4 — WARP + VISUALISE
        # ============================================================
        if last_ordered is not None:
            if rect_found:
                # Save clean warp — all drawing goes on a copy
                last_warped = cv2.warpPerspective(frame, last_M, (WARP_W, WARP_H))

            warped_draw = last_warped.copy()

            if rect_found:
                # Draw order: occupancy → paths → highlights → robot markers → grid
                draw_occupancy_layer(warped_draw, occ_map, GRID_ROWS, GRID_COLS)

                for robot_id, path in robot_paths.items():
                    draw_path(warped_draw, path, GRID_ROWS, GRID_COLS)

                for robot_id, (wx, wy, grid_row, grid_col,
                               angle, fwd_w, cam_cx, cam_cy) in detected_robots.items():

                    highlight_cell(warped_draw, grid_row, grid_col,
                                   GRID_ROWS, GRID_COLS, color=(0, 200, 80))

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

                # Grid lines always on top
                draw_grid(warped_draw, GRID_ROWS, GRID_COLS)
                project_grid_back(display, last_ordered,
                                  GRID_ROWS, GRID_COLS,
                                  WARP_W, WARP_H, last_M_inv)

            # Warped window — always open, never closes
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
        draw_hud(display, rect_found, manual_lock, fps, replan_count)
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
