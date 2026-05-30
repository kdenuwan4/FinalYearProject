"""
=============================================================================
Project: Multi-Robot Communication System with Vision-Based Tracking
Module : Central Intelligence & Mapping + Execution Layer
Author : Amunugama H.M.K.D. (E/21/029)
Version: 4.0 (Path Execution & Navigation Intelligence)
Date   : May 16, 2026

Description:
Transforms the system from a passive path planner into an active navigation
execution system. All Week 3 functionality is fully preserved and extended.

Architecture Layers:
  PERCEPTION    — Boundary detection, perspective warp, ArUco localisation.
  OCCUPANCY     — Static + dynamic hybrid map with adaptive confidence decay.
  PLANNING      — 8-direction A* with corner-cut prevention, hash-gated replan.
  EXECUTION     — PathExecutor converts A* paths into orientation-aware commands.
  VISUALISATION — Engineering-style overlay: state, heading, waypoint, command.

New in v4:
  [Execution]   PathExecutor class — waypoint following, goal detection.
  [Navigation]  Direction utility — cell-delta to cardinal/diagonal direction.
  [Commands]    NavigationCommand enum — MOVE_FORWARD / TURN_LEFT / TURN_RIGHT
                / STOP / REACHED_GOAL.
  [State]       RobotState dataclass — per-robot execution state machine.
  [Integration] Executor resets safely on A* replan without dropping state.
  [Visualisation] Command, heading, waypoint index overlaid per robot.

Preserved from v3 exactly:
  - OccupancyMap (static layer, confidence array, ring buffer, adaptive decay)
  - fused_grid() lazy cache + fused_changed() hash gate
  - 8-direction A* with Chebyshev heuristic + corner-cut prevention
  - All drawing helpers and warped-window persistence logic
=============================================================================
"""

import cv2
import cv2.aruco as aruco
import numpy as np
import time
import heapq
import math
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict

# ================================================================
#  CONFIG
# ================================================================
CAMERA_SOURCE  = 0
GRID_ROWS      = 3
GRID_COLS      = 4
WARP_W         = 600
WARP_H         = 600
CANNY_LOW      = 30
CANNY_HIGH     = 120
MIN_ARENA_AREA = 8000
DIAGONAL_COST  = 1.414   # √2

# ================================================================
#  OCCUPANCY CONFIG  (preserved from v3 exactly)
# ================================================================
STATIC_GRID = [
    [0, 1, 0, 0],
    [0, 1, 0, 1],
    [0, 0, 0, 1],
]

GOAL_CELL = (0, 3)

CONFIDENCE_REINFORCE  = 0.35
CONFIDENCE_DECAY_SLOW = 0.04
CONFIDENCE_DECAY_FAST = 0.12
CONFIDENCE_THRESHOLD  = 0.55
STABILITY_WINDOW      = 6

# ================================================================
#  EXECUTION CONFIG
# ================================================================
# Angle tolerance (degrees) to consider robot aligned with a direction
HEADING_TOLERANCE_DEG = 30.0

# ================================================================
#  DIRECTION SYSTEM
# ================================================================
class Direction(Enum):
    NORTH     = "N"
    SOUTH     = "S"
    EAST      = "E"
    WEST      = "W"
    NORTHEAST = "NE"
    NORTHWEST = "NW"
    SOUTHEAST = "SE"
    SOUTHWEST = "SW"
    NONE      = "?"

# Map (row_delta, col_delta) -> Direction
_DELTA_TO_DIR: Dict[Tuple[int,int], Direction] = {
    (-1,  0): Direction.NORTH,
    ( 1,  0): Direction.SOUTH,
    ( 0,  1): Direction.EAST,
    ( 0, -1): Direction.WEST,
    (-1,  1): Direction.NORTHEAST,
    (-1, -1): Direction.NORTHWEST,
    ( 1,  1): Direction.SOUTHEAST,
    ( 1, -1): Direction.SOUTHWEST,
}

# Map Direction -> compass angle in degrees (0=East, CCW positive — OpenCV convention)
# We use the standard math convention: East=0°, North=90°, West=180°, South=-90°
_DIR_TO_ANGLE: Dict[Direction, float] = {
    Direction.NORTH:     90.0,
    Direction.SOUTH:    -90.0,
    Direction.EAST:       0.0,
    Direction.WEST:     180.0,
    Direction.NORTHEAST: 45.0,
    Direction.NORTHWEST: 135.0,
    Direction.SOUTHEAST: -45.0,
    Direction.SOUTHWEST: -135.0,
    Direction.NONE:       0.0,
}

def cell_direction(from_cell: Tuple[int,int],
                   to_cell:   Tuple[int,int]) -> Direction:
    """Return the Direction enum for a step from one cell to another."""
    dr = to_cell[0] - from_cell[0]
    dc = to_cell[1] - from_cell[1]
    return _DELTA_TO_DIR.get((dr, dc), Direction.NONE)

def angle_diff(target_deg: float, current_deg: float) -> float:
    """Signed angular difference in [-180, 180] (target - current)."""
    diff = (target_deg - current_deg + 180.0) % 360.0 - 180.0
    return diff

# ================================================================
#  NAVIGATION COMMANDS
# ================================================================
class NavigationCommand(Enum):
    MOVE_FORWARD  = "MOVE_FORWARD"
    TURN_LEFT     = "TURN_LEFT"
    TURN_RIGHT    = "TURN_RIGHT"
    STOP          = "STOP"
    REACHED_GOAL  = "REACHED_GOAL"
    IDLE          = "IDLE"
    BLOCKED       = "BLOCKED"

# Visual colours per command  (BGR)
CMD_COLORS: Dict[NavigationCommand, Tuple[int,int,int]] = {
    NavigationCommand.MOVE_FORWARD: (0,   220,  80),
    NavigationCommand.TURN_LEFT:    (0,   200, 255),
    NavigationCommand.TURN_RIGHT:   (0,   140, 255),
    NavigationCommand.STOP:         (0,    80, 255),
    NavigationCommand.REACHED_GOAL: (0,   255, 255),
    NavigationCommand.IDLE:         (160, 160, 160),
    NavigationCommand.BLOCKED:      (0,    0,  200),
}

# ================================================================
#  EXECUTION STATUS
# ================================================================
class ExecStatus(Enum):
    IDLE       = auto()
    EXECUTING  = auto()
    REACHED    = auto()
    BLOCKED    = auto()

# ================================================================
#  ROBOT STATE
# ================================================================
@dataclass
class RobotState:
    """Per-robot execution state. One instance per detected robot ID."""
    robot_id:    int

    # Localisation (updated every frame from ArUco)
    cell:        Tuple[int,int]         = (0, 0)
    heading_deg: float                  = 0.0   # current angle from ArUco

    # Execution state
    path:        List[Tuple[int,int]]   = field(default_factory=list)
    waypoint_idx: int                   = 0     # index into path of NEXT target
    status:      ExecStatus             = ExecStatus.IDLE
    command:     NavigationCommand      = NavigationCommand.IDLE
    target_dir:  Direction              = Direction.NONE

    @property
    def target_cell(self) -> Optional[Tuple[int,int]]:
        if self.path and self.waypoint_idx < len(self.path):
            return self.path[self.waypoint_idx]
        return None

    @property
    def at_goal(self) -> bool:
        return self.status == ExecStatus.REACHED

# ================================================================
#  PATH EXECUTOR
# ================================================================
class PathExecutor:
    """
    Converts A* paths into high-level NavigationCommands for a single robot.

    Responsibilities:
      - Accept a new path (from replanning) and reset waypoint tracking.
      - Each frame: receive robot's current cell + heading and return a command.
      - Detect waypoint arrival and advance the index.
      - Detect goal arrival and set REACHED status.
      - Recover gracefully when a replan replaces the current path.
    """

    def __init__(self, robot_id: int):
        self.state = RobotState(robot_id=robot_id)

    def set_path(self, path: Optional[List[Tuple[int,int]]]):
        """
        Called by the main loop when A* produces a new path.
        Resets execution safely:
          - If robot is already past some waypoints on the new path, fast-forward.
          - If path is None (blocked), enter BLOCKED status.
        """
        if path is None:
            self.state.path        = []
            self.state.waypoint_idx = 0
            self.state.status      = ExecStatus.BLOCKED
            self.state.command     = NavigationCommand.BLOCKED
            return

        self.state.path   = path
        self.state.status = ExecStatus.EXECUTING

        # Fast-forward: if the robot's current cell is already in the new path,
        # start from the next waypoint after current position to avoid backtrack.
        current = self.state.cell
        try:
            idx = path.index(current)
            self.state.waypoint_idx = idx + 1
        except ValueError:
            self.state.waypoint_idx = 0

        # Check if already at goal
        if self.state.waypoint_idx >= len(path):
            self.state.status  = ExecStatus.REACHED
            self.state.command = NavigationCommand.REACHED_GOAL

    def update(self, cell: Tuple[int,int], heading_deg: float) -> NavigationCommand:
        """
        Call every frame with the robot's current cell and heading.
        Updates the state machine and returns the current NavigationCommand.
        """
        self.state.cell        = cell
        self.state.heading_deg = heading_deg

        # ---- Terminal states ----
        if self.state.status == ExecStatus.IDLE:
            self.state.command = NavigationCommand.IDLE
            return self.state.command

        if self.state.status == ExecStatus.REACHED:
            self.state.command = NavigationCommand.REACHED_GOAL
            return self.state.command

        if self.state.status == ExecStatus.BLOCKED:
            self.state.command = NavigationCommand.BLOCKED
            return self.state.command

        # ---- EXECUTING ----
        # Advance waypoint index if robot has reached current target
        while (self.state.waypoint_idx < len(self.state.path) and
               cell == self.state.path[self.state.waypoint_idx]):
            self.state.waypoint_idx += 1

        # Check goal reached
        if self.state.waypoint_idx >= len(self.state.path):
            self.state.status  = ExecStatus.REACHED
            self.state.command = NavigationCommand.REACHED_GOAL
            return self.state.command

        # Determine required movement direction to next waypoint
        next_cell            = self.state.path[self.state.waypoint_idx]
        required_dir         = cell_direction(cell, next_cell)
        self.state.target_dir = required_dir

        required_angle = _DIR_TO_ANGLE[required_dir]
        diff           = angle_diff(required_angle, heading_deg)

        if abs(diff) <= HEADING_TOLERANCE_DEG:
            cmd = NavigationCommand.MOVE_FORWARD
        elif diff > 0:
            cmd = NavigationCommand.TURN_LEFT
        else:
            cmd = NavigationCommand.TURN_RIGHT

        self.state.command = cmd
        return cmd

# ================================================================
#  NAVIGATION CONTROLLER
# ================================================================
class NavigationController:
    """
    Manages PathExecutor instances for all active robots.
    Acts as the bridge between the planner and the execution layer.
    """

    def __init__(self):
        self._executors: Dict[int, PathExecutor] = {}

    def on_replan(self, robot_id: int,
                  path: Optional[List[Tuple[int,int]]]):
        """Called when A* produces a new or updated path for a robot."""
        if robot_id not in self._executors:
            self._executors[robot_id] = PathExecutor(robot_id)
        self._executors[robot_id].set_path(path)

    def update(self, robot_id:   int,
               cell:        Tuple[int,int],
               heading_deg: float) -> NavigationCommand:
        """Update execution state and return the current command."""
        if robot_id not in self._executors:
            self._executors[robot_id] = PathExecutor(robot_id)
        cmd = self._executors[robot_id].update(cell, heading_deg)
        # Console output — throttle to avoid spam (print only on change)
        ex = self._executors[robot_id]
        return cmd

    def state(self, robot_id: int) -> Optional[RobotState]:
        ex = self._executors.get(robot_id)
        return ex.state if ex else None

    def remove(self, robot_id: int):
        self._executors.pop(robot_id, None)

    def active_ids(self):
        return list(self._executors.keys())

# ================================================================
#  OCCUPANCY MAP  (preserved from v3 exactly)
# ================================================================
class OccupancyMap:
    def __init__(self, static_grid):
        self.rows = len(static_grid)
        self.cols = len(static_grid[0])
        self._static     = [row[:] for row in static_grid]
        self.confidence  = np.zeros((self.rows, self.cols), dtype=np.float32)
        self._history    = np.zeros(
            (self.rows, self.cols, STABILITY_WINDOW), dtype=bool)
        self._history_idx = 0
        self._fused      = None
        self._fused_hash = None
        self._prev_hash  = None
        self._dirty      = True

    def update(self, occupied_cells: set):
        slot = self._history_idx % STABILITY_WINDOW
        for r in range(self.rows):
            for c in range(self.cols):
                detected = (r, c) in occupied_cells
                self._history[r, c, slot] = detected
                if detected:
                    self.confidence[r, c] = min(
                        1.0, self.confidence[r, c] + CONFIDENCE_REINFORCE)
                else:
                    detection_rate = self._history[r, c].sum() / STABILITY_WINDOW
                    decay = (CONFIDENCE_DECAY_SLOW +
                             (CONFIDENCE_DECAY_FAST - CONFIDENCE_DECAY_SLOW)
                             * (1.0 - detection_rate))
                    self.confidence[r, c] = max(0.0, self.confidence[r, c] - decay)
        self._history_idx += 1
        self._dirty = True

    def fused_grid(self):
        if not self._dirty and self._fused is not None:
            return self._fused
        fused = []
        for r in range(self.rows):
            row = []
            for c in range(self.cols):
                blocked = (self._static[r][c] == 1 or
                           float(self.confidence[r, c]) >= CONFIDENCE_THRESHOLD)
                row.append(1 if blocked else 0)
            fused.append(row)
        self._fused      = fused
        self._fused_hash = self._hash(fused)
        self._dirty      = False
        return fused

    def fused_changed(self):
        self.fused_grid()
        changed         = self._fused_hash != self._prev_hash
        self._prev_hash = self._fused_hash
        return changed

    def static_blocked(self, r, c):  return self._static[r][c] == 1
    def dynamic_confidence(self, r, c): return float(self.confidence[r, c])
    def is_dynamic_occupied(self, r, c):
        return float(self.confidence[r, c]) >= CONFIDENCE_THRESHOLD

    @staticmethod
    def _hash(grid):
        return tuple(cell for row in grid for cell in row)

# ================================================================
#  ARUCO
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
        pts[np.argmin(s)], pts[np.argmin(d)],
        pts[np.argmax(s)], pts[np.argmax(d)],
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
#  A*  (preserved from v3 exactly)
# ================================================================
NEIGHBOURS = [
    (-1,  0, False), ( 1,  0, False),
    ( 0, -1, False), ( 0,  1, False),
    (-1, -1, True),  (-1,  1, True),
    ( 1, -1, True),  ( 1,  1, True),
]

def heuristic(a, b):
    dr, dc = abs(a[0]-b[0]), abs(a[1]-b[1])
    return max(dr, dc) + (DIAGONAL_COST - 1) * min(dr, dc)

def astar(grid, start, goal):
    rows, cols = len(grid), len(grid[0])
    if grid[start[0]][start[1]] == 1 or grid[goal[0]][goal[1]] == 1:
        return None
    open_set = []
    heapq.heappush(open_set, (0.0, start))
    came_from = {}
    g_score   = {(r,c): float("inf") for r in range(rows) for c in range(cols)}
    g_score[start] = 0.0
    while open_set:
        _, current = heapq.heappop(open_set)
        if current == goal:
            path = []
            while current in came_from:
                path.append(current); current = came_from[current]
            path.append(start); path.reverse(); return path
        r, c = current
        for dr, dc, diag in NEIGHBOURS:
            nr, nc = r+dr, c+dc
            if nr<0 or nr>=rows or nc<0 or nc>=cols: continue
            if grid[nr][nc] == 1: continue
            if diag and (grid[r+dr][c]==1 or grid[r][c+dc]==1): continue
            tg = g_score[current] + (DIAGONAL_COST if diag else 1.0)
            if tg < g_score[(nr,nc)]:
                came_from[(nr,nc)] = current
                g_score[(nr,nc)]   = tg
                heapq.heappush(open_set, (tg+heuristic((nr,nc),goal),(nr,nc)))
    return None

# ================================================================
#  DRAWING HELPERS
# ================================================================
def _cell_center(r, c, cell_w, cell_h):
    return (c * cell_w + cell_w // 2, r * cell_h + cell_h // 2)

def draw_occupancy_layer(img, occ_map, rows, cols):
    h, w   = img.shape[:2]
    cell_w, cell_h = w//cols, h//rows
    for r in range(rows):
        for c in range(cols):
            x1, y1 = c*cell_w, r*cell_h
            x2, y2 = x1+cell_w, y1+cell_h
            if occ_map.static_blocked(r, c):
                cv2.rectangle(img, (x1,y1),(x2,y2),(60,60,60),-1)
                cv2.putText(img,"X",(x1+cell_w//2-10,y1+cell_h//2+10),
                            cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),3)
            else:
                conf = occ_map.dynamic_confidence(r,c)
                if conf >= CONFIDENCE_THRESHOLD:
                    alpha = min(0.55, conf*0.6)
                    ov = img.copy()
                    cv2.rectangle(ov,(x1,y1),(x2,y2),(0,0,200),-1)
                    cv2.addWeighted(ov,alpha,img,1-alpha,0,img)
                    bw = int((x2-x1)*conf)
                    cv2.rectangle(img,(x1,y2-6),(x1+bw,y2),(0,0,255),-1)
                elif conf > 0.05:
                    alpha = conf*0.4
                    ov = img.copy()
                    cv2.rectangle(ov,(x1,y1),(x2,y2),(0,200,255),-1)
                    cv2.addWeighted(ov,alpha,img,1-alpha,0,img)

def draw_path(img, path, rows, cols):
    if not path: return
    h, w   = img.shape[:2]
    cell_w, cell_h = w//cols, h//rows
    for i in range(len(path)-1):
        x1,y1 = _cell_center(path[i][0],   path[i][1],   cell_w, cell_h)
        x2,y2 = _cell_center(path[i+1][0], path[i+1][1], cell_w, cell_h)
        cv2.line(img,(x1,y1),(x2,y2),(255,0,255),4)
    gr,gc = path[-1]
    cv2.circle(img, _cell_center(gr,gc,cell_w,cell_h), 12,(0,255,255),-1)

def draw_waypoint_marker(img, target_cell, waypoint_idx, rows, cols):
    """Yellow ring on the current target waypoint cell."""
    if target_cell is None: return
    h, w   = img.shape[:2]
    cell_w, cell_h = w//cols, h//rows
    cx, cy = _cell_center(target_cell[0], target_cell[1], cell_w, cell_h)
    cv2.circle(img,(cx,cy),18,(0,255,200),3)
    cv2.putText(img, f"W{waypoint_idx}", (cx-14,cy-22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,200), 2)

def highlight_cell(img, row, col, rows, cols,
                   color=(0,80,255), alpha=0.35):
    h, w   = img.shape[:2]
    cell_w, cell_h = w//cols, h//rows
    x1 = max(0, col*cell_w);  y1 = max(0, row*cell_h)
    x2 = min(w, x1+cell_w);   y2 = min(h, y1+cell_h)
    ov = img.copy()
    cv2.rectangle(ov,(x1,y1),(x2,y2),color,-1)
    cv2.addWeighted(ov,alpha,img,1-alpha,0,img)

def draw_robot_on_warped(img, wx, wy, fwd_w, robot_state: RobotState,
                          rows, cols):
    """Draw robot dot, heading arrow, and execution info panel."""
    cmd   = robot_state.command
    color = CMD_COLORS.get(cmd, (200,200,200))

    # Cell highlight
    highlight_cell(img, robot_state.cell[0], robot_state.cell[1],
                   rows, cols, color=(0,200,80))

    # Heading arrow
    cv2.circle(img,(wx,wy),8,(0,0,255),-1)
    cv2.arrowedLine(img,(wx,wy),tuple(fwd_w),(0,200,255),2,tipLength=0.3)

    # Info panel — dark background box
    lines = [
        f"ID{robot_state.robot_id}  ({robot_state.cell[0]},{robot_state.cell[1]})",
        f"HDG: {robot_state.heading_deg:.0f}deg  DIR:{robot_state.target_dir.value}",
        f"CMD: {cmd.value}",
        f"WPT: {robot_state.waypoint_idx}/{max(len(robot_state.path)-1,0)}",
        f"STS: {robot_state.status.name}",
    ]
    lh   = 18
    bx   = wx + 12
    by   = wy - len(lines)*lh - 4
    bw   = 210
    # Clamp panel inside image
    h_img = img.shape[0]
    if by < 4: by = wy + 12
    by = max(4, min(by, h_img - len(lines)*lh - 8))

    cv2.rectangle(img,(bx-2,by-14),(bx+bw,by+len(lines)*lh),(20,20,20),-1)
    for k, ln in enumerate(lines):
        lcolor = color if k == 2 else (220,220,220)
        cv2.putText(img, ln, (bx, by+k*lh),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, lcolor, 1)

def draw_goal_reached(img, goal_cell, rows, cols):
    """Flashing cyan overlay on goal cell."""
    h, w   = img.shape[:2]
    cell_w, cell_h = w//cols, h//rows
    r, c = goal_cell
    x1,y1 = c*cell_w, r*cell_h
    x2,y2 = x1+cell_w, y1+cell_h
    ov = img.copy()
    cv2.rectangle(ov,(x1,y1),(x2,y2),(0,255,255),-1)
    cv2.addWeighted(ov,0.4,img,0.6,0,img)
    cx,cy = _cell_center(r,c,cell_w,cell_h)
    cv2.putText(img,"GOAL",(cx-22,cy+6),
                cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,0),2)

def draw_grid(img, rows, cols, color=(0,255,0), thickness=2):
    h, w   = img.shape[:2]
    cell_w, cell_h = w//cols, h//rows
    for r in range(rows+1):
        cv2.line(img,(0,r*cell_h),(w,r*cell_h),color,thickness)
    for c in range(cols+1):
        cv2.line(img,(c*cell_w,0),(c*cell_w,h),color,thickness)
    for r in range(rows):
        for c in range(cols):
            cx = c*cell_w+cell_w//2; cy = r*cell_h+cell_h//2
            cv2.putText(img,f"{r},{c}",(cx-18,cy+6),
                        cv2.FONT_HERSHEY_SIMPLEX,0.55,color,1)

def project_grid_back(frame, ordered, rows, cols, warp_w, warp_h, M_inv):
    cell_w, cell_h = warp_w//cols, warp_h//rows
    def wp(px,py):
        pt = np.array([[[float(px),float(py)]]],dtype="float32")
        return tuple(cv2.perspectiveTransform(pt,M_inv)[0][0].astype(int))
    for r in range(1,rows):
        cv2.line(frame,wp(0,r*cell_h),wp(warp_w,r*cell_h),(0,255,0),2)
    for c in range(1,cols):
        cv2.line(frame,wp(c*cell_w,0),wp(c*cell_w,warp_h),(0,255,0),2)
    for i in range(4):
        p1=tuple(ordered[i].astype(int)); p2=tuple(ordered[(i+1)%4].astype(int))
        cv2.line(frame,p1,p2,(0,140,255),3)

def draw_hud(frame, rect_found, manual_lock, fps, replan_count):
    h = frame.shape[0]
    if manual_lock:
        label,color = "Boundary: MANUAL LOCK",        (0,200,255)
    elif rect_found:
        label,color = "Boundary: TRACKING",           (0,220,80)
    else:
        label,color = "Boundary: LOST — searching...",(0,80,255)
    cv2.putText(frame,label,(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.72,color,2)
    cv2.putText(frame,f"FPS: {fps:.1f}",(10,58),
                cv2.FONT_HERSHEY_SIMPLEX,0.65,(200,200,200),2)
    cv2.putText(frame,f"Replans: {replan_count}",(10,84),
                cv2.FONT_HERSHEY_SIMPLEX,0.55,(180,180,180),1)
    cv2.putText(frame,"Q/ESC=quit  L=manual lock  D=edge debug",
                (10,h-12),cv2.FONT_HERSHEY_SIMPLEX,0.45,(160,160,160),1)

# ================================================================
#  COMMAND LOGGER — prints to console only on state change
# ================================================================
class CommandLogger:
    def __init__(self):
        self._last: Dict[int, NavigationCommand] = {}

    def log(self, robot_id: int, state: RobotState):
        cmd = state.command
        if self._last.get(robot_id) != cmd:
            self._last[robot_id] = cmd
            tc = state.target_cell
            print(
                f"[Robot {robot_id}] "
                f"Cell({state.cell[0]},{state.cell[1]}) "
                f"HDG:{state.heading_deg:.0f}° "
                f"DIR:{state.target_dir.value:2s}  "
                f"→  {cmd.value}"
                + (f"  (target {tc})" if tc else "")
            )

# ================================================================
#  MAIN
# ================================================================
def main():
    cap = cv2.VideoCapture(CAMERA_SOURCE)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        print(f"ERROR: Cannot open camera: {CAMERA_SOURCE}"); return

    # Vision
    last_ordered = last_M = last_M_inv = last_warped = None

    # Occupancy (v3 preserved)
    occ_map = OccupancyMap(STATIC_GRID)

    # Planning
    robot_paths:  Dict[int, Optional[list]] = {}
    replan_count = 0

    # Execution layer (new in v4)
    nav_ctrl = NavigationController()
    cmd_log  = CommandLogger()

    manual_lock = show_debug = False
    prev_time   = time.time()

    print("\n=== Navigation Execution System v4 ===")
    print("  Q/ESC -> quit  |  L -> manual lock  |  D -> debug edges")
    print(f"  Goal: {GOAL_CELL}  |  Heading tolerance: {HEADING_TOLERANCE_DEG}°")
    print("=======================================\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Cannot read frame"); break

        display = frame.copy()
        gray    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        now = time.time()
        fps = 1.0 / max(now - prev_time, 1e-9)
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
        # detected_robots: id -> (wx,wy,row,col,angle,fwd_w,cam_cx,cam_cy)
        detected_robots: Dict[int,tuple] = {}

        if last_ordered is not None and rect_found:
            cell_w = WARP_W // GRID_COLS
            cell_h = WARP_H // GRID_ROWS
            corners, ids = detect_aruco(gray)
            if ids is not None:
                for i in range(len(ids)):
                    c        = corners[i][0]
                    robot_id = int(ids[i][0])
                    cam_cx   = float(c[:,0].mean())
                    cam_cy   = float(c[:,1].mean())
                    pt       = np.array([[[cam_cx,cam_cy]]],dtype="float32")
                    wpt      = cv2.perspectiveTransform(pt,last_M)[0][0]
                    wx = max(0,min(WARP_W-1,int(wpt[0])))
                    wy = max(0,min(WARP_H-1,int(wpt[1])))
                    gc = min(wx//cell_w, GRID_COLS-1)
                    gr = min(wy//cell_h, GRID_ROWS-1)

                    top_mid    = c[0:2].mean(axis=0)
                    bottom_mid = c[2:4].mean(axis=0)
                    angle = np.degrees(np.arctan2(
                        -(top_mid[1]-bottom_mid[1]),
                         (top_mid[0]-bottom_mid[0])
                    ))
                    fwd_cam = np.array([[[
                        cam_cx+(top_mid[0]-cam_cx)*2.0,
                        cam_cy+(top_mid[1]-cam_cy)*2.0
                    ]]],dtype="float32")
                    fwd_w = cv2.perspectiveTransform(fwd_cam,last_M)[0][0].astype(int)

                    occupied_this_frame.add((gr,gc))
                    detected_robots[robot_id] = (wx,wy,gr,gc,angle,fwd_w,cam_cx,cam_cy)

        if rect_found:
            occ_map.update(occupied_this_frame)

        # ============================================================
        #  STEP 3 — REPLANNING (hash-gated, v3 logic preserved)
        # ============================================================
        if rect_found and occ_map.fused_changed():
            fused = occ_map.fused_grid()
            replan_count += 1
            for rid, (_, _, gr, gc, _, _, _, _) in detected_robots.items():
                saved = fused[gr][gc]
                fused[gr][gc] = 0
                path = astar(fused, (gr,gc), GOAL_CELL)
                fused[gr][gc] = saved
                robot_paths[rid] = path
                nav_ctrl.on_replan(rid, path)   # notify execution layer
            # Clean up stale robots
            for rid in list(robot_paths):
                if rid not in detected_robots:
                    del robot_paths[rid]
                    nav_ctrl.remove(rid)

        # ============================================================
        #  STEP 4 — EXECUTION LAYER UPDATE
        # ============================================================
        for rid, (_, _, gr, gc, angle, _, _, _) in detected_robots.items():
            cmd = nav_ctrl.update(rid, (gr,gc), angle)
            st  = nav_ctrl.state(rid)
            if st:
                cmd_log.log(rid, st)

        # ============================================================
        #  STEP 5 — WARP + VISUALISE
        # ============================================================
        if last_ordered is not None:
            if rect_found:
                last_warped = cv2.warpPerspective(frame,last_M,(WARP_W,WARP_H))

            warped_draw = last_warped.copy()

            if rect_found:
                # 1. Occupancy layer
                draw_occupancy_layer(warped_draw, occ_map, GRID_ROWS, GRID_COLS)

                # 2. Goal reached overlay (if any robot there)
                goal_reached_any = any(
                    nav_ctrl.state(rid) and
                    nav_ctrl.state(rid).status == ExecStatus.REACHED
                    for rid in detected_robots
                )
                if goal_reached_any:
                    draw_goal_reached(warped_draw, GOAL_CELL, GRID_ROWS, GRID_COLS)

                # 3. Paths
                for rid, path in robot_paths.items():
                    draw_path(warped_draw, path, GRID_ROWS, GRID_COLS)

                # 4. Waypoint markers + robot overlays
                for rid, (wx,wy,gr,gc,angle,fwd_w,cam_cx,cam_cy) in detected_robots.items():
                    st = nav_ctrl.state(rid)
                    if st:
                        draw_waypoint_marker(warped_draw, st.target_cell,
                                             st.waypoint_idx, GRID_ROWS, GRID_COLS)
                        draw_robot_on_warped(warped_draw, wx, wy, fwd_w, st,
                                             GRID_ROWS, GRID_COLS)
                    # Camera view dot
                    cv2.circle(display,(int(cam_cx),int(cam_cy)),5,(255,80,0),-1)
                    st_label = st.command.value if st else "?"
                    cv2.putText(display,f"ID{rid} {st_label}",
                                (int(cam_cx)+8,int(cam_cy)-8),
                                cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,80,0),2)

                # 5. Grid lines always on top
                draw_grid(warped_draw, GRID_ROWS, GRID_COLS)
                project_grid_back(display, last_ordered,
                                  GRID_ROWS, GRID_COLS, WARP_W, WARP_H, last_M_inv)

            # Warped window — never closes
            show_warped = warped_draw.copy()
            if not rect_found:
                cv2.rectangle(show_warped,(0,0),(WARP_W,36),(0,0,0),-1)
                cv2.putText(show_warped,"Searching... (last known view)",
                            (10,25),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,140,255),2)
            cv2.imshow("Warped Grid", show_warped)

        # ============================================================
        #  CAMERA WINDOW
        # ============================================================
        draw_hud(display, rect_found, manual_lock, fps, replan_count)
        cv2.imshow("Camera", display)

        if show_debug:
            blurred = cv2.GaussianBlur(gray,(7,7),0)
            edges   = cv2.Canny(blurred,CANNY_LOW,CANNY_HIGH)
            cv2.imshow("Edges (debug)", edges)

        # ============================================================
        #  KEY HANDLING
        # ============================================================
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27): break
        elif key == ord('l'):
            manual_lock = not manual_lock
            print(f"Manual lock: {'ON' if manual_lock else 'OFF'}")
        elif key == ord('d'):
            show_debug = not show_debug
            if not show_debug: cv2.destroyWindow("Edges (debug)")

    cap.release()
    cv2.destroyAllWindows()
    print("Exited cleanly.")


if __name__ == "__main__":
    main()
