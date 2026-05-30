"""
=============================================================================
Project: Multi-Robot Communication System with Vision-Based Tracking
Module : PC Navigation Controller + Serial LED Communication
Author : Amunugama H.M.K.D. (E/21/029)
Version: 6.0 (PC / Arduino Split Architecture)
Date   : May 30, 2026

Description:
PC-side controller for the vision, mapping, planning, and navigation stack.
This program starts from pathPlanning_v5.py and keeps the OpenCV camera input,
boundary detection, perspective transform, ArUco localisation, OccupancyMap,
A* planning, NavigationController, PathExecutor, visual overlays, and LED panel
simulation window. Physical LED GPIO control is separated and handled by an
Arduino over USB serial.

System Architecture:
  PC (Python/OpenCV) -> Serial USB -> Arduino LED Panel Controller
  -> Physical 2x2 LEDs -> Robot camera reads LED patterns

Layer Order:
  PERCEPTION -> OCCUPANCY -> PLANNING -> EXECUTION -> COMMUNICATION -> VISUALISATION

PC Responsibilities:
  - Camera input and warped arena visualisation.
  - Boundary detection, perspective transform, and ArUco localisation.
  - Occupancy mapping with adaptive confidence decay.
  - 8-direction A* path planning and hash-gated replanning.
  - Navigation execution logic and NavigationCommand generation.
  - Serial packet transmission and LED simulation window for debugging.

Arduino Responsibilities:
  - Receive simple serial packets such as R1:1010.
  - Decode the four LED bits A/B/C/D.
  - Drive the physical 2x2 LED panel using GPIO pins.

Why split PC and Arduino:
  - OpenCV, A*, and perception require PC processing power.
  - Arduino is excellent for deterministic real-time GPIO control.
  - Python should not directly drive Arduino GPIO; it should send commands.
  - Serial USB is simple, debuggable, and reliable for PC-to-Arduino control.
  - Modular separation mirrors real robotics systems and supports later
    Raspberry Pi migration.

Serial Packet Format:
  R<robot_id>:ABCD\n
Examples:
  R1:1010  -> Robot 1 receives LED pattern A=1, B=0, C=1, D=0.
  R1:1100  -> Robot 1 receives LED pattern A=1, B=1, C=0, D=0.
  R1:1111  -> Robot 1 receives LED pattern A=1, B=1, C=1, D=1.

Panel Layout:
  [ A ][ B ]
  [ C ][ D ]

Preserved from pathPlanning_v5.py:
  - OccupancyMap static/dynamic fusion with adaptive confidence decay.
  - 8-direction A* planning with corner-cut prevention and hash-gated replans.
  - NavigationCommand, RobotState, PathExecutor, NavigationController.
  - Execution state machine, waypoint tracking, goal detection.
  - Warped grid, HUD, path, robot overlay, and LED panel simulation window.
=============================================================================
"""

import cv2
import cv2.aruco as aruco
import numpy as np
import time
import heapq
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict

try:
    import serial
    import serial.tools.list_ports
    from serial import SerialException
except ImportError:
    serial = None
    SerialException = OSError

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
DIAGONAL_COST  = 1.414

# ================================================================
#  SERIAL COMMUNICATION CONFIG
# ================================================================
# The PC performs camera processing, mapping, A* planning, and command
# generation. The Arduino owns physical GPIO because it can drive LEDs with
# deterministic timing and without tying robot hardware to the OpenCV process.
SERIAL_PORT = "COM5"
SERIAL_BAUD = 115200

# ================================================================
#  OCCUPANCY CONFIG  (preserved from v3 exactly)
# ================================================================
STATIC_GRID = [
    [0, 1, 0, 0],
    [0, 1, 0, 1],
    [0, 0, 0, 1],
]

GOAL_CELL = (2, 2)

CONFIDENCE_REINFORCE  = 0.35
CONFIDENCE_DECAY_SLOW = 0.04
CONFIDENCE_DECAY_FAST = 0.12
CONFIDENCE_THRESHOLD  = 0.55
STABILITY_WINDOW      = 6

# ================================================================
#  EXECUTION CONFIG
# ================================================================
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

_DIR_TO_ANGLE: Dict[Direction, float] = {
    Direction.NORTH:      90.0,
    Direction.SOUTH:     -90.0,
    Direction.EAST:        0.0,
    Direction.WEST:      180.0,
    Direction.NORTHEAST:  45.0,
    Direction.NORTHWEST: 135.0,
    Direction.SOUTHEAST: -45.0,
    Direction.SOUTHWEST:-135.0,
    Direction.NONE:        0.0,
}

def cell_direction(from_cell: Tuple[int,int],
                   to_cell:   Tuple[int,int]) -> Direction:
    dr = to_cell[0] - from_cell[0]
    dc = to_cell[1] - from_cell[1]
    return _DELTA_TO_DIR.get((dr, dc), Direction.NONE)

def angle_diff(target_deg: float, current_deg: float) -> float:
    return (target_deg - current_deg + 180.0) % 360.0 - 180.0

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

CMD_COLORS: Dict[NavigationCommand, Tuple[int,int,int]] = {
    NavigationCommand.MOVE_FORWARD: (0,   220,  80),
    NavigationCommand.TURN_LEFT:    (0,   200, 255),
    NavigationCommand.TURN_RIGHT:   (0,   140, 255),
    NavigationCommand.STOP:         (0,    80, 255),
    NavigationCommand.REACHED_GOAL: (0,   255, 255),
    NavigationCommand.IDLE:         (160, 160, 160),
    NavigationCommand.BLOCKED:      (0,     0, 200),
}

# ================================================================
#  LED COMMUNICATION CONFIG
#
#  Panel layout (wall-mounted, robot reads with onboard camera):
#
#    [ A ][ B ]   <- top row
#    [ C ][ D ]   <- bottom row
#
#  Each LED is 1 (ON) or 0 (OFF).
#  All 7 commands have UNIQUE patterns — no two share a code.
#
#  Pattern design rationale:
#    MOVE_FORWARD  1010  Left column ON  → vertical stripe = go
#    TURN_LEFT     0011  Bottom row ON   → bottom-heavy = lean left
#    TURN_RIGHT    1100  Top row ON      → top-heavy = lean right
#    STOP          0101  Diagonal B/C    → cross-pattern = halt
#    REACHED_GOAL  1111  All ON          → full bright = done
#    BLOCKED       0110  Inner diagonal  → different from STOP
#    IDLE          0000  All OFF         → no light = no action
# ================================================================
LED_PATTERNS: Dict[NavigationCommand, Tuple[int,int,int,int]] = {
    #                              A  B  C  D
    NavigationCommand.MOVE_FORWARD: (1, 0, 1, 0),  # 1010
    NavigationCommand.TURN_LEFT:    (0, 0, 1, 1),  # 0011
    NavigationCommand.TURN_RIGHT:   (1, 1, 0, 0),  # 1100
    NavigationCommand.STOP:         (0, 1, 0, 1),  # 0101  ← fixed (was 0000)
    NavigationCommand.REACHED_GOAL: (1, 1, 1, 1),  # 1111
    NavigationCommand.BLOCKED:      (0, 1, 1, 0),  # 0110
    NavigationCommand.IDLE:         (0, 0, 0, 0),  # 0000
}

# Reverse lookup: pattern tuple -> command (for robot-side decoding)
PATTERN_TO_CMD: Dict[Tuple[int,int,int,int], NavigationCommand] = {
    v: k for k, v in LED_PATTERNS.items()
}

# ================================================================
#  EXECUTION STATUS
# ================================================================
class ExecStatus(Enum):
    IDLE      = auto()
    EXECUTING = auto()
    REACHED   = auto()
    BLOCKED   = auto()

# ================================================================
#  ROBOT STATE  (preserved from v4)
# ================================================================
@dataclass
class RobotState:
    robot_id:     int
    cell:         Tuple[int,int]       = (0, 0)
    heading_deg:  float                = 0.0
    path:         List[Tuple[int,int]] = field(default_factory=list)
    waypoint_idx: int                  = 0
    status:       ExecStatus           = ExecStatus.IDLE
    command:      NavigationCommand    = NavigationCommand.IDLE
    target_dir:   Direction            = Direction.NONE

    @property
    def target_cell(self) -> Optional[Tuple[int,int]]:
        if self.path and self.waypoint_idx < len(self.path):
            return self.path[self.waypoint_idx]
        return None

    @property
    def at_goal(self) -> bool:
        return self.status == ExecStatus.REACHED

# ================================================================
#  LED PANEL STATE
# ================================================================
@dataclass
class LEDPanelState:
    """
    Tracks the current state of the wall-mounted 2x2 LED panel.
    One instance per robot channel (for multi-robot multiplexing).
    """
    robot_id:       int                    = 0
    pattern:        Tuple[int,int,int,int] = (0, 0, 0, 0)
    active_command: NavigationCommand      = NavigationCommand.IDLE
    last_update_ts: float                  = 0.0

    @property
    def A(self): return self.pattern[0]
    @property
    def B(self): return self.pattern[1]
    @property
    def C(self): return self.pattern[2]
    @property
    def D(self): return self.pattern[3]

# ================================================================
#  PATH EXECUTOR  (preserved from v4 exactly)
# ================================================================
class PathExecutor:
    def __init__(self, robot_id: int):
        self.state = RobotState(robot_id=robot_id)

    def set_path(self, path: Optional[List[Tuple[int,int]]]):
        if path is None:
            self.state.path         = []
            self.state.waypoint_idx = 0
            self.state.status       = ExecStatus.BLOCKED
            self.state.command      = NavigationCommand.BLOCKED
            return
        self.state.path   = path
        self.state.status = ExecStatus.EXECUTING
        current = self.state.cell
        try:
            idx = path.index(current)
            self.state.waypoint_idx = idx + 1
        except ValueError:
            self.state.waypoint_idx = 0
        if self.state.waypoint_idx >= len(path):
            self.state.status  = ExecStatus.REACHED
            self.state.command = NavigationCommand.REACHED_GOAL

    def update(self, cell: Tuple[int,int],
               heading_deg: float) -> NavigationCommand:
        self.state.cell        = cell
        self.state.heading_deg = heading_deg

        if self.state.status == ExecStatus.IDLE:
            self.state.command = NavigationCommand.IDLE
            return self.state.command
        if self.state.status == ExecStatus.REACHED:
            self.state.command = NavigationCommand.REACHED_GOAL
            return self.state.command
        if self.state.status == ExecStatus.BLOCKED:
            self.state.command = NavigationCommand.BLOCKED
            return self.state.command

        while (self.state.waypoint_idx < len(self.state.path) and
               cell == self.state.path[self.state.waypoint_idx]):
            self.state.waypoint_idx += 1

        if self.state.waypoint_idx >= len(self.state.path):
            self.state.status  = ExecStatus.REACHED
            self.state.command = NavigationCommand.REACHED_GOAL
            return self.state.command

        next_cell             = self.state.path[self.state.waypoint_idx]
        required_dir          = cell_direction(cell, next_cell)
        self.state.target_dir = required_dir
        required_angle        = _DIR_TO_ANGLE[required_dir]
        diff                  = angle_diff(required_angle, heading_deg)

        if abs(diff) <= HEADING_TOLERANCE_DEG:
            cmd = NavigationCommand.MOVE_FORWARD
        elif diff > 0:
            cmd = NavigationCommand.TURN_LEFT
        else:
            cmd = NavigationCommand.TURN_RIGHT

        self.state.command = cmd
        return cmd

# ================================================================
#  NAVIGATION CONTROLLER  (preserved from v4 exactly)
# ================================================================
class NavigationController:
    def __init__(self):
        self._executors: Dict[int, PathExecutor] = {}

    def on_replan(self, robot_id: int,
                  path: Optional[List[Tuple[int,int]]]):
        if robot_id not in self._executors:
            self._executors[robot_id] = PathExecutor(robot_id)
        self._executors[robot_id].set_path(path)

    def update(self, robot_id: int,
               cell: Tuple[int,int],
               heading_deg: float) -> NavigationCommand:
        if robot_id not in self._executors:
            self._executors[robot_id] = PathExecutor(robot_id)
        return self._executors[robot_id].update(cell, heading_deg)

    def state(self, robot_id: int) -> Optional[RobotState]:
        ex = self._executors.get(robot_id)
        return ex.state if ex else None

    def remove(self, robot_id: int):
        self._executors.pop(robot_id, None)

    def active_ids(self):
        return list(self._executors.keys())

# ================================================================
#  SERIAL LED BRIDGE
#
#  Serial USB is the clean boundary between the PC and Arduino.
#  The PC should not directly drive GPIO pins because it is busy with OpenCV,
#  A* planning, and visualisation. Arduino is separated so it can perform the
#  simple but timing-sensitive LED output task reliably.
# ================================================================
class SerialLEDBridge:
    """
    Sends NavigationCommand as a single byte over USB serial to Arduino.
    The byte value is the 4-bit LED pattern interpreted as an integer.
    """

    CMD_BYTES: Dict[NavigationCommand, int] = {
        NavigationCommand.MOVE_FORWARD:  0x0A,  # 1010
        NavigationCommand.TURN_LEFT:     0x03,  # 0011
        NavigationCommand.TURN_RIGHT:    0x0C,  # 1100
        NavigationCommand.STOP:          0x05,  # 0101
        NavigationCommand.REACHED_GOAL:  0x0F,  # 1111
        NavigationCommand.BLOCKED:       0x06,  # 0110
        NavigationCommand.IDLE:          0x00,  # 0000
    }

    def __init__(self, port: Optional[str] = None, baud: int = 9600):
        self.port  = port or self._auto_detect()
        self.baud  = baud
        self._ser  = None
        self._last = None

        if serial is None:
            print("[Serial] pyserial is not installed - running in simulation mode")
            return

        if not self.port:
            print("[Serial] No Arduino found - running in simulation mode")
            return

        try:
            self._ser = serial.Serial(self.port, self.baud, timeout=1)
            time.sleep(2.0)  # Wait for Arduino reset after serial connect.
            print(f"[Serial] Connected on {self.port} at {self.baud} baud")
        except (SerialException, OSError) as exc:
            self._ser = None
            print(f"[Serial] Failed to open {self.port}: {exc}")
            print("[Serial] Running in simulation mode")

    def send(self, command: NavigationCommand):
        byte_val = self.CMD_BYTES.get(command, 0x00)

        # Suppress send if command has not changed.
        if byte_val == self._last:
            return
        self._last = byte_val

        if self._ser and self._ser.is_open:
            try:
                self._ser.write(bytes([byte_val]))
                print(f"[Serial] Sent 0x{byte_val:02X} -> {command.value}")
            except (SerialException, OSError) as exc:
                print(f"[Serial] Write failed: {exc}")
                try:
                    self._ser.close()
                except (SerialException, OSError, AttributeError):
                    pass
                self._ser = None
        else:
            print(f"[Serial SIM] 0x{byte_val:02X} -> {command.value}")

    def close(self):
        if not (self._ser and self._ser.is_open):
            return
        try:
            self._ser.write(bytes([0x00]))  # Send IDLE before closing.
            self._ser.close()
            print("[Serial] Port closed")
        except (SerialException, OSError):
            pass

    @staticmethod
    def _auto_detect() -> Optional[str]:
        """Find the first Arduino-like port automatically."""
        if serial is None:
            return None

        for port in serial.tools.list_ports.comports():
            desc = port.description.lower()
            if any(k in desc for k in
                   ["arduino", "ch340", "cp210", "ftdi", "usb serial"]):
                print(f"[Serial] Auto-detected: {port.device} ({port.description})")
                return port.device
        return None

class LEDPanelController:
    """
    Manages LED panel state for one robot channel.
    Translates NavigationCommand -> 4-bit pattern -> serial packet.
    """

    def __init__(self, robot_id: int):
        self.panel_state = LEDPanelState(robot_id=robot_id)

    def set_command(self, command: NavigationCommand):
        pattern = LED_PATTERNS.get(command, (0, 0, 0, 0))

        # Suppress write if nothing changed
        if (pattern == self.panel_state.pattern and
                command == self.panel_state.active_command):
            return

        self.panel_state.pattern        = pattern
        self.panel_state.active_command = command
        self.panel_state.last_update_ts = time.time()

class LEDCommunicationController:
    """
    Manages one LEDPanelController per robot.

    Wall panel architecture note:
      The physical panel persists regardless of robot visibility.
      When a robot disappears from camera, its panel channel receives STOP.
      Panels are NOT destroyed when robots go out of frame.
    """

    def __init__(self):
        self._panels: Dict[int, LEDPanelController] = {}

    def set_command(self, robot_id: int, command: NavigationCommand):
        if robot_id not in self._panels:
            self._panels[robot_id] = LEDPanelController(robot_id)
        self._panels[robot_id].set_command(command)

    def set_stop_all(self):
        """Broadcast STOP to all active panel channels."""
        for rid in self._panels:
            self._panels[rid].set_command(NavigationCommand.STOP)

    def state(self, robot_id: int) -> Optional[LEDPanelState]:
        panel = self._panels.get(robot_id)
        return panel.panel_state if panel else None

    def all_states(self) -> Dict[int, LEDPanelState]:
        return {rid: p.panel_state for rid, p in self._panels.items()}

    # Panel channels persist — no remove() on robot disappear

# ================================================================
#  OCCUPANCY MAP  (preserved from v3 exactly)
# ================================================================
class OccupancyMap:
    def __init__(self, static_grid):
        self.rows = len(static_grid)
        self.cols = len(static_grid[0])
        self._static      = [row[:] for row in static_grid]
        self.confidence   = np.zeros((self.rows, self.cols), dtype=np.float32)
        self._history     = np.zeros(
            (self.rows, self.cols, STABILITY_WINDOW), dtype=bool)
        self._history_idx = 0
        self._fused       = None
        self._fused_hash  = None
        self._prev_hash   = None
        self._dirty       = True

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
                    dr = self._history[r, c].sum() / STABILITY_WINDOW
                    decay = (CONFIDENCE_DECAY_SLOW +
                             (CONFIDENCE_DECAY_FAST - CONFIDENCE_DECAY_SLOW)
                             * (1.0 - dr))
                    self.confidence[r, c] = max(
                        0.0, self.confidence[r, c] - decay)
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
    blurred  = cv2.GaussianBlur(gray, (7, 7), 0)
    edges    = cv2.Canny(blurred, CANNY_LOW, CANNY_HIGH)
    kernel   = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    edges    = cv2.dilate(edges, kernel, iterations=2)
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
    g_score   = {(r, c): float("inf")
                 for r in range(rows) for c in range(cols)}
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
        for dr, dc, diag in NEIGHBOURS:
            nr, nc = r+dr, c+dc
            if nr < 0 or nr >= rows or nc < 0 or nc >= cols:
                continue
            if grid[nr][nc] == 1:
                continue
            if diag and (grid[r+dr][c] == 1 or grid[r][c+dc] == 1):
                continue
            tg = g_score[current] + (DIAGONAL_COST if diag else 1.0)
            if tg < g_score[(nr, nc)]:
                came_from[(nr, nc)] = current
                g_score[(nr, nc)]   = tg
                heapq.heappush(open_set,
                               (tg + heuristic((nr, nc), goal), (nr, nc)))
    return None

# ================================================================
#  DRAWING HELPERS
# ================================================================
def _cell_center(r, c, cell_w, cell_h):
    return (c * cell_w + cell_w // 2, r * cell_h + cell_h // 2)

def draw_occupancy_layer(img, occ_map, rows, cols):
    h, w         = img.shape[:2]
    cell_w, cell_h = w // cols, h // rows
    for r in range(rows):
        for c in range(cols):
            x1, y1 = c * cell_w, r * cell_h
            x2, y2 = x1 + cell_w, y1 + cell_h
            if occ_map.static_blocked(r, c):
                cv2.rectangle(img, (x1, y1), (x2, y2), (60, 60, 60), -1)
                cv2.putText(img, "X",
                            (x1 + cell_w//2 - 10, y1 + cell_h//2 + 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            else:
                conf = occ_map.dynamic_confidence(r, c)
                if conf >= CONFIDENCE_THRESHOLD:
                    alpha = min(0.55, conf * 0.6)
                    ov    = img.copy()
                    cv2.rectangle(ov, (x1, y1), (x2, y2), (0, 0, 200), -1)
                    cv2.addWeighted(ov, alpha, img, 1 - alpha, 0, img)
                    bw = int((x2 - x1) * conf)
                    cv2.rectangle(img, (x1, y2-6), (x1+bw, y2), (0, 0, 255), -1)
                elif conf > 0.05:
                    alpha = conf * 0.4
                    ov    = img.copy()
                    cv2.rectangle(ov, (x1, y1), (x2, y2), (0, 200, 255), -1)
                    cv2.addWeighted(ov, alpha, img, 1 - alpha, 0, img)

def draw_path(img, path, rows, cols):
    if not path:
        return
    h, w         = img.shape[:2]
    cell_w, cell_h = w // cols, h // rows
    for i in range(len(path) - 1):
        x1, y1 = _cell_center(path[i][0],   path[i][1],   cell_w, cell_h)
        x2, y2 = _cell_center(path[i+1][0], path[i+1][1], cell_w, cell_h)
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 4)
    gr, gc = path[-1]
    cv2.circle(img, _cell_center(gr, gc, cell_w, cell_h), 12, (0, 255, 255), -1)

def draw_waypoint_marker(img, target_cell, waypoint_idx, rows, cols):
    if target_cell is None:
        return
    h, w         = img.shape[:2]
    cell_w, cell_h = w // cols, h // rows
    cx, cy = _cell_center(target_cell[0], target_cell[1], cell_w, cell_h)
    cv2.circle(img, (cx, cy), 18, (0, 255, 200), 3)
    cv2.putText(img, f"W{waypoint_idx}", (cx-14, cy-22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 200), 2)

def highlight_cell(img, row, col, rows, cols,
                   color=(0, 80, 255), alpha=0.35):
    h, w         = img.shape[:2]
    cell_w, cell_h = w // cols, h // rows
    x1 = max(0, col * cell_w);    y1 = max(0, row * cell_h)
    x2 = min(w, x1 + cell_w);     y2 = min(h, y1 + cell_h)
    ov = img.copy()
    cv2.rectangle(ov, (x1, y1), (x2, y2), color, -1)
    cv2.addWeighted(ov, alpha, img, 1 - alpha, 0, img)

def draw_led_grid_inline(img, pattern: Tuple[int,int,int,int],
                         origin_x: int, origin_y: int,
                         led_size: int = 14, gap: int = 4):
    """
    Draw a visual 2×2 LED grid at a given pixel position inside an image.
    ON  = bright green circle
    OFF = dark grey circle
    Used inside the robot info panel on the warped view.
    """
    a, b, c, d = pattern
    leds = [(a, origin_x,          origin_y),
            (b, origin_x+led_size+gap, origin_y),
            (c, origin_x,          origin_y+led_size+gap),
            (d, origin_x+led_size+gap, origin_y+led_size+gap)]
    for state, lx, ly in leds:
        color  = (0, 220, 60) if state else (50, 50, 50)
        center = (lx + led_size//2, ly + led_size//2)
        cv2.circle(img, center, led_size//2, color, -1)
        cv2.circle(img, center, led_size//2, (80, 80, 80), 1)

def draw_robot_on_warped(img, wx, wy, fwd_w,
                          robot_state: RobotState,
                          rows, cols,
                          led_state: Optional[LEDPanelState] = None):
    """Robot dot, heading arrow, and engineering info panel with LED grid."""
    cmd    = robot_state.command
    color  = CMD_COLORS.get(cmd, (200, 200, 200))
    pattern = (led_state.pattern if led_state
                else LED_PATTERNS.get(cmd, (0, 0, 0, 0)))

    highlight_cell(img, robot_state.cell[0], robot_state.cell[1],
                   rows, cols, color=(0, 200, 80))

    cv2.circle(img, (wx, wy), 8, (0, 0, 255), -1)
    cv2.arrowedLine(img, (wx, wy), tuple(fwd_w),
                    (0, 200, 255), 2, tipLength=0.3)

    lines = [
        f"ID{robot_state.robot_id}  ({robot_state.cell[0]},{robot_state.cell[1]})",
        f"HDG:{robot_state.heading_deg:.0f}  DIR:{robot_state.target_dir.value}",
        f"CMD: {cmd.value}",
        f"WPT: {robot_state.waypoint_idx}/{max(len(robot_state.path)-1,0)}",
        f"STS: {robot_state.status.name}",
        "",  # blank line — LED grid drawn here graphically
    ]
    lh  = 18
    bx  = wx + 12
    by  = wy - len(lines) * lh - 30
    bw  = 220

    h_img = img.shape[0]
    if by < 4:
        by = wy + 12
    by = max(4, min(by, h_img - len(lines) * lh - 40))

    cv2.rectangle(img, (bx-2, by-14),
                  (bx+bw, by+len(lines)*lh+28), (20, 20, 20), -1)

    for k, ln in enumerate(lines):
        lcolor = color if k == 2 else (220, 220, 220)
        cv2.putText(img, ln, (bx, by + k * lh),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, lcolor, 1)

    # LED visual grid at bottom of panel
    led_y = by + len(lines) * lh - 6
    cv2.putText(img, "LED:", (bx, led_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)
    draw_led_grid_inline(img, pattern, bx + 38, led_y - 12)

def draw_goal_reached(img, goal_cell, rows, cols):
    h, w         = img.shape[:2]
    cell_w, cell_h = w // cols, h // rows
    r, c = goal_cell
    x1, y1 = c * cell_w, r * cell_h
    x2, y2 = x1 + cell_w, y1 + cell_h
    ov = img.copy()
    cv2.rectangle(ov, (x1, y1), (x2, y2), (0, 255, 255), -1)
    cv2.addWeighted(ov, 0.4, img, 0.6, 0, img)
    cx, cy = _cell_center(r, c, cell_w, cell_h)
    cv2.putText(img, "GOAL", (cx-22, cy+6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

def draw_grid(img, rows, cols, color=(0, 255, 0), thickness=2):
    h, w         = img.shape[:2]
    cell_w, cell_h = w // cols, h // rows
    for r in range(rows + 1):
        cv2.line(img, (0, r*cell_h), (w, r*cell_h), color, thickness)
    for c in range(cols + 1):
        cv2.line(img, (c*cell_w, 0), (c*cell_w, h), color, thickness)
    for r in range(rows):
        for c in range(cols):
            cx = c * cell_w + cell_w // 2
            cy = r * cell_h + cell_h // 2
            cv2.putText(img, f"{r},{c}", (cx-18, cy+6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)

def project_grid_back(frame, ordered, rows, cols, warp_w, warp_h, M_inv):
    cell_w, cell_h = warp_w // cols, warp_h // rows
    def wp(px, py):
        pt = np.array([[[float(px), float(py)]]], dtype="float32")
        return tuple(cv2.perspectiveTransform(pt, M_inv)[0][0].astype(int))
    for r in range(1, rows):
        cv2.line(frame, wp(0, r*cell_h), wp(warp_w, r*cell_h), (0, 255, 0), 2)
    for c in range(1, cols):
        cv2.line(frame, wp(c*cell_w, 0), wp(c*cell_w, warp_h), (0, 255, 0), 2)
    for i in range(4):
        p1 = tuple(ordered[i].astype(int))
        p2 = tuple(ordered[(i+1) % 4].astype(int))
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
                (10, h-12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1)

# ================================================================
#  LED PANEL SIMULATION WINDOW
#
#  Renders a large visual representation of the wall-mounted panel.
#  This is what the robot's onboard camera would see on the wall.
#  Updated every frame with the current active pattern.
# ================================================================
PANEL_WIN_W = 300
PANEL_WIN_H = 340

def render_led_panel_window(led_states: Dict[int, LEDPanelState]) -> np.ndarray:
    """
    Render a simulation of the physical wall-mounted LED panel.
    Shows one section per robot ID.
    ON  = large bright green circle
    OFF = large dark circle with dim outline
    """
    n      = max(len(led_states), 1)
    height = PANEL_WIN_H * n
    canvas = np.zeros((height, PANEL_WIN_W, 3), dtype=np.uint8)
    canvas[:] = (30, 30, 30)

    for idx, (rid, ps) in enumerate(sorted(led_states.items())):
        y_off  = idx * PANEL_WIN_H
        cmd    = ps.active_command
        color  = CMD_COLORS.get(cmd, (160, 160, 160))
        pattern = ps.pattern
        a, b, c, d = pattern

        # Section header
        cv2.rectangle(canvas, (0, y_off), (PANEL_WIN_W, y_off+36),
                      (50, 50, 50), -1)
        cv2.putText(canvas, f"Robot {rid}  —  {cmd.value}",
                    (10, y_off+24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)

        # Large LED circles
        r_led  = 50   # radius
        gap    = 20
        left   = PANEL_WIN_W // 2 - r_led - gap // 2
        right  = PANEL_WIN_W // 2 + gap // 2
        top    = y_off + 60
        bottom = top + r_led * 2 + gap

        leds = [
            (a, left,  top),     # A — top left
            (b, right, top),     # B — top right
            (c, left,  bottom),  # C — bottom left
            (d, right, bottom),  # D — bottom right
        ]
        labels = ["A", "B", "C", "D"]

        for (state, lx, ly), label in zip(leds, labels):
            center = (lx + r_led, ly + r_led)
            fill   = (0, 200, 50) if state else (40, 40, 40)
            border = color if state else (80, 80, 80)
            cv2.circle(canvas, center, r_led, fill, -1)
            cv2.circle(canvas, center, r_led, border, 3)
            cv2.putText(canvas, label,
                        (center[0]-8, center[1]+6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (200, 200, 200) if state else (100, 100, 100), 2)

        # Binary readout
        bin_str = f"[{a}][{b}]  [A][B]"
        bin_str2= f"[{c}][{d}]  [C][D]"
        ty = bottom + r_led * 2 + 16
        cv2.putText(canvas, bin_str,  (20, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 160, 160), 1)
        cv2.putText(canvas, bin_str2, (20, ty+20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 160, 160), 1)

    return canvas

# ================================================================
#  COMMAND LOGGER
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
                f"[Robot {robot_id:2d}] "
                f"Cell({state.cell[0]},{state.cell[1]}) "
                f"HDG:{state.heading_deg:6.1f}°  "
                f"DIR:{state.target_dir.value:2s}  "
                f"→  {cmd.value}"
                + (f"  → WPT{tc}" if tc else "")
            )

# ================================================================
#  STARTUP REFERENCE TABLE
# ================================================================
def print_led_reference():
    print("\n  LED PATTERN REFERENCE")
    print("  ┌─────────────────┬──────┬────────────────┐")
    print("  │ Command         │ ABCD │ Panel          │")
    print("  ├─────────────────┼──────┼────────────────┤")
    for cmd, pat in LED_PATTERNS.items():
        a, b, c, d = pat
        panel = f"[{a}][{b}] / [{c}][{d}]"
        print(f"  │ {cmd.value:<15s}  │ {''.join(map(str,pat))}  │ {panel:<14s} │")
    print("  └─────────────────┴──────┴────────────────┘\n")

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
    last_ordered = last_M = last_M_inv = last_warped = None

    # Occupancy
    occ_map = OccupancyMap(STATIC_GRID)

    # Planning
    robot_paths:  Dict[int, Optional[list]] = {}
    replan_count = 0

    # Execution
    nav_ctrl = NavigationController()
    cmd_log  = CommandLogger()

    # Communication (LED panel — wall-mounted, persists independently)
    # LED simulation state remains in the PC for overlays and demonstrations.
    # The serial bridge sends the physical LED command byte to Arduino.
    led_ctrl      = LEDCommunicationController()
    serial_bridge = SerialLEDBridge(SERIAL_PORT, SERIAL_BAUD)

    stopped_missing_ids = set()

    manual_lock = show_debug = False
    prev_time   = time.time()

    print("\n=== PC Navigation Controller + Arduino LED Link ===")
    print(f"  Serial LED: {SERIAL_PORT} @ {SERIAL_BAUD}")
    print(f"  Goal: {GOAL_CELL}  |  Heading tolerance: {HEADING_TOLERANCE_DEG}°")
    print("  Q/ESC=quit  L=manual lock  D=debug edges")
    print_led_reference()

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
        detected_robots: Dict[int, tuple] = {}

        if last_ordered is not None and rect_found:
            cell_w = WARP_W // GRID_COLS
            cell_h = WARP_H // GRID_ROWS
            corners, ids = detect_aruco(gray)
            if ids is not None:
                for i in range(len(ids)):
                    c        = corners[i][0]
                    robot_id = int(ids[i][0])
                    cam_cx   = float(c[:, 0].mean())
                    cam_cy   = float(c[:, 1].mean())
                    pt       = np.array([[[cam_cx, cam_cy]]], dtype="float32")
                    wpt      = cv2.perspectiveTransform(pt, last_M)[0][0]
                    wx = max(0, min(WARP_W-1, int(wpt[0])))
                    wy = max(0, min(WARP_H-1, int(wpt[1])))
                    gc = min(wx // cell_w, GRID_COLS-1)
                    gr = min(wy // cell_h, GRID_ROWS-1)

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

                    occupied_this_frame.add((gr, gc))
                    detected_robots[robot_id] = (
                        wx, wy, gr, gc, angle, fwd_w, cam_cx, cam_cy)

        if rect_found:
            occ_map.update(occupied_this_frame)

        # ============================================================
        #  STEP 3 — REPLANNING
        # ============================================================
        if rect_found and occ_map.fused_changed():
            fused        = occ_map.fused_grid()
            replan_count += 1
            for rid, (_, _, gr, gc, _, _, _, _) in detected_robots.items():
                saved = fused[gr][gc]
                fused[gr][gc] = 0
                path = astar(fused, (gr, gc), GOAL_CELL)
                fused[gr][gc] = saved
                robot_paths[rid] = path
                nav_ctrl.on_replan(rid, path)
            for rid in list(robot_paths):
                if rid not in detected_robots:
                    del robot_paths[rid]
                    nav_ctrl.remove(rid)
                    # Panel NOT removed — wall panel persists with last state

        # ============================================================
        #  STEP 4 — EXECUTION UPDATE
        # ============================================================
        for rid, (_, _, gr, gc, angle, _, _, _) in detected_robots.items():
            cmd = nav_ctrl.update(rid, (gr, gc), angle)
            # Forward to LED communication layer
            led_ctrl.set_command(rid, cmd)
            serial_bridge.send(cmd)
            stopped_missing_ids.discard(rid)
            st = nav_ctrl.state(rid)
            if st:
                cmd_log.log(rid, st)

        # When robot disappears — send STOP to its panel channel
        for rid in led_ctrl.all_states():
            if rid not in detected_robots and rid not in stopped_missing_ids:
                led_ctrl.set_command(rid, NavigationCommand.STOP)
                serial_bridge.send(NavigationCommand.STOP)
                stopped_missing_ids.add(rid)

        # ============================================================
        #  STEP 5 — WARP + VISUALISE
        # ============================================================
        if last_ordered is not None:
            if rect_found:
                last_warped = cv2.warpPerspective(
                    frame, last_M, (WARP_W, WARP_H))

            warped_draw = last_warped.copy()

            if rect_found:
                draw_occupancy_layer(warped_draw, occ_map, GRID_ROWS, GRID_COLS)

                goal_reached_any = any(
                    nav_ctrl.state(rid) and
                    nav_ctrl.state(rid).status == ExecStatus.REACHED
                    for rid in detected_robots
                )
                if goal_reached_any:
                    draw_goal_reached(warped_draw, GOAL_CELL, GRID_ROWS, GRID_COLS)

                for rid, path in robot_paths.items():
                    draw_path(warped_draw, path, GRID_ROWS, GRID_COLS)

                for rid, (wx, wy, gr, gc, angle,
                          fwd_w, cam_cx, cam_cy) in detected_robots.items():
                    st     = nav_ctrl.state(rid)
                    led_st = led_ctrl.state(rid)
                    if st:
                        draw_waypoint_marker(warped_draw, st.target_cell,
                                             st.waypoint_idx,
                                             GRID_ROWS, GRID_COLS)
                        draw_robot_on_warped(warped_draw, wx, wy, fwd_w, st,
                                             GRID_ROWS, GRID_COLS, led_st)
                        st_label = st.command.value
                    else:
                        st_label = "?"
                    cv2.circle(display, (int(cam_cx), int(cam_cy)),
                               5, (255, 80, 0), -1)
                    cv2.putText(display, f"ID{rid} {st_label}",
                                (int(cam_cx)+8, int(cam_cy)-8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 80, 0), 2)

                draw_grid(warped_draw, GRID_ROWS, GRID_COLS)
                project_grid_back(display, last_ordered,
                                  GRID_ROWS, GRID_COLS,
                                  WARP_W, WARP_H, last_M_inv)

            show_warped = warped_draw.copy()
            if not rect_found:
                cv2.rectangle(show_warped, (0, 0), (WARP_W, 36), (0, 0, 0), -1)
                cv2.putText(show_warped, "Searching... (last known view)",
                            (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0, 140, 255), 2)
            cv2.imshow("Warped Grid", show_warped)

        # ============================================================
        #  LED PANEL SIMULATION WINDOW
        # ============================================================
        all_led = led_ctrl.all_states()
        if all_led:
            panel_img = render_led_panel_window(all_led)
            cv2.imshow("LED Panel (Wall)", panel_img)

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
            led_ctrl.set_stop_all()
            serial_bridge.send(NavigationCommand.IDLE)
            break
        elif key == ord('l'):
            manual_lock = not manual_lock
            print(f"Manual lock: {'ON' if manual_lock else 'OFF'}")
        elif key == ord('d'):
            show_debug = not show_debug
            if not show_debug:
                cv2.destroyWindow("Edges (debug)")

    cap.release()
    serial_bridge.close()
    cv2.destroyAllWindows()
    print("Exited cleanly.")


if __name__ == "__main__":
    main()
