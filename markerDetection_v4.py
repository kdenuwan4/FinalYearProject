"""
=============================================================================
Project: Multi-Robot Communication System with Vision-Based Tracking (VLC)
Module : Central Intelligence & Mapping
Author : Amunugama H.M.K.D. (E/21/029)
Version: 4.0 (State Estimation & Kinematic Smoothing)
Date   : April 29, 2026

Description:
This version resolves frame-to-frame jitter and establishes the foundational 
kinematic state required for advanced path planning and multi-agent coordination. 
It introduces an Exponential Moving Average (EMA) filter for spatial smoothing, 
calculates robot orientation (heading), and derives linear velocity. The system 
now maintains a persistent state dictionary, providing a stable, structured 
data pipeline for the AI decision-making layer.

Upgrades from Previous Version:
- [Architecture] Introduced `robot_states` dictionary for cross-frame memory.
- [Kinematics] Added heading angle (theta) calculation using marker edge vectors.
- [Kinematics] Implemented real-time velocity (vx, vy) tracking using frame delta t.
- [Signal Processing] Applied an EMA filter (SMOOTHING_ALPHA = 0.7) to eliminate jitter.
- [Visualization] Added dynamic vector arrows indicating real-time robot heading.
=============================================================================
"""

import cv2
import cv2.aruco as aruco
import numpy as np
import math
import time

# ----------------------------------------
# CONFIGURATION
# ----------------------------------------
CAMERA_INDEX = 0
VALID_IDS = [1, 2]   # your robots only
SMOOTHING_ALPHA = 0.7

# ----------------------------------------
# INITIALIZE CAMERA
# ----------------------------------------
cap = cv2.VideoCapture(CAMERA_INDEX)

# Load ArUco dictionary
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
parameters = aruco.DetectorParameters()

# ----------------------------------------
# STATE STORAGE (IMPORTANT)
# ----------------------------------------
robot_states = {}

prev_time = time.time()

# ----------------------------------------
# MAIN LOOP
# ----------------------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    current_time = time.time()
    dt = current_time - prev_time
    prev_time = current_time

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect markers
    corners, ids, rejected = aruco.detectMarkers(
        gray,
        aruco_dict,
        parameters=parameters
    )

    if ids is not None:

        for i in range(len(ids)):
            robot_id = int(ids[i][0])

            # Ignore unwanted markers
            if robot_id not in VALID_IDS:
                continue

            c = corners[i][0]  # 4 corner points

            # ----------------------------------------
            # POSITION (CENTER OF MARKER)
            # ----------------------------------------
            cx = int(c[:, 0].mean())
            cy = int(c[:, 1].mean())

            # ----------------------------------------
            # ORIENTATION (HEADING ANGLE)
            # Using top edge: corner 0 -> corner 1
            # ----------------------------------------
            dx = c[1][0] - c[0][0]
            dy = c[1][1] - c[0][1]

            theta = math.atan2(dy, dx)  # radians

            # ----------------------------------------
            # SMOOTHING (REMOVE JITTER)
            # ----------------------------------------
            if robot_id in robot_states:
                prev = robot_states[robot_id]

                cx = int(SMOOTHING_ALPHA * prev["x"] + (1 - SMOOTHING_ALPHA) * cx)
                cy = int(SMOOTHING_ALPHA * prev["y"] + (1 - SMOOTHING_ALPHA) * cy)
                theta = SMOOTHING_ALPHA * prev["theta"] + (1 - SMOOTHING_ALPHA) * theta

                vx = (cx - prev["x"]) / dt if dt > 0 else 0
                vy = (cy - prev["y"]) / dt if dt > 0 else 0
            else:
                vx, vy = 0, 0

            # ----------------------------------------
            # STORE STATE
            # ----------------------------------------
            robot_states[robot_id] = {
                "x": cx,
                "y": cy,
                "theta": theta,
                "vx": vx,
                "vy": vy,
                "last_seen": current_time
            }

            # ----------------------------------------
            # VISUALIZATION
            # ----------------------------------------
            aruco.drawDetectedMarkers(frame, [corners[i]], ids[i:i+1])

            # Draw center
            cv2.circle(frame, (cx, cy), 6, (0, 0, 255), -1)

            # Draw direction arrow
            arrow_length = 50
            end_x = int(cx + arrow_length * math.cos(theta))
            end_y = int(cy + arrow_length * math.sin(theta))
            cv2.line(frame, (cx, cy), (end_x, end_y), (255, 0, 0), 2)

            # Display info
            cv2.putText(
                frame,
                f"ID:{robot_id} X:{cx} Y:{cy}",
                (cx + 10, cy),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2
            )

    cv2.imshow("Tracking System", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()