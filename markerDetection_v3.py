"""
=============================================================================
Project: Multi-Robot Communication System with Vision-Based Tracking (VLC)
Module : Central Intelligence & Mapping
Author : Amunugama H.M.K.D. (E/21/029)
Version: 3.0 (ID Filtering & Noise Reduction)
Date   : April 28, 2026

Description:
This script refines the real-time tracking loop by introducing a strict 
whitelist filter for valid robot IDs. This prevents the system from 
processing false positives, ambient geometric noise, or rogue ArUco markers 
in the environment, ensuring the central controller only calculates paths 
for active, verified agents.

Upgrades from Previous Version:
- [Data Integrity] Introduced `VALID_IDS` list to strictly isolate Robot 1 and 2.
- [Robustness] Added logic to bypass non-system markers, reducing false tracking.
- [UI Update] Changed display window name to "ArUco Tracking System".
- [UI Update] Adjusted text overlay offset to prevent overlapping the center dot.
=============================================================================
"""

import cv2
import cv2.aruco as aruco

# -----------------------------
# CONFIGURATION
# -----------------------------
cap = cv2.VideoCapture(0)

aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
parameters = aruco.DetectorParameters()

VALID_IDS = [1, 2]  # ONLY your robots

# -----------------------------
# MAIN LOOP
# -----------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    corners, ids, rejected = aruco.detectMarkers(
        gray,
        aruco_dict,
        parameters=parameters
    )

    if ids is not None:

        for i in range(len(ids)):
            robot_id = ids[i][0]

            # FILTER INVALID IDS
            if robot_id not in VALID_IDS:
                continue

            c = corners[i][0]

            # Center position
            cx = int(c[:, 0].mean())
            cy = int(c[:, 1].mean())

            # Draw marker boundary
            aruco.drawDetectedMarkers(frame, [corners[i]], ids[i:i+1])

            # Draw center point
            cv2.circle(frame, (cx, cy), 6, (0, 0, 255), -1)

            # Display info
            cv2.putText(
                frame,
                f"Robot {robot_id} ({cx},{cy})",
                (cx + 10, cy),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

            # Print for debugging
            print(f"Robot {robot_id} -> X:{cx}, Y:{cy}")

    # Show result
    cv2.imshow("ArUco Tracking System", frame)

    # ESC to exit
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()