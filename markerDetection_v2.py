"""
=============================================================================
Project: Multi-Robot Communication System with Vision-Based Tracking (VLC)
Module : Central Intelligence & Mapping
Author : Amunugama H.M.K.D. (E/21/029)
Version: 2.0 (Data Extraction Update)
Date   : April 26, 2026

Description:
This script updates the baseline tracking loop to extract and output raw 
telemetry data. It detects ArUco markers, calculates their geometric center, 
and streams the (X, Y) coordinates to the console, laying the groundwork 
for data transmission to the central path-planning logic.

Upgrades from Previous Version:
- [Data Pipeline] Added real-time console logging of X, Y coordinates and ID.
- [UI/UX] Increased text scale and adjusted overlay format for better visibility.
- [Refactoring] Improved function line-breaks and descriptive comments.
=============================================================================
"""

import cv2
import cv2.aruco as aruco

# Open camera
cap = cv2.VideoCapture(0)

# Load ArUco dictionary (MUST match your markers)
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)

# Detector parameters
parameters = aruco.DetectorParameters()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to grayscale (better detection)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect markers
    corners, ids, rejected = aruco.detectMarkers(
        gray,
        aruco_dict,
        parameters=parameters
    )

    # If markers are found
    if ids is not None:
        aruco.drawDetectedMarkers(frame, corners, ids)

        for i in range(len(ids)):
            c = corners[i][0]

            # Center of marker
            cx = int(c[:, 0].mean())
            cy = int(c[:, 1].mean())

            # Draw center point
            cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

            # Show ID + position
            cv2.putText(
                frame,
                f"ID:{ids[i][0]} ({cx},{cy})",
                (cx, cy - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

            print(f"Robot {ids[i][0]} -> X:{cx}, Y:{cy}")

    # Show output
    cv2.imshow("Aruco Detection", frame)

    # Press ESC to exit
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()