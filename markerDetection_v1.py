"""
=============================================================================
Project: Multi-Robot Communication System with Vision-Based Tracking (VLC)
Module : Central Intelligence & Mapping
Author : Amunugama H.M.K.D. (E/21/029)
Version: 1.0 (Baseline Tracking Loop)
Date   : April 26, 2026

Description:
This script serves as the baseline implementation for the real-time robot 
perception layer. It initializes a live video feed from the overhead camera, 
detects ArUco markers (DICT_4X4_50), and mathematically extracts the 2D 
centroid (X, Y pixel coordinates) of each detected robot. 

"""

import cv2
import cv2.aruco as aruco

# Initialize overhead camera feed (0 is the default webcam)
cap = cv2.VideoCapture(0)

# Load ArUco dictionary to match the printed robot markers
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
parameters = aruco.DetectorParameters()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to grayscale for faster processing
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect markers in the current frame
    corners, ids, rejected = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

    if ids is not None:
        # Draw bounding boxes around detected markers
        aruco.drawDetectedMarkers(frame, corners, ids)

        for i in range(len(ids)):
            c = corners[i][0]

            # Calculate the geometric center (centroid) of the marker
            cx = int(c[:, 0].mean())
            cy = int(c[:, 1].mean())

            # Draw center point (Red dot)
            cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

            # Display Robot ID and its current (X, Y) pixel position
            cv2.putText(frame, f"ID {ids[i][0]} ({cx},{cy})",
                        (cx, cy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        2)

    # Render the live tracking view
    cv2.imshow("Aruco Tracking", frame)

    # Break loop if 'ESC' is pressed
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()