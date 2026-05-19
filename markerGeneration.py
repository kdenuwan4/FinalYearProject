"""
=============================================================================
Project: Multi-Robot Communication System with Vision-Based Tracking (VLC)
Module : Central Intelligence & Mapping
Author : Amunugama H.M.K.D. (E/21/029)
Version: 1.0 (Utility Script)
Date   : April 25, 2026

Description:
A setup utility script used to generate ArUco markers from the 4X4_50 
dictionary. These markers are attached to the mobile robots to provide 
unique IDs and directional (yaw) data for the overhead camera tracking system.

Usage/Notes:
- Run once during system setup.
- Generates 500x500 pixel PNG images for Robot ID 1 and Robot ID 2.
- Markers must be printed and physically mounted to the robots.
=============================================================================
"""

import cv2
import cv2.aruco as aruco

# Initialize the ArUco dictionary (using a small dictionary to reduce false positives)
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)

# Generate markers for the robots
for i in range(1, 3):  # IDs 1 and 2 for your robots
    # Generate a 500x500 pixel marker
    marker = aruco.generateImageMarker(aruco_dict, i, 500)

    # Save to local directory
    filename = f"marker_{i}.png"
    cv2.imwrite(filename, marker)

print("Markers successfully generated and saved!")