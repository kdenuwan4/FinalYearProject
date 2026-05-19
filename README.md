# Central Intelligence & Mapping 
**Part of the Multi-Robot Communication System FYP**

This repository contains the Central Intelligence and Mapping subsystem for a deterministic multi-robot coordination project. Acting as the "brain" and "eyes" of the swarm, this layer utilizes an overhead camera array to process real-time environment data, discretize the physical workspace, and compute optimal, collision-free trajectories for autonomous agents.

### Core Technologies
* **Language:** Python 3
* **Computer Vision:** OpenCV, ArUco Fiducial Tracking, NumPy
* **Algorithms:** A* Path Planning, Perspective Transformation (Homography), Kinematic State Estimation

### Key Capabilities
* **Dynamic Perception Pipeline:** Continuously detects arena boundaries and applies homographic warping to convert angled video feeds into a normalized, top-down 2D coordinate space.
* **Real-Time Localization:** Utilizes ArUco marker detection to extract precise spatial coordinates $(x, y)$ and heading orientation $(\theta)$ for multiple robots simultaneously.
* **Fault-Tolerant State Estimation:** Implements persistent state memory to freeze telemetry and maintain grid stability during temporary visual occlusions or lighting failures.
* **Centralized A* Path Planning:** Generates optimal, 8-directional routes mapped onto a dynamic occupancy grid, featuring strict corner-cut prevention and physical collision avoidance.
