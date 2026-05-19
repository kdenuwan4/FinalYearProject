# FINAL YEAR PROJECT – WEEK 1
**Title:** Camera-Based Arena Detection and Robot Position Mapping  
**Module:** Central Intelligence & Mapping  
**Author:** Amunugama H.M.K.D. (E/21/029)  

## 1. Objective
The objective of this week was to develop the core perception layer of the system using computer vision. The system must observe a real-world robot arena through an overhead camera and convert that raw camera feed into a structured computational representation. This representation will later be used for robot navigation, decision-making, communication, and multi-robot coordination.

To achieve this, the system was designed to detect the physical arena boundary, convert the angled camera view into a top-down perspective, and divide the space into a logical grid. Furthermore, it must detect robots using visual markers, estimate their precise position and orientation, and maintain stable tracking even during temporary visual disruptions. This week mainly focused on building a reliable mapping and localization pipeline.

## 2. System Overview
The developed system transforms the raw camera feed into a structured environment model through multiple processing stages. This creates a reliable bridge between the physical robot environment and a normalized computational space. The complete pipeline performs the following operations:

* **Arena Boundary Detection:** The system continuously identifies the physical boundary of the working area from the raw camera feed.
* **Perspective Transformation:** It applies mathematical homography to generate a flattened, top-down view of the angled camera view.
* **Grid Mapping:** The flattened arena is then discretized into a 3×3 logical coordinate grid for navigation.
* **Marker Detection & Localization:** Active robots are identified via ArUco markers, allowing the system to calculate their precise positions and facing orientations.
* **State Persistence:** The system is engineered to maintain stable operation and memory of the arena even if the boundary is temporarily obscured.

## 3. Core Processing Pipeline

### 3.1 Arena Boundary Detection
The first stage of the system is detecting the working arena from the live camera image. The camera frame is first converted into grayscale. Removing color information simplifies the image and allows edge-based operations to work more reliably. Next, a Gaussian blur is applied to reduce small image noise and unwanted texture variations that could create unstable edges.

After preprocessing, Canny edge detection is used to extract strong image boundaries. Since real-world chalk or tape boundaries may contain small gaps, dilation is applied to connect fragmented edge segments. Contours are then extracted from the processed image, and the system searches for the largest convex quadrilateral, which is assumed to represent the arena boundary.

Once detected, the four corner points are mathematically reordered into a fixed sequence: Top-Left → Top-Right → Bottom-Right → Bottom-Left. Consistent corner ordering is extremely important because perspective transformation depends entirely on correct point correspondence.

### 3.2 Perspective Transformation (Homography)
After detecting the arena boundary, the system applies a perspective transformation. Internally, OpenCV computes a homography matrix that maps the detected quadrilateral into a normalized 600×600 square. This transformation successfully removes the distortion caused by the camera angle and perspective. 

This transformation is crucial because it ensures the arena appears as a perfect top-down square. Consequently, position calculations become geometrically consistent, grid division becomes mathematically simpler, and the robot tracking logic becomes entirely independent of the physical camera's mounting tilt. The warped arena effectively becomes the system’s normalized world coordinate space.

### 3.3 Grid Mapping
The warped arena is divided into a 3×3 logical grid. Instead of working directly with raw pixel coordinates, the system converts robot positions into grid indices represented strictly as (row, column). This abstraction is important because future navigation and decision-making algorithms operate more effectively on discrete logical cells than on raw image coordinates. The grid structure created this week becomes the foundation for future path-planning and robot coordination modules.

### 3.4 Robot Detection Using ArUco Markers
Robots are identified using ArUco markers. Each robot contains a unique marker ID, which allows the system to distinguish multiple robots individually. For every detected marker, the system calculates the marker centroid to determine the robot's physical position, and analyzes the marker's orientation to determine the robot's heading direction.

The centroid is calculated by averaging the marker corner coordinates. The orientation is estimated by comparing the marker’s top edge and bottom edge geometry, which generates a direction vector representing the robot’s heading. This orientation estimation is critical because movement commands, such as “move forward,” depend heavily on the robot's heading, not just its spatial position.

### 3.5 Coordinate Mapping into Warped Space
Robot positions are initially detected in raw camera coordinates. To ensure consistency with the arena model, the system transforms these robot coordinates into the warped top-down coordinate space using the exact same homography matrix used for the arena transformation. 

Once transformed, the arena geometry and robot positions exist in the exact same normalized coordinate system. This ensures that robot locations can be mapped directly into the discrete grid cells, making spatial reasoning reliable and consistent. The system then successfully determines the robot’s active grid cell based on these warped coordinates.

## 4. Major System Upgrade – Continuous Detection Model

### 4.1 Limitation of the Original Fixed-Lock Design
The original implementation relied on a one-time detection approach: the system would detect the arena boundary once, compute the transformation matrix, and lock that boundary permanently. Although computationally simple, this approach failed whenever the camera position changed slightly. Since the transformation matrix depended entirely on the original boundary detection, even small camera movements caused incorrect mapping and inaccurate robot projections.

### 4.2 Continuous Detection Architecture
To solve this limitation, the system was upgraded into a continuous detection model. Instead of locking the boundary permanently, the system now re-detects the arena boundary in every single frame. The updated behavior ensures that if the boundary is detected successfully, the system updates its transformation matrices. However, if the boundary is temporarily lost, the system preserves the last valid transformation state instead of resetting. This allows the grid system to dynamically follow camera movement.

### 4.3 Persistent State Memory Design
The upgraded system introduces persistent internal state variables to handle visual disruptions. It utilizes `last_ordered` to remember the last valid boundary corners, `last_M` for the latest valid homography matrix, and `last_warped` to store the last cleanly flattened frame. These variables act as a robust fallback memory system. If detection temporarily fails due to occlusion or lighting variation, the system does not immediately collapse; it continues operating using this last reliable state, significantly improving stability.

### 4.4 Graceful Fallback Behaviour
When boundary detection fails temporarily, the warped window remains open and preserves the last valid warped frame. The system displays a "Searching..." status message and automatically recovers full tracking the moment the boundary reappears. This prevents sudden flickering and unstable visualization. More importantly, it allows the perception pipeline to tolerate short-term visual failures without resetting the entire system.

### 4.5 Critical Reliability Design Decision
A very important engineering decision was implemented in the upgraded system: robot positions are only updated when the arena boundary is successfully detected in the current frame. This decision aggressively prevents incorrect robot projections. If the camera moved while the system was still using an outdated transformation matrix, the robot coordinates in the warped view would become physically incorrect. Therefore, during temporary boundary loss, the warped grid remains visible using the last valid frame, but robot projections are intentionally disabled until live detection recovers.

### 4.6 Manual Lock Mode
A manual lock feature was also introduced. Pressing the 'L' key freezes the current boundary and temporarily disables continuous re-detection. This mode is highly useful in scenarios where the camera is completely stable, the environment is strictly controlled, and reducing CPU usage is preferred. The system can easily resume live tracking by toggling the manual lock off.

## 5. Visualization and Debugging Features
Several visualization tools were added to simplify debugging and verification. The system includes multiple real-time overlays, such as a grid drawn on both the warped arena and the raw camera view. It also visualizes robot positions, draws directional arrows for their orientation, highlights the specific grid cells they currently occupy, and provides an FPS counter to monitor system performance. These overlays allow for the real-time verification of geometric correctness and tracking accuracy.

## 6. Key Engineering Takeaways
This week provided practical experience with several important computer vision and robotics concepts. These include image preprocessing techniques, edge detection, perspective transformation using homography, coordinate-space normalization, marker-based localization, and the integration of multiple perception subsystems into a real-time state estimator. 

One particularly important engineering lesson learned during this week was that a reliable robotics system must prioritize correct data over continuous output. Maintaining stable and trustworthy state information is far more important than displaying constantly changing, but potentially incorrect, information.

## 7. Current System Capabilities
At the end of Week 1, the system can successfully detect the arena boundary dynamically, adapt to camera movement, and generate a top-down warped arena view. It accurately divides the arena into logical grid cells, detects robots using ArUco markers, estimates their precise position and orientation, and maps them into structured grid coordinates. Furthermore, it maintains stable operation during temporary detection failures and recovers automatically after visual interruptions. This perception layer forms a highly reliable foundation for the next stage of the project, which includes path planning, robot coordination, and communication logic.