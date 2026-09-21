\# Wisconsin Autonomous Perception Challenge



\## Method



This project estimates the ego-vehicle trajectory using a fixed traffic light as a world reference.



For each frame with available depth data, the provided traffic-light bounding box is used to find the center pixel of the light. A small patch around that center is sampled from the corresponding XYZ depth array. Invalid depth values are removed, and the median XYZ position of the remaining points is used to reduce sensitivity to noisy stereo depth.



Because the traffic light is stationary in the world, changes in its measured position relative to the camera represent motion of the ego vehicle in the opposite direction.



The camera-coordinate measurements are projected onto the ground plane, converted so that camera-right corresponds to negative world-Y, and rotated so that the initial line from the car to the traffic light defines the +X world direction. The traffic light is then placed at the world origin, and the ego-vehicle trajectory is plotted in the X-Y plane.



\## Assumptions / Data Handling



\- Only frames containing both a depth file and a valid traffic-light bounding box are used.

\- Bounding boxes containing all zeros are treated as invalid.

\- Non-finite depth values such as `inf` and `NaN` are ignored.

\- The first three values of each XYZ pixel are used as the 3D coordinates.

\- A median over a small local patch is used instead of relying on a single depth pixel.



\## Outputs



\- `trajectory.png` — static bird's-eye-view trajectory

\- `trajectory.mp4` — animated trajectory over time

\- `main.py` — source code

