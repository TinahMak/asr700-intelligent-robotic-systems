# Component A notes
- Robot: TurtleBot3Burger (differential drive), Webots R2025a
- LiDAR (LDS-01): 360 beams, 360 deg FOV, max range 3.5 m
- Time step: see console line "Time step:" when the controller starts
- Arena: 4 m x 4 m, wall height 0.5 m (0.1 m was below the scan plane, so walls were invisible to the LiDAR)
- Obstacles: box1..box6 at (0,0.5) (1,-0.5) (-0.5,1) (0.8,1) (-1,0) (0.3,-1)

## A1 findings: pose estimation and mapping
- LiDAR (LDS-01): beam 0 points BACKWARDS and the index runs clockwise. Robot-frame angle = pi - 2*pi*i/360. Found by comparing a scan at a known pose with the world (report/scan_start.png).
- Compass: at heading 0 it reads north along robot +y, so heading = atan2(c[0], c[1]).
- Wheels-only odometry heading drifts: after 150 s it differs from the compass by about 0.67 rad (38 deg), and the map is smeared (report/map_wheels_only.png).
- Fused pose (wheel distance + compass heading) matched the true position to a few cm at t = 152 s: fused (-0.56, -1.77), true about (-0.56, -1.8) read from the 3D view.
- Occupancy grid: 80 x 80 cells, 0.05 m, log-odds with L_OCC = +0.85, L_FREE = -0.40, clamp +/-5 (report/occupancy_map.png).
- Limitations: LiDAR is about 3 cm behind the robot centre (ignored); box interiors stay unknown; simple reactive wandering covers the arena unevenly.

## A1 findings: camera and fusion
- The Burger has no camera. One was added in the extension slot (R2025a). In this setup only the camera node `cam_c` (rotation 0 1 0 1.5708) looks forward; the unrotated one looks at the sky (report/camera_orientation_test.png). The live overlay is unreliable, so saved images were used to find the orientation.
- OpenCV: HSV masks for boxes (H 10-20, S>=80, V 40-170), shadows (V<45), walls (H 100-125, V<112), then morphological open/close.
- Ranging: pinhole model, camera 0.12 m above the floor, horizontal FOV 1.0 rad (fx = 293 px). depth = 0.12 * fx / (row - 120).
- Fusion: per image column, camera range vs the LiDAR beam at the same bearing; agreement within 0.30 m -> inverse-variance fusion (sigma LiDAR 0.02 m, camera 5% of range); otherwise the LiDAR is trusted.
- Limitations: colour thresholds depend on the lighting and floor texture; a shadowed box side can be mistaken for wall; far-range estimates are sensitive to a one-pixel error.
