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

## Component A (summary of results, from the runs)
- Closed loop in Webots (A* path, PID Kp=8, Ki=0, Kd=0.8, fused pose): 29.8 s, mean cross-track 1.4 cm,
  max 7.0 cm (at path corners: the lookahead cuts them), final distance to goal 8.5 cm.
  Offline simulation of the same controller: 29.5 s, mean 1.5 cm -> the kinematic model is validated.
- PID tuning: P Kp=1 slow (settle 3.46 s); high P Kp=12 overshoots 7.9 %; PD removes the overshoot;
  adding I (Ki=1) made it worse (9.7 % overshoot, 7.52 s settle) because there is no steady-state error and
  the integral winds up; tuned (8, 0, 0.8): rise 0.54 s, overshoot 0 %, settle 0.90 s.
- Derivative term gives +-0.3 cm ripple (amplifies compass/wheel quantisation noise).
- Webots warning "requested velocity 6.67 exceeds maxVelocity" fixed by capping wheel speed at 6.6 rad/s.

## Component B (summary of results)
- Multi-robot world: 4 TurtleBot3 Burgers, Emitter/Receiver on each, every robot heard all 3 others.
- Average consensus on a 4-ring (eps=0.3, round every 0.2 s): states reach the average (0,0) within ~3 s;
  disagreement falls geometrically (straight line on the log plot). Robots end in a ring of 0.35 m round the centre.
- PSO task allocation: see report/pso_results.txt. Basic PSO converges prematurely (swarm collapses after ~600
  evaluations); ring topology + inertia decay + reflecting borders fixed most of it. Brute force takes ~1 s for
  4^8 options, so PSO only pays off for larger problems.
- Neural network (NumPy MLP 46-32-16-3) on LiDAR scans, 5863 samples after cleaning: test accuracy 90.7 %
  (majority baseline 44.3 %). Box class weakest (precision 0.76, recall 0.82). Accuracy by distance ahead:
  99.4 % (<0.5 m), 94.6 % (0.5-0.8), 75.0 % (0.8-1.2), 92.0 % (1.2-2.0), 100 % (>2.0).
  Earlier 20-minute dataset: 94.2 % test accuracy but only 22 box samples in the test set (noisy estimate).
- Data issues found: the wander controller sometimes spun in place in a corner (1338 scans capped); LiDAR noise
  made a scan-difference duplicate filter useless, a pose-based filter worked; block-wise split needed because
  consecutive scans are almost identical.
- Centralised vs distributed: see report/central_vs_distributed.txt (ring needs ~N^2 rounds, centralised needs 2
  rounds but the leader receives N-1 messages per round and is a single point of failure).
