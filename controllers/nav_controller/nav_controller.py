"""nav_controller.py - Component A: wander, build an occupancy grid, log the path.

Beam convention found by testing: beam 0 points BACKWARDS and the index runs
clockwise, so beam 180 is straight ahead, ~90 is left and ~270 is right.
"""
import math
import numpy as np
from controller import Robot
from odometry import Odometry
from occupancy_grid import OccupancyGrid

FORWARD_SPEED = 3.0     # rad/s  (about 0.1 m/s)
TURN_SPEED = 2.0        # rad/s  (wheels opposite directions)
FRONT_STOP = 0.30       # m  start turning below this
FRONT_CLEAR = 0.50      # m  stop turning above this
GRID_EVERY = 4          # steps between grid updates
SAVE_EVERY = 5.0        # s of simulated time between saves
PRINT_EVERY = 1.0

robot = Robot()
time_step = int(robot.getBasicTimeStep())

left_motor = robot.getDevice("left wheel motor")
right_motor = robot.getDevice("right wheel motor")
lidar = robot.getDevice("LDS-01")
left_sensor = robot.getDevice("left wheel sensor")
right_sensor = robot.getDevice("right wheel sensor")
for motor in (left_motor, right_motor):
    motor.setPosition(float("inf"))
    motor.setVelocity(0.0)
lidar.enable(time_step)
left_sensor.enable(time_step)
right_sensor.enable(time_step)
compass = robot.getDevice("compass")
compass.enable(time_step)
camera = robot.getDevice("cam_c")   # forward-facing camera (see report/camera_orientation_test.png)
camera.enable(time_step)
print(f"Camera: {camera.getWidth()} x {camera.getHeight()}, FOV {camera.getFov():.2f} rad")
max_range = lidar.getMaxRange()

grid = OccupancyGrid()
odom = None
path = []
path_odom = []
turn_dir = 0            # 0 = driving, +1 = turning left, -1 = turning right
step_count = 0
print_steps = max(1, int(PRINT_EVERY * 1000 / time_step))
save_steps = max(1, int(SAVE_EVERY * 1000 / time_step))

while robot.step(time_step) != -1:
    l_pos = left_sensor.getValue()
    r_pos = right_sensor.getValue()
    cv = compass.getValues()
    heading = math.atan2(cv[0], cv[1])     # world heading from the compass (rad); this compass reads north along +y at heading 0
    if odom is None:
        odom = Odometry(l_pos, r_pos, x=-1.5, y=-1.5, theta=0.0)       # wheels only
        fused = Odometry(l_pos, r_pos, x=-1.5, y=-1.5, theta=heading)  # wheels + compass
    else:
        odom.update(l_pos, r_pos)
        fused.update(l_pos, r_pos, heading=heading)
    pose = (fused.x, fused.y, fused.theta)

    ranges = lidar.getRangeImage()
    r = np.array(ranges)
    r = np.where(np.isfinite(r), r, max_range)       # no hit -> treat as max range

    # ---- Perception: update the map and log the path ----
    if step_count % GRID_EVERY == 0:
        grid.update(pose, ranges, max_range)
        path.append(pose)
        path_odom.append((odom.x, odom.y, odom.theta))

    # ---- Reactive wandering ----
    front = float(np.min(r[150:211]))
    left_clear = float(np.mean(r[60:121]))
    right_clear = float(np.mean(r[240:301]))
    if turn_dir == 0:
        if front < FRONT_STOP:
            turn_dir = 1 if left_clear >= right_clear else -1
    elif front > FRONT_CLEAR:
        turn_dir = 0

    if turn_dir == 0:
        left_motor.setVelocity(FORWARD_SPEED)
        right_motor.setVelocity(FORWARD_SPEED)
    else:
        left_motor.setVelocity(-turn_dir * TURN_SPEED)
        right_motor.setVelocity(turn_dir * TURN_SPEED)

    if step_count == 30:
        camera.saveImage("../../data/camera_start.png", 100)
        print("Saved data/camera_start.png")
    # ---- Logging and saving ----
    if step_count % print_steps == 0:
        print(f"t={robot.getTime():6.1f}s  front={front:.2f} m  "
              f"fused=({fused.x:.2f}, {fused.y:.2f}, {fused.theta:.2f})  "
              f"wheels-only theta={odom.theta:.2f}")
    if step_count % save_steps == 0 and step_count > 0:
        np.save("../../data/occupancy_logodds.npy", grid.logodds)
        np.save("../../data/odom_path.npy", np.array(path))
        np.save("../../data/odom_path_wheels.npy", np.array(path_odom))
    step_count += 1
