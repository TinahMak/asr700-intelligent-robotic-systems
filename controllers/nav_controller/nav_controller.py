"""nav_controller.py - Step 0 sanity check for the Component A robot.

Purpose: prove that Python, Webots and the TurtleBot3Burger's LiDAR and
motors all work together. This is NOT the final navigation controller.
The perception, planning and PID code will replace it step by step.

What it does:
  1. Finds the two wheel motors and the LiDAR by name.
  2. Drives the robot forward slowly.
  3. Prints the nearest LiDAR reading once per second.
  4. Stops when something is closer than STOP_DISTANCE.
"""

import math
from controller import Robot

# ---- Settings you can change -------------------------------------------
WHEEL_SPEED = 2.0       # rad/s. The Burger's maximum is about 6.67 rad/s.
STOP_DISTANCE = 0.20    # metres. Stop if anything is closer than this.
PRINT_EVERY = 1.0       # seconds between printed LiDAR readings.
# -------------------------------------------------------------------------

robot = Robot()
time_step = int(robot.getBasicTimeStep())   # simulation step in ms

# ---- Devices. These names come from the TurtleBot3Burger model. ----------
left_motor = robot.getDevice("left wheel motor")
right_motor = robot.getDevice("right wheel motor")
lidar = robot.getDevice("LDS-01")

# getDevice() returns None when the name is wrong, so fail with a clear message.
for name, device in (("left wheel motor", left_motor),
                     ("right wheel motor", right_motor),
                     ("LDS-01", lidar)):
    if device is None:
        raise RuntimeError(f"Device '{name}' not found. Check its name in the scene tree.")

# ---- Motor setup. Velocity control needs position = infinity. ------------
for motor in (left_motor, right_motor):
    motor.setPosition(float("inf"))
    motor.setVelocity(0.0)

# ---- LiDAR setup. It returns nothing until it is enabled. ----------------
lidar.enable(time_step)
n_beams = lidar.getHorizontalResolution()
fov_deg = math.degrees(lidar.getFov())
max_range = lidar.getMaxRange()
print(f"LiDAR: {n_beams} beams, FOV {fov_deg:.0f} deg, max range {max_range} m")
print(f"Time step: {time_step} ms")

steps_per_print = max(1, int(PRINT_EVERY * 1000 / time_step))
step_count = 0
stopped = False

# ---- Main loop: read sensors -> decide -> command motors -> step. --------
while robot.step(time_step) != -1:
    ranges = lidar.getRangeImage()                      # list of distances in metres
    valid = [r for r in ranges if math.isfinite(r)]     # drop inf (no hit) and nan
    nearest = min(valid) if valid else float("inf")

    if nearest < STOP_DISTANCE:
        left_motor.setVelocity(0.0)
        right_motor.setVelocity(0.0)
        if not stopped:
            print(f"Obstacle at {nearest:.2f} m - stopping.")
            stopped = True
    else:
        left_motor.setVelocity(WHEEL_SPEED)
        right_motor.setVelocity(WHEEL_SPEED)

    if step_count % steps_per_print == 0:
        print(f"t={robot.getTime():5.1f}s  nearest obstacle: {nearest:.2f} m")
    step_count += 1
