"""collect_controller.py - wanders around component_a.wbt and logs LiDAR scans with the fused pose.

Output: data/lidar_dataset.npz with scans (N x 360, metres, inf = no return), poses (N x 3: x, y, theta)
and t (sample times). One sample every SAMPLE_EVERY seconds of simulated time.
The labels for the neural network are computed later, offline, from the known world geometry.
"""
import math
import random
import numpy as np
from controller import Robot
from odometry import Odometry

FORWARD_SPEED, TURN_SPEED = 3.0, 2.0          # wheel speeds, rad/s
FRONT_STOP, FRONT_CLEAR = 0.30, 0.55          # start turning below 0.30 m, stop turning above 0.55 m
SAMPLE_EVERY, T_END = 0.5, 1200.0             # s of simulated time
START_X, START_Y = -1.5, -1.5

random.seed(1)
robot = Robot()
dt_ms = int(robot.getBasicTimeStep())
left_m, right_m = robot.getDevice("left wheel motor"), robot.getDevice("right wheel motor")
for m in (left_m, right_m):
    m.setPosition(float("inf"))
    m.setVelocity(0.0)
left_s, right_s = robot.getDevice("left wheel sensor"), robot.getDevice("right wheel sensor")
lidar, compass = robot.getDevice("LDS-01"), robot.getDevice("compass")
for dev in (left_s, right_s, lidar, compass):
    dev.enable(dt_ms)

robot.step(dt_ms)
cv = compass.getValues()
odom = Odometry(left_s.getValue(), right_s.getValue(), START_X, START_Y, math.atan2(cv[0], cv[1]))

scans, poses, times = [], [], []
state, turn_dir, turn_start, turn_until = "forward", 1, 0.0, 0.0
next_rand_turn = random.uniform(8, 20)
next_sample, next_save, finished = 0.0, 30.0, False


def save():
    np.savez_compressed("../../data/lidar_dataset.npz", scans=np.array(scans, dtype=np.float32),
                        poses=np.array(poses), t=np.array(times))


while robot.step(dt_ms) != -1:
    t = robot.getTime()
    cv = compass.getValues()
    out = odom.update(left_s.getValue(), right_s.getValue(), math.atan2(cv[0], cv[1]))
    x, y, th = out if out is not None else (odom.x, odom.y, odom.theta)

    if finished:
        continue
    raw = np.array(lidar.getRangeImage(), dtype=float)
    r = np.where(np.isfinite(raw), raw, 3.5)
    front = r[150:211].min()

    if t >= SAMPLE_EVERY and t >= next_sample:
        scans.append(raw)
        poses.append((x, y, th))
        times.append(t)
        next_sample += SAMPLE_EVERY

    # wander: drive forward, turn away from obstacles (hysteresis) and sometimes turn at random
    if state == "forward":
        if front < FRONT_STOP:
            state, turn_dir, turn_start = "turn", random.choice((1, -1)), t
        elif t >= next_rand_turn:
            state, turn_dir, turn_until = "randturn", random.choice((1, -1)), t + random.uniform(0.4, 1.5)
    elif state == "turn":
        if front > FRONT_CLEAR:
            state, next_rand_turn = "forward", t + random.uniform(8, 20)
        elif t - turn_start > 5.0:                # stuck in a corner: try the other direction
            turn_dir, turn_start = -turn_dir, t
    elif state == "randturn":
        if t >= turn_until:
            state, next_rand_turn = "forward", t + random.uniform(8, 20)

    if state == "forward":
        wl = wr = FORWARD_SPEED
    else:
        wl, wr = -turn_dir * TURN_SPEED, turn_dir * TURN_SPEED
    if state == "randturn" and front < FRONT_STOP:
        state = "turn"
        turn_start = t
    left_m.setVelocity(wl)
    right_m.setVelocity(wr)

    if t >= next_save:
        save()
        print(f"t={t:6.1f}s samples={len(scans)} pos=({x:5.2f},{y:5.2f}) state={state}")
        next_save += 30.0
    if t >= T_END:
        left_m.setVelocity(0.0)
        right_m.setVelocity(0.0)
        save()
        print(f"FINISHED: saved {len(scans)} samples to data/lidar_dataset.npz")
        finished = True
