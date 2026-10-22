"""follow_controller.py - closed-loop path following in Webots.

Pipeline: A* path (from the occupancy map) -> lookahead target -> heading error -> PID -> turn rate omega
-> inverse kinematics -> wheel motor speeds. The pose comes from wheel odometry fused with the compass.
The actual path is logged to data/actual_path_webots.npy for the planned-vs-actual plot.
"""
import math
import numpy as np
from controller import Robot
from odometry import Odometry

# ---- constants (same as notebooks/pid_tuning.py) ---------------------------------------
R, L, W_MAX = 0.033, 0.160, 6.6
OMEGA_MAX = 2 * R * W_MAX / L
V_CRUISE, LOOKAHEAD, GOAL_TOL = 0.15, 0.25, 0.08
START_X, START_Y = -1.5, -1.5          # robot start in component_a.wbt
MAX_TIME = 150.0                       # s of simulated time before giving up


def wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


def body_to_wheels(v, omega):
    wr = (v + omega * L / 2) / R
    wl = (v - omega * L / 2) / R
    peak = max(abs(wl), abs(wr))
    if peak > W_MAX:                   # scale both so the curvature is preserved
        wl, wr = wl * W_MAX / peak, wr * W_MAX / peak
    return wl, wr


class PID:
    def __init__(self, kp, ki, kd, out_limit, dt, i_limit=1.0):
        self.kp, self.ki, self.kd, self.lim, self.dt, self.il = kp, ki, kd, out_limit, dt, i_limit
        self.i, self.prev = 0.0, None

    def step(self, err):
        self.i = float(np.clip(self.i + err * self.dt, -self.il, self.il))
        d = 0.0 if self.prev is None else (err - self.prev) / self.dt
        self.prev = err
        return float(np.clip(self.kp * err + self.ki * self.i + self.kd * d, -self.lim, self.lim))


robot = Robot()
dt_ms = int(robot.getBasicTimeStep())
DT = dt_ms / 1000.0

left_m = robot.getDevice("left wheel motor")
right_m = robot.getDevice("right wheel motor")
for m in (left_m, right_m):
    m.setPosition(float("inf"))        # velocity control
    m.setVelocity(0.0)
left_s = robot.getDevice("left wheel sensor")
right_s = robot.getDevice("right wheel sensor")
left_s.enable(dt_ms)
right_s.enable(dt_ms)
compass = robot.getDevice("compass")
compass.enable(dt_ms)

# ---- load the plan and the tuned gains -------------------------------------------------
path = np.load("../../data/planned_path_astar.npy")
KP, KI, KD = np.load("../../data/pid_gains.npy")
pid = PID(KP, KI, KD, OMEGA_MAX, DT)
print(f"Loaded path with {len(path)} points, gains Kp={KP} Ki={KI} Kd={KD}")
print(f"Path starts at ({path[0][0]:.2f}, {path[0][1]:.2f}); robot starts at ({START_X}, {START_Y})")

dense = [path[0]]
for a, b in zip(path[:-1], path[1:]):
    n = max(1, int(np.hypot(*(b - a)) / 0.02))
    for k in range(1, n + 1):
        dense.append(a + (b - a) * k / n)
dense = np.array(dense)

robot.step(dt_ms)                      # first step so the sensors hold valid values
cv = compass.getValues()
odom = Odometry(left_s.getValue(), right_s.getValue(), START_X, START_Y, math.atan2(cv[0], cv[1]))

trace, idx, next_print, done = [], 0, 0.0, False
while robot.step(dt_ms) != -1:
    t = robot.getTime()
    cv = compass.getValues()
    heading = math.atan2(cv[0], cv[1])
    out = odom.update(left_s.getValue(), right_s.getValue(), heading)
    x, y, th = out if out is not None else (odom.x, odom.y, odom.theta)
    if not done:
        trace.append((t, x, y, th))
    if not done and len(trace) % 150 == 0:
        np.save("../../data/actual_path_webots.npy", np.array(trace))

    if not done:
        d = np.hypot(dense[idx:, 0] - x, dense[idx:, 1] - y)
        idx += int(np.argmin(d))
        j = idx
        while j < len(dense) - 1 and math.hypot(dense[j][0] - x, dense[j][1] - y) < LOOKAHEAD:
            j += 1
        err = wrap(math.atan2(dense[j][1] - y, dense[j][0] - x) - th)
        omega = pid.step(err)
        dist_goal = math.hypot(dense[-1][0] - x, dense[-1][1] - y)
        v = min(V_CRUISE * max(0.0, math.cos(err)) ** 2, 0.6 * dist_goal)
        if dist_goal < GOAL_TOL or t > MAX_TIME:
            done = True
            left_m.setVelocity(0.0)
            right_m.setVelocity(0.0)
            np.save("../../data/actual_path_webots.npy", np.array(trace))
            print(f"FINISHED at t={t:.1f} s, distance to goal {dist_goal * 100:.1f} cm. "
                  f"Saved data/actual_path_webots.npy")
        else:
            wl, wr = body_to_wheels(v, omega)
            left_m.setVelocity(wl)
            right_m.setVelocity(wr)
        if t >= next_print:
            print(f"t={t:5.1f}s pos=({x:5.2f},{y:5.2f}) heading={math.degrees(th):6.1f} deg "
                  f"err={math.degrees(err):6.1f} deg dist_goal={dist_goal:.2f} m")
            next_print += 2.0
