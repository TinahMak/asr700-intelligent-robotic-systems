"""pid_tuning.py - differential-drive kinematics, PID heading control, gain tuning and path following.

Runs offline (no Webots). The simulated robot uses the TurtleBot3 Burger's real limits and a first-order
lag on the wheel speeds, so overshoot and oscillation behave like a real motor-driven robot.
"""
import itertools
import math
import numpy as np
import matplotlib.pyplot as plt

# ---- Robot constants (TurtleBot3 Burger) ---------------------------------------------
R = 0.033                    # m, wheel radius
L = 0.160                    # m, wheel separation
W_MAX = 6.67                 # rad/s, maximum wheel speed
V_MAX = R * W_MAX            # m/s, about 0.22
OMEGA_MAX = 2 * R * W_MAX / L    # rad/s, turning on the spot, about 2.75
DT = 0.032                   # s, Webots time step
TAU = 0.15                   # s, wheel-speed lag (assumed motor response)


# ---- 1. Differential-drive kinematics -----------------------------------------------
def wheels_to_body(w_left, w_right):
    """Wheel angular speeds (rad/s) -> forward speed v (m/s) and turn rate omega (rad/s)."""
    v = R * (w_right + w_left) / 2
    omega = R * (w_right - w_left) / L
    return v, omega


def body_to_wheels(v, omega):
    """Inverse kinematics. If a wheel would exceed W_MAX, both are scaled down together so the
    curvature (omega / v) is kept."""
    w_right = (v + omega * L / 2) / R
    w_left = (v - omega * L / 2) / R
    peak = max(abs(w_left), abs(w_right))
    if peak > W_MAX:
        w_left, w_right = w_left * W_MAX / peak, w_right * W_MAX / peak
    return w_left, w_right


class Robot:
    def __init__(self, x=0.0, y=0.0, theta=0.0):
        self.x, self.y, self.theta = x, y, theta
        self.w_left = 0.0
        self.w_right = 0.0

    def step(self, v_cmd, omega_cmd):
        cmd_l, cmd_r = body_to_wheels(v_cmd, omega_cmd)
        self.w_left += (cmd_l - self.w_left) * DT / TAU          # motor lag
        self.w_right += (cmd_r - self.w_right) * DT / TAU
        v, omega = wheels_to_body(self.w_left, self.w_right)
        self.x += v * math.cos(self.theta + omega * DT / 2) * DT   # mid-step heading
        self.y += v * math.sin(self.theta + omega * DT / 2) * DT
        self.theta += omega * DT


def wrap(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


# ---- 2. PID controller -----------------------------------------------------------------
class PID:
    def __init__(self, kp, ki, kd, out_limit, i_limit=1.0):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.out_limit, self.i_limit = out_limit, i_limit
        self.integral, self.prev_err = 0.0, None

    def step(self, err):
        self.integral = float(np.clip(self.integral + err * DT, -self.i_limit, self.i_limit))   # anti-windup
        deriv = 0.0 if self.prev_err is None else (err - self.prev_err) / DT
        self.prev_err = err
        out = self.kp * err + self.ki * self.integral + self.kd * deriv
        return float(np.clip(out, -self.out_limit, self.out_limit))


# ---- 3. Heading step response (turn on the spot by 90 degrees) ------------------------
TARGET = math.pi / 2
T_END = 8.0


def heading_step(kp, ki, kd):
    robot, pid = Robot(), PID(kp, ki, kd, OMEGA_MAX)
    t = np.arange(0, T_END, DT)
    th = np.zeros_like(t)
    for i in range(len(t)):
        th[i] = robot.theta
        robot.step(0.0, pid.step(wrap(TARGET - robot.theta)))
    return t, th


def metrics(t, th):
    final = TARGET
    try:
        t10 = t[np.where(th >= 0.1 * final)[0][0]]
        t90 = t[np.where(th >= 0.9 * final)[0][0]]
        rise = t90 - t10
    except IndexError:
        rise = float("nan")
    overshoot = max(0.0, (th.max() - final) / final * 100)
    outside = np.where(np.abs(th - final) > 0.02 * final)[0]
    settle = t[outside[-1]] + DT if len(outside) else 0.0
    if len(outside) and outside[-1] >= len(t) - 1:
        settle = float("nan")                          # never settled
    iae = float(np.sum(np.abs(th - final)) * DT)
    return rise, overshoot, settle, iae


manual = {
    "P  Kp=1":              (1.0, 0.0, 0.0),
    "P  Kp=4":              (4.0, 0.0, 0.0),
    "P  Kp=12 (high gain)": (12.0, 0.0, 0.0),
    "PD Kp=4 Kd=0.4":       (4.0, 0.0, 0.4),
    "PID Kp=4 Ki=1 Kd=0.4": (4.0, 1.0, 0.4),
}

# small grid search; score = settling time + penalty for overshoot (lower is better)
best, best_score = None, float("inf")
for kp, ki, kd in itertools.product([2, 3, 4, 6, 8], [0.0, 0.2, 0.5], [0.0, 0.2, 0.4, 0.8]):
    t, th = heading_step(kp, ki, kd)
    rise, os_, settle, iae = metrics(t, th)
    if math.isnan(settle):
        continue
    score = settle + 0.1 * os_
    if score < best_score:
        best, best_score = (kp, ki, kd), score
manual["Tuned (grid search) " + str(best)] = best

print(f"{'gains':36s} {'rise (s)':>9} {'overshoot %':>12} {'settle 2% (s)':>14} {'IAE':>8}")
curves = {}
for name, (kp, ki, kd) in manual.items():
    t, th = heading_step(kp, ki, kd)
    rise, os_, settle, iae = metrics(t, th)
    curves[name] = (t, th)
    print(f"{name:36s} {rise:9.2f} {os_:12.1f} {settle:14.2f} {iae:8.3f}")

plt.figure(figsize=(9, 5))
for name, (t, th) in curves.items():
    plt.plot(t, np.degrees(th), label=name, linewidth=2.2 if name.startswith("Tuned") else 1.2)
plt.axhline(90, color="k", linestyle=":", label="target 90 deg")
plt.xlabel("time (s)")
plt.ylabel("heading (deg)")
plt.title("Heading step response for different PID gains")
plt.grid(True)
plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig("report/pid_step_response.png", dpi=150)

# ---- 4. Path following with the tuned heading PID ------------------------------------
KP, KI, KD = best
V_CRUISE = 0.15              # m/s
LOOKAHEAD = 0.25             # m
GOAL_TOL = 0.08              # m

path = np.load("data/planned_path_astar.npy")                 # (N, 2) world coordinates from A*
# densify the path (every ~2 cm) so the lookahead moves smoothly
dense = [path[0]]
for a, b in zip(path[:-1], path[1:]):
    n = max(1, int(np.hypot(*(b - a)) / 0.02))
    for k in range(1, n + 1):
        dense.append(a + (b - a) * k / n)
dense = np.array(dense)

robot = Robot(path[0][0], path[0][1], 0.0)
pid = PID(KP, KI, KD, OMEGA_MAX)
trace, idx = [], 0
for step in range(int(120 / DT)):
    trace.append((robot.x, robot.y))
    # closest path point (searching only forward), then the lookahead point beyond it
    d = np.hypot(dense[idx:, 0] - robot.x, dense[idx:, 1] - robot.y)
    idx += int(np.argmin(d))
    j = idx
    while j < len(dense) - 1 and np.hypot(*(dense[j] - [robot.x, robot.y])) < LOOKAHEAD:
        j += 1
    tx, ty = dense[j]
    err = wrap(math.atan2(ty - robot.y, tx - robot.x) - robot.theta)
    omega = pid.step(err)
    dist_goal = math.hypot(dense[-1][0] - robot.x, dense[-1][1] - robot.y)
    v = V_CRUISE * max(0.0, math.cos(err)) ** 2           # slow down when not facing the target
    v = min(v, 0.6 * dist_goal)                           # and when arriving
    if dist_goal < GOAL_TOL:
        break
    robot.step(v, omega)
trace = np.array(trace)

# cross-track error = distance from each actual point to the nearest planned point
xt = np.array([np.min(np.hypot(dense[:, 0] - p[0], dense[:, 1] - p[1])) for p in trace])
print(f"\nPath following (gains Kp={KP}, Ki={KI}, Kd={KD}, cruise {V_CRUISE} m/s)")
print(f"  time to goal       : {len(trace) * DT:.1f} s")
print(f"  mean cross-track   : {xt.mean() * 100:.1f} cm")
print(f"  max cross-track    : {xt.max() * 100:.1f} cm")
print(f"  final distance to goal: {math.hypot(*(dense[-1] - trace[-1])) * 100:.1f} cm")

fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 5))
a1.plot(path[:, 0], path[:, 1], "b--", label="planned (A*)")
a1.plot(trace[:, 0], trace[:, 1], "r-", label="simulated actual")
a1.plot(*path[0], "go", label="start")
a1.plot(*path[-1], "k*", markersize=12, label="goal")
a1.set_aspect("equal")
a1.grid(True)
a1.set_xlabel("x (m)")
a1.set_ylabel("y (m)")
a1.set_title("Planned vs simulated path")
a1.legend()
a2.plot(np.arange(len(xt)) * DT, xt * 100)
a2.set_xlabel("time (s)")
a2.set_ylabel("cross-track error (cm)")
a2.set_title("Distance from the planned path")
a2.grid(True)
plt.tight_layout()
plt.savefig("report/path_following_sim.png", dpi=150)
np.save("data/pid_gains.npy", np.array(best))
plt.show()
