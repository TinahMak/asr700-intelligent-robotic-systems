"""consensus_controller.py - distributed average consensus (rendezvous) on a ring of 4 robots.

Each robot only hears its two ring neighbours. State z_i = position - formation offset. Every
round: z_i += EPS * sum_j (z_j - z_i). The robot drives (PID heading control from Component A)
to the point z_i + d_i. Logs data/consensus_robot<i>.npy: t, x, y, zx, zy, messages_sent.
"""
import math
import numpy as np
from controller import Robot
from odometry import Odometry

R, L, W_MAX = 0.033, 0.160, 6.6
OMEGA_MAX = 2 * R * W_MAX / L
N, EPS, TX_PERIOD = 4, 0.3, 0.2          # robots, consensus step size, seconds between rounds
RING_RADIUS, V_CRUISE, STOP_DIST, T_END = 0.35, 0.15, 0.05, 45.0


def wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


def body_to_wheels(v, omega):
    wr, wl = (v + omega * L / 2) / R, (v - omega * L / 2) / R
    peak = max(abs(wl), abs(wr))
    if peak > W_MAX:
        wl, wr = wl * W_MAX / peak, wr * W_MAX / peak
    return wl, wr


class PID:
    def __init__(self, kp, ki, kd, lim, dt, il=1.0):
        self.kp, self.ki, self.kd, self.lim, self.dt, self.il = kp, ki, kd, lim, dt, il
        self.i, self.prev = 0.0, None

    def step(self, err):
        self.i = float(np.clip(self.i + err * self.dt, -self.il, self.il))
        d = 0.0 if self.prev is None else (err - self.prev) / self.dt
        self.prev = err
        return float(np.clip(self.kp * err + self.ki * self.i + self.kd * d, -self.lim, self.lim))


robot = Robot()
dt_ms = int(robot.getBasicTimeStep())
DT = dt_ms / 1000.0
name = robot.getName()
me = int(name.replace("robot", ""))
nbrs = {(me - 1) % N, (me + 1) % N}                       # ring topology
x0, y0 = [float(v) for v in robot.getCustomData().split()]
norm = math.hypot(x0, y0)
d = (RING_RADIUS * x0 / norm, RING_RADIUS * y0 / norm)    # formation offset (own corner direction)
z = [x0 - d[0], y0 - d[1]]                                # consensus state

left_m, right_m = robot.getDevice("left wheel motor"), robot.getDevice("right wheel motor")
for m in (left_m, right_m):
    m.setPosition(float("inf"))
    m.setVelocity(0.0)
left_s, right_s = robot.getDevice("left wheel sensor"), robot.getDevice("right wheel sensor")
compass, emitter, receiver = robot.getDevice("compass"), robot.getDevice("emitter"), robot.getDevice("receiver")
for dev in (left_s, right_s, compass, receiver):
    dev.enable(dt_ms)

KP, KI, KD = np.load("../../data/pid_gains.npy")
pid = PID(KP, KI, KD, OMEGA_MAX, DT)

robot.step(dt_ms)
cv = compass.getValues()
odom = Odometry(left_s.getValue(), right_s.getValue(), x0, y0, math.atan2(cv[0], cv[1]))

heard, log, msgs, next_tx, next_save, saved_final = {}, [], 0, 0.0, 5.0, False
while robot.step(dt_ms) != -1:
    t = robot.getTime()
    cv = compass.getValues()
    out = odom.update(left_s.getValue(), right_s.getValue(), math.atan2(cv[0], cv[1]))
    x, y, th = out if out is not None else (odom.x, odom.y, odom.theta)

    while receiver.getQueueLength() > 0:                  # read neighbours' states
        j, zx, zy = receiver.getString().split(",")
        receiver.nextPacket()
        if int(j) in nbrs:
            heard[int(j)] = (float(zx), float(zy))

    if t >= next_tx:                                      # one consensus round
        if len(heard) == len(nbrs):
            z[0] += EPS * sum(v[0] - z[0] for v in heard.values())
            z[1] += EPS * sum(v[1] - z[1] for v in heard.values())
        emitter.send(f"{me},{z[0]:.5f},{z[1]:.5f}".encode())
        msgs += 1
        next_tx += TX_PERIOD

    tx, ty = z[0] + d[0], z[1] + d[1]                     # drive to the consensus target
    dist = math.hypot(tx - x, ty - y)
    if dist > STOP_DIST:
        err = wrap(math.atan2(ty - y, tx - x) - th)
        omega = pid.step(err)
        v = min(V_CRUISE * max(0.0, math.cos(err)) ** 2, 0.6 * dist)
        wl, wr = body_to_wheels(v, omega)
    else:
        wl = wr = 0.0
    left_m.setVelocity(wl)
    right_m.setVelocity(wr)

    if t < T_END:
        log.append((t, x, y, z[0], z[1], msgs))
        if t >= next_save:
            np.save(f"../../data/consensus_robot{me}.npy", np.array(log))
            print(f"[{name}] t={t:4.1f}s pos=({x:5.2f},{y:5.2f}) z=({z[0]:6.3f},{z[1]:6.3f}) dist_to_target={dist:.2f} m")
            next_save += 5.0
    elif not saved_final:
        np.save(f"../../data/consensus_robot{me}.npy", np.array(log))
        print(f"[{name}] FINISHED t={t:.1f}s final pos=({x:.2f},{y:.2f}) messages sent={msgs}")
        saved_final = True
