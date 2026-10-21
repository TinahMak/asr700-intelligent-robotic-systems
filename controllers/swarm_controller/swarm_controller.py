"""swarm_controller.py - step B1: every robot estimates its pose (wheels + compass), broadcasts it
with the Emitter, and listens to the others with the Receiver. Robots do not move yet."""
import math
from controller import Robot
from odometry import Odometry

robot = Robot()
dt_ms = int(robot.getBasicTimeStep())
name = robot.getName()
x0, y0 = [float(v) for v in robot.getCustomData().split()]     # start position from the world file

left_s = robot.getDevice("left wheel sensor")
right_s = robot.getDevice("right wheel sensor")
compass = robot.getDevice("compass")
emitter = robot.getDevice("emitter")
receiver = robot.getDevice("receiver")
for d in (left_s, right_s, compass, receiver):
    d.enable(dt_ms)

robot.step(dt_ms)
cv = compass.getValues()
odom = Odometry(left_s.getValue(), right_s.getValue(), x0, y0, math.atan2(cv[0], cv[1]))

neighbours = {}                                                # name -> (x, y)
next_tx, next_print = 0.0, 0.0
while robot.step(dt_ms) != -1:
    t = robot.getTime()
    cv = compass.getValues()
    out = odom.update(left_s.getValue(), right_s.getValue(), math.atan2(cv[0], cv[1]))
    x, y, th = out if out is not None else (odom.x, odom.y, odom.theta)

    if t >= next_tx:                                           # broadcast at 2 Hz
        emitter.send(f"{name},{x:.3f},{y:.3f}".encode())
        next_tx += 0.5
    while receiver.getQueueLength() > 0:
        msg = receiver.getString()
        receiver.nextPacket()
        other, ox, oy = msg.split(",")
        if other != name:
            neighbours[other] = (float(ox), float(oy))
    if t >= next_print:
        seen = ", ".join(f"{k}=({v[0]:.1f},{v[1]:.1f})" for k, v in sorted(neighbours.items()))
        print(f"[{name}] t={t:4.1f}s pos=({x:5.2f},{y:5.2f}) hears {len(neighbours)}: {seen}")
        next_print += 2.0
