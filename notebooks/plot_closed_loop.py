"""plot_closed_loop.py - planned (A*) vs actual (Webots) path, plus cross-track error."""
import numpy as np
import matplotlib.pyplot as plt

planned = np.load("data/planned_path_astar.npy")
log = np.load("data/actual_path_webots.npy")           # columns: t, x, y, theta
logodds = np.load("data/occupancy_map_final.npy")
# keep the log only up to the moment the robot first reaches the goal tolerance
d_goal = np.hypot(log[:, 1] - planned[-1][0], log[:, 2] - planned[-1][1])
end = int(np.argmax(d_goal < 0.085)) if (d_goal < 0.085).any() else len(log) - 1
log = log[:end + 1]
t, ax_, ay_ = log[:, 0], log[:, 1], log[:, 2]

# cross-track error: distance from each actual point to the nearest point on the densified plan
dense = [planned[0]]
for a, b in zip(planned[:-1], planned[1:]):
    n = max(1, int(np.hypot(*(b - a)) / 0.01))
    for k in range(1, n + 1):
        dense.append(a + (b - a) * k / n)
dense = np.array(dense)
xt = np.array([np.min(np.hypot(dense[:, 0] - x, dense[:, 1] - y)) for x, y in zip(ax_, ay_)])

print("Closed-loop result in Webots")
print(f"  time to goal       : {t[-1]:.1f} s")
print(f"  mean cross-track   : {xt.mean() * 100:.1f} cm")
print(f"  max cross-track    : {xt.max() * 100:.1f} cm")
print(f"  final dist to goal : {np.hypot(*(planned[-1] - log[-1, 1:3])) * 100:.1f} cm")

fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5.5))
a1.imshow(logodds > 0, cmap="Greys", origin="lower", extent=[-2, 2, -2, 2], alpha=0.5)
a1.plot(planned[:, 0], planned[:, 1], "b--", linewidth=2, label="planned (A*)")
a1.plot(ax_, ay_, "r-", linewidth=1.5, label="actual (Webots, fused pose)")
a1.plot(*planned[0], "go", label="start")
a1.plot(*planned[-1], "k*", markersize=14, label="goal")
a1.set_aspect("equal")
a1.set_xlabel("x (m)")
a1.set_ylabel("y (m)")
a1.set_title("Planned vs actual path in the occupancy map")
a1.legend(loc="upper left", fontsize=8)
a2.plot(t, xt * 100)
a2.set_xlabel("time (s)")
a2.set_ylabel("cross-track error (cm)")
a2.set_title("Distance from the planned path")
a2.grid(True)
plt.tight_layout()
plt.savefig("report/closed_loop_webots.png", dpi=150)
plt.show()
