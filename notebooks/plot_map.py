"""plot_map.py - show the saved occupancy grid and the odometry path."""
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, "controllers/nav_controller")
from occupancy_grid import ARENA_MIN, ARENA_SIZE

logodds = np.load("data/occupancy_logodds.npy")
path = np.load("data/odom_path.npy")
prob = 1.0 - 1.0 / (1.0 + np.exp(logodds))

extent = [ARENA_MIN, ARENA_MIN + ARENA_SIZE, ARENA_MIN, ARENA_MIN + ARENA_SIZE]
plt.figure(figsize=(7, 7))
plt.imshow(prob, origin="lower", extent=extent, cmap="gray_r", vmin=0, vmax=1)
wheels = np.load("data/odom_path_wheels.npy")
plt.plot(wheels[:, 0], wheels[:, 1], "b--", linewidth=1, label="wheels only")
plt.plot(path[:, 0], path[:, 1], "r-", linewidth=1, label="wheels + compass")
plt.colorbar(label="P(occupied)")
plt.xlabel("x (m)")
plt.ylabel("y (m)")
plt.title("Occupancy grid and robot path")
plt.legend()
plt.savefig("report/occupancy_map.png", dpi=150)
plt.show()
