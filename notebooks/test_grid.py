"""test_grid.py - build an occupancy grid from the saved scan and plot it."""
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, "controllers/nav_controller")
from occupancy_grid import OccupancyGrid, ARENA_MIN, ARENA_SIZE

ranges = np.load("data/scan_start.npy")
pose = (-1.435, -1.5, 0.0)              # odometry pose at the time of the scan

grid = OccupancyGrid()
grid.update(pose, list(ranges), max_range=3.5)

extent = [ARENA_MIN, ARENA_MIN + ARENA_SIZE, ARENA_MIN, ARENA_MIN + ARENA_SIZE]
plt.figure(figsize=(6, 6))
plt.imshow(grid.probability(), origin="lower", extent=extent, cmap="gray_r", vmin=0, vmax=1)
plt.plot(pose[0], pose[1], "r^", markersize=10, label="robot")
plt.colorbar(label="P(occupied)")
plt.xlabel("x (m)")
plt.ylabel("y (m)")
plt.title("Occupancy grid from one scan")
plt.legend()
plt.savefig("report/grid_one_scan.png", dpi=150)
plt.show()
