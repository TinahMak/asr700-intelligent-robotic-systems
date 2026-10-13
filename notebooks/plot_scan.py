"""plot_scan.py - plot one saved LiDAR scan in the robot frame."""
import numpy as np
import matplotlib.pyplot as plt

ranges = np.load("data/scan_start.npy")      # 360 distances in metres, one per beam
n = len(ranges)

# TODO 1: angle of each beam in radians. Beam i points at i * (2*pi / n).
angles = np.arange(n) * (2 * np.pi / n)

# TODO 2: boolean mask that keeps only finite readings (drop inf = no hit).
valid = np.isfinite(ranges)

# TODO 3: polar -> Cartesian for the valid beams only.
#   x = r * cos(theta)      y = r * sin(theta)
x = ranges[valid] * np.cos(np.pi - angles[valid])
y = ranges[valid] * np.sin(np.pi - angles[valid])

plt.figure(figsize=(6, 6))
plt.scatter(x, y, s=8, label="LiDAR hits")
plt.plot(0, 0, "r^", markersize=12, label="robot")
plt.axis("equal")
plt.grid(True)
plt.xlabel("x (m)")
plt.ylabel("y (m)")
plt.title("LiDAR scan in the robot frame")
plt.legend()
plt.savefig("report/scan_start.png", dpi=150)
plt.show()
