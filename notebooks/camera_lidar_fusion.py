"""camera_lidar_fusion.py - OpenCV obstacle detection on the camera image, fused with the LiDAR scan.

Pipeline
  1. HSV colour masks: wooden boxes (orange-brown), shadowed box sides (very dark), walls (blue-grey).
  2. Morphological open/close to remove speckle and fill gaps.
  3. For every 8th image column, the lowest mask pixel is where the object meets the floor.
     A pinhole model with a level camera at height CAM_H turns that pixel row into a distance:
         depth z = CAM_H * FX / (row - CY)        lateral = (col - CX) * z / FX
  4. Each camera point is converted to the robot frame, then to a LiDAR beam index, and compared with
     the LiDAR range for the same bearing.
  5. Fusion: where the sensors agree (within GATE) the ranges are combined with inverse-variance weights.
"""
import math
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt

IMG_PATH = "data/camera_start.png"
SCAN_PATH = "data/scan_start.npy"

W, H_IMG = 320, 240
FOV = 1.0                                   # rad, horizontal field of view of the camera
FX = (W / 2) / math.tan(FOV / 2)            # focal length in pixels
CX, CY = W / 2, H_IMG / 2
CAM_H = 0.12                                # m, camera height above the floor
CAM_X = 0.05                                # m, camera position ahead of the robot centre
LIDAR_X = -0.03                             # m, LiDAR position relative to the robot centre
SIGMA_LIDAR = 0.02                          # m, LiDAR range noise (assumed)
SIGMA_CAM_REL = 0.05                        # camera range noise as a fraction of the range (assumed)
GATE = 0.30                                 # m, maximum disagreement to call the sensors consistent
LIDAR_MAX = 3.5

img = cv2.imread(IMG_PATH)
if img is None:
    sys.exit(f"Cannot read {IMG_PATH}. Run from the project folder.")
img = cv2.resize(img, (W, H_IMG))
scan = np.load(SCAN_PATH)

# ---- 1-2. Colour masks (OpenCV hue runs 0-179) ------------------------------------
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
Hh, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]
box_face = (Hh >= 10) & (Hh <= 20) & (S >= 80) & (V >= 40) & (V <= 170)   # brown wood
shadow = V < 45                                                            # shadowed box side
wall = (Hh >= 100) & (Hh <= 125) & (V < 112)                               # blue-grey wall


def clean(mask):
    m = mask.astype(np.uint8) * 255
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9)))


box_mask = clean(box_face | shadow)
wall_mask = clean(wall)
any_mask = cv2.bitwise_or(box_mask, wall_mask)

# ---- Annotated picture: bounding boxes around the detected boxes -------------------
annotated = img.copy()
contours, _ = cv2.findContours(box_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
for c in contours:
    if cv2.contourArea(c) < 150:
        continue
    x, y, w, h = cv2.boundingRect(c)
    cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(annotated, "box", (x, max(12, y - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
wc, _ = cv2.findContours(wall_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
for c in wc:
    if cv2.contourArea(c) < 400:
        continue
    x, y, w, h = cv2.boundingRect(c)
    cv2.rectangle(annotated, (x, y), (x + w, y + h), (255, 160, 0), 2)
    cv2.putText(annotated, "wall", (x + 3, y + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 160, 0), 1)
cv2.imwrite("report/camera_detections.png", annotated)


# ---- 3-4. Camera range per column, compared with the LiDAR --------------------------
def camera_range(u, v):
    z = CAM_H * FX / (v - CY)                # depth along the optical axis
    lateral = (u - CX) * z / FX              # to the right
    fwd, left = CAM_X + z, -lateral          # robot frame (x forward, y left)
    return math.hypot(fwd - LIDAR_X, left), math.atan2(left, fwd - LIDAR_X)


def lidar_range(phi):
    # beam 0 points backwards and the index runs clockwise: phi = pi - 2*pi*i/n
    n = len(scan)
    i = int(round((math.pi - phi) / (2 * math.pi / n))) % n
    vals = [scan[(i + k) % n] for k in (-1, 0, 1)]
    vals = [x for x in vals if math.isfinite(x)]
    return min(vals) if vals else float("inf")


rows = []
for u in range(4, W, 8):
    ys = np.where(any_mask[:, u] > 0)[0]
    if len(ys) == 0:
        continue
    vb = int(ys.max())
    if vb <= CY + 2 or vb >= H_IMG - 2:      # above the horizon or touching the bottom edge
        continue
    r_cam, phi = camera_range(u, vb)
    cls = "box" if box_mask[vb, u] > 0 else "wall"
    r_lid = lidar_range(phi)
    if not math.isfinite(r_lid):
        status = "camera-only" if r_cam > LIDAR_MAX - 0.2 else "disagree"
        fused = r_cam if status == "camera-only" else float("nan")
    elif abs(r_cam - r_lid) < GATE:
        status = "confirmed"
        wl, wc_ = 1 / SIGMA_LIDAR ** 2, 1 / (SIGMA_CAM_REL * r_cam) ** 2
        fused = (wl * r_lid + wc_ * r_cam) / (wl + wc_)
    else:
        status = "disagree"
        fused = r_lid                          # trust the LiDAR when they disagree
    rows.append((u, math.degrees(phi), cls, r_cam, r_lid, fused, status))

print(f"{'col':>4} {'bearing':>10} {'class':>6} {'camera':>7} {'lidar':>7} {'fused':>7}  status")
for u, b, cls, rc, rl, f, st in rows:
    print(f"{u:4d} {b:7.1f} deg {cls:>6} {rc:7.2f} {rl:7.2f} {f:7.2f}  {st}")

ok = [r for r in rows if r[6] == "confirmed"]
err = np.array([r[3] - r[4] for r in ok])
print("\nSummary")
print(f"  columns with a detection : {len(rows)}")
for st in ("confirmed", "camera-only", "disagree"):
    print(f"  {st:12s}: {sum(1 for r in rows if r[6] == st)}")
if len(err):
    print(f"  camera-vs-LiDAR error on confirmed columns: mean {err.mean():+.3f} m, "
          f"RMSE {np.sqrt((err ** 2).mean()):.3f} m, max {np.abs(err).max():.3f} m")

# ---- Plot ---------------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
ax1.imshow(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
ax1.set_title("OpenCV detections (green = box, blue = wall)")
ax1.axis("off")
bearings = [r[1] for r in rows]
ax2.plot(bearings, [min(r[4], 4) for r in rows], "k-", label="LiDAR")
colors = {"confirmed": "tab:green", "camera-only": "tab:blue", "disagree": "tab:red"}
for st, col in colors.items():
    pts = [(r[1], r[3]) for r in rows if r[6] == st]
    if pts:
        ax2.scatter(*zip(*pts), c=col, s=25, label=f"camera ({st})")
ax2.plot([r[1] for r in rows if math.isfinite(r[5])], [r[5] for r in rows if math.isfinite(r[5])],
         "m--", linewidth=1, label="fused")
ax2.set_xlabel("bearing in robot frame (deg, left positive)")
ax2.set_ylabel("range (m)")
ax2.set_title("Camera vs LiDAR range along the field of view")
ax2.invert_xaxis()
ax2.grid(True)
ax2.legend()
plt.tight_layout()
plt.savefig("report/camera_lidar_fusion.png", dpi=150)
plt.show()
