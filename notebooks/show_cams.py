"""show_cams.py - compare the four test camera orientations side by side."""
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

files = [("camera (no rotation)", "camera_start.png"),
         ("cam_b (pitched down 0.5 rad)", "cam_b.png"),
         ("cam_c (rotated +90 deg about y)", "cam_c.png"),
         ("cam_d (rotated -90 deg about y)", "cam_d.png")]
fig, axes = plt.subplots(2, 2, figsize=(9, 7))
for ax, (title, f) in zip(axes.ravel(), files):
    ax.imshow(mpimg.imread("data/" + f))
    ax.set_title(title)
    ax.axis("off")
plt.tight_layout()
plt.savefig("report/camera_orientation_test.png", dpi=120)
plt.show()
