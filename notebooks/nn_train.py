"""nn_train.py - neural network that classifies what the robot is facing from a raw LiDAR scan.

Classes: 0 = clear (nothing within 1.2 m ahead), 1 = box, 2 = wall.
Data   : data/lidar_dataset.npz (scans + fused poses logged by collect_controller in Webots).
Labels : derived automatically from the known world geometry (boxes parsed from the .wbt file) and
         the robot pose; the network itself only sees the scan, never the pose.
Model  : multilayer perceptron 46 -> 32 -> 16 -> 3 (ReLU, softmax), implemented in NumPy
         (backpropagation + Adam, class-weighted cross-entropy, early stopping).
"""
import re
import numpy as np
import matplotlib.pyplot as plt

CLASSES = ["clear", "box", "wall"]
CLEAR_R, MAX_R, ARENA_HALF = 1.2, 3.5, 2.0
rng = np.random.default_rng(0)

# ---- 1. load data and the box layout from the world file ------------------------------------------
data = np.load("data/lidar_dataset.npz")
scans, poses = data["scans"].astype(float), data["poses"]
print(f"Loaded {len(scans)} scans of {scans.shape[1]} beams")
scans = np.where(np.isfinite(scans), np.minimum(scans, MAX_R), MAX_R)

# drop near-duplicate consecutive scans: for part of the run the robot was stuck turning in a corner,
# which produced hundreds of identical scans that would leak between train and test
kept, anchor, run = [0], 0, 1
for n in range(1, len(scans)):
    if np.hypot(poses[n, 0] - poses[anchor, 0], poses[n, 1] - poses[anchor, 1]) < 0.03:
        run += 1                              # still at the same spot (turning in place)
        if run <= 8:                          # keep at most 8 scans (about one sweep) per spot
            kept.append(n)
    else:
        anchor, run = n, 1
        kept.append(n)
print(f"Removed {len(scans) - len(kept)} scans from long turn-in-place runs; {len(kept)} left")
scans, poses = scans[kept], poses[kept]

world = open("worlds/component_a.wbt").read()
boxes = []
for m in re.finditer(r"WoodenBox\s*\{(.*?)\n\}", world, re.S):
    body = m.group(1)
    tr = re.search(r"translation\s+(\S+)\s+(\S+)\s+(\S+)", body)
    sz = re.search(r"size\s+(\S+)\s+(\S+)\s+(\S+)", body)
    rot = re.search(r"rotation\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)", body)
    if not tr:
        continue
    half = float(sz.group(1)) / 2 if sz else 0.3        # WoodenBox default size is 0.6 m
    if rot and abs(float(rot.group(4))) > 1e-3:
        print("WARNING: a rotated box was found; labels assume axis-aligned boxes")
    boxes.append((float(tr.group(1)), float(tr.group(2)), half))
print("Boxes (x, y, half-size):", boxes)

# ---- 2. labels from geometry ------------------------------------------------------------------------
FRONT = np.arange(170, 191)                              # beams straight ahead (beam 180 = forward)
labels = np.full(len(scans), -1)
for n in range(len(scans)):
    x, y, th = poses[n]
    k = FRONT[np.argmin(scans[n, FRONT])]
    r = scans[n, k]
    if r >= CLEAR_R:
        labels[n] = 0
        continue
    ang = th + (np.pi - 2 * np.pi * k / 360)
    hx, hy = x + r * np.cos(ang), y + r * np.sin(ang)
    if any(abs(hx - bx) < h + 0.08 and abs(hy - by) < h + 0.08 for bx, by, h in boxes):
        labels[n] = 1
    elif abs(hx) > ARENA_HALF - 0.1 or abs(hy) > ARENA_HALF - 0.1:
        labels[n] = 2
keep = labels >= 0
print(f"Discarded {np.sum(~keep)} ambiguous scans")
scans, labels = scans[keep], labels[keep]
print("Class counts:", {c: int(np.sum(labels == i)) for i, c in enumerate(CLASSES)})

# ---- 3. features and a leakage-free split --------------------------------------------------------
cols = np.arange(90, 271, 4)                             # front 180 degrees, every 4th beam -> 46 features
X = scans[:, cols] / MAX_R
# consecutive scans are almost identical, so split in blocks of 40 scans (20 s) to avoid leakage
block = np.arange(len(X)) // 40
ub = rng.permutation(np.unique(block))
n_tr, n_va = int(0.70 * len(ub)), int(0.15 * len(ub))
which = np.empty(len(X), dtype=int)
for b_i, b in enumerate(ub):
    which[block == b] = 0 if b_i < n_tr else (1 if b_i < n_tr + n_va else 2)
Xtr, ytr = X[which == 0], labels[which == 0]
Xva, yva = X[which == 1], labels[which == 1]
Xte, yte = X[which == 2], labels[which == 2]
fr_te = scans[which == 2][:, FRONT].min(1)      # distance to the nearest object straight ahead
mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
Xtr, Xva, Xte = (Xtr - mu) / sd, (Xva - mu) / sd, (Xte - mu) / sd
print(f"split train/val/test = {len(Xtr)}/{len(Xva)}/{len(Xte)}")

# ---- 4. the network -------------------------------------------------------------------------------------
sizes = [X.shape[1], 32, 16, 3]
P = {}
for i in range(3):
    P[f"W{i}"] = rng.normal(0, np.sqrt(2 / sizes[i]), (sizes[i], sizes[i + 1]))   # He initialisation
    P[f"b{i}"] = np.zeros(sizes[i + 1])


def forward(X_, P_):
    a0 = X_
    a1 = np.maximum(0, a0 @ P_["W0"] + P_["b0"])
    a2 = np.maximum(0, a1 @ P_["W1"] + P_["b1"])
    z = a2 @ P_["W2"] + P_["b2"]
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return a0, a1, a2, e / e.sum(axis=1, keepdims=True)


counts = np.bincount(ytr, minlength=3).astype(float)
cw = counts.sum() / (3 * np.maximum(counts, 1))          # class weights: rare classes count more


def loss_acc(X_, y_):
    p = forward(X_, P)[3]
    l = -np.mean(cw[y_] * np.log(p[np.arange(len(y_)), y_] + 1e-12))
    return l, np.mean(p.argmax(1) == y_)


m = {k: np.zeros_like(v) for k, v in P.items()}
v = {k: np.zeros_like(x) for k, x in P.items()}
LR, B1, B2, L2, BATCH, EPOCHS, PATIENCE = 3e-3, 0.9, 0.999, 1e-4, 64, 400, 40
hist = {"tr_loss": [], "va_loss": [], "tr_acc": [], "va_acc": []}
best_va, best_P, wait, t = np.inf, None, 0, 0
for ep in range(EPOCHS):
    order = rng.permutation(len(Xtr))
    for s in range(0, len(Xtr), BATCH):
        idx = order[s:s + BATCH]
        xb, yb = Xtr[idx], ytr[idx]
        a0, a1, a2, p = forward(xb, P)
        dz = p.copy()
        dz[np.arange(len(yb)), yb] -= 1
        dz *= (cw[yb] / len(yb))[:, None]
        g = {}
        g["W2"], g["b2"] = a2.T @ dz + L2 * P["W2"], dz.sum(0)
        d2 = (dz @ P["W2"].T) * (a2 > 0)
        g["W1"], g["b1"] = a1.T @ d2 + L2 * P["W1"], d2.sum(0)
        d1 = (d2 @ P["W1"].T) * (a1 > 0)
        g["W0"], g["b0"] = a0.T @ d1 + L2 * P["W0"], d1.sum(0)
        t += 1
        for k in P:                                       # Adam update
            m[k] = B1 * m[k] + (1 - B1) * g[k]
            v[k] = B2 * v[k] + (1 - B2) * g[k] ** 2
            P[k] -= LR * (m[k] / (1 - B1 ** t)) / (np.sqrt(v[k] / (1 - B2 ** t)) + 1e-8)
    tl, ta = loss_acc(Xtr, ytr)
    vl, va = loss_acc(Xva, yva)
    for k_, val in zip(hist, (tl, vl, ta, va)):
        hist[k_].append(val)
    if vl < best_va - 1e-5:
        best_va, best_P, wait = vl, {k: x.copy() for k, x in P.items()}, 0
    else:
        wait += 1
        if wait >= PATIENCE:
            print(f"early stopping at epoch {ep + 1} (best validation loss {best_va:.4f})")
            break
P = best_P

# ---- 5. evaluation on the held-out test blocks -----------------------------------------------------
pred = forward(Xte, P)[3].argmax(1)
cm = np.zeros((3, 3), int)
for a, b in zip(yte, pred):
    cm[a, b] += 1
acc = np.trace(cm) / cm.sum()
maj = np.mean(yte == np.bincount(ytr).argmax())
print(f"\nTest accuracy: {acc * 100:.1f} %   (majority-class baseline: {maj * 100:.1f} %)")
print("\nConfusion matrix (rows = true, columns = predicted):")
print(f"{'':8s}" + "".join(f"{c:>8s}" for c in CLASSES))
for i, c in enumerate(CLASSES):
    print(f"{c:8s}" + "".join(f"{cm[i, j]:8d}" for j in range(3)))
print(f"\n{'class':8s} {'precision':>10} {'recall':>8} {'F1':>8} {'support':>8}")
for i, c in enumerate(CLASSES):
    pr = cm[i, i] / max(1, cm[:, i].sum())
    rc = cm[i, i] / max(1, cm[i].sum())
    print(f"{c:8s} {pr:10.3f} {rc:8.3f} {2 * pr * rc / max(1e-9, pr + rc):8.3f} {cm[i].sum():8d}")

print("\nTest accuracy by distance to the nearest object straight ahead:")
for lo, hi in ((0, 0.5), (0.5, 0.8), (0.8, 1.2), (1.2, 2.0), (2.0, 3.6)):
    sel = (fr_te >= lo) & (fr_te < hi)
    if sel.sum():
        print(f"  {lo:.1f}-{hi:.1f} m: {np.mean(pred[sel] == yte[sel]) * 100:5.1f} %  ({int(sel.sum())} scans)")

np.savez("data/nn_weights.npz", mu=mu, sd=sd, **P)
fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
cmn = cm / np.maximum(1, cm.sum(1, keepdims=True))
ax[0].imshow(cmn, cmap="Blues", vmin=0, vmax=1)
for i in range(3):
    for j in range(3):
        ax[0].text(j, i, f"{cm[i, j]}\n({cmn[i, j] * 100:.0f}%)", ha="center", va="center",
                   color="white" if cmn[i, j] > 0.5 else "black")
ax[0].set_xticks(range(3)); ax[0].set_xticklabels(CLASSES)
ax[0].set_yticks(range(3)); ax[0].set_yticklabels(CLASSES)
ax[0].set_xlabel("predicted"); ax[0].set_ylabel("true"); ax[0].set_title(f"Confusion matrix (test, acc {acc * 100:.1f} %)")
ax[1].plot(hist["tr_loss"], label="train"); ax[1].plot(hist["va_loss"], label="validation")
ax[1].set_xlabel("epoch"); ax[1].set_ylabel("weighted cross-entropy"); ax[1].set_title("Loss"); ax[1].grid(True); ax[1].legend()
ax[2].plot(hist["tr_acc"], label="train"); ax[2].plot(hist["va_acc"], label="validation")
ax[2].set_xlabel("epoch"); ax[2].set_ylabel("accuracy"); ax[2].set_title("Accuracy"); ax[2].grid(True); ax[2].legend()
plt.tight_layout()
plt.savefig("report/nn_results.png", dpi=150)
plt.show()
