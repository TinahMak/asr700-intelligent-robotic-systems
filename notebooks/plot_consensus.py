"""plot_consensus.py - trajectories, consensus states and disagreement for the 4-robot ring."""
import numpy as np
import matplotlib.pyplot as plt

logs = [np.load(f"data/consensus_robot{i}.npy") for i in range(4)]
n = min(len(l) for l in logs)
logs = [l[:n] for l in logs]
t = logs[0][:, 0]
Z = np.stack([l[:, 3:5] for l in logs])                  # (robot, time, xy)
zbar = Z.mean(axis=0)                                    # the average is invariant
disagree = np.max(np.linalg.norm(Z - zbar, axis=2), axis=0)

# theory: ring Laplacian, EPS = 0.3
Lap = np.array([[2, -1, 0, -1], [-1, 2, -1, 0], [0, -1, 2, -1], [-1, 0, -1, 2]], float)
lam = np.sort(np.linalg.eigvalsh(Lap))
rate = max(abs(1 - 0.3 * lam[1]), abs(1 - 0.3 * lam[-1]))
print(f"Laplacian eigenvalues {np.round(lam, 2)}, lambda2={lam[1]:.2f}, theoretical rate/round={rate:.2f}")
hit = np.where(disagree < 0.01)[0]
print(f"consensus average z: ({zbar[-1][0]:.3f}, {zbar[-1][1]:.3f}) (expected (0, 0))")
print(f"time until disagreement < 1 cm: {t[hit[0]]:.1f} s" if len(hit) else "did not reach 1 cm")
print(f"total messages sent by all robots: {int(sum(l[-1, 5] for l in logs))}")
for i, l in enumerate(logs):
    print(f"robot{i}: start ({l[0,1]:.2f},{l[0,2]:.2f}) -> final ({l[-1,1]:.2f},{l[-1,2]:.2f})")

fig, ax = plt.subplots(1, 3, figsize=(16, 5))
for i, l in enumerate(logs):
    ax[0].plot(l[:, 1], l[:, 2], label=f"robot{i}")
    ax[0].plot(l[0, 1], l[0, 2], "go")
    ax[0].plot(l[-1, 1], l[-1, 2], "rs")
    ax[1].plot(t, l[:, 3], label=f"z{i} x")
    ax[1].plot(t, l[:, 4], "--", label=f"z{i} y")
ax[0].set_aspect("equal"); ax[0].grid(True); ax[0].legend(fontsize=8)
ax[0].set_title("Robot paths (green = start, red = end)")
ax[1].set_title("Consensus states converge to the average"); ax[1].set_xlabel("time (s)"); ax[1].grid(True)
ax[1].legend(fontsize=6, ncol=2)
ax[2].semilogy(t, np.maximum(disagree, 1e-6)); ax[2].grid(True, which="both")
ax[2].set_title("Disagreement max|z_i - mean| (log scale)"); ax[2].set_xlabel("time (s)")
plt.tight_layout()
plt.savefig("report/consensus.png", dpi=150)
plt.show()
