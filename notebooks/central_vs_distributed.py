"""central_vs_distributed.py - centralised vs distributed coordination for the multi-robot team.

Task: all N robots must agree on a common point (the average of their positions), as in the Webots
rendezvous experiment. Three architectures are compared in a round-based simulation (one round =
one communication period, 0.2 s in Webots):
  centralised          : star. Round 1, everyone sends its position to the leader; round 2, the leader
                         broadcasts the average. 2 rounds, 2(N-1) messages, but the leader is a
                         bottleneck and a single point of failure.
  distributed, ring    : average consensus, every robot talks only to its 2 ring neighbours.
  distributed, complete: average consensus with all-to-all links (step size 1/N).
Experiments: (1) scaling with the team size N, (2) failure of one robot after round 3.
"""
import numpy as np
import matplotlib.pyplot as plt

TX_PERIOD = 0.2          # s per communication round (as in consensus_controller.py)
TOL = 0.01               # converged when every robot is within 1 cm of the target
rng = np.random.default_rng(0)


def adjacency(kind, n):
    A = np.zeros((n, n))
    if kind == "ring":
        for i in range(n):
            A[i, (i + 1) % n] = A[(i + 1) % n, i] = 1
    elif kind == "complete":
        A = np.ones((n, n)) - np.eye(n)
    return A


def run_consensus(kind, z0, max_rounds=60000, fail=None, fail_round=3, record=False):
    """Average consensus z <- z + eps*(A z - deg z) with eps = 1/(d_max+1). Optionally robot `fail` dies."""
    n = len(z0)
    A = adjacency(kind, n)
    eps = 1.0 / (A.sum(1).max() + 1)
    alive = np.ones(n, bool)
    z = z0.copy()
    target = z0.mean(0)
    errs = []
    for k in range(1, max_rounds + 1):
        if fail is not None and k == fail_round + 1:
            alive[fail] = False
            target_surv_true = z0[alive].mean(0)            # true mean of the survivors' initial positions
        Aa = A * alive[None, :] * alive[:, None]            # links to a dead robot are gone
        z = np.where(alive[:, None], z + eps * (Aa @ z - Aa.sum(1, keepdims=True) * z), z)
        err = np.max(np.linalg.norm(z[alive] - z[alive].mean(0), axis=1))   # disagreement among survivors
        errs.append(err)
        if fail is None and np.max(np.linalg.norm(z - target, axis=1)) < TOL:
            return k, errs, z
        if fail is not None and k > fail_round + 3000:
            break
    if fail is not None:
        return k, errs, (z[alive].mean(0), target_surv_true)
    return k, errs, z


# ---------------- Experiment 1: scaling ------------------------------------------------------------
sizes = [4, 8, 16, 32, 64]
rows = []
for n in sizes:
    z0 = rng.uniform(-1.5, 1.5, (n, 2))
    r_ring, _, _ = run_consensus("ring", z0)
    r_comp, _, _ = run_consensus("complete", z0)
    rows.append(dict(n=n,
                     c_rounds=2, c_msgs=2 * (n - 1), c_load=n - 1,
                     r_rounds=r_ring, r_msgs=r_ring * 2 * n, r_load=2,
                     k_rounds=r_comp, k_msgs=r_comp * n * (n - 1), k_load=n - 1))
print(f"{'N':>3} | {'CENTRALISED':^28} | {'DISTRIBUTED ring':^32} | {'DISTRIBUTED complete':^28}")
print(f"{'':>3} | {'rounds':>6} {'time s':>7} {'msgs':>7} {'leader rx':>9} | {'rounds':>6} {'time s':>8} {'msgs':>9} {'max rx':>6} | "
      f"{'rounds':>6} {'time s':>7} {'msgs':>8}")
for r in rows:
    print(f"{r['n']:3d} | {r['c_rounds']:6d} {r['c_rounds'] * TX_PERIOD:7.1f} {r['c_msgs']:7d} {r['c_load']:9d} | "
          f"{r['r_rounds']:6d} {r['r_rounds'] * TX_PERIOD:8.1f} {r['r_msgs']:9d} {r['r_load']:6d} | "
          f"{r['k_rounds']:6d} {r['k_rounds'] * TX_PERIOD:7.1f} {r['k_msgs']:8d}")

# ---------------- Experiment 2: one robot fails after round 3 (N = 8) -----------------------------
n = 8
z0 = rng.uniform(-1.5, 1.5, (n, 2))
leader = 0
disagree0 = np.max(np.linalg.norm(z0 - z0.mean(0), axis=1))
print(f"\nFailure experiment, N = {n}, one robot dies after round 3 (initial disagreement {disagree0:.2f} m)")

# centralised, leader dies: followers never receive the average -> disagreement stays
c_leader_final = disagree0
# centralised, a follower dies: leader times out and averages the survivors -> agreement, exact survivor mean
c_follower_final, c_follower_bias = 0.0, 0.0
# distributed ring, one robot dies (a non-adjacent robot is picked): ring becomes a line, still connected
_, errs_ring, (zbar, ztrue) = run_consensus("ring", z0, fail=3)
d_final = errs_ring[-1]
d_bias = float(np.linalg.norm(zbar - ztrue))
print(f"{'architecture / failure':42s} {'final disagreement (m)':>24} {'error of agreed point (m)':>27}")
print(f"{'centralised, LEADER fails':42s} {c_leader_final:24.3f} {'no agreement':>27}")
print(f"{'centralised, a follower fails':42s} {c_follower_final:24.3f} {c_follower_bias:27.3f}")
print(f"{'distributed ring, robot 3 fails':42s} {d_final:24.5f} {d_bias:27.3f}")

# ---------------- plots -------------------------------------------------------------------------------
N_ = np.array([r["n"] for r in rows])
fig, ax = plt.subplots(2, 2, figsize=(12, 9))
ax[0, 0].loglog(N_, [r["c_rounds"] * TX_PERIOD for r in rows], "o-", label="centralised (star)")
ax[0, 0].loglog(N_, [r["r_rounds"] * TX_PERIOD for r in rows], "s-", label="distributed, ring")
ax[0, 0].loglog(N_, [r["k_rounds"] * TX_PERIOD for r in rows], "^-", label="distributed, complete")
ax[0, 0].set_title("Time to agree within 1 cm"); ax[0, 0].set_xlabel("robots N"); ax[0, 0].set_ylabel("seconds")
ax[0, 1].loglog(N_, [r["c_msgs"] for r in rows], "o-", label="centralised")
ax[0, 1].loglog(N_, [r["r_msgs"] for r in rows], "s-", label="distributed, ring")
ax[0, 1].loglog(N_, [r["k_msgs"] for r in rows], "^-", label="distributed, complete")
ax[0, 1].set_title("Total messages sent until agreement"); ax[0, 1].set_xlabel("robots N"); ax[0, 1].set_ylabel("messages")
ax[1, 0].loglog(N_, [r["c_load"] for r in rows], "o-", label="centralised: leader")
ax[1, 0].loglog(N_, [r["r_load"] for r in rows], "s-", label="ring: every robot")
ax[1, 0].set_title("Messages the busiest robot must receive per round"); ax[1, 0].set_xlabel("robots N"); ax[1, 0].set_ylabel("messages / round")
for a in (ax[0, 0], ax[0, 1], ax[1, 0]):
    a.grid(True, which="both"); a.legend()
rounds_axis = np.arange(1, 41)
ax[1, 1].semilogy(rounds_axis, np.maximum(errs_ring[:40], 1e-6), label="distributed ring (robot 3 dies after round 3)")
ax[1, 1].semilogy(rounds_axis, [disagree0 if k > 2 else disagree0 for k in rounds_axis], "r--", label="centralised, leader dies (model: stuck)")
ax[1, 1].semilogy(rounds_axis, [disagree0 if k < 2 else 1e-6 for k in rounds_axis], "g-.", label="centralised, follower dies (model: agrees after round 2)")
ax[1, 1].set_title(f"Disagreement after a robot failure (N={n})"); ax[1, 1].set_xlabel("round"); ax[1, 1].set_ylabel("max distance from the others (m)")
ax[1, 1].grid(True, which="both"); ax[1, 1].legend(fontsize=8)
plt.tight_layout()
plt.savefig("report/central_vs_distributed.png", dpi=150)
plt.show()
