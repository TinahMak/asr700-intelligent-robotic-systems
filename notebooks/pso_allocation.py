"""pso_allocation.py - Particle Swarm Optimisation for multi-robot task allocation.

4 robots (starting on the rendezvous ring) must visit 8 tasks. A particle encodes, for every
task, a continuous number in [0, 4); floor() gives the robot index. Fitness = makespan (longest
robot route) + 0.1 * total length. Compared with brute force (optimum), greedy and random search.

Two PSO variants are compared:
  basic    : global-best topology, constant inertia 0.72, c1=c2=1.5, clamping at the borders
  improved : ring-neighbourhood best (each particle only sees 2 neighbours, which keeps the swarm
             diverse for longer), inertia decaying 0.9 -> 0.4, c1=c2=2, reflecting borders, 60 particles
"""
import itertools
import json
import time
import numpy as np
import matplotlib.pyplot as plt

ROBOTS = np.array([(-0.25, -0.25), (0.25, -0.25), (0.25, 0.25), (-0.25, 0.25)])
TASKS = np.array([(-1.4, -1.2), (-1.2, 1.3), (1.4, 1.2), (1.3, -1.3),
                  (0.0, 1.5), (0.0, -1.5), (-1.5, 0.1), (1.5, 0.1)])
NR, NT = len(ROBOTS), len(TASKS)
D_RT = np.linalg.norm(ROBOTS[:, None, :] - TASKS[None, :, :], axis=2)     # robot -> task
D_TT = np.linalg.norm(TASKS[:, None, :] - TASKS[None, :, :], axis=2)      # task -> task


def route(r, idxs):
    """Nearest-neighbour visiting order of the tasks idxs for robot r. Returns (length, order)."""
    left, order, length, cur = list(idxs), [], 0.0, None
    while left:
        if cur is None:
            k = min(left, key=lambda j: D_RT[r][j])
            length += D_RT[r][k]
        else:
            k = min(left, key=lambda j: D_TT[cur][j])
            length += D_TT[cur][k]
        order.append(k)
        left.remove(k)
        cur = k
    return length, order


def cost(assign):
    lengths = [route(r, [j for j in range(NT) if assign[j] == r])[0] for r in range(NR)]
    return max(lengths) + 0.1 * sum(lengths)


def decode(p):
    return np.minimum(np.floor(p).astype(int), NR - 1)


def pso(seed, n_particles, iters, w0, w1, c1, c2, ring, reflect, vmax=1.5):
    rng = np.random.default_rng(seed)
    pos = rng.uniform(0, NR, (n_particles, NT))
    vel = rng.uniform(-1, 1, (n_particles, NT))
    pbest, pbest_f = pos.copy(), np.array([cost(decode(p)) for p in pos])
    hist = [pbest_f.min()]
    for it in range(iters):
        w = w0 + (w1 - w0) * it / max(1, iters - 1)               # inertia schedule
        if ring:                                                  # best of itself + 2 ring neighbours
            guide = np.empty_like(pos)
            for i in range(n_particles):
                nb = [(i - 1) % n_particles, i, (i + 1) % n_particles]
                guide[i] = pbest[nb[int(np.argmin(pbest_f[nb]))]]
        else:                                                     # global best
            guide = pbest[pbest_f.argmin()]
        r1, r2 = rng.random(pos.shape), rng.random(pos.shape)
        vel = np.clip(w * vel + c1 * r1 * (pbest - pos) + c2 * r2 * (guide - pos), -vmax, vmax)
        pos = pos + vel
        if reflect:                                               # bounce off the borders
            lo, hi = pos < 0, pos >= NR
            pos[lo] = -pos[lo]
            pos[hi] = 2 * NR - pos[hi] - 1e-9
            vel[lo | hi] *= -1
        pos = np.clip(pos, 0, NR - 1e-9)
        f = np.array([cost(decode(p)) for p in pos])
        better = f < pbest_f
        pbest[better], pbest_f[better] = pos[better], f[better]
        hist.append(pbest_f.min())
    i = int(pbest_f.argmin())
    return decode(pbest[i]), pbest_f[i], np.array(hist)


def random_search(seed, budget, every):
    rng = np.random.default_rng(seed)
    best, hist, best_a = np.inf, [], None
    for i in range(budget):
        a = rng.integers(0, NR, NT)
        c = cost(a)
        if c < best:
            best, best_a = c, a
        if (i + 1) % every == 0:
            hist.append(best)
    return best_a, best, np.array(hist)


# ---- optimum by brute force ----------------------------------------------------------------
t0 = time.time()
opt_f, opt_a = np.inf, None
for a in itertools.product(range(NR), repeat=NT):
    c = cost(a)
    if c < opt_f:
        opt_f, opt_a = c, np.array(a)
print(f"Brute force over {NR ** NT} assignments took {time.time() - t0:.1f} s: optimum cost {opt_f:.3f}")

greedy_a = D_RT.argmin(axis=0)                  # every task goes to the robot that starts nearest to it
greedy_f = cost(greedy_a)

RUNS, ITERS = 20, 60
BASIC = dict(n_particles=30, iters=ITERS, w0=0.72, w1=0.72, c1=1.5, c2=1.5, ring=False, reflect=False)
IMPROVED = dict(n_particles=60, iters=ITERS, w0=0.9, w1=0.4, c1=2.0, c2=2.0, ring=True, reflect=True)
basic_res = [pso(s, **BASIC) for s in range(RUNS)]
impr_res = [pso(s, **IMPROVED) for s in range(RUNS)]
budget_basic = BASIC["n_particles"] * (ITERS + 1)
budget_impr = IMPROVED["n_particles"] * (ITERS + 1)
rs_res = [random_search(100 + s, budget_impr, IMPROVED["n_particles"]) for s in range(RUNS)]
rs_basic_f = np.array([random_search(200 + s, budget_basic, BASIC["n_particles"])[1] for s in range(RUNS)])
f_of = lambda res: np.array([r[1] for r in res])
basic_f, impr_f, rs_f = f_of(basic_res), f_of(impr_res), f_of(rs_res)
ok = lambda f: int(np.sum(f <= opt_f + 1e-9))

print(f"\n{'method':34s} {'evals':>6} {'mean cost':>10} {'best':>7} {'worst':>7} {'hit optimum':>12}")
print(f"{'brute force (optimum)':34s} {NR ** NT:6d} {opt_f:10.3f}")
print(f"{'greedy nearest robot':34s} {NR:6d} {greedy_f:10.3f}")
print(f"{'random search (basic budget)':34s} {budget_basic:6d} {rs_basic_f.mean():10.3f} {rs_basic_f.min():7.3f} {rs_basic_f.max():7.3f} {ok(rs_basic_f):9d}/{RUNS}")
print(f"{'PSO basic (global best)':34s} {budget_basic:6d} {basic_f.mean():10.3f} {basic_f.min():7.3f} {basic_f.max():7.3f} {ok(basic_f):9d}/{RUNS}")
print(f"{'random search (improved budget)':34s} {budget_impr:6d} {rs_f.mean():10.3f} {rs_f.min():7.3f} {rs_f.max():7.3f} {ok(rs_f):9d}/{RUNS}")
print(f"{'PSO improved (ring, decay)':34s} {budget_impr:6d} {impr_f.mean():10.3f} {impr_f.min():7.3f} {impr_f.max():7.3f} {ok(impr_f):9d}/{RUNS}")

# plan for the robots: the best assignment found by the improved PSO
best_i = int(impr_f.argmin())
best_a = impr_res[best_i][0]
print("\nBest improved-PSO assignment (task -> robot):", best_a.tolist(), f"cost {impr_f[best_i]:.3f}")
routes, plan = {}, {}
for r in range(NR):
    length, order = route(r, [j for j in range(NT) if best_a[j] == r])
    routes[r] = order
    plan[f"robot{r}"] = {"tasks": order, "waypoints": TASKS[order].tolist(), "length": round(length, 3)}
    print(f"  robot{r}: tasks {order}, route length {length:.2f} m")
json.dump({"assignment": best_a.tolist(), "plan": plan}, open("data/task_assignment.json", "w"), indent=1)

# ---- plots ----------------------------------------------------------------------------------
fig, ax = plt.subplots(1, 2, figsize=(14, 6))
cols = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
for r in range(NR):
    pts = np.array([ROBOTS[r]] + [TASKS[j] for j in routes[r]])
    ax[0].plot(pts[:, 0], pts[:, 1], "-o", color=cols[r], label=f"robot{r} ({plan[f'robot{r}']['length']} m)")
    ax[0].plot(*ROBOTS[r], "s", color=cols[r], markersize=11)
for j in range(NT):
    ax[0].annotate(f"T{j}", TASKS[j] + 0.05, fontsize=10)
ax[0].set_xlim(-2, 2); ax[0].set_ylim(-2, 2); ax[0].set_aspect("equal"); ax[0].grid(True)
ax[0].legend(); ax[0].set_title("PSO task allocation (squares = robot start, circles = tasks)")

for res, bud, n, colr, lab in ((basic_res, budget_basic, BASIC["n_particles"], "tab:blue", "PSO basic"),
                               (impr_res, budget_impr, IMPROVED["n_particles"], "tab:purple", "PSO improved")):
    H = np.stack([r[2] for r in res])
    ev = n * np.arange(1, ITERS + 2)
    ax[1].plot(ev, H.mean(axis=0), color=colr, label=f"{lab} (mean of {RUNS} runs)")
    ax[1].fill_between(ev, H.min(axis=0), H.max(axis=0), color=colr, alpha=0.15)
R_ = np.stack([r[2] for r in rs_res])
ev = IMPROVED["n_particles"] * np.arange(1, ITERS + 2)
ax[1].plot(ev, R_.mean(axis=0), "r-", label="random search (mean)")
ax[1].axhline(opt_f, color="k", linestyle=":", label="optimum (brute force)")
ax[1].axhline(greedy_f, color="g", linestyle="--", label="greedy")
ax[1].set_xlabel("cost-function evaluations"); ax[1].set_ylabel("best cost found")
ax[1].set_title("Convergence: PSO variants vs random search"); ax[1].grid(True); ax[1].legend()
plt.tight_layout()
plt.savefig("report/pso_allocation.png", dpi=150)
plt.show()
