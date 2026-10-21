"""planning.py - Dijkstra vs A* on the occupancy grid (8-connected, inflated obstacles)."""
import heapq
import math
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import binary_dilation

ARENA_MIN = -2.0          # m, world coordinate of grid corner
RES = 0.05                # m per cell
ROBOT_RADIUS = 0.15       # m, obstacle inflation (Burger radius ~0.105 m plus margin)
SQ2 = math.sqrt(2)

# ---- 1. Occupancy map -> blocked cells --------------------------------------
logodds = np.load("data/occupancy_map_final.npy")
occupied = logodds > 0.0                       # P(occupied) > 0.5 (unknown cells stay free)
n = occupied.shape[0]
k = int(math.ceil(ROBOT_RADIUS / RES))         # inflation radius in cells
yy, xx = np.ogrid[-k:k + 1, -k:k + 1]
disk = xx ** 2 + yy ** 2 <= k ** 2
blocked = binary_dilation(occupied, structure=disk)
blocked[:k, :] = blocked[-k:, :] = True        # keep the robot inside the arena
blocked[:, :k] = blocked[:, -k:] = True
extent = [ARENA_MIN, ARENA_MIN + n * RES, ARENA_MIN, ARENA_MIN + n * RES]


def world_to_cell(x, y):
    return int((y - ARENA_MIN) / RES), int((x - ARENA_MIN) / RES)    # (row, col)


def cell_to_world(r, c):
    return ARENA_MIN + (c + 0.5) * RES, ARENA_MIN + (r + 0.5) * RES


# ---- 2. Search (use_heuristic=False -> Dijkstra, True -> A*) ----------------
MOVES = [(-1, 0, 1), (1, 0, 1), (0, -1, 1), (0, 1, 1),
         (-1, -1, SQ2), (-1, 1, SQ2), (1, -1, SQ2), (1, 1, SQ2)]


def search(start, goal, use_heuristic):
    def h(cell):
        if not use_heuristic:
            return 0.0
        dr, dc = abs(cell[0] - goal[0]), abs(cell[1] - goal[1])
        return (dr + dc) + (SQ2 - 2) * min(dr, dc)        # octile distance (admissible)

    g = {start: 0.0}
    parent = {}
    heap = [(h(start), start)]
    closed = set()
    found = False
    while heap:
        _, cur = heapq.heappop(heap)
        if cur in closed:
            continue
        closed.add(cur)
        if cur == goal:
            found = True
            break
        r, c = cur
        for dr, dc, cost in MOVES:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < n and 0 <= nc < n) or blocked[nr, nc]:
                continue
            if dr and dc and (blocked[r + dr, c] or blocked[r, c + dc]):
                continue                                    # no cutting corners
            ng = g[cur] + cost
            if ng < g.get((nr, nc), float("inf")):
                g[(nr, nc)] = ng
                parent[(nr, nc)] = cur
                heapq.heappush(heap, (ng + h((nr, nc)), (nr, nc)))
    if not found:
        return None
    path = [goal]
    while path[-1] != start:
        path.append(parent[path[-1]])
    path.reverse()
    return path, len(closed), g[goal] * RES, closed


# ---- 3. Scenarios -----------------------------------------------------------
scenarios = [("S1 SW -> NE", (-1.5, -1.5), (1.5, 1.5)),
             ("S2 NW -> SE", (-1.5, 1.5), (1.5, -1.5)),
             ("S3 W -> E",   (-1.5, 0.0), (1.5, 0.0))]

print(f"{'scenario':14s} {'planner':9s} {'length (m)':>10s} {'expanded':>9s} {'time (ms)':>10s}")
results = {}
for name, s_xy, g_xy in scenarios:
    s, g_ = world_to_cell(*s_xy), world_to_cell(*g_xy)
    if blocked[s] or blocked[g_]:
        print(f"{name}: start or goal is blocked - skipped")
        continue
    for planner, use_h in (("Dijkstra", False), ("A*", True)):
        t0 = time.perf_counter()
        res = search(s, g_, use_h)
        ms = (time.perf_counter() - t0) * 1000
        if res is None:
            print(f"{name:14s} {planner:9s} no path")
            continue
        path, expanded, length, closed = res
        results[(name, planner)] = res
        print(f"{name:14s} {planner:9s} {length:10.2f} {expanded:9d} {ms:10.1f}")

# ---- 4. Plot scenario 1 and save the A* path for the controller -------------
key = scenarios[0][0]
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
for ax, planner in zip(axes, ("Dijkstra", "A*")):
    if (key, planner) not in results:
        continue
    path, expanded, length, closed = results[(key, planner)]
    img = np.ones((n, n, 3))
    img[blocked] = (0.2, 0.2, 0.2)
    for r, c in closed:
        img[r, c] = (0.75, 0.85, 1.0)
    ax.imshow(img, origin="lower", extent=extent)
    px, py = zip(*[cell_to_world(r, c) for r, c in path])
    ax.plot(px, py, "r-", linewidth=2)
    ax.plot(px[0], py[0], "go", markersize=9, label="start")
    ax.plot(px[-1], py[-1], "b*", markersize=14, label="goal")
    ax.set_title(f"{planner}: {length:.2f} m, {expanded} cells expanded")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.legend(loc="lower right")
plt.suptitle(f"{key}: explored cells (light blue), inflated obstacles (dark), path (red)")
plt.tight_layout()
plt.savefig("report/planning_comparison.png", dpi=150)

if (key, "A*") in results:
    pts = [cell_to_world(r, c) for r, c in results[(key, "A*")][0]]
    np.save("data/planned_path_astar.npy", np.array(pts))
    print("Saved data/planned_path_astar.npy (A* path, world coordinates)")
plt.show()
