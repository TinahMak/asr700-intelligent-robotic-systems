"""occupancy_grid.py - log-odds occupancy grid built from LiDAR scans."""
import math
import numpy as np

ARENA_MIN = -2.0      # m, world coordinate of the grid corner (x and y)
ARENA_SIZE = 4.0      # m
RES = 0.05            # m per cell
L_OCC = 0.85          # log-odds added to a cell where a beam ends
L_FREE = -0.40        # log-odds added to cells a beam passes through
L_CLAMP = 5.0


class OccupancyGrid:
    def __init__(self):
        self.n = int(round(ARENA_SIZE / RES))        # cells per side (80)
        self.logodds = np.zeros((self.n, self.n))    # row = y index, col = x index

    def world_to_cell(self, x, y):
        """World (x, y) in metres -> (row, col), or None if outside the arena."""
        col = int(math.floor((x - ARENA_MIN) / RES))
        row = int(math.floor((y - ARENA_MIN) / RES))
        if 0 <= row < self.n and 0 <= col < self.n:
            return row, col
        return None

    def update(self, pose, ranges, max_range):
        """pose = (x, y, theta). ranges = list of 360 LiDAR distances."""
        px, py, ptheta = pose
        n_beams = len(ranges)
        for i, r in enumerate(ranges):
            if not math.isfinite(r):
                continue                              # no hit: skip this beam
            # Beam angle in the world frame (beam 0 points backwards, index runs clockwise).
            beam_angle = ptheta + (math.pi - i * 2 * math.pi / n_beams)
            c, s = math.cos(beam_angle), math.sin(beam_angle)

            # Free cells: sample along the beam in half-cell steps, then keep each cell once.
            free = set()
            t = 0.0
            while t < r:
                cell = self.world_to_cell(px + t * c, py + t * s)
                if cell is not None:
                    free.add(cell)
                t += RES / 2

            # Occupied cell: where the beam ended (only if it hit something real).
            hit = None
            if r < max_range:
                hit = self.world_to_cell(px + r * c, py + r * s)
                free.discard(hit)                     # the hit cell is occupied, not free

            for cell in free:
                self.logodds[cell] += L_FREE
            if hit is not None:
                self.logodds[hit] += L_OCC

        np.clip(self.logodds, -L_CLAMP, L_CLAMP, out=self.logodds)

    def probability(self):
        """Convert log-odds to probability of occupied (0..1)."""
        return 1.0 - 1.0 / (1.0 + np.exp(self.logodds))
