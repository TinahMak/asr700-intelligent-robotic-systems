"""odometry.py - dead-reckoning pose estimate for a differential-drive robot."""
import math

WHEEL_RADIUS = 0.033      # m   (TurtleBot3 Burger)
WHEEL_SEPARATION = 0.160  # m   (distance between the two wheels)


class Odometry:
    def __init__(self, left_pos, right_pos, x=0.0, y=0.0, theta=0.0):
        self.prev_left = left_pos
        self.prev_right = right_pos
        self.x, self.y, self.theta = x, y, theta

    def update(self, left_pos, right_pos, heading=None):
        """Call once per step with the encoder angles (rad).
        heading=None: heading comes from the wheels (drifts).
        heading=value: heading comes from an absolute sensor (e.g. compass)."""
        d_left = WHEEL_RADIUS * (left_pos - self.prev_left)
        d_right = WHEEL_RADIUS * (right_pos - self.prev_right)
        d = (d_left + d_right) / 2

        if heading is None:
            d_theta = (d_right - d_left) / WHEEL_SEPARATION
        else:
            # shortest signed angle from the old heading to the measured one
            d_theta = math.atan2(math.sin(heading - self.theta),
                                 math.cos(heading - self.theta))

        self.x += d * math.cos(self.theta + d_theta / 2)
        self.y += d * math.sin(self.theta + d_theta / 2)
        self.theta = self.theta + d_theta if heading is None else heading

        self.prev_left, self.prev_right = left_pos, right_pos
        return self.x, self.y, self.theta
