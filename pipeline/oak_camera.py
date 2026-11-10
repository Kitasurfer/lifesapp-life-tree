"""One fixed camera for every age — numpy only (Blender and the 2D previews both use it).

Straight verticals (no tilt): a horizontal camera at eye height with lens shift, framed on the
FULL-age tree, so frames line up pixel for pixel and the stem stands on the centre line.
"""
from dataclasses import dataclass

import numpy as np

LENS_MM, SENSOR_MM = 50.0, 36.0
TOP, BOTTOM, SIDE = 0.05, 0.10, 0.06   # margins as shares of the frame; BOTTOM is the ground line
EYE_M = 1.6


@dataclass(frozen=True)
class Framing:
    frame_h: float       # metres of the y = 0 plane covered by the frame height
    distance: float      # camera stands at (x_c, -distance, EYE_M) and looks along +y
    x_c: float

    @property
    def shift_y(self) -> float:
        return ((0.5 - BOTTOM) * self.frame_h - EYE_M) / self.frame_h


def solve(full_points: np.ndarray, width: int, height: int) -> Framing:
    """Framing is solved in projection, so crown parts nearer the camera (which look bigger) still fit."""
    aspect = width / height
    pts = full_points[np.random.default_rng(0).choice(len(full_points), min(len(full_points), 200_000), replace=False)]
    x_c = 0.0  # the stem stands on the centre line of every frame
    distance = 40.0
    for _ in range(6):
        k = distance / (distance + pts[:, 1])           # perspective scale towards the y=0 plane
        x = x_c + (pts[:, 0] - x_c) * k
        z = EYE_M + (pts[:, 2] - EYE_M) * k
        xmin, xmax = np.percentile(x, [0.02, 99.98])
        zmax = float(np.percentile(z, 99.98))
        frame_h = max(zmax / (1 - TOP - BOTTOM), 2 * max(abs(xmin), abs(xmax)) / (1 - 2 * SIDE) / aspect)
        distance = frame_h / (SENSOR_MM / LENS_MM)
    return Framing(float(frame_h), float(distance), x_c)


def project(points: np.ndarray, framing: Framing, width: int, height: int):
    """World points → (pixel x, pixel y, distance along the view axis)."""
    depth = framing.distance + points[:, 1]
    k = framing.distance / depth
    x = (points[:, 0] - framing.x_c) * k
    z = EYE_M + (points[:, 2] - EYE_M) * k
    px = width / 2 + x / framing.frame_h * height
    py = height * (1 - BOTTOM) - z / framing.frame_h * height
    return px, py, depth
