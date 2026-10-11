"""Reverse-growth model for a scanned mature tree (TreeQSM cylinders) — numpy only.

The scan is the tree at its full age (age 100 on the app's human-life axis). A younger tree is
derived from the SAME skeleton, so every age shows one individual oak:

* framework    — a cylinder exists once the age's tip threshold allows it: it is kept when at
  least T(age) terminal tips lie downstream of it at full age. Young = trunk + main limbs (the
  crown outline is already the oak's own), finer twigs fill in year by year; cut ends become
  growing tips. (Trimming a veteran from the root up does NOT work — its lower trunk lost its
  young crown to self-pruning long ago, so you get a bare pole.)
* scale        — the kept framework is scaled so the height follows an oak height curve
  (Chapman–Richards): size is botanical, shape is the scanned individual.
* radius       — measured radius × scale^1.5 (stem allometry), redistributed by the pipe model
  over the tips that are alive at this age, never thicker than the scaled measurement.

No bpy here: the Blender build script and the 2D preview both import this module.
"""
from dataclasses import dataclass

import numpy as np

FULL_AGE = 100.0


@dataclass(frozen=True)
class Skeleton:
    start: np.ndarray   # (n, 3) cylinder start, metres, stem base at origin, z up
    axis: np.ndarray    # (n, 3) unit axis
    length: np.ndarray  # (n,)
    radius: np.ndarray  # (n,)
    parent: np.ndarray  # (n,) parent cylinder index, -1 for the root cylinder
    order: np.ndarray   # (n,) branch order, 0 = trunk
    branch: np.ndarray  # (n,) branch id; a cylinder continues its parent's chain when ids match


_FIELDS = ("start", "axis", "length", "radius", "parent", "order", "branch")


def load_npz(path) -> Skeleton:
    d = np.load(path)
    return Skeleton(*(d[f] for f in _FIELDS))


def save_npz(skel: Skeleton, path) -> None:
    np.savez_compressed(path, **{f: getattr(skel, f) for f in _FIELDS})


def load_qsm_mat(path) -> Skeleton:
    """TreeQSM v2.0 .mat → Skeleton (needs scipy; not available inside Blender)."""
    from scipy.io import loadmat

    m = loadmat(path, squeeze_me=True, struct_as_record=False)
    start = np.asarray(m["Sta"], dtype=np.float64)
    parent = np.asarray(m["CPar"], dtype=np.int64) - 1  # 1-based, 0 = none → -1
    root = int(np.flatnonzero(parent < 0)[0])
    boc = np.asarray(m["BoC"], dtype=np.int64)          # [branch id, branch order, index in branch]
    return Skeleton(
        start=start - start[root],
        axis=np.asarray(m["Axe"], dtype=np.float64),
        length=np.asarray(m["Len"], dtype=np.float64),
        radius=np.asarray(m["Rad"], dtype=np.float64),
        parent=parent,
        order=boc[:, 1],
        branch=boc[:, 0] - 1,
    )


def rotate_z(skel: Skeleton, degrees: float) -> Skeleton:
    """Turn the tree about its stem (choose which side faces the camera)."""
    a = np.radians(degrees)
    rot = np.array([[np.cos(a), -np.sin(a), 0.0], [np.sin(a), np.cos(a), 0.0], [0.0, 0.0, 1.0]])
    return Skeleton(skel.start @ rot.T, skel.axis @ rot.T, skel.length, skel.radius, skel.parent,
                    skel.order, skel.branch)


def topo_order(parent: np.ndarray) -> np.ndarray:
    """Cylinder indices with every parent before its children."""
    n = len(parent)
    depth = np.full(n, -1)
    depth[parent < 0] = 0
    remaining = np.flatnonzero(depth < 0)
    while remaining.size:
        ready = remaining[depth[parent[remaining]] >= 0]
        depth[ready] = depth[parent[ready]] + 1
        remaining = np.flatnonzero(depth < 0)
    return np.argsort(depth, kind="stable")


@dataclass(frozen=True)
class Precomputed:
    order_idx: np.ndarray    # topological order
    tips_full: np.ndarray    # terminal cylinders downstream at full age


def precompute(skel: Skeleton) -> Precomputed:
    idx = topo_order(skel.parent)
    return Precomputed(idx, _tips_downstream(skel.parent, idx, np.ones(len(idx), bool)))


def _tips_downstream(parent: np.ndarray, idx: np.ndarray, alive: np.ndarray) -> np.ndarray:
    has_alive_child = np.zeros(len(parent), bool)
    kids = alive & (parent >= 0)
    has_alive_child[parent[kids]] = True
    tips = (alive & ~has_alive_child).astype(np.float64)
    for i in idx[::-1]:  # children before parents
        p = parent[i]
        if p >= 0 and alive[i]:
            tips[p] += tips[i]
    return tips


def height_fraction(age: float, k: float = 0.035, p: float = 1.4) -> float:
    """Oak height relative to the height at FULL_AGE (Chapman–Richards)."""
    cr = lambda t: (1.0 - np.exp(-k * t)) ** p
    return float(cr(max(age, 0.0)) / cr(FULL_AGE))


def tip_threshold(age: float, total_tips: float, gamma: float = 0.45) -> float:
    """Minimum full-age downstream tips for a cylinder to exist at `age` (log-interpolated)."""
    f = float(np.clip(age / FULL_AGE, 0.0, 1.0)) ** gamma
    return float(np.exp(np.log(total_tips) * (1.0 - f)))


@dataclass(frozen=True)
class AgedTree:
    """Cylinders alive at one age, already scaled. `cyl` maps back to Skeleton indices."""
    cyl: np.ndarray
    start: np.ndarray
    end: np.ndarray
    radius: np.ndarray
    is_tip: np.ndarray       # cylinder ends in a living growing tip (leaves go here)
    order: np.ndarray
    scale: float


def grow(skel: Skeleton, pre: Precomputed, age: float, min_radius: float = 0.0015) -> AgedTree:
    total = float(pre.tips_full.max())
    alive = pre.tips_full >= tip_threshold(age, total) - 1e-9
    tips = _tips_downstream(skel.parent, pre.order_idx, alive)

    full_end = skel.start + skel.axis * skel.length[:, None]
    full_height = float(full_end[:, 2].max())
    kept_height = max(float(full_end[alive, 2].max()), 1e-3)
    scale = height_fraction(age) * full_height / kept_height

    share_alive = np.divide(tips, pre.tips_full, out=np.zeros_like(tips), where=pre.tips_full > 0)
    share_total = tips.max() / total
    radius = skel.radius * scale ** 1.5 * np.sqrt(share_alive / share_total)
    radius = np.minimum(radius, skel.radius * scale)

    has_alive_child = np.zeros(len(alive), bool)
    kids = alive & (skel.parent >= 0)
    has_alive_child[skel.parent[kids]] = True
    return AgedTree(
        cyl=np.flatnonzero(alive),
        start=skel.start[alive] * scale,
        end=full_end[alive] * scale,
        radius=np.maximum(radius[alive], min_radius),
        is_tip=(alive & ~has_alive_child)[alive],
        order=skel.order[alive],
        scale=scale,
    )


@dataclass(frozen=True)
class Chain:
    """One continuous branch as a polyline — what Blender turns into a single tube."""
    points: np.ndarray   # (k, 3)
    radii: np.ndarray    # (k,)
    order: int
    ends_in_tip: bool


def branch_chains(skel: Skeleton, tree: AgedTree, smooth_passes: int = 2) -> list:
    """Join the alive cylinders of each branch into one smooth polyline.

    QSM cylinders are fitted independently, so consecutive ones are slightly offset; drawing them
    as separate tubes leaves steps on the trunk. Joints are averaged, the line is smoothed, radii
    never grow towards the tip, and a side branch starts inside its parent (projected onto the
    parent axis) so it emerges from the bark instead of floating next to it.
    """
    local = {int(c): i for i, c in enumerate(tree.cyl)}
    children = {}
    for i, c in enumerate(tree.cyl):
        p = int(skel.parent[c])
        if p in local and skel.branch[p] == skel.branch[c]:
            children[local[p]] = i
    continues = set(children.values())

    chains = []
    for head in range(len(tree.cyl)):
        if head in continues:
            continue
        run = [head]
        while run[-1] in children:
            run.append(children[run[-1]])
        run = np.array(run)

        joints = (tree.end[run[:-1]] + tree.start[run[1:]]) / 2.0
        points = np.vstack([tree.start[run[:1]], joints, tree.end[run[-1:]]])
        radii = np.concatenate([tree.radius[run[:1]], tree.radius[run]])
        for _ in range(smooth_passes if len(points) > 2 else 0):
            points[1:-1] = (points[:-2] + 2.0 * points[1:-1] + points[2:]) / 4.0
        radii = np.minimum.accumulate(radii)

        p = int(skel.parent[tree.cyl[run[0]]])
        if p in local:
            a, b = tree.start[local[p]], tree.end[local[p]]
            seg = b - a
            t = np.clip(np.dot(points[0] - a, seg) / max(np.dot(seg, seg), 1e-12), 0.0, 1.0)
            points = np.vstack([a + t * seg, points])
            radii = np.concatenate([radii[:1], radii])

        chains.append(Chain(points, radii, int(tree.order[run[0]]), bool(tree.is_tip[run[-1]])))
    return chains
