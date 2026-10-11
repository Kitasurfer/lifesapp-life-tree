"""Geometry for one age of the oak — numpy only, so it is testable without Blender.

* tubes  — every branch chain becomes one tube (rings along the polyline, parallel-transport
           frames), UVs in metres of bark so the texture keeps its physical size at every age;
           the trunk gets a lobed root flare and a few surface roots.
* leaves — oak-leaf cards clustered on the scan's real twig ends (scaled to the age); leaf size is
           the real 7–14 cm at every age (only the tree scales, leaves don't).

Blender (blender/render_age.py) just turns these arrays into meshes.
"""
from dataclasses import dataclass

import numpy as np

import oak_growth as og

BARK_TILE_M = 0.6          # metres of bark covered by one bark texture tile
LEAF_LENGTH_M = 0.11
LEAVES_AT_FULL_AGE = 280_000


@dataclass(frozen=True)
class MeshData:
    verts: np.ndarray        # (v, 3)
    quads: np.ndarray        # (f, 4) vertex indices
    loop_uv: np.ndarray      # (f * 4, 2) per face corner, in quad order
    point_value: np.ndarray  # (v,) tubes: radius in metres; leaves: random 0..1 per leaf


def _resample(points: np.ndarray, radii: np.ndarray, max_seg: float):
    seg = np.linalg.norm(np.diff(points, axis=0), axis=1)
    parts = np.maximum(1, np.ceil(seg / max_seg).astype(int))
    if parts.max() == 1:
        return points, radii
    t = np.concatenate([np.arange(n) / n for n in parts])
    i = np.repeat(np.arange(len(seg)), parts)
    p = points[i] + (points[i + 1] - points[i]) * t[:, None]
    r = radii[i] + (radii[i + 1] - radii[i]) * t
    return np.vstack([p, points[-1:]]), np.concatenate([r, radii[-1:]])


def _frames(points: np.ndarray):
    tang = np.gradient(points, axis=0)
    tang /= np.maximum(np.linalg.norm(tang, axis=1, keepdims=True), 1e-9)
    ref = np.array([1.0, 0.0, 0.0]) if abs(tang[0, 0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    normal = np.empty_like(points)
    n = np.cross(tang[0], ref)
    n /= np.linalg.norm(n)
    for k in range(len(points)):  # parallel transport: no twisting along the branch
        n = n - tang[k] * np.dot(n, tang[k])
        n /= max(np.linalg.norm(n), 1e-9)
        normal[k] = n
    return tang, normal, np.cross(tang, normal)


def _flare(theta: np.ndarray, z: float, r0: float, phase: float) -> np.ndarray:
    """Root-flare multiplier for a trunk ring at height z: wide lobed buttress at the ground."""
    fade = np.exp(-max(z, 0.0) / (2.2 * r0))
    lobes = np.maximum(np.cos(5.0 * theta + phase), 0.0) ** 2
    return 1.0 + fade * (0.45 + 0.55 * lobes)


def root_chains(trunk_radius: float, scale: float, rng: np.random.Generator) -> list:
    """A handful of surface roots running out from the flare and diving into the soil."""
    roots = []
    for k in range(7):
        ang = k * 2 * np.pi / 7 + rng.uniform(-0.25, 0.25)
        reach = trunk_radius * rng.uniform(3.0, 5.0) + 0.25 * scale
        t = np.linspace(0.0, 1.0, 9)
        wobble = 0.08 * reach * np.sin(t * np.pi * rng.uniform(1.5, 3.0))
        d = np.array([np.cos(ang), np.sin(ang), 0.0])
        side = np.array([-d[1], d[0], 0.0])
        pts = (d[None] * (trunk_radius * 0.4 + reach * t)[:, None] + side[None] * wobble[:, None])
        radii = trunk_radius * rng.uniform(0.3, 0.42) * (1 - t) ** 1.2 + 0.004
        pts[:, 2] = radii * (0.45 - 1.6 * t ** 2)  # half-buried hump that dives into the soil
        roots.append(og.Chain(pts, radii, order=-1, ends_in_tip=False))  # -1: not the trunk
    return roots


def tube_mesh(chains: list, trunk_flare: bool = True) -> MeshData:
    verts, quads, uvs, values = [], [], [], []
    base = 0
    trunk_r0 = max(c.radii[0] for c in chains)
    phase = 0.7
    for c_idx, chain in enumerate(chains):
        r_max = float(chain.radii.max())
        points, radii = _resample(chain.points, chain.radii, max(0.06, 3.0 * r_max) if r_max < 0.15 else 0.12)
        if len(points) < 2:
            continue
        sides = int(np.clip(round(6 + r_max * 36), 5, 26))
        tang, normal, binormal = _frames(points)
        theta = np.linspace(0.0, 2 * np.pi, sides + 1)  # seam vertex duplicated for clean UVs
        cos, sin = np.cos(theta), np.sin(theta)
        is_trunk = trunk_flare and chain.order == 0 and np.linalg.norm(points[0, :2]) < 0.05 and points[0, 2] < 0.05
        ring_r = np.repeat(radii[:, None], sides + 1, axis=1)
        if is_trunk:
            for k in range(len(points)):
                ring_r[k] *= _flare(theta, points[k, 2], trunk_r0, phase)
        ring = (points[:, None, :]
                + ring_r[..., None] * (cos[None, :, None] * normal[:, None, :] + sin[None, :, None] * binormal[:, None, :]))
        n_rings = len(points)
        verts.append(ring.reshape(-1, 3))
        values.append(np.repeat(radii, sides + 1))

        u_repeat = max(1.0, round(2 * np.pi * r_max / BARK_TILE_M))
        v = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]) / BARK_TILE_M
        v = v + (c_idx * 0.37) % 1.0  # neighbours don't start on the same texel row
        u = theta / (2 * np.pi) * u_repeat

        k = np.arange(n_rings - 1)[:, None]
        s = np.arange(sides)[None, :]
        a = base + k * (sides + 1) + s
        q = np.stack([a, a + 1, a + sides + 2, a + sides + 1], axis=-1).reshape(-1, 4)
        quads.append(q)
        ku = np.broadcast_to(s, (n_rings - 1, sides)).ravel()
        kv = np.broadcast_to(k, (n_rings - 1, sides)).ravel()
        uvs.append(np.stack([
            np.stack([u[ku], v[kv]], -1), np.stack([u[ku + 1], v[kv]], -1),
            np.stack([u[ku + 1], v[kv + 1]], -1), np.stack([u[ku], v[kv + 1]], -1),
        ], axis=1).reshape(-1, 2))
        base += n_rings * (sides + 1)
    return MeshData(np.vstack(verts), np.vstack(quads), np.vstack(uvs), np.concatenate(values))


@dataclass(frozen=True)
class LeafCard:
    """One leaf cut out of the atlas: UV box and where the stalk sits inside it (0..1)."""
    u0: float
    v0: float
    u1: float
    v1: float
    stem_x: float
    stem_y: float
    aspect: float            # width / length


def twig_anchors(skel: og.Skeleton, scale: float):
    """Where the leaves hang: the real twig ends of the scanned oak, scaled to this age.

    Using the scan's own twig tips (not spheres around the few tips a young framework has) keeps
    the crown shaped like this oak at every age — a young crown is the future crown, smaller —
    and avoids "broccoli" puffs.
    """
    has_child = np.zeros(len(skel.parent), bool)
    has_child[skel.parent[skel.parent >= 0]] = True
    tip = ~has_child
    ends = (skel.start[tip] + skel.axis[tip] * skel.length[tip, None]) * scale
    return ends, skel.axis[tip]


def leaf_quads(anchors: np.ndarray, anchor_dirs: np.ndarray, age: float, scale: float, cards: list,
               rng: np.random.Generator, leaf_scale: float = 1.0) -> MeshData:
    if age <= 0 or not len(anchors):
        return MeshData(np.zeros((0, 3)), np.zeros((0, 4), int), np.zeros((0, 2)), np.zeros(0))
    total = LEAVES_AT_FULL_AGE * og.height_fraction(age) ** 2
    per_anchor = max(2, int(round(total / len(anchors))))
    cluster_r = max(0.38 * scale, 0.12)

    n = per_anchor * len(anchors)
    owner = np.repeat(np.arange(len(anchors)), per_anchor)
    dirs = anchor_dirs[owner]
    gauss = rng.normal(size=(n, 3))
    gauss /= np.linalg.norm(gauss, axis=1, keepdims=True)
    offset = gauss * (rng.uniform(0, 1, n) ** (1 / 3) * cluster_r)[:, None] * np.array([1.0, 1.0, 0.75])
    # Leaves sit along the last part of the twig, not only at its very end.
    stalk = anchors[owner] - dirs * (rng.uniform(0.0, 1.2, n) * cluster_r)[:, None] + offset

    up = np.array([0.0, 0.0, 1.0])
    outward = offset / np.maximum(np.linalg.norm(offset, axis=1, keepdims=True), 1e-9)
    normal = up + 0.5 * outward + 0.6 * rng.normal(size=(n, 3))
    normal /= np.linalg.norm(normal, axis=1, keepdims=True)
    axis = outward + 0.6 * dirs - 0.35 * up + 0.4 * rng.normal(size=(n, 3))
    axis -= normal * np.sum(axis * normal, axis=1, keepdims=True)
    axis /= np.maximum(np.linalg.norm(axis, axis=1, keepdims=True), 1e-9)
    width_dir = np.cross(normal, axis)

    card_idx = rng.integers(0, len(cards), n)
    c = np.array([[k.u0, k.v0, k.u1, k.v1, k.stem_x, k.stem_y, k.aspect] for k in cards])[card_idx]
    length = LEAF_LENGTH_M * leaf_scale * rng.uniform(0.75, 1.25, n)
    width = length * c[:, 6]

    corners_xy = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    along = (corners_xy[None, :, 1] - c[:, None, 5]) * length[:, None]
    across = (corners_xy[None, :, 0] - c[:, None, 4]) * width[:, None]
    verts = stalk[:, None, :] + axis[:, None, :] * along[..., None] + width_dir[:, None, :] * across[..., None]
    uv = np.stack([
        np.stack([c[:, 0], c[:, 1]], -1), np.stack([c[:, 2], c[:, 1]], -1),
        np.stack([c[:, 2], c[:, 3]], -1), np.stack([c[:, 0], c[:, 3]], -1),
    ], axis=1)
    quads = np.arange(n * 4).reshape(n, 4)
    return MeshData(verts.reshape(-1, 3), quads, uv.reshape(-1, 2), np.repeat(rng.uniform(0, 1, n), 4))


def _boost(chains: list, trunk_boost: float) -> list:
    """Art direction: a heavier trunk and main limbs (order 0 / 1) than the scan's measurement."""
    if trunk_boost == 1.0:
        return chains
    by_order = {0: trunk_boost, 1: 1.0 + (trunk_boost - 1.0) * 0.5}
    return [og.Chain(c.points, c.radii * by_order.get(c.order, 1.0), c.order, c.ends_in_tip) for c in chains]


def build_age(skel: og.Skeleton, pre: og.Precomputed, age: float, cards: list, seed: int = 195,
              leaf_scale: float = 1.0, trunk_boost: float = 1.0):
    """Everything the renderer needs for one age: (bark tubes, leaves, aged tree)."""
    rng = np.random.default_rng(seed + int(round(age * 10)))
    tree = og.grow(skel, pre, age)
    chains = _boost(og.branch_chains(skel, tree), trunk_boost)
    trunk_r = float(tree.radius.max()) * trunk_boost
    chains = chains + root_chains(trunk_r, tree.scale, np.random.default_rng(seed))
    anchors, anchor_dirs = twig_anchors(skel, tree.scale)
    return tube_mesh(chains), leaf_quads(anchors, anchor_dirs, age, tree.scale, cards, rng, leaf_scale), tree
