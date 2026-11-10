"""Geometry for one age of the oak — numpy only, so it is testable without Blender.

* tubes  — every branch chain becomes one tube (rings along the polyline, parallel-transport
           frames), UVs in metres of bark so the texture keeps its physical size at every age;
           the trunk gets buttresses, fluting and forked surface roots.
* leaves — oak-leaf cards clustered on the scan's real twig ends (scaled to the age); leaf size is
           the real 7–14 cm at every age (only the tree scales, leaves don't).

Blender (blender/render_age.py) just turns these arrays into meshes.
"""
from dataclasses import dataclass

import numpy as np

import oak_growth as og

BARK_TILE_M = 0.6          # metres of bark covered by one bark texture tile on thin wood
BARK_TILE_PER_RADIUS = 1.5  # thick wood: the tile grows with the radius, so fissures read from afar
LEAF_LENGTH_M = 0.11
LEAVES_AT_FULL_AGE = 450_000


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


@dataclass(frozen=True)
class RootLayout:
    """Where the buttresses stand around the stem: the flare lobes and the roots share it."""
    angles: np.ndarray       # world angle of each buttress, radians
    strength: np.ndarray     # 0..1, how heavy each one is


def root_layout(seed: int, count: int = 9) -> RootLayout:
    rng = np.random.default_rng(seed)
    angles = (np.arange(count) + rng.uniform(-0.3, 0.3, count)) * 2 * np.pi / count
    return RootLayout(angles, rng.uniform(0.55, 1.0, count))


def _flare(world_angle: np.ndarray, z: float, r0: float, layout: RootLayout) -> np.ndarray:
    """Radius multiplier for a trunk ring at height z.

    At the ground the stem swells and throws out sharp buttresses (one per root); higher up they
    broaden into the shallow fluting of an old oak stem and die out towards the first limbs.
    """
    h = max(z, 0.0) / r0
    swell = 0.55 * np.exp(-h / 1.4)
    sharp = 2.5 + 14.0 * np.exp(-h / 1.2)
    amp = 1.25 * np.exp(-h / 2.4) + 0.12 * np.exp(-h / 9.0)
    d = world_angle[:, None] - layout.angles[None, :]
    lobes = (layout.strength[None, :] * np.exp(sharp * (np.cos(d) - 1.0))).max(axis=1)
    ribs = 0.035 * np.exp(-h / 12.0) * np.cos(19.0 * world_angle + 1.7 * np.sin(0.9 * h))
    return 1.0 + swell + amp * lobes + ribs


def root_chains(trunk_radius: float, scale: float, layout: RootLayout, rng: np.random.Generator) -> list:
    """Buttress roots: each leaves its flare lobe as a ridge, snakes outwards and dives into the soil.

    The heavy ones fork once. Only what is above the ground is seen (the ground is a holdout).
    """
    roots = []

    def one(ang: float, strength: float, start: np.ndarray, r_start: float, reach: float, steps: int = 16):
        t = np.linspace(0.0, 1.0, steps)
        d = np.array([np.cos(ang), np.sin(ang), 0.0])
        side = np.array([-d[1], d[0], 0.0])
        wobble = 0.10 * reach * np.sin(t * np.pi * rng.uniform(1.2, 2.6) + rng.uniform(0, 6.28)) * t
        pts = start[None] + d[None] * (reach * t)[:, None] + side[None] * wobble[:, None]
        radii = r_start * (1 - t ** 1.6) ** 0.8 + 0.01
        humps = 0.18 * np.sin(t * np.pi * rng.uniform(2.0, 3.5) + rng.uniform(0, 6.28))
        pts[:, 2] = radii * (0.95 + humps - 2.4 * t ** 1.5)  # rides high at the stem, then sinks
        roots.append(og.Chain(pts, radii, order=-1, ends_in_tip=False))  # -1: not the trunk
        return pts, radii

    for ang, strength in zip(layout.angles, layout.strength):
        d = np.array([np.cos(ang), np.sin(ang), 0.0])
        reach = trunk_radius * (3.5 + 3.5 * strength) + 0.3 * scale
        r_start = trunk_radius * (0.34 + 0.26 * strength)
        pts, radii = one(ang, strength, d * trunk_radius * 0.75, r_start, reach)
        if strength > 0.72:
            k = 6
            fork = ang + rng.choice([-1.0, 1.0]) * rng.uniform(0.45, 0.8)
            one(fork, strength, pts[k] * np.array([1.0, 1.0, 0.0]), radii[k] * 0.7, reach * 0.55, steps=10)
    return roots


def _smooth(values: np.ndarray, passes: int) -> np.ndarray:
    out = values.astype(np.float64).copy()
    for _ in range(passes if len(out) > 2 else 0):
        out[1:-1] = (out[:-2] + 2.0 * out[1:-1] + out[2:]) / 4.0
    return out


def tube_mesh(chains: list, layout: RootLayout, trunk_flare: bool = True) -> MeshData:
    verts, quads, uvs, values = [], [], [], []
    base = 0
    trunk_r0 = max(c.radii[0] for c in chains)
    for c_idx, chain in enumerate(chains):
        r_max = float(chain.radii.max())
        # The scan's cylinders are ~2 m long with one radius each: smooth the taper before and after
        # resampling, or every change of slope shows as a ring on the stem.
        points, radii = _resample(chain.points, _smooth(chain.radii, 2),
                                  max(0.06, 3.0 * r_max) if r_max < 0.15 else 0.12)
        if len(points) < 2:
            continue
        radii = _smooth(radii, 60 if r_max > 0.08 else 3)
        # Every piece of the stem (the scan splits it into several branches) takes the same fluting,
        # otherwise each joint shows as a ring.
        is_trunk = trunk_flare and chain.order == 0 and r_max > 0.25 * trunk_r0
        sides = 80 if is_trunk else int(np.clip(round(6 + r_max * 36), 5, 28))
        tang, normal, binormal = _frames(points)
        theta = np.linspace(0.0, 2 * np.pi, sides + 1)  # seam vertex duplicated for clean UVs
        cos, sin = np.cos(theta), np.sin(theta)
        ring_r = np.repeat(radii[:, None], sides + 1, axis=1)
        if is_trunk:
            for k in range(len(points)):
                out = cos[:, None] * normal[k][None, :] + sin[:, None] * binormal[k][None, :]
                ring_r[k] *= _flare(np.arctan2(out[:, 1], out[:, 0]), points[k, 2], trunk_r0, layout)
        ring = (points[:, None, :]
                + ring_r[..., None] * (cos[None, :, None] * normal[:, None, :] + sin[None, :, None] * binormal[:, None, :]))
        n_rings = len(points)
        verts.append(ring.reshape(-1, 3))
        # Roots are old wood whatever their thickness: they take the fissured bark, not the smooth one.
        values.append(np.repeat(np.maximum(radii, 0.12) if chain.order < 0 else radii, sides + 1))

        tile = max(BARK_TILE_M, BARK_TILE_PER_RADIUS * r_max)
        u_repeat = max(1.0, round(2 * np.pi * r_max / tile))
        v = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]) / tile
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
    cluster_r = max(0.46 * scale, 0.12)

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
    """Art direction: a heavier trunk than the scan's measurement.

    The extra thickness fades out with branch order (trunk → limbs → boughs), so a massive trunk
    still flows into its limbs instead of snapping to thin branches at the first fork.
    """
    if trunk_boost == 1.0:
        return chains
    share = {0: 1.0, 1: 0.7, 2: 0.38, 3: 0.16}
    return [og.Chain(c.points, c.radii * (1.0 + (trunk_boost - 1.0) * share.get(c.order, 0.0)),
                     c.order, c.ends_in_tip) for c in chains]


def build_age(skel: og.Skeleton, pre: og.Precomputed, age: float, cards: list, seed: int = 195,
              leaf_scale: float = 1.0, trunk_boost: float = 1.0):
    """Everything the renderer needs for one age: (bark tubes, leaves, aged tree)."""
    rng = np.random.default_rng(seed + int(round(age * 10)))
    tree = og.grow(skel, pre, age)
    chains = _boost(og.branch_chains(skel, tree), trunk_boost)
    trunk_r = float(tree.radius.max()) * trunk_boost
    layout = root_layout(seed)
    chains = chains + root_chains(trunk_r, tree.scale, layout, np.random.default_rng(seed))
    anchors, anchor_dirs = twig_anchors(skel, tree.scale)
    return tube_mesh(chains, layout), leaf_quads(anchors, anchor_dirs, age, tree.scale, cards, rng, leaf_scale), tree
