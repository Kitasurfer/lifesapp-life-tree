"""Geometry for one age of the oak — numpy only, so it is testable without Blender.

* tubes  — every branch chain becomes one tube (rings along the polyline, parallel-transport
           frames), UVs in metres of bark so the texture keeps its physical size at every age.
           The stem is a veteran's: a heavy bole that holds its girth up to the main fork, an
           irregular cross-section (flutes, burrs), a root flare of a few uneven buttresses that
           run out into low root ridges and go under the soil. Limbs leave it through collars.
* leaves — oak-leaf cards in rosettes on short shoots at the scan's real twig ends. The crown is
           not a shell of evenly spread leaves: twig ends in the shaded interior carry little,
           and a low-frequency field leaves whole pockets bare, so foliage gathers into pads with
           sky and limbs between them. Leaf size is real at every age (only the tree scales).
           One fixed set of leaves for all ages: a leaf that exists at 40 is the same leaf at 80.

Blender (blender/render_age.py) just turns these arrays into meshes.
"""
from dataclasses import dataclass, field

import numpy as np

import oak_growth as og

BARK_TILE_M = 0.6           # metres of bark covered by one bark texture tile on thin wood
BARK_TILE_PER_RADIUS = 0.8  # thick wood: the tile grows with the radius, but stays fine enough to give scale
LEAF_LENGTH_M = 0.11
LEAVES_AT_FULL_AGE = 260_000
MIN_TWIG_RADIUS_M = 0.008   # thinner twigs vanish between pixels and leaves seem to float
BOLE_SHARE = 0.27           # the bole (stem below the main fork) as a share of the tree's height


@dataclass(frozen=True)
class MeshData:
    verts: np.ndarray        # (v, 3)
    quads: np.ndarray        # (f, 4) vertex indices
    loop_uv: np.ndarray      # (f * 4, 2) per face corner, in quad order
    point_value: np.ndarray  # (v,) tubes: radius in metres; leaves: random 0..1 per leaf
    extra: dict = field(default_factory=dict)      # more per-vertex floats, by attribute name
    loop_uv_m: np.ndarray = None                   # tubes: (f * 4, 2) bark coordinates in metres


def _resample(points: np.ndarray, radii: np.ndarray, max_seg):
    """Split segments longer than `max_seg` (one length, or one per segment)."""
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


def _smoothstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


@dataclass(frozen=True)
class StemShape:
    """The individual shape of this stem: where its buttresses stand and how its bark bulges."""
    angles: np.ndarray       # world angle of each buttress root, radians
    strength: np.ndarray     # 0..1, how heavy each one is
    bend: np.ndarray         # radians: a root swings aside on its way out
    phases: np.ndarray       # phases of the cross-section harmonics
    burrs: np.ndarray        # (b, 5): angle, height (in stem radii), angular size, height size, amplitude


def stem_shape(seed: int, count: int = 7) -> StemShape:
    """Evenly spaced, equally heavy buttress roots: the foot of the stem is symmetric, and mirrored
    left to right in the frame. None points straight at the camera (it looks along +y, so the
    camera stands at -90 degrees): seen end-on a root is a lump, seen from the side it is a root."""
    rng = np.random.default_rng(seed + 7)
    step = 2 * np.pi / count
    angles = -np.pi / 2 + step * (np.arange(count) + 0.5)
    strength = np.full(count, 0.85)
    bend = np.zeros(count)
    burrs = np.stack([rng.uniform(0, 2 * np.pi, 7), rng.uniform(1.0, 5.5, 7), rng.uniform(0.25, 0.5, 7),
                      rng.uniform(0.3, 0.7, 7), rng.uniform(0.015, 0.04, 7)], axis=1)
    return StemShape(angles, strength, bend, rng.uniform(0, 2 * np.pi, 8), burrs)


FOOT_HEIGHT = 0.5           # in stem radii: the root flare sweeps out fast below this (finer rings there)


def _stem_profile(angle: np.ndarray, z: np.ndarray, shape: StemShape) -> np.ndarray:
    """Radius multiplier for stem rings: `angle` (k, s) world angles, `z` (k,) heights in stem radii.

    One surface from the bole down into the soil. The bole is irregular (oval, shallow flutes,
    burrs); towards the ground it swells into a root flare of a few uneven buttresses. The roots
    are the stem's own surface, not pipes laid against it: no seam, no root ends on the ground.
    """
    h = np.maximum(z, 0.0)[:, None]
    ph = shape.phases
    flutes = np.exp(-h / 7.0) * (0.020 * np.cos(7 * angle + ph[0] + 0.15 * h)
                                 + 0.015 * np.cos(11 * angle + ph[1] - 0.11 * h)
                                 + 0.010 * np.cos(17 * angle + ph[2] + 0.07 * h))
    oval = 0.012 * np.cos(2 * angle + ph[3] + 0.2 * h) + 0.008 * np.cos(3 * angle + ph[4] - 0.3 * h)
    burr = np.zeros_like(angle)
    for a, bh, sa, sh, amp_b in shape.burrs:
        da = np.angle(np.exp(1j * (angle - a)))
        burr += amp_b * np.exp(-(da / sa) ** 2 - ((h - bh) / sh) ** 2)
    swell = 0.13 * np.exp(-h / 0.9) + 0.10 * np.exp(-h / 3.0)

    # Buttress roots: ridges that stand out further and narrower the lower they are, in the concave
    # sweep of a root flare, and meet the soil steeply. Just above the soil each one rolls under
    # (a rounded toe), so the foot of the stem has no knife edge along the ground.
    hh = h[:, :, None]
    strength = shape.strength[None, None, :]
    d = angle[:, :, None] - shape.angles[None, None, :] - shape.bend[None, None, :] * np.exp(-hh / 0.45)
    sharp = 5.0 + (12.0 + 8.0 * strength) * np.exp(-hh / 1.3)
    out = (strength * (1.00 * np.exp(-hh / 0.70) + 0.50 * np.exp(-hh / 0.26) + 0.17 * np.exp(-hh / 3.0))
           * (1.0 - 0.22 * np.exp(-hh / 0.05)))
    ridge = (out * np.exp(sharp * (np.cos(d) - 1.0))).max(axis=2)
    toe = 1.0 - 0.06 * np.exp(-h / 0.05)
    return (1.0 + swell + flutes + oval + burr) * toe + ridge


def _bole(chain: og.Chain, height: float) -> og.Chain:
    """A veteran's bole holds its girth up to the main fork (the limbs above carry most of the
    wood: by the pipe model the stem under them is far thicker than the scan's leaning-stem fit),
    then hands it over to the limbs within a short distance."""
    z = chain.points[:, 2]
    top = BOLE_SHARE * height
    r0 = float(chain.radii[0])
    hold = np.where(z < top, 1.0 - 0.22 * (z / top), 0.78 * np.exp(-(z - top) / (0.10 * height)))
    radii = np.maximum(chain.radii, r0 * hold)
    foot = chain.points[0] - np.array([0.0, 0.0, 0.25 * r0])  # the stem and its roots go on under the soil
    return og.Chain(np.vstack([foot[None], chain.points]), np.concatenate([radii[:1], radii]), chain.order,
                    chain.ends_in_tip, chain.parent_radius, chain.parent_order)


def _closed_end(chain: og.Chain) -> og.Chain:
    """A branch that ends thick (a snag, or a limb the scan lost) ends in a healed-over round
    stub instead of an open pipe."""
    r = float(chain.radii[-1])
    if r < 0.025 or chain.order < 0:
        return chain
    d = chain.points[-1] - chain.points[-2]
    d /= max(np.linalg.norm(d), 1e-9)
    more = chain.points[-1][None] + d[None] * (r * np.array([0.45, 0.85, 1.1, 1.25]))[:, None]
    return og.Chain(np.vstack([chain.points, more]), np.concatenate([chain.radii, r * np.array([0.9, 0.65, 0.35, 0.02])]),
                    chain.order, chain.ends_in_tip, chain.parent_radius, chain.parent_order)


def _smooth(values: np.ndarray, passes: int) -> np.ndarray:
    out = values.astype(np.float64).copy()
    for _ in range(passes if len(out) > 2 else 0):
        out[1:-1] = (out[:-2] + 2.0 * out[1:-1] + out[2:]) / 4.0
    return out


def _collar(points: np.ndarray, radii: np.ndarray, chain: og.Chain) -> np.ndarray:
    """A limb leaves its parent through a collar: it is thickest where it comes out of the bark."""
    if chain.parent_radius <= 0 or chain.order < 1:
        return radii
    r0 = float(radii[0])
    if r0 < 0.03:
        return radii
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))])
    out_of_bark = np.maximum(s - chain.parent_radius, 0.0)
    swell = 1.0 + 0.55 * np.exp(-out_of_bark / (1.6 * r0))
    return np.minimum(radii * swell, np.maximum(radii, 0.85 * chain.parent_radius))


def tube_mesh(chains: list, shape: StemShape, trunk_flare: bool = True) -> MeshData:
    verts, quads, uvs, uvs_m, values, seams = [], [], [], [], [], []
    base = 0
    trunk_r0 = max(c.radii[0] for c in chains)
    tree_height = max(float(c.points[:, 2].max()) for c in chains)
    for c_idx, chain in enumerate(chains):
        r_max = float(chain.radii.max())
        # Every piece of the stem (the scan splits it into several branches) takes the same shape.
        is_trunk = trunk_flare and chain.order == 0 and r_max > 0.25 * trunk_r0
        thick = r_max > 0.15
        # The scan's cylinders are ~2 m long with one radius each: smooth the taper before and after
        # resampling, or every change of slope shows as a ring on the stem.
        radii0 = _smooth(chain.radii, 2)
        if thick:
            # Thick wood is displaced by its bark in the render, so it is meshed about as fine as a
            # pixel of the frame (3-4.5 cm) where it is thick; the thin end only needs to stay round.
            local = np.maximum(radii0[:-1], radii0[1:])
            step = np.where(local > 0.45, 0.03, np.where(local > 0.15, 0.045, 0.10))
            if is_trunk:  # the flare sweeps out fast near the soil: it needs rings close together
                step = np.where(chain.points[:-1, 2] < FOOT_HEIGHT * trunk_r0, 0.015, step)
        else:
            step = max(0.12, 4.0 * r_max)
            if is_trunk:  # a sapling's root flare is a few centimetres high
                step = np.where(chain.points[:-1, 2] < 3.0 * trunk_r0, max(0.2 * trunk_r0, 0.008), step)
        points, radii = _resample(chain.points, radii0, step)
        if len(points) < 2:
            continue
        if thick:  # and the axis: a kink between two scan cylinders is a crease across the bark
            keep = points[[0, -1]].copy()
            for _ in range(int(min(400, (1.2 * r_max / 0.04) ** 2))):
                points[1:-1] = (points[:-2] + 2.0 * points[1:-1] + points[2:]) / 4.0
            points[[0, -1]] = keep
        if is_trunk:
            # The bole is a plumb column: below the main fork its axis is the z axis exactly (the
            # leader's wander above the fork must not pull the stem aside), then it eases into the leader.
            ease = _smoothstep((points[:, 2] - BOLE_SHARE * tree_height) / (0.14 * tree_height))
            points[:, :2] *= ease[:, None]
        radii = _smooth(radii, 200 if thick else 3)
        radii = _collar(points, radii, chain)
        if is_trunk:
            sides = int(np.clip(round(2 * np.pi * r_max / 0.024), 80, 480))
        elif thick:
            sides = int(np.clip(round(2 * np.pi * r_max / 0.04), 24, 120))
        else:
            sides = int(np.clip(round(4 + r_max * 110), 4, 20))
        tang, normal, binormal = _frames(points)
        theta = np.linspace(0.0, 2 * np.pi, sides + 1)  # seam vertex duplicated for clean UVs
        cos, sin = np.cos(theta), np.sin(theta)
        ring_r = np.repeat(radii[:, None], sides + 1, axis=1)
        if is_trunk:
            out = cos[None, :, None] * normal[:, None, :] + sin[None, :, None] * binormal[:, None, :]
            ring_r = ring_r * _stem_profile(np.arctan2(out[..., 1], out[..., 0]), points[:, 2] / trunk_r0, shape)
        ring = (points[:, None, :]
                + ring_r[..., None] * (cos[None, :, None] * normal[:, None, :] + sin[None, :, None] * binormal[:, None, :]))
        n_rings = len(points)
        verts.append(ring.reshape(-1, 3))
        values.append(np.repeat(radii, sides + 1))
        # The furrow pattern is not periodic round the wood: it fades out at the UV seam, or the
        # displaced bark would tear open along it.
        seams.append(np.tile(_smoothstep(np.abs(np.sin(theta / 2.0)) / 0.12), n_rings))

        tile = max(BARK_TILE_M, BARK_TILE_PER_RADIUS * r_max)
        u_repeat = max(1.0, round(2 * np.pi * r_max / tile))
        along = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))])
        v = along / tile + (c_idx * 0.37) % 1.0  # neighbours don't start on the same texel row
        u = theta / (2 * np.pi) * u_repeat
        u_m = theta * r_max + c_idx * 1.7        # metres round the wood and along it: furrows keep
        v_m = along + c_idx * 0.9                # their real spacing whatever the tile size

        k = np.arange(n_rings - 1)[:, None]
        s = np.arange(sides)[None, :]
        a = base + k * (sides + 1) + s
        q = np.stack([a, a + 1, a + sides + 2, a + sides + 1], axis=-1).reshape(-1, 4)
        quads.append(q)
        ku = np.broadcast_to(s, (n_rings - 1, sides)).ravel()
        kv = np.broadcast_to(k, (n_rings - 1, sides)).ravel()
        for store, (uu, vv) in ((uvs, (u, v)), (uvs_m, (u_m, v_m))):
            store.append(np.stack([
                np.stack([uu[ku], vv[kv]], -1), np.stack([uu[ku + 1], vv[kv]], -1),
                np.stack([uu[ku + 1], vv[kv + 1]], -1), np.stack([uu[ku], vv[kv + 1]], -1),
            ], axis=1).reshape(-1, 2))
        base += n_rings * (sides + 1)
    return MeshData(np.vstack(verts), np.vstack(quads), np.vstack(uvs), np.concatenate(values),
                    {"seam": np.concatenate(seams)}, np.vstack(uvs_m))


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


SHOOTS_PER_TWIG = 8         # candidate shoots at one twig end; how many are in leaf depends on the place
LEAVES_PER_SHOOT = 8        # at most


def _field(points: np.ndarray, rng: np.random.Generator, wavelengths, waves: int = 10) -> np.ndarray:
    """Smooth random field over space, 0..1 (a sum of plane waves): neighbours get similar values."""
    d = rng.normal(size=(waves, 3))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    lam = rng.uniform(wavelengths[0], wavelengths[1], waves)
    v = np.cos(points @ (d * (2 * np.pi / lam)[:, None]).T + rng.uniform(0, 2 * np.pi, waves)).sum(axis=1)
    return 0.5 + 0.5 * np.tanh(v / np.sqrt(waves / 2.0))


def _shade_depth(tips: np.ndarray, radius: float = 2.6) -> np.ndarray:
    """How deep in its own foliage a twig end sits, 0 (open to the sky) .. 1 (buried).

    Counts the twig ends that stand between this one and the light (up and away from the stem)
    within `radius`. It is local, so every lobe of the crown has its own lit outside.
    """
    centre = np.array([np.median(tips[:, 0]), np.median(tips[:, 1]), np.percentile(tips[:, 2], 35)])
    light = (tips - centre) * np.array([1.0, 1.0, 0.7])
    light /= np.maximum(np.linalg.norm(light, axis=1, keepdims=True), 1e-9)
    light += np.array([0.0, 0.0, 0.55])
    light /= np.linalg.norm(light, axis=1, keepdims=True)
    count = np.zeros(len(tips))
    for a in range(0, len(tips), 400):
        rel = tips[None, :, :] - tips[a:a + 400, None, :]
        dist = np.linalg.norm(rel, axis=2)
        ahead = np.einsum("ijk,ik->ij", rel, light[a:a + 400])
        count[a:a + 400] = ((dist < radius) & (ahead > 0.35 * dist)).sum(axis=1)
    return np.clip(count / max(np.percentile(count, 90), 1.0), 0.0, 1.0)


@dataclass(frozen=True)
class Foliage:
    """Every leaf the oak will ever carry, in the full-age tree's frame. Ages only choose from it."""
    tip_cyl: np.ndarray      # (t,) skeleton cylinder that ends in this twig end
    tip_pos: np.ndarray      # (t, 3) full-age position
    tip_dir: np.ndarray      # (t, 3)
    light_dir: np.ndarray    # (t, 3) where the open sky is, seen from the twig end
    depth: np.ndarray        # (t,) 0 lit outside .. 1 shaded inside
    tint: np.ndarray         # (t,) 0..1, smooth over the crown: neighbouring pads share a tone
    weight: np.ndarray       # (t,) how much foliage this place carries, 0 = a gap
    rank: np.ndarray         # (t, SHOOTS_PER_TWIG) 0..1: a shoot is in leaf when rank < weight * density(age)
    density: float           # solved so that the full-age tree carries LEAVES_AT_FULL_AGE
    seed: int


def plan_foliage(skel: og.Skeleton, seed: int, leaves: int = LEAVES_AT_FULL_AGE, gaps: float = 1.0) -> Foliage:
    """`gaps` 0 = foliage on every twig end (a closed shell), 1 = the default pads and openings."""
    has_child = np.zeros(len(skel.parent), bool)
    has_child[skel.parent[skel.parent >= 0]] = True
    tip_cyl = np.flatnonzero(~has_child)
    pos = skel.start[tip_cyl] + skel.axis[tip_cyl] * skel.length[tip_cyl, None]
    rng = np.random.default_rng(seed + 31)

    depth = _shade_depth(pos)
    centre = np.array([np.median(pos[:, 0]), np.median(pos[:, 1]), np.percentile(pos[:, 2], 35)])
    light = (pos - centre) * np.array([1.0, 1.0, 0.7])
    light /= np.maximum(np.linalg.norm(light, axis=1, keepdims=True), 1e-9)

    # Pads and openings: two scales of the same field — boughs (5-9 m) and sprays (2-3.5 m).
    pads = 0.4 * _field(pos, rng, (5.0, 9.0)) + 0.6 * _field(pos, rng, (2.2, 3.8))
    cut = np.quantile(pads, 0.30 * gaps)
    open_air = _smoothstep((pads - cut) / 0.10) if gaps > 0 else np.ones(len(pos))
    shaded = 1.0 - (1.0 - 0.10) * _smoothstep((depth - 0.30) / 0.45) * min(gaps, 1.0)
    # The outside of the crown is never bare (bare twigs above the foliage read as die-back): an
    # opening there is thin foliage one can see into, not a hole.
    skin = 0.4 * (1.0 - _smoothstep(depth / 0.25))
    weight = np.maximum(open_air, skin) * shaded * rng.uniform(0.75, 1.25, len(pos))
    tint = _field(pos, rng, (3.0, 7.0))
    rank = rng.uniform(0.0, 1.0, (len(pos), SHOOTS_PER_TWIG))

    mean_leaves = (LEAVES_PER_SHOOT + 4) / 2.0
    lo, hi = 0.0, 50.0
    for _ in range(40):  # density such that the expected number of leaves at full age is `leaves`
        mid = (lo + hi) / 2.0
        n = np.clip(weight * mid, 0.0, 1.0).sum() * SHOOTS_PER_TWIG * mean_leaves
        lo, hi = (mid, hi) if n < leaves else (lo, mid)
    return Foliage(tip_cyl, pos, skel.axis[tip_cyl], light, depth, tint, weight, rank, hi, seed)


def leaf_share(age: float) -> float:
    """Share of the full-age foliage carried at `age`: the crown's surface grows with height²,
    a young tree is a little leafier than that."""
    return og.height_fraction(age) ** 1.7


def shoots_in_leaf(foliage: Foliage, age: float) -> np.ndarray:
    """(t, SHOOTS_PER_TWIG) bool. Grows monotonically with age: leaves are added, never reshuffled."""
    if age <= 0:
        return np.zeros_like(foliage.rank, bool)
    return foliage.rank < (np.clip(foliage.weight * foliage.density, 0.0, 1.0) * leaf_share(age))[:, None]


def leafy_cylinders(skel: og.Skeleton, foliage: Foliage, in_leaf: np.ndarray) -> np.ndarray:
    mask = np.zeros(len(skel.parent), bool)
    mask[foliage.tip_cyl[in_leaf.any(axis=1)]] = True
    return mask


def leaf_quads(foliage: Foliage, in_leaf: np.ndarray, scale: float, cards: list, leaf_scale: float = 1.0) -> MeshData:
    """Leaf cards for the shoots in leaf, tree scaled by `scale` (leaves and shoots keep real size).

    A shoot stands on the last stretch of its twig, turns towards the light, and carries a spiral
    rosette of leaves crowded towards its end, blades turned up to the sky — as oak shoots do.
    """
    t, k, m = len(foliage.tip_pos), SHOOTS_PER_TWIG, LEAVES_PER_SHOOT
    if not in_leaf.any():
        return MeshData(np.zeros((0, 3)), np.zeros((0, 4), int), np.zeros((0, 2)), np.zeros(0),
                        {"leaf_depth": np.zeros(0), "leaf_tint": np.zeros(0)})
    # All randomness is drawn for every candidate leaf in a fixed order, then masked: the same
    # leaf has the same place, size and tilt at every age.
    rng = np.random.default_rng(foliage.seed + 57)
    back = rng.uniform(0.0, 0.9, (t, k))
    shoot_noise = rng.normal(size=(t, k, 3))
    shoot_len = rng.uniform(0.22, 0.60, (t, k))
    shoot_phase = rng.uniform(0, 2 * np.pi, (t, k))
    shoot_size = rng.uniform(0.85, 1.15, (t, k))
    shoot_tint = rng.normal(0.0, 0.10, (t, k))
    leaves_on = rng.integers(4, m + 1, (t, k))
    spread = rng.uniform(0.6, 1.3, (t, k, m))
    leaf_noise = rng.normal(size=(t, k, m, 3))
    leaf_size = rng.uniform(0.7, 1.3, (t, k, m))
    leaf_rand = rng.uniform(0.0, 1.0, (t, k, m))
    turned = rng.uniform(0.0, 1.0, (t, k, m)) < 0.12
    card_idx = rng.integers(0, len(cards), (t, k, m))

    it, ik, im = np.nonzero(in_leaf[:, :, None] & (np.arange(m)[None, None, :] < leaves_on[:, :, None]))
    n = len(it)
    up = np.array([0.0, 0.0, 1.0])
    reach = float(np.clip(scale * 1.3, 0.35, 1.0))  # a sapling's shoots are short, like its twigs
    tip_dir, light = foliage.tip_dir[it], foliage.light_dir[it]

    shoot = 0.55 * tip_dir + 0.55 * light + 0.3 * up + 0.75 * shoot_noise[it, ik]
    shoot /= np.maximum(np.linalg.norm(shoot, axis=1, keepdims=True), 1e-9)
    base = foliage.tip_pos[it] * scale - tip_dir * (back[it, ik] * reach)[:, None]
    f = 0.2 + 0.8 * np.sqrt((im + 0.5) / leaves_on[it, ik])          # crowded towards the shoot's end
    stalk = base + shoot * (shoot_len[it, ik] * reach * f)[:, None]

    e1 = np.cross(shoot, up)
    e1 /= np.maximum(np.linalg.norm(e1, axis=1, keepdims=True), 1e-9)
    e2 = np.cross(shoot, e1)
    az = im * 2.39996 + shoot_phase[it, ik]                          # golden angle: a spiral rosette
    out = np.cos(az)[:, None] * e1 + np.sin(az)[:, None] * e2
    tilt = np.radians(75.0 - 45.0 * f) * spread[it, ik, im]           # lower leaves stand off, end leaves point on
    axis = np.cos(tilt)[:, None] * shoot + np.sin(tilt)[:, None] * out - 0.22 * up + 0.2 * leaf_noise[it, ik, im]
    axis /= np.maximum(np.linalg.norm(axis, axis=1, keepdims=True), 1e-9)
    normal = up + 0.35 * light + np.where(turned[it, ik, im], 1.6, 0.45)[:, None] * leaf_noise[it, ik, im][:, ::-1]
    normal -= axis * np.sum(normal * axis, axis=1, keepdims=True)
    normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-9)
    width_dir = np.cross(axis, normal)   # card faces along `normal`: its front is the leaf's upper side

    c = np.array([[q.u0, q.v0, q.u1, q.v1, q.stem_x, q.stem_y, q.aspect] for q in cards])[card_idx[it, ik, im]]
    depth = foliage.depth[it]
    # Shade leaves in the interior are larger than sun leaves on the outside.
    length = LEAF_LENGTH_M * leaf_scale * leaf_size[it, ik, im] * shoot_size[it, ik] * (0.9 + 0.25 * depth)
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
    tint = np.clip(foliage.tint[it] + shoot_tint[it, ik], 0.0, 1.0)
    return MeshData(verts.reshape(-1, 3), quads, uv.reshape(-1, 2), np.repeat(leaf_rand[it, ik, im], 4),
                    {"leaf_depth": np.repeat(depth, 4), "leaf_tint": np.repeat(tint, 4)})


def girth(age: float) -> float:
    """Extra stem girth of the old tree: height stops, the stem keeps laying on wood."""
    return 1.0 + 0.16 * float(_smoothstep((age - 30.0) / 60.0))


def _boost(chains: list, trunk_boost: float) -> list:
    """Art direction: a heavier trunk than the scan's measurement.

    The extra thickness fades out with branch order (trunk → limbs → boughs), so a massive trunk
    still flows into its limbs instead of snapping to thin branches at the first fork.
    """
    share = {0: 1.0, 1: 0.62, 2: 0.36, 3: 0.16}
    gain = lambda order: 1.0 + (trunk_boost - 1.0) * share.get(order, 0.0)

    out = []
    for c in chains:
        boosted, parent_radius = c.radii * gain(c.order), c.parent_radius * gain(c.parent_order)
        length = float(np.linalg.norm(np.diff(c.points, axis=0), axis=1).sum())
        if c.order > 0 and length - parent_radius < 8.0 * boosted[0]:
            # A short snag (measured from where it leaves the bark) stays a thin dead branch:
            # boosted and collared, it is a horn on the stem.
            boosted = np.minimum(boosted, max((length - parent_radius) / 16.0, 0.03))
            parent_radius = 0.0
        out.append(og.Chain(np.asarray(c.points), np.maximum(boosted, MIN_TWIG_RADIUS_M), c.order, c.ends_in_tip,
                            parent_radius, c.parent_order))
    return out


def build_age(skel: og.Skeleton, pre: og.Precomputed, age: float, cards: list, seed: int = 195,
              leaf_scale: float = 1.0, trunk_boost: float = 1.0, leaves: int = LEAVES_AT_FULL_AGE,
              gaps: float = 1.0):
    """Everything the renderer needs for one age: (bark tubes, leaves, aged tree)."""
    foliage = plan_foliage(skel, seed, leaves, gaps)
    in_leaf = shoots_in_leaf(foliage, age)
    tree = og.grow(skel, pre, age, leafy=leafy_cylinders(skel, foliage, in_leaf))
    boost = trunk_boost * girth(age)
    chains = _boost(og.branch_chains(skel, tree), boost)
    height = float(tree.end[:, 2].max())
    stem = max(range(len(chains)), key=lambda i: chains[i].radii[0])
    chains = [_closed_end(c) for c in chains]
    chains[stem] = _bole(chains[stem], height)
    shape = stem_shape(seed)
    return tube_mesh(chains, shape), leaf_quads(foliage, in_leaf, tree.scale, cards, leaf_scale), tree
