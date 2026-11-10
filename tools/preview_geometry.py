#!/usr/bin/env python3
"""Fast look at the geometry of one age without Blender: same camera, flat shading, no shadows.

Good for the things geometry decides — silhouette, stem and root shapes, how foliage gathers into
pads and where the openings are. Light and materials only show in the Cycles render.

    python3 tools/leaf_atlas.py textures/leaf textures/leaf_cards.json   # once
    python3 tools/preview_geometry.py --age 100 --out out/geo_100.png [--zoom 3 --centre 0.5 0.85]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))
import oak_camera as oc  # noqa: E402
import oak_growth as og  # noqa: E402
import oak_mesh as om  # noqa: E402

SUN = np.array([-40.0, -30.0, 27.0])
SUN = SUN / np.linalg.norm(SUN)
VIEW = np.array([0.0, 1.0, 0.0])


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--npz", type=Path, default=ROOT / "data/oak195.npz")
    p.add_argument("--cards", type=Path, default=ROOT / "textures/leaf_cards.json")
    p.add_argument("--age", type=float, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--width", type=int, default=1080)
    p.add_argument("--height", type=int, default=1440)
    p.add_argument("--zoom", type=float, default=1.0)
    p.add_argument("--centre", type=float, nargs=2, default=(0.5, 0.5), help="frame point kept in the middle when zoomed")
    p.add_argument("--leaf-scale", type=float, default=1.8)
    p.add_argument("--trunk-boost", type=float, default=3.0)
    p.add_argument("--leaves", type=int, default=om.LEAVES_AT_FULL_AGE)
    p.add_argument("--gaps", type=float, default=1.0)
    p.add_argument("--no-leaves", action="store_true")
    args = p.parse_args()

    cards = [om.LeafCard(**c) for c in json.loads(args.cards.read_text())]
    skel = og.straighten(og.load_npz(args.npz))
    pre = og.precompute(skel)
    look = dict(leaf_scale=args.leaf_scale, trunk_boost=args.trunk_boost, leaves=args.leaves, gaps=args.gaps)
    full_tubes, full_leaves, _ = om.build_age(skel, pre, og.FULL_AGE, cards, **look)
    framing = oc.solve(np.vstack([full_tubes.verts, full_leaves.verts]), args.width, args.height)
    tubes, leaves, tree = (full_tubes, full_leaves, None) if args.age == og.FULL_AGE else om.build_age(skel, pre, args.age, cards, **look)

    w, h, zoom = args.width, args.height, args.zoom
    off = np.array([w / 2 - args.centre[0] * w * zoom, h / 2 - args.centre[1] * h * zoom])

    def to_px(verts):
        x, y, depth = oc.project(verts, framing, w, h)
        return np.stack([x * zoom + off[0], y * zoom + off[1]], axis=-1), depth

    polys, depths, colours = [], [], []

    xy, depth = to_px(tubes.verts)
    q = tubes.quads
    v = tubes.verts[q]
    normal = np.cross(v[:, 1] - v[:, 0], v[:, 3] - v[:, 0])
    normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-12)
    if np.mean(np.sum(normal[: len(normal) // 50] * (v[: len(normal) // 50, 0] * [1, 1, 0]), axis=1)) < 0:
        normal = -normal
    front = (normal @ VIEW) < 0.05
    light = 0.38 + 0.75 * np.clip(normal @ SUN, 0, 1) + 0.12 * np.clip(normal[:, 2], 0, 1)
    old = np.clip((tubes.point_value[q[:, 0]] - 0.012) / 0.08, 0, 1)[:, None]
    bark = (np.array([92, 80, 66]) * (1 - old) + np.array([104, 84, 64]) * old) * light[:, None]
    polys.append(xy[q][front])
    depths.append(depth[q].mean(axis=1)[front])
    colours.append(bark[front])

    if len(leaves.quads) and not args.no_leaves:
        xy, depth = to_px(leaves.verts)
        q = leaves.quads
        v = leaves.verts[q]
        # A leaf, not its card: stalk, two shoulders, tip.
        stalk, tip = (xy[q[:, 0]] + xy[q[:, 1]]) / 2, (xy[q[:, 2]] + xy[q[:, 3]]) / 2
        lo_l, lo_r = xy[q[:, 0]] * 0.65 + xy[q[:, 3]] * 0.35, xy[q[:, 1]] * 0.65 + xy[q[:, 2]] * 0.35
        hi_l, hi_r = xy[q[:, 0]] * 0.25 + xy[q[:, 3]] * 0.75, xy[q[:, 1]] * 0.25 + xy[q[:, 2]] * 0.75
        shrink = lambda a, b: a * 0.85 + b * 0.15
        polys.append(np.stack([stalk, shrink(lo_r, lo_l), shrink(hi_r, hi_l), tip, shrink(hi_l, hi_r), shrink(lo_l, lo_r)], axis=1))
        depths.append(depth[q].mean(axis=1))
        normal = np.cross(v[:, 1] - v[:, 0], v[:, 3] - v[:, 0])
        normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-12)
        lit = np.abs(normal @ SUN)
        d, tint = leaves.extra["leaf_depth"][q[:, 0]], leaves.extra["leaf_tint"][q[:, 0]]
        rand = leaves.point_value[q[:, 0]]
        value = (0.55 + 0.6 * lit) * (1.0 - 0.55 * d) * (0.9 + 0.2 * rand)
        green = np.stack([58 + 50 * tint, 112 + 20 * tint, 34 - 8 * tint], axis=1) * value[:, None]
        colours.append(green)

    polys = np.concatenate([np.pad(k, ((0, 0), (0, 6 - k.shape[1]), (0, 0)), mode="edge") for k in polys])
    depths, colours = np.concatenate(depths), np.concatenate(colours)
    inside = ((polys[..., 0].max(1) > 0) & (polys[..., 0].min(1) < w) & (polys[..., 1].max(1) > 0) & (polys[..., 1].min(1) < h))
    polys, depths, colours = polys[inside], depths[inside], np.clip(colours[inside], 0, 255).astype(np.uint8)
    order = np.argsort(-depths)

    ss = 2
    img = Image.new("RGB", (w * ss, h * ss), (226, 232, 238))
    draw = ImageDraw.Draw(img)
    ground = h * (1 - oc.BOTTOM) * zoom + off[1]
    flat = (polys * ss).reshape(len(polys), -1)
    for i in order:
        draw.polygon(flat[i].tolist(), fill=tuple(colours[i]))
    draw.rectangle([0, ground * ss, w * ss, h * ss], fill=(206, 214, 204))
    img = img.resize((w, h), Image.LANCZOS)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    img.save(args.out)
    print(args.out, "tube quads", len(tubes.quads), "leaves", len(leaves.quads),
          "frame_h", round(framing.frame_h, 2), "scale", None if tree is None else round(tree.scale, 3))


if __name__ == "__main__":
    main()
