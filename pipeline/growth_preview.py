#!/usr/bin/env python3
"""2D check of the reverse-growth model before any Blender work.

One fixed camera for every age (like the final renders): the full-age tree is drawn as a faint mint
silhouette, the tree at `age` on top in bark/leaf colours.

    python -I growth_preview.py --mat data/wytham/....mat --out strip.png 5 15 30 50 80 100
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))  # run with -I: sibling module only
import oak_growth as og  # noqa: E402

H = 900
MINT = (78, 238, 176)


def draw_tree(draw: ImageDraw.ImageDraw, skel: og.Skeleton, tree: og.AgedTree, px: float, cx: float,
              colour, leaf=None) -> None:
    to_px = lambda p: (cx + p[0] * px, H - 20 - p[2] * px)
    chains = sorted(og.branch_chains(skel, tree), key=lambda c: -c.radii[0])  # thick first
    for chain in chains:
        for a, b, r in zip(chain.points[:-1], chain.points[1:], chain.radii[1:]):
            draw.line([to_px(a), to_px(b)], fill=colour, width=max(1, int(round(2 * r * px))), joint="curve")
    if leaf:
        r = max(1.2, 0.18 * px * tree.scale)
        for chain in chains:
            if chain.ends_in_tip:
                x, y = to_px(chain.points[-1])
                draw.ellipse([x - r, y - r, x + r, y + r], fill=leaf)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mat", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--azimuth", type=float, default=0.0, help="camera azimuth, degrees")
    parser.add_argument("ages", nargs="+", type=float)
    args = parser.parse_args()

    skel = og.rotate_z(og.load_qsm_mat(args.mat), args.azimuth)
    pre = og.precompute(skel)
    full = og.grow(skel, pre, og.FULL_AGE)
    xs = np.concatenate([full.start[:, 0], full.end[:, 0]])
    height = full.end[:, 2].max()
    px = (H - 40) / height
    width = int((xs.max() - xs.min()) * px) + 40
    cx = 20 - xs.min() * px

    tiles = []
    for age in args.ages:
        img = Image.new("RGB", (width, H), (5, 8, 7))
        d = ImageDraw.Draw(img, "RGBA")
        draw_tree(d, skel, full, px, cx, MINT + (40,))
        tree = og.grow(skel, pre, age)
        draw_tree(d, skel, tree, px, cx, (150, 118, 88, 255), leaf=(96, 170, 70, 200))
        d.text((10, 10), f"age {age:g}  scale {tree.scale:.2f}  cyl {len(tree.radius)}  tips {int(tree.is_tip.sum())}",
               fill=(230, 230, 230))
        tiles.append(img)
    sheet = Image.new("RGB", (width * len(tiles), H))
    for i, t in enumerate(tiles):
        sheet.paste(t, (i * width, 0))
    sheet.save(args.out)
    print(args.out, sheet.size)


if __name__ == "__main__":
    main()
