#!/usr/bin/env python3
"""Contact sheet of candidate trees: front + side projections of each TLS point cloud.

Used once to pick which scanned oak becomes the Life Tree; needs numpy + Pillow.

    python -I preview_clouds.py --out sheet.png data/wytham/wytham_winter_6.ply ...
"""
import argparse
import re
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

VIEW_H = 820
MINT = np.array([78, 238, 176], dtype=np.float32)  # #4EEEB0


def read_ply_xyz(path: Path) -> np.ndarray:
    raw = path.read_bytes()
    header_end = raw.index(b"end_header\n") + len(b"end_header\n")
    header = raw[:header_end].decode("ascii")
    count = int(re.search(r"element vertex (\d+)", header).group(1))
    if "binary_little_endian" not in header or header.count("property float") != 3:
        raise ValueError(f"{path.name}: expected binary float x/y/z")
    return np.frombuffer(raw, dtype="<f4", count=count * 3, offset=header_end).reshape(-1, 3)


def splat(u: np.ndarray, v: np.ndarray, scale: float, width: int) -> Image.Image:
    density = np.zeros((VIEW_H, width), dtype=np.float32)
    x = np.clip((u * scale + width / 2).astype(int), 0, width - 1)
    y = np.clip((VIEW_H - 1 - v * scale).astype(int), 0, VIEW_H - 1)
    np.add.at(density, (y, x), 1.0)
    tone = np.log1p(density) / max(np.log1p(density).max(), 1e-6)
    rgb = (tone[..., None] ** 0.6) * MINT
    return Image.fromarray(rgb.clip(0, 255).astype(np.uint8))


def render_tree(path: Path, azimuth: float = 0.0) -> Image.Image:
    pts = read_ply_xyz(path).astype(np.float64)
    ground = np.percentile(pts[:, 2], 0.5)
    base = pts[pts[:, 2] < ground + 1.3]  # trunk slice ~breast height → stem centre
    cx, cy = np.median(base[:, 0]), np.median(base[:, 1])
    a = np.radians(azimuth)
    x0, y0, z = pts[:, 0] - cx, pts[:, 1] - cy, pts[:, 2] - ground
    x, y = x0 * np.cos(a) - y0 * np.sin(a), x0 * np.sin(a) + y0 * np.cos(a)
    height = np.percentile(z, 99.9)
    scale = (VIEW_H * 0.92) / height
    reach = max(np.percentile(np.abs(x), 99.9), np.percentile(np.abs(y), 99.9))
    width = int(2 * reach * scale) + 40
    front, side = splat(x, z, scale, width), splat(y, z, scale, width)
    tile = Image.new("RGB", (width * 2 + 12, VIEW_H + 40), "black")
    tile.paste(front, (0, 40))
    tile.paste(side, (width + 12, 40))
    label = re.search(r"wytham_winter_(\w+)\.ply", path.name).group(1)
    ImageDraw.Draw(tile).text((10, 10), f"oak #{label}  h={height:.1f} m  azimuth {azimuth:g} | {azimuth + 90:g}", fill=(230, 230, 230))
    return tile


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--azimuths", nargs="+", type=float, default=[0.0],
                        help="camera azimuths in degrees; each tile shows azimuth and azimuth+90")
    parser.add_argument("clouds", nargs="+", type=Path)
    args = parser.parse_args()
    tiles = [render_tree(p, az) for p in args.clouds for az in args.azimuths]
    cols = 2
    rows = (len(tiles) + cols - 1) // cols
    tw = max(t.size[0] for t in tiles)
    th = max(t.size[1] for t in tiles)
    sheet = Image.new("RGB", (tw * cols, th * rows), "black")
    for i, tile in enumerate(tiles):
        sheet.paste(tile, ((i % cols) * tw, (i // cols) * th))
    sheet.save(args.out)
    print(args.out, sheet.size)


if __name__ == "__main__":
    main()
