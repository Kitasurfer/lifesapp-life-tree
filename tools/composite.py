#!/usr/bin/env python3
"""Preview the frames the way the app will show them.

Each age: OLED-black card + soft mint glow, the full-age frame as a faint blurred mint silhouette
("where you can grow"), the age frame on top. Also writes a labelled contact sheet.

    python3 composite.py <frames dir with oak_XXX.png> <out dir>
"""
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

MINT = np.array([78, 238, 176], dtype=np.float32)


def load(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGBA"), dtype=np.float32) / 255.0


def compose(frame: np.ndarray, silhouette_alpha: np.ndarray) -> Image.Image:
    h, w = frame.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    d = np.hypot((xx - w * 0.5) / w, (yy - h * 0.45) / h)
    glow = np.clip(1 - d / 0.6, 0, 1) ** 2 * 0.22
    rgb = np.zeros((h, w, 3), np.float32) + glow[..., None] * MINT / 255.0

    sil = Image.fromarray((silhouette_alpha * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(max(2, w // 300)))
    sil_a = np.asarray(sil, np.float32) / 255.0 * 0.24
    rgb = rgb * (1 - sil_a[..., None]) + (MINT / 255.0) * sil_a[..., None]

    a = frame[..., 3:4]
    rgb = rgb * (1 - a) + frame[..., :3] * a
    return Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8))


def main() -> None:
    frames_dir, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    frames = sorted(frames_dir.rglob("oak_*.png"), key=lambda p: int(re.search(r"oak_(\d+)", p.name).group(1)))
    full = next((p for p in frames if p.name == "oak_100.png"), frames[-1])
    silhouette = load(full)[..., 3]

    tiles = []
    for p in frames:
        img = compose(load(p), silhouette)
        img.save(out / f"app_{p.stem}.png")
        age = int(re.search(r"oak_(\d+)", p.name).group(1))
        ImageDraw.Draw(img).text((14, 12), f"{age}", fill=(230, 230, 230))
        tiles.append(img)
    w, h = tiles[0].size
    sheet = Image.new("RGB", (w * len(tiles), h))
    for i, t in enumerate(tiles):
        sheet.paste(t, (i * w, 0))
    sheet.save(out / "sheet.png")
    print(out / "sheet.png", sheet.size)


if __name__ == "__main__":
    main()
