#!/usr/bin/env python3
"""Cut the leaf atlas into cards: one UV box per leaf + where its stalk is.

Reads the atlas opacity map, labels the opaque islands, keeps the large ones (the leaves), and
writes JSON consumed by blender/render_age.py. The stalk is the lowest opaque pixel of each island
(atlas leaves point up, stalk down).

    python3 leaf_atlas.py textures/leaf textures/leaf_cards.json
"""
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage


def main() -> None:
    leaf_dir, out = Path(sys.argv[1]), Path(sys.argv[2])
    opacity_path = next(p for p in leaf_dir.iterdir() if "Opacity" in p.name)
    alpha = np.asarray(Image.open(opacity_path).convert("L"), dtype=np.float32) / 255.0
    h, w = alpha.shape
    labels, count = ndimage.label(alpha > 0.5)
    sizes = ndimage.sum(np.ones_like(alpha), labels, index=np.arange(1, count + 1))
    keep = [i + 1 for i in np.argsort(-sizes) if sizes[i] > 0.15 * sizes.max()]

    cards = []
    for lab in keep:
        ys, xs = np.nonzero(labels == lab)
        pad = 4
        x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad + 1, w)
        y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad + 1, h)
        stem_row = ys.max()
        stem_col = int(np.median(xs[ys >= stem_row - 2]))
        cards.append({
            # Blender UV: v = 0 at the bottom of the image.
            "u0": x0 / w, "u1": x1 / w, "v0": 1 - y1 / h, "v1": 1 - y0 / h,
            "stem_x": (stem_col - x0) / (x1 - x0),
            "stem_y": (y1 - stem_row) / (y1 - y0),
            "aspect": (x1 - x0) / (y1 - y0),
        })
    out.write_text(json.dumps(cards, indent=1))
    print(f"{len(cards)} leaf cards from {opacity_path.name} ({w}x{h})")


if __name__ == "__main__":
    main()
