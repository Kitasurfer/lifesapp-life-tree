#!/usr/bin/env python3
"""TreeQSM .mat → skeleton .npz that Blender's bundled Python can read (it has numpy, not scipy).

The azimuth is baked in here, so the Blender scene's camera always looks along +Y.

    python -I qsm_to_npz.py --azimuth 45 data/wytham/wytham_winter_195-....mat oak195.npz
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # run with -I: sibling module only
import oak_growth as og  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--azimuth", type=float, default=0.0)
    parser.add_argument("mat", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    skel = og.rotate_z(og.load_qsm_mat(args.mat), args.azimuth)
    og.save_npz(skel, args.out)
    height = float((skel.start + skel.axis * skel.length[:, None])[:, 2].max())
    print(f"{args.out}: {len(skel.length)} cylinders, {skel.branch.max() + 1} branches, height {height:.1f} m")


if __name__ == "__main__":
    main()
