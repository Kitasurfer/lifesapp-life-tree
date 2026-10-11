#!/usr/bin/env python3
"""Fetch individual oak trees from the Wytham Woods TLS dataset (Zenodo 7307956, CC BY 4.0).

The full dataset is 42.8 GB; this pulls only the requested trees out of the remote zips with
HTTP range requests (remotezip), so a few hundred MB land on disk instead.

Calders, K. et al. (2022) "Laser scanning reveals potential underestimation of biomass carbon in
temperate forest", Ecological Solutions and Evidence. Data: https://zenodo.org/records/7307956

Usage (venv with remotezip, outside the repo):
    python -I fetch_wytham.py --out ~/LifesAppArt/life-tree/data/wytham 6 195 12
"""
import argparse
import re
import sys
from pathlib import Path

from remotezip import RemoteZip

RECORD = "https://zenodo.org/api/records/7307956/files/%s/content"
CLOUDS = "DATA_clouds_ply.zip"
QSMS = "DATA_QSM_opt.zip"


def fetch(zip_name: str, pattern: re.Pattern, out_dir: Path, first_only: bool) -> list[Path]:
    saved = []
    with RemoteZip(RECORD % zip_name) as archive:
        matches = sorted(i.filename for i in archive.infolist() if pattern.match(i.filename))
        for name in matches[:1] if first_only else matches:
            target = out_dir / Path(name).name
            if not target.exists():
                target.write_bytes(archive.read(name))
            saved.append(target)
    return saved


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("tree_ids", nargs="+")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    for tree_id in args.tree_ids:
        cloud = re.compile(rf"^DATA_clouds_ply/wytham_winter_{re.escape(tree_id)}\.ply$")
        # Ten QSM replicates per tree; one is enough for the branch skeleton.
        qsm = re.compile(rf"^DATA_QSM_opt/wytham_winter_{re.escape(tree_id)}-[\d.\-]+\.mat$")
        files = fetch(CLOUDS, cloud, args.out, first_only=True) + fetch(QSMS, qsm, args.out, first_only=True)
        if len(files) != 2:
            print(f"tree {tree_id}: expected cloud + QSM, got {files}", file=sys.stderr)
            return 1
        print(tree_id, *(f"{p.name} ({p.stat().st_size // 1_000_000} MB)" for p in files))
    return 0


if __name__ == "__main__":
    sys.exit(main())
