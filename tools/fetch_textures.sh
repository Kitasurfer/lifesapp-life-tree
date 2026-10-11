#!/usr/bin/env bash
# CC0 photoscans used by the render (downloaded on the CI runner, never committed):
#   bark   — Poly Haven "Jolcham Oak Bark 01" 2K   https://polyhaven.com/a/jolcham_oak_bark_01
#   leaves — ambientCG "LeafSet016" (green oak) 2K  https://ambientcg.com/view?id=LeafSet016
set -euo pipefail
out="${1:?usage: fetch_textures.sh <dir>}"
mkdir -p "$out/bark" "$out/leaf"

ph="https://dl.polyhaven.org/file/ph-assets/Textures/jpg/2k/jolcham_oak_bark_01"
for map in diff rough nor_gl disp; do
  f="$out/bark/jolcham_oak_bark_01_${map}_2k.jpg"
  [ -s "$f" ] || curl -fsSL --retry 3 -o "$f" "$ph/jolcham_oak_bark_01_${map}_2k.jpg"
done

if ! ls "$out"/leaf/*Color* >/dev/null 2>&1; then
  curl -fsSL --retry 3 -o "$out/leaf/leafset.zip" "https://ambientcg.com/get?file=LeafSet016_2K-JPG.zip"
  unzip -oq "$out/leaf/leafset.zip" -d "$out/leaf"
  rm "$out/leaf/leafset.zip"
fi
ls -la "$out/bark" "$out/leaf"
