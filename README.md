# LifesApp — Life Tree renders

Source of the «Древо жизни» frames in the LifesApp onboarding: one **real laser-scanned English oak**
(*Quercus robur*), grown backwards from its scan so every age shows the same individual tree, rendered
with Blender Cycles on GitHub Actions (nothing to install locally).

The app shows the frame for the user's age and, behind it, the full-age frame as a mint silhouette —
the tree they can still grow into.

## How a frame is made

1. `pipeline/oak_growth.py` — reverse growth of the scanned skeleton: young = trunk + main limbs of the
   same oak, finer twigs appear with age; height follows an oak height curve, stem thickness follows
   allometry + the pipe model. Twigs that carry leaves at an age exist at that age.
2. `pipeline/oak_mesh.py` — bark tubes and leaves. The stem is a veteran's: a bole that holds its girth
   to the main fork and keeps thickening after the height has stopped, an irregular section, a root
   flare of a few uneven buttresses (the stem's own surface, no root pipes), limbs leaving through
   collars. Leaves are oak-leaf cards in rosettes on shoots at the scan's real twig ends; the shaded
   interior and low-frequency pockets carry little, so foliage gathers into pads with sky and limbs
   between them. One fixed set of leaves serves every age — a leaf at 40 is the same leaf at 80.
3. `blender/render_age.py` — builds the scene from scratch: photoscanned bark with furrows displaced
   into old wood, leaves toned by their depth in the crown (paler underside, light shining through),
   sun + graded sky dome + meadow bounce, transparent film cut at the ground, one camera for all ages
   (`pipeline/oak_camera.py`).
4. `tools/composite.py` — preview exactly as in the app (dark card, glow, silhouette).
5. `tools/preview_geometry.py` — the geometry of one age in seconds, without Blender (silhouette, stem,
   where the foliage gathers); light and materials only show in the Cycles render.

## Run

Actions → **render** → *Run workflow* (inputs: ages, samples, resolution %, extra flags), or:

```bash
gh workflow run render -f ages=20,40,60,80 -f samples=128 -f percent=100
gh run download --name preview
```

Each age renders as its own job in parallel; artifacts: `frame-<age>` (RGBA PNG + JSON stats) and
`preview` (app-style composites + contact sheet).

## Data and licences

| Asset | Source | Licence |
|-------|--------|---------|
| Oak skeleton `data/oak195.npz` (tree 195, derived from the TreeQSM cylinder model) | Wytham Woods terrestrial laser scanning, [Zenodo 7307956](https://zenodo.org/records/7307956) | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |
| Bark: *Jolcham Oak Bark 01* | [Poly Haven](https://polyhaven.com/a/jolcham_oak_bark_01) | CC0 |
| Leaves: *LeafSet016* (oak) | [ambientCG](https://ambientcg.com/view?id=LeafSet016) | CC0 |

Oak skeleton © Calders, K., Verbeeck, H., Burt, A., Origo, N., Nightingale, J., Malhi, Y., Wilkes, P.,
Raumonen, P., Bunce, R. G. H., Disney, M. (2022). *Laser scanning reveals potential underestimation of
biomass carbon in temperate forest.* Ecological Solutions and Evidence. Used under CC BY 4.0; the tree
was rotated (azimuth 45°) and converted to `.npz`; all growth stages are derived renders.

Textures are downloaded by the workflow and are not stored in this repository.
