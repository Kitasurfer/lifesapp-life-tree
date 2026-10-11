# LifesApp — Life Tree renders

Source of the «Древо жизни» frames in the LifesApp onboarding: one **real laser-scanned English oak**
(*Quercus robur*), grown backwards from its scan so every age shows the same individual tree, rendered
with Blender Cycles on GitHub Actions (nothing to install locally).

The app shows the frame for the user's age and, behind it, the full-age frame as a mint silhouette —
the tree they can still grow into.

## How a frame is made

1. `pipeline/oak_growth.py` — reverse growth of the scanned skeleton: young = trunk + main limbs of the
   same oak, finer twigs appear with age; height follows an oak height curve, stem thickness follows
   allometry + the pipe model.
2. `pipeline/oak_mesh.py` — bark tubes (UVs in metres of bark, lobed root flare, surface roots) and
   clumps of real-size oak-leaf cards at the growing tips.
3. `blender/render_age.py` — builds the scene from scratch: photoscanned bark and leaves, warm key
   light, mint (#4EEEB0) back light, transparent film + shadow catcher, one camera for all ages.
4. `tools/composite.py` — preview exactly as in the app (dark card, glow, silhouette).

## Run

Actions → **render** → *Run workflow* (inputs: ages, samples, resolution %, extra flags), or:

```bash
gh workflow run render -f ages=10,30,50,80 -f samples=64 -f percent=50
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
