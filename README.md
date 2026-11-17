# wS1 barrel registration (method D2)

Local registration of the mouse whisker barrel field (wS1, SSp-bfd) in cleared light-sheet brains.

A whole-brain registration gets the cortex roughly right but misplaces individual barrels, typically by half a barrel
to 1.5 barrels. This package finds the barrels in the brain's own autofluorescence channel (ch0) and fits the Blue
Brain barrel map onto them. The output is:

- **Named barrel columns in sample space** (C2, D1, …; every layer of a barrel carries the barrel's name).
- **Allen CCF coordinates for every voxel in and around the barrel field.** These come from the barrel matches, not
  from the coarse registration.
- **Optionally, a table of detected cells in wS1 with their CCF position, barrel and layer.**

The coarse registration (brainreg to the Perens LSFM atlas at 20 µm, or to the Allen barrel atlas at 10 µm) is used
only to find the barrel field, the hemisphere, and the pia and white-matter surfaces.

## How it works

Per hemisphere:

1. **Own flatmap.** A Laplace depth between the registered pia and white matter → streamlines to the L4 level →
   flat coordinates on the L4 surface (PCA plane, anterior / dorsal). The columns follow the brain's curvature.
2. **Fixed template.** The atlas L4 barrels drawn in the Blue Brain flatmap of the CCF, in the same plane convention.
   It is built once (`build-template`) and comes with an inverse lookup (flat position, depth) → CCF voxel.
3. **L4 slab.** Depth is measured below the *image* pia. The surface is first closed with a 400 µm ball, so notches,
   dents and tears in the surface are bridged and the slab does not dip into them (see "Surface damage" below).
   - The bottom of the barrel row (L4/L5 boundary) is found in ch0 for every column of the own flatmap. The detection
     is the same as in the brain-space depth analysis of the paper (`l4bottom.py`). Columns are grouped into 100 µm
     patches and pooled over 200 µm; in each pooled, detrended ch0 depth profile, the bottom is where the barrel peak
     has fallen halfway to the trough below it.
   - The slab is sampled at a fixed offset above that bottom, so it follows the brain's own L4.
   - Two candidate offsets are tried: best band-pass contrast, and best match to the barrel pattern. The full fit runs
     at both, and the offset with more matched barrels wins.
4. **Image barrels.** Artefacts (lesions, vessels) are masked, then band-pass → local maxima + watershed → barrel
   centroids.
5. **Search.** Every scale (0.7–1.4) × rotation (±20°) × shift (±600 µm) of the synthetic barrel pattern is scored
   against the slab; the top 5 placements are kept.
6. **Matching.** For each candidate: CPD affine → CPD non-rigid → one-to-one matching → re-pairing → leave-one-out
   cleaning.
   - Pattern correlation alone cannot tell a correct naming from one shifted by a row once the grid is pinned onto the
     blobs.
   - So among near-tied candidates (within 0.02), the one whose matches an affine map explains best wins.
7. **Warp.** A thin-plate spline through the matches, with smoothing chosen by leave-one-out. Outside the field it
   blends into a near-affine fit.
8. **Back to 3D.** Each voxel takes the template barrel whose convex hull in the flat plane contains its corrected
   flat coordinate. This is the column definition of Blue Brain's barrel annotations. Depth is mapped piecewise
   linearly per column:
   - image pia → atlas pia;
   - image L4 bottom → atlas L4 bottom (both per flat pixel);
   - brain white matter → atlas white matter.

   The brain's white matter is the deeper of two estimates from the coarse registration: the deepest registered
   isocortex below the image pia, and the registered cortical thickness. The thickness does not move when the
   registered cortex is shifted against the image (in MH064 left it is 300 µm too high), so the columns are not cut
   short by such a shift. Columns run from the image pia to that depth.
9. **CCF.** (Corrected flat coordinate, mapped depth) → inverse flatmap → CCF voxel. The Blue Brain flatmap covers one
   CCF hemisphere, so the other brain hemisphere is mirrored onto it (the annotation is symmetric).

A longer explanation with figures is in [docs/D2_explained.md](docs/D2_explained.md). The matching mathematics (search,
CPD, Hungarian, leave-one-out, thin-plate splines) is explained from scratch in
[docs/methods_explained.md](docs/methods_explained.md). It is also on the share, next to
the step-by-step figures of the closely related method D:
`/mnt/lsens-analysis/Lana_Smith/neuropixels/local_barrel_registration/D2_explained/` and `…/D_explained/`.
`scripts/explain_d2.py` redraws the D2 figures for any brain.

## Install

```bash
pip install -e .                 # numpy, scipy, scikit-image, pandas, tifffile, matplotlib, pillow, h5py
pip install -e ".[rawpoints]"    # only if some brains lack downsampled.points (adds brainglobe-space)
```

On the lab server, the existing environment already has everything:
`/raid0/lmsmith/local_barrel_reg/C_flatmap2d/venv/bin/python -m barrelreg …` (run from the repo folder).

## Inputs

| What | Where it comes from |
|---|---|
| brainreg output per brain | `registered_atlas.tiff`, `registered_hemispheres.tiff`, `deformation_field_{0,1,2}.tiff`, downsampled ch0, at 20 or 10 µm. Any atlas with Allen region IDs works (tested with `perens_lsfm_mouse_20um` and `allen_mouse_bluebrain_barrels_10um`). |
| `allen_mouse_bluebrain_barrels_10um` | brainglobe atlas folder (needs `annotation.tiff`) |
| `flatmap_float.nrrd` | Blue Brain flatmap of the CCF, 10 µm |
| annotation of the registration atlas (`.npy` or `.tiff`) | only for a consistency check of the deformation fields, and for the cell-count comparison |
| `registration_params.csv` | one row per brain: `brain` plus the columns used in `dir_pattern` (see `examples/`) |
| cells (optional) | `<cells_dir>/<ID>/downsampled.points`, brainreg downsampled space |

The structure table of the barrel atlas ships with the package (`barrelreg/data/structures.csv`).

## Configure

```bash
cp config.example.toml config.toml   # then edit the paths
```

`dir_pattern` locates each brain's brainreg folder. `{data_root}`, `{brain}` and any column of
`registration_params.csv` are filled in, so brains with different registration parameters can be mixed.

Set `voxel_um` to the brainreg output resolution (20 or 10 µm) and `registration_atlas_um` to the atlas resolution.
`config.allen_AB_MH.toml` with `examples/registration_params_allen_AB_MH.csv` is a working setup for brains registered
to `allen_mouse_bluebrain_barrels_10um` (the AB / MH brains). The template does not depend on the brains, so
`template.npz` and the flatmap crop can be copied between work folders.

## Run

```bash
python -m barrelreg -c config.toml convert-flatmap      # once: crop the flatmap (~5 min, ~10 GB RAM)
python -m barrelreg -c config.toml build-template       # once: template.npz (~2 min; rerun if it predates the L4 bottom)
python -m barrelreg -c config.toml run MS261 MS262      # or no IDs = every brain in registration_params.csv
python -m barrelreg -c config.toml run -j 8             # all brains, 8 in parallel (logs in work_dir/logs/)
python -m barrelreg -c config.toml summarize            # summary.csv + contact_sheet_D2.png
python -m barrelreg -c config.toml export               # overview figures + cells -> CCF
python -m barrelreg -c config.toml export --copy        # ... and copy the results to export_dir
```

A hemisphere takes about 9 minutes on the shared server (median over 42), so 15–20 minutes per brain. `run --hemisphere 1` refits a single hemisphere, but note that
`barrels_corrected_full.tif` and the CCF fields are rewritten with that hemisphere only.

## Outputs (`work_dir/`)

| File | Content |
|---|---|
| `<ID>/barrels_corrected_full.tif` | uint8 barrel index, same grid as the downsampled ch0 (both hemispheres). Names in `barrel_index.csv`: 1–29 = A1…E8, then Alpha, Beta, Delta, Gamma. |
| `<ID>/ccf_field_{0,1,2}.tif` | CCF position in mm, brainreg's deformation-field convention for the 10 µm Allen atlas (voxel = value × 100). NaN outside the barrel-field columns. |
| `<ID>/h<1\|2>/metrics.json` | fit quality (see QC) and parameters |
| `<ID>/h<1\|2>/flatcoords.npz` | flat coordinate of every working voxel (raw and corrected); L4 bottom (image and atlas) and white-matter depth per flat pixel |
| `<ID>/h<1\|2>/l4_bottom_patches.csv` | L4 bottom per 100 µm flat patch (raw = own peak, NaN where none was found) |
| `<ID>/h<1\|2>/<ID>_h<hv>_flat_fit.png` | flat L4 view: initial placement / matches / corrected barrels |
| `<ID>/<ID>_coronal.png`, `<ID>/<ID>_D2_overview.png` | coronal slices through C2; per-mouse overview |
| `<ID>/<ID>_wS1_cells_ccf.csv` | cells in SSp-bfd: CCF voxel / µm, structure, barrel, layer, `fit_ok` |
| `summary.csv`, `contact_sheet_D2.png` | one row / tile per hemisphere |
| `all_mice_wS1_cells_ccf.csv`, `wS1_cell_counts.csv` | cells over all exported mice |

**Hemispheres:** brainglobe value 1 = left (high axis 2), 2 = right.

## Quality control

Check every hemisphere in `summary.csv` and the contact sheet:

- **`contrast_after < 0.3`: failed fit.** The template was not placed on the barrels. Do not use it; export sets
  `fit_ok = False` for its cells. Good fits are typically ≥ 0.6.
- **`naming_ambiguous = True`: check by eye.** Another candidate with a different naming scored almost as well
  (margin < 0.02). The affine tie-break decided between them.
- **`jac_min > 0`: no folding.** Always true so far.
- **`ccf_label_agreement`: tests the inverse lookup, not the names.** It is the fraction of labelled voxels whose
  mapped CCF voxel carries the same barrel (typically 0.92–0.94).
- **`wm_registered_below_pia_um` vs `wm_registered_thickness_um`:** a large difference means the registered cortex is
  shifted against the image surface; the deeper one is used (`brain_wm_depth_um`).
- **`l4_bottom_found_frac`: L4 bottom coverage.** The fraction of flat patches with their own barrel peak; the rest are
  filled from neighbours. `l4_bottom_um` / `atlas_l4_bottom_um` are the medians over the barrel field, and
  `l4_offset_um` is the slab position relative to the bottom.
- **`field_check`: should be 1.000.** It confirms the deformation fields and the registered atlas agree.
- **`surface_bridged_frac`: surface damage.** The fraction of the L4 band whose depth changed by more than 30 µm when
  the surface was closed. Intact hemispheres are around 0.02–0.06. Above ~0.1 the surface is notched or torn in the
  barrel field; look at the coronal figure.

### Surface damage

Depth below the pia is the distance to the image's tissue edge. Taken as the distance to the *nearest* edge, a notch or
tear in the surface pulls everything near it upwards: the L4 slab and the layer boundaries bow into the damage, and the
columns under it get the wrong layers. D2 therefore measures depth from the surface after a morphological closing with
a 400 µm ball (`SURF_CLOSE_UM` in `register.py`), i.e. from where the surface would be. Notches and tears up to roughly
0.8 mm across are bridged; the convex surface of intact cortex does not move. This assumes the tissue in a notch was
torn away, not pressed down. Missing tissue stays missing: the columns there simply start lower, with the correct
layer.

Known limits:
- Barrel names come from the matching criterion; there is no independent ground truth. The best check is whisker
  responses on recording tracks.
- CCF coordinates exist only in and around the barrel field.
- Lesions, tears or a coarse registration more than ~600 µm off can make a fit fail. These show up as
  `contrast_after < 0.3`.
- Surface damage wider than ~0.8 mm is not bridged.

## Step-by-step report per brain

```bash
BARRELREG_DEBUG=1 python -m barrelreg -c config.toml run MS261 --hemisphere 1   # also saves h1/debug.npz + debug3d.npz
python scripts/brain_report.py -c config.toml MS261 --tracks tracks_summary.csv  # work_dir/reports/<ID>_h1_D2_report.pdf
```

`BARRELREG_DEBUG=1` saves every intermediate step of each hemisphere (about 250 MB per hemisphere; the fit is
unchanged). The report has one page per step on that brain's own data: inputs, own flatmap, template, depth and L4
bottom, slab choice and image barrels, the search landscape and top placements, the point matching of every candidate,
the warp, the 3D columns and depth mapping, and, with `--tracks`, each Neuropixels wS1 track with its barrels along the
probe under brainreg and under D2.

## Check against a previous run

```bash
python scripts/compare_outputs.py <work_dir_a> <work_dir_b> MS261
```

## Code

| Module | Content |
|---|---|
| `register.py` | the per-brain pipeline (steps 1 and 3–9) |
| `template.py` | template and inverse flatmap (step 2) |
| `flatmap.py` | Laplace depth, streamlines, plane frame, flat-grid binning, convex-hull barrel columns, flatmap conversion |
| `l4bottom.py` | L4 bottom (L4/L5 boundary) per flat column from ch0 |
| `imageproc.py` | band-pass, tissue / artefact masks, barrel segmentation |
| `search.py` | synthetic pattern, exhaustive similarity search |
| `pointset.py` | CPD, matching, thin-plate splines, overlap metrics |
| `export.py`, `summarize.py`, `plotting.py` | cells → CCF, tables, figures |
| `atlas.py`, `config.py` | structure tables, paths |

## Provenance

This is method D2 from the local barrel registration project (Lana Smith, EPFL), packaged from
`/raid0/lmsmith/local_barrel_reg/D2_perens/` (snapshot of `run_d2.py` / `export.py` from 2026-10-30 11:22). The
functions come unchanged from methods A (point set), C (flatmap) and D (hybrid). Added since: 10 µm / Allen-registered
input, depth below the closed surface (surface damage), and the white-matter depth as the deeper of the two
registration estimates. The method comparison and the
validation on 24 Allen-registered brains are in `/raid0/lmsmith/local_barrel_reg/SUMMARY.md`.
