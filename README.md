# wS1 barrel registration (method D2)

Local registration of the mouse whisker barrel field (wS1, SSp-bfd) in cleared light-sheet brains.

A whole-brain registration gets the cortex roughly right but misplaces individual barrels, typically by half a barrel
to 1.5 barrels. This package finds the barrels in the brain's own L4 image and fits the Blue Brain barrel map onto
them. The output is:

- **Named barrel columns in sample space** (C2, D1, …; every layer of a barrel carries the barrel's name).
- **Allen CCF coordinates for every voxel in and around the barrel field.** These come from the barrel matches, not
  from the coarse registration.
- **Optionally, a table of detected cells in wS1 with their CCF position, barrel and layer.**

The coarse registration (brainreg to the Perens LSFM atlas at 20 µm, or to the Allen barrel atlas at 10 µm) is used
only to find the barrel field, the hemisphere, and the pia and white-matter surfaces.

**Documentation:** the [wiki](https://github.com/abisi/wS1_barrel_registration/wiki) has the full workflow, the review page with screenshots, every
metric, the methods and troubleshooting. Background notes: [D2_explained.md](docs/D2_explained.md) (pipeline with
figures), [methods_explained.md](docs/methods_explained.md) (the mathematics from first principles).

## How it works

Per hemisphere:

1. **Own flatmap.** Laplace depth between the registered pia and white matter → streamlines to L4 → flat coordinates
   on the L4 surface, following the brain's curvature.
2. **Template.** The atlas L4 barrels (33, A1–E8 and Alpha–Delta) in the Blue Brain flatmap of the CCF.
3. **L4 slab.** The L4/L5 boundary is found in the image per flat patch; the slab at the offset with the best band-pass
   contrast and the one best matching the barrel pattern are both fitted, and the fit with more matched barrels wins
   (optionally more offsets, chosen by barrel contrast: `BARRELREG_FIT_OFFSETS`, `BARRELREG_PICK=contrast`). Depth is
   measured below the closed image surface, so notches and tears are bridged.
4. **Search.** Every similarity (scale, rotation, shift) of the barrel pattern is scored against the slab.
5. **Matching.** CPD (affine, then non-rigid) → one-to-one matching → leave-one-out cleaning (optionally followed by a
   contrast refinement, `BARRELREG_CONTRAST_REFINE=1`).
6. **Warp.** A thin-plate spline through the matches.
7. **Back to 3D and CCF.** Each voxel takes the barrel whose flat convex hull contains it; depth is mapped per column
   (pia, L4 bottom, white matter); the inverse flatmap gives the CCF position.

When the automatic search fails, two fallbacks: **sharpened** images (unsharp mask, for blurred brains) and a
**seeded** fit (barrel centres clicked by hand on the review page replace the search). See
[Methods](https://github.com/abisi/wS1_barrel_registration/wiki/Methods).

## Workflow at a glance

```
first pass (wide ch0 / ch1, lesion mask where needed)  →  review page: decide per brain
   →  refits for the rest (sharpened, seeded)  →  review again  →  export (project folder + mouse Anatomy folders)
```

Every step is a script in this repository; no agent or account is needed. Step-by-step commands:
[Workflow](https://github.com/abisi/wS1_barrel_registration/wiki/Workflow).

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
| brainreg output per brain | `registered_atlas.tiff`, `registered_hemispheres.tiff`, `deformation_field_{0,1,2}.tiff`, downsampled ch0 (and ch1), at 20 or 10 µm. Any atlas with Allen region IDs works (tested with `perens_lsfm_mouse_20um` and `allen_mouse_bluebrain_barrels_10um`). |
| `allen_mouse_bluebrain_barrels_10um` | brainglobe atlas folder (needs `annotation.tiff`) |
| `flatmap_float.nrrd` | Blue Brain flatmap of the CCF, 10 µm |
| annotation of the registration atlas (`.npy` or `.tiff`) | only for a consistency check of the deformation fields, and for the cell-count comparison |
| `registration_params.csv` | one row per brain: `brain` plus the columns used in `dir_pattern` (see `examples/`), or a `reg_dir` column with the full path |
| cells (optional) | `<cells_dir>/<ID>/downsampled.points`, brainreg downsampled space |

## Configure

```bash
cp config.example.toml config.toml   # then edit the paths
```

`dir_pattern` locates each brain's brainreg folder; `{data_root}`, `{brain}` and any column of
`registration_params.csv` are filled in. Set `voxel_um` to the brainreg output resolution and `registration_atlas_um`
to the atlas resolution. `config.allen_AB_MH.toml` with `examples/registration_params_allen_AB_MH.csv` is a working
setup for the AB / MH brains (Allen barrel atlas, 10 µm).

## Run

```bash
python -m barrelreg -c config.toml convert-flatmap      # once: crop the flatmap (~5 min, ~10 GB RAM)
python -m barrelreg -c config.toml build-template       # once: template.npz
python -m barrelreg -c config.toml run MS261 MS262      # or no IDs = every brain; -j 8 in parallel; --hemisphere 1
python -m barrelreg -c config.toml summarize            # summary.csv + contact sheet
python -m barrelreg -c config.toml export               # overview figures + cells -> CCF
```

Several variants per brain (one folder each), as used for the AB / MH dataset:

```bash
scripts/run_variants.sh wide    BASE RUN/variants  <brains>            # wide search, ch0 and ch1
scripts/run_variants.sh lesion  BASE RUN/variants  <brains>            # + lesion mask
scripts/run_variants.sh sharpen BASE REDO/variants <brains>            # + unsharp mask x1, x2
scripts/run_variants.sh seeded  BASE REDO/variants --seeds seeds.json <brains>
```

A hemisphere takes about 9 minutes per variant and 15–30 GB RAM. `run --hemisphere 1` fits one hemisphere, and
`barrels_corrected_full.tif` and the CCF fields then contain that hemisphere only (the AB / MH runs are left only).
`run_variants.sh` uses the AB / MH settings (wide search, slab offsets −350…+150 µm chosen by contrast, contrast
refinement); plain `run` uses the package defaults (±20° / 0.7–1.4 / ±600 µm search, two slab offsets, most matches).

## Review and export

```bash
python scripts/review_gui.py --variants RUN/variants REDO/variants --config RUN/config.toml --out PAGE
python scripts/review_server.py PAGE                  # http://localhost:8000; decisions saved in PAGE/db/
python scripts/db_to_files.py PAGE/db --picks-csv picks.csv --seeds-json seeds.json
python scripts/export_user_picks.py --run RUN --variants REDO/variants --picks picks.csv --out FINAL --data-root DATA
```

The review page shows all fits of a brain on one canvas (overlay or flicker), their metrics, and the barrel each probe
track lands in; it records a decision per brain (use a fit / global is better / redo) and collects seed clicks. The
same page can also be published as a claude.ai page. See [Review page](https://github.com/abisi/wS1_barrel_registration/wiki/Review-page).

## Outputs

| File | Content |
|---|---|
| `<ID>/results/barrels_corrected_full.tif` | uint8 barrel index in sample space (names in `barrel_index.csv`) |
| `<ID>/results/ccf_field_{0,1,2}.tif` | corrected CCF position (mm, brainreg deformation-field convention); NaN outside the barrel columns |
| `<ID>/results/h<1\|2>/metrics.json` | fit quality and settings |
| `<ID>/report/` | flat fit, coronal and per-track figures; step-by-step report (`scripts/brain_report.py`, needs `BARRELREG_DEBUG=1`) |
| `FINAL/<ID>/registration_info.json` | the decision for this brain (`use_local` + variant, `use_global`, `redo`) and its metrics |
| `<data>/<mouse>/<session>/Anatomy/wS1_barrel_registration/` | the mouse's final fit, copied for analysis |

Hemispheres: brainglobe value 1 = left, 2 = right. Details: [Outputs and export](https://github.com/abisi/wS1_barrel_registration/wiki/Outputs-and-export).

## Quality control

- **`contrast_after` < 0.3: failed fit** (good fits ≥ 0.6).
- **`naming_ambiguous`: check by eye**; a fit shifted by one row can still score well.
- **`n_matched`, `err_tps_loo_um`**: more matches and lower leave-one-out error = better constrained.
- **`surface_bridged_frac` > 0.1**: damaged surface. **`field_check`** should be 1.000.
- Seeded brains: score fits against the clicks with `scripts/click_agreement.py`.

All fields: [Metrics](https://github.com/abisi/wS1_barrel_registration/wiki/Metrics).

## Code

| Module | Content |
|---|---|
| `register.py` | the per-brain pipeline; environment switches `BARRELREG_*` (channel, lesion mask, search, slab depths, sharpen, seeds, debug) |
| `template.py` | template and inverse flatmap |
| `flatmap.py` | Laplace depth, streamlines, flat grid, convex-hull barrel columns |
| `l4bottom.py` | L4 bottom (L4/L5 boundary) per flat column |
| `imageproc.py` | band-pass, tissue / artefact / lesion masks, barrel segmentation |
| `search.py` | synthetic pattern, exhaustive similarity search |
| `pointset.py` | CPD, matching, thin-plate splines, overlap and contrast metrics |
| `refine.py` | contrast refinement of the matched fit |
| `export.py`, `summarize.py`, `plotting.py` | cells → CCF, tables, figures |

| Script | Use |
|---|---|
| `run_variants.sh` | fit a set of variants (wide, lesion, sharpen, seeded) for some brains |
| `lesion_fraction.py`, `build_final.py` | which brains need the lesion variant; automatic choice among variants |
| `review_gui.py`, `review_gui.html`, `review_server.py` | build the review page; serve it locally with saving |
| `seed_images.py` | seed images (used by `review_gui.py --seed-brains`) |
| `db_to_files.py` | review-page decisions / seeds → picks CSV / seeds JSON |
| `click_agreement.py` | score fits against clicked seeds |
| `export_user_picks.py` | final project folder + Anatomy folders |
| `brain_report.py`, `paper_figure.py`, `explain_d2.py` | step-by-step report, figures |
| `map_channels_example.py`, `map_units_compare.py` | map probe channels / units through the final fits |
| `compare_outputs.py`, `compare_runs.py`, `edge_check.py` | checks between runs |
| `make_wiki_screenshots.py` | screenshots of the review page for the wiki |

## Provenance

This is method D2 from the local barrel registration project (Lana Smith, EPFL), packaged from
`/raid0/lmsmith/local_barrel_reg/D2_perens/` (snapshot of `run_d2.py` / `export.py` from 2026-10-30 11:22). The
functions come unchanged from methods A (point set), C (flatmap) and D (hybrid). Added since: 10 µm / Allen-registered
input, depth below the closed surface, white-matter depth as the deeper of two registration estimates, slab depth by
contrast, wide search, lesion mask, contrast refinement, sharpening, seeded fits, the review page and the user-picked
export. The method comparison and the validation on 24 Allen-registered brains are in
`/raid0/lmsmith/local_barrel_reg/SUMMARY.md`.
