# Changelog

Changes in this fork (abisi/wS1_barrel_registration) relative to the source repository
[LanaSmith1313/wS1_barrel_registration](https://github.com/LanaSmith1313/wS1_barrel_registration) (base `9f5cbc4`,
2026-11-05). The source has no commits after the base, so the history is linear.

## Breaking changes

- **Output layout.** Fit outputs go to `<work_dir>/<ID>/results/` (`barrels_corrected_full.tif`, `ccf_field_*.tif`,
  `h<n>/`) and figures to `<ID>/report/`, instead of everything in `<ID>/`. The package and scripts still read the old
  layout; external code that reads `<ID>/barrels_corrected_full.tif` must use `<ID>/results/`.
- **`metrics.json`**: `contrast_after_ch<c>` per image channel and `contrast_both` (mean over the channels present)
  are added; consumers that assumed exactly the source's fields are unaffected, but `contrast_both` is a mean over all
  channels, not only ch0 and ch1.

## Defaults

Running `python -m barrelreg run` without any `BARRELREG_*` variable gives the source's fit: search ±20°, scale
0.7–1.4, shift ±600 µm, 5 candidates; slab offsets −300…+50 µm; the contrast and pattern depths fitted, most matched
barrels wins; no refinement. Differences with the defaults:

- every fit is also scored on every other image channel present (extra metrics, no change to the fit);
- input volumes are opened read-only (the source opened them writable, which risks modifying brainreg files);
- a degenerate point match at an extreme depth skips that candidate instead of stopping the hemisphere;
- a brain without the requested channel stops with a clear error; a brain with a single channel is fitted and scored
  on it (the 2026-11-17 to 2026-11-22 versions required a ch1 file).

## New options (environment variables, off by default)

| Variable | Effect |
|---|---|
| `BARRELREG_SEARCH=wide` | search ±30°, scale 0.6–1.5, shift ±800 µm, 8 candidates, each polished on a fine local grid |
| `BARRELREG_OFFSETS_RANGE=lo,hi` | slab offsets searched (default −300,50) |
| `BARRELREG_FIT_OFFSETS=a,b,…` | slab offsets fitted in addition to the contrast / pattern picks |
| `BARRELREG_PICK=contrast` | choose the fitted depth by barrel contrast instead of most matched barrels |
| `BARRELREG_CH=<c>` | fit on image channel c (default 0) |
| `BARRELREG_LESION=1` | mask large lesions and their rims (`imageproc.lesion_mask`) |
| `BARRELREG_CONTRAST_REFINE=1` | after the matched fit, re-rank placements by contrast and move the barrels to maximise it (`refine.py`); kept when it gains > 0.02 |
| `BARRELREG_SHARPEN=a[,σ]` | unsharp mask on every channel (σ default 60 µm) |
| `BARRELREG_SEEDS=file.json` | seeded fit: the similarity through clicked barrel centres replaces the search |
| `BARRELREG_NEAR_UM`, `BARRELREG_MARGIN_UM` | larger flat area / crop around the barrel field |
| `BARRELREG_DEBUG=1` | save every intermediate step (`debug.npz`, `debug3d.npz`) |

New `metrics.json` fields: `channels`, `contrast_after_ch<c>`, `contrast_both`, `depth_fits`, `depth_pick`,
`lesion_mask`, `near_um`, `margin_um`, `sharpen`, `seeded`, `seed_barrels`, `seed_residual_um`, `contrast_refine`,
`refined`, `contrast_matched_fit`, `contrast_refined_fit`, `refine_placement`.

## New scripts

| Script | Use |
|---|---|
| `run_variants.sh` | fit a set of variants (wide, lesion, sharpen, seeded) on every channel present |
| `lesion_fraction.py`, `build_final.py` | which brains need the lesion variant; provisional choice among variants |
| `review_gui.py`, `review_gui.html`, `review_server.py` | review page: compare fits, decide, click seeds; local server saving JSON (no agent needed) |
| `seed_images.py`, `db_to_files.py`, `click_agreement.py` | seed images; decisions / seeds to files; fits scored against clicks |
| `export_user_picks.py` | final project folder and per-mouse Anatomy folders |
| `brain_report.py`, `paper_figure.py`, `explain_d2.py`, `export_pick_figures.py` | step-by-step reports and figures |
| `variant_gallery.py`, `slab_gallery.py`, `compare_runs.py`, `compare_figures.py`, `rescue_report.py`, `edge_check.py`, `accept_fit.py` | inspection and run comparison |
| `map_channels_example.py`, `map_units_compare.py` | map probe channels / units through the final fits |
| `migrate_layout.sh` | move old runs to the results / report layout |
| `make_wiki_screenshots.py` | screenshots for the wiki |

## Documentation

README rewritten as an overview; detailed documentation in the
[GitHub wiki](https://github.com/abisi/wS1_barrel_registration/wiki) (workflow, review page, metrics, methods,
outputs, troubleshooting).

## AB / MH dataset (2026-11)

101 brains, left hemisphere only (`--hemisphere 1`), fitted with `run_variants.sh` settings (wide search, slab
offsets −350…+150 µm picked by contrast, contrast refinement), sharpened and seeded refits for difficult brains, and
the final fit chosen per brain by eye on the review page.
