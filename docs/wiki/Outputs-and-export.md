# Outputs and export

## One fit

Each fit lives in `<variants>/<variant>/<brain>/`:

| File | Content |
|---|---|
| `results/barrels_corrected_full.tif` | uint8 barrel index per voxel, brainreg downsampled sample space (same grid as the downsampled ch0). Names in `barrel_index.csv`: 1–29 = A1…E8, then Alpha, Beta, Delta, Gamma; 0 = outside. |
| `results/ccf_field_{0,1,2}.tif` | corrected Allen CCF position of every voxel in mm, brainreg deformation-field convention (10 µm CCF voxel = value × 100); NaN outside the barrel-field columns |
| `results/h1/metrics.json` | fit quality and settings ([Metrics](Metrics.md)) |
| `results/h1/flatcoords.npz` | flat coordinate of every working voxel (raw and corrected), L4 bottom and white-matter depth per flat pixel |
| `results/h1/l4_bottom_patches.csv` | L4 bottom per 100 µm flat patch |
| `results/h1/debug.npz`, `debug3d.npz` | every intermediate step (with `BARRELREG_DEBUG=1`; used by the review page and reports; not exported) |
| `report/` | flat fit and coronal figures, step-by-step report PDF, per-track paper figures |

## Final export

`scripts/export_user_picks.py` turns the decisions into two copies.

### 1. Project folder (`--out`, e.g. `D2_final_user_picked/`)

One folder per **brain**:

- `use_local`: `results/` (the files above except the debug dumps), `report/`, `registration_info.json`.
- `use_global` / `redo`: `registration_info.json` only.

Plus `final_registration_table.csv` (one row per brain), `anatomy_export.csv` (one row per session folder written),
the picks CSV it was built from, and `README.txt`.

`registration_info.json`:

```json
{
 "brain": "AB077", "mouse_id": "AB077", "hemisphere": 1,
 "decision": "use_local", "note": "", "picked_variant": "wide_ch1", "auto_variant": "wide_ch1",
 "picked_at": "2026-11-20T11:37:15.498Z", "exported": "2026-11-20", "run": "D2_all_brains",
 "source_dir": ".../D2_all_brains/variants/wide_ch1/AB077",
 "contrast_after": 0.83, "contrast_both": 0.83, "n_matched": 24, "err_tps_loo_um": 45.5, "err_initial_um": 105.7,
 "l4_offset_um": -100, "naming_ambiguous": false, "ccf_label_agreement": 0.90, "image_channel": 1, "lesion_mask": false
}
```

### 2. Anatomy folders (`--data-root`)

One brain per **mouse**, copied into every session folder of the mouse that has an `Anatomy` folder:
`<data-root>/<mouse>/<session>/Anatomy/wS1_barrel_registration/`

- Files: `barrels_corrected_full.tif`, `ccf_field_{0,1,2}.tif`, `barrel_index.csv`, `metrics.json`,
  `l4_bottom_patches.csv`, report images, `registration_info.json` (with `other_brains` and `project_dir`),
  `README.txt`.
- Several brains of one mouse (e.g. `MH038` and `MH038cmdii`): the `use_local` brain is used (highest contrast if
  several); otherwise `registration_info.json` says `use_global` (or `redo`).
- Mice without an Anatomy folder are listed as `NO ANATOMY FOLDER` in `anatomy_export.csv`.

Run with `--dry-run` first: it prints the decision counts and the mice with several brains, writes nothing.
Rerunning overwrites `wS1_barrel_registration/` in every listed session folder.

## Using the results

For a unit or channel with a global CCF position (e.g. NWB `ccf_atlas_{ap,dv,ml}`, µm on the 10 µm grid):

1. Read the session's `registration_info.json`. `use_global` → keep the global labels.
2. `use_local`: every sample voxel near the barrel field has a global CCF position (brainreg
   `deformation_field_{0,1,2}` × 100) and a corrected one (`ccf_field_{0,1,2}` × 100). A k-d tree on the global
   positions finds the unit's sample voxel; there read the corrected CCF position, the barrel
   (`barrels_corrected_full.tif` + `barrel_index.csv`) and the layer (structure of the barrel atlas at the corrected
   CCF voxel). Outside the barrel columns `ccf_field` is NaN: keep the global position.

Worked examples: `scripts/map_channels_example.py` (channels of a few probes, global vs refined plots) and
`scripts/map_units_compare.py` (every unit in the barrel columns over all brains, keyed by `mouse_id`, `session_id`,
`electrode_group`, `cluster_id`; does area or layer change besides the barrel).

Cells (whole-brain cell detection): `python -m barrelreg -c config.toml export` writes
`<ID>_wS1_cells_ccf.csv` (CCF voxel / µm, structure, barrel, layer, `fit_ok`) per brain and
`all_mice_wS1_cells_ccf.csv`.

## Reports and figures

```bash
BARRELREG_DEBUG=1 python -m barrelreg -c config.toml run MS261 --hemisphere 1    # saves results/h1/debug*.npz
python scripts/brain_report.py -c config.toml MS261 --tracks tracks_summary.csv  # <ID>/report/<ID>_h1_D2_report.pdf
python scripts/paper_figure.py -c config.toml MS261 --tracks tracks_summary.csv  # one figure per wS1 track
python scripts/explain_d2.py ...                                                 # the D2_explained figures for a brain
```

- **Step-by-step report**: one page per step on the brain's own data (inputs, own flatmap, template, depth and L4
  bottom, slab choice and image barrels, search landscape and top placements, point matching of every candidate, warp,
  3D columns and depth mapping, and with `--tracks` each wS1 probe track with its barrels under brainreg and D2).
  Every page explains what the step does and how to read it.
- **Publication panels** (in `<ID>/report/`, png 300 dpi, pdf, svg with editable text): the flat L4 slab alone, with
  the brainreg barrels, with the refined barrels, and brainreg / refined side by side (`_side_by_side`;
  `_side_by_side_signal` adds the probe-dye channel and a ring at each track's L4 crossing). Outlines are coloured by
  row; brainreg dashed, refined solid.
- **Comparison with another run**: `python scripts/compare_outputs.py <work_dir_a> <work_dir_b> MS261`.
