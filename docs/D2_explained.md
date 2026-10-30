# How D2 works

D2 names the barrels of the mouse whisker barrel field (wS1) in a cleared light-sheet brain and gives every voxel of
the barrel field an Allen CCF coordinate. The coordinate comes from the barrels the method finds in the brain's own
autofluorescence (ch0), not from the whole-brain registration.

The figures `D2_0`–`D2_3` sit next to this file, and `D2_example_MS261_*.png` show a finished result. The matching
steps that D2 shares with method D are illustrated in `../D_explained/` (`D3`–`D7`; on the share:
`/mnt/lsens-analysis/Lana_Smith/neuropixels/local_barrel_registration/D_explained/`). The code is the `barrelreg`
package (`/home/lmsmith/wS1_barrel_registration`); the figures are made by its `scripts/explain_d2.py`.

## Why it is needed

- **Whole-brain registration misplaces barrels.** It gets the cortex roughly right, but in the barrel field it is
  typically off by half a barrel to 1.5 barrels (median about 150 µm, up to 500 µm). Under such a registration only
  about a third of the barrels carry the right name.
- **The barrels are visible.** In L4 of ch0 they appear as bright patches separated by dark septa, so the atlas
  barrel map can be fitted onto them directly.

## The steps (`D2_0_pipeline.png`)

**1. Coarse registration (brainreg to the Perens LSFM atlas at 20 µm, or to the Allen barrel atlas at 10 µm).**
It is used only to find the hemisphere, the rough position of SSp-bfd, and the pia and white-matter surfaces. Its
barrel placement is not used.

**2. The brain's own flatmap (`D2_1_own_flatmap.png`).**
- A depth coordinate is solved between the registered pia (0) and white matter (1), a Laplace equation on the 20 µm
  crop.
- Every cortical voxel is followed along the depth gradient (its streamline) to the L4 level, and the L4 surface is
  laid flat (PCA plane, axes anterior and dorsal).
- So each voxel gets a flat position. Lines of constant flat position run along the columns and follow the curvature
  of this brain's cortex.
- Everything is then interpolated to 10 µm.

**3. The fixed template (`D2_2_template.png`), built once from the CCF.**
- The atlas L4 barrels (`allen_mouse_bluebrain_barrels_10um`) are drawn in the Blue Brain flatmap of the CCF and
  mapped linearly onto an L4 plane with the same convention and origin as step 2.
- The template therefore starts at the same place as the brain's flatmap, and the fit only has to correct the
  difference.
- It also stores the atlas cortical thickness and the atlas L4 bottom per flat pixel.
- It stores an inverse lookup: (flat position, atlas depth) → CCF voxel.
- The Blue Brain flatmap covers one CCF hemisphere; the right brain hemisphere is mirrored onto it (the annotation is
  symmetric).

**4. L4 slab** (figure `D3` shows the method D version). Depth is measured below the *image* pia, not the registered
one. The image surface is first closed with a 400 µm ball, so the depth under a notch or tear is measured from where
the surface would be (see "Surface damage").
- **L4 bottom.** The bottom of the barrel row (the L4/L5 boundary) is found in ch0 for every column of the own
  flatmap, with the detection used in the brain-space depth analysis of the paper. Columns are grouped into 100 µm
  patches and pooled over 200 µm. In each pooled ch0 profile against normalised depth, the slow pia-to-white-matter
  trend is removed; the bottom is where the barrel peak has fallen halfway to the trough below it. Patches without a
  peak are filled from their neighbours.
- **Slab.** It is sampled at a fixed offset above that bottom, so it follows the brain's own L4 instead of a fixed
  depth.
- **Two candidate offsets** are tried: best band-pass contrast, and best match to the barrel pattern. The whole fit
  runs at both, and the one that matches more barrels wins.

**5. Image barrels** (`D4`). Lesions, vessels and debris are masked. The slab is band-passed and split into blobs by
local maxima and watershed. The blob centres are the landmarks; at this point they have no names.

**6. Search and matching** (`D5`, `D6`).
- **Search:** every scale (0.7–1.4) × rotation (±20°) × shift (±600 µm) of a synthetic barrel pattern (barrels
  bright, septa dark) is scored against the slab, and the 5 best placements are kept.
- **Matching:** from each placement, the template barrels are matched one-to-one to image barrels (CPD affine, then
  non-rigid, Hungarian matching, re-pairing, leave-one-out cleaning).
- **Choosing a candidate:** once the template is pinned onto the blobs, a naming shifted by one row can correlate with
  the image as well as the right one. So among candidates within 0.02 of the best correlation, D2 keeps the one whose
  matches an affine map explains best. A naming error forces rows to shift against each other, which an affine map
  cannot do.

**7. Warp** (`D7`). A thin-plate spline through the matched barrel centres, with stiffness chosen by leave-one-out.
Outside the barrel field it blends into a near-affine fit.

**8. Back to 3D (`D2_3_depth_mapping.png`).**
- Each voxel takes the template barrel whose convex hull in the flat plane contains its corrected flat position, so
  the columns follow the streamlines. This is how Blue Brain's barrel annotations define a column
  (`atlas-enhancement/barrel-annotations/annotations_to_positions.py`); where two hulls overlap, the nearest barrel
  wins.
- Depth is mapped piecewise linearly with three anchors per column:
  - image pia → atlas pia;
  - image L4 bottom (found in ch0) → atlas L4 bottom (the deepest barrel L4 voxel per template pixel);
  - the brain's white matter → atlas white matter.
- The brain's white matter per column is the deeper of two estimates from the coarse registration: the deepest
  registered isocortex below the image pia, and the registered cortical thickness. The first moves with any shift of
  the registered cortex against the image; the second does not. In MH064 (left) the registered cortex sits 300 µm
  above the real surface, and the first estimate alone stopped the columns 300 µm short. Taking the deeper one means
  a shifted registration cannot cut the columns short; the overshoot is at most the surface offset (typically
  20–60 µm). The registered labels no longer cut the columns either, since they are shifted in the same way.
- Columns run from the image pia down to that depth.
- Method D used one factor (image L4 ÷ atlas L4) for the whole column. That stopped columns short wherever the cortex
  below L4 is thicker than in the atlas; the right panel of `D2_3` shows where that happens.

**9. Into CCF.** (Corrected flat position, mapped atlas depth) → inverse lookup → CCF voxel. The result is stored like
brainreg's deformation fields (`ccf_field_{0,1,2}.tif`, mm), so cells can be mapped exactly as brainreg maps them.

## Surface damage

The depth below the pia is the distance to the tissue edge in the image. Measured to the *nearest* edge, a notch, dent
or tear in the surface pulls everything around it upwards: the L4 slab and the layer boundaries bow into the damage, and
the columns under it get the wrong layers. D2 measures depth from the surface after a morphological closing with a
400 µm ball instead. Damage up to roughly 0.8 mm across is bridged, while the convex surface of intact cortex does not
move. This assumes the tissue in a notch was torn away, not pressed down. Tissue that is missing stays missing; the
columns there simply start lower, with the correct layer. `surface_bridged_frac` in the metrics is the fraction of the
L4 band whose depth changed by more than 30 µm; intact hemispheres are near 0.02–0.06, notched ones above 0.1.

## What D2 differs from D in

| | D | D2 |
|---|---|---|
| Coarse registration | brainreg to the Allen barrel atlas | brainreg to any atlas with Allen region IDs (Perens 20 µm or Allen 10 µm) |
| Flat coordinates | Blue Brain flatmap, read through the registration | the brain's own flatmap (Laplace streamlines) |
| Template | atlas barrels in the Blue Brain flatmap | the same, as a fixed 2D picture in a matching plane |
| Search range | shift ±400 µm, scale 0.8–1.4 | shift ±600 µm, scale 0.7–1.4 (rougher start) |
| Column depth | one factor per column | three anchors: pia, L4, white matter (deeper of two registration estimates) |
| Depth below the pia | nearest tissue edge | below the closed surface (notches and tears bridged) |
| CCF coordinates | through the Allen registration | through the barrel matches and the inverse flatmap |

## Using the results

- **Per mouse:** `barrels_corrected_full.tif` (barrel index per voxel, downsampled grid),
  `ccf_field_{0,1,2}.tif`, `<ID>_D2_overview.png` and `<ID>_wS1_cells_ccf.csv`.
- **Cell table:** each detected cell in SSp-bfd with its CCF position, structure, barrel and layer.
- **`fit_ok`:** exclude cells with `fit_ok = False`. Those come from hemispheres whose fit failed (contrast after
  < 0.3).
- **Hemispheres:** brainglobe value 1 = left (high axis 2), 2 = right.
- **Ambiguous names:** hemispheres flagged `naming_ambiguous` had a close runner-up with different names. Look at
  their overview figure before relying on barrel identity.

## Limits

- **No ground truth for the names.** Barrel names come from the matching criterion. The best independent check would
  be whisker responses on recording tracks.
- **CCF coordinates only in and around the barrel field.** Elsewhere the fields are NaN.
- **Some fits fail.** Lesions, tears, or a coarse registration more than ~600 µm off can make a fit fail; these show
  up as low contrast after the fit.
- **Layer boundaries are partly approximate.** The L4/L5 boundary is seen in the image (barrel bottom); the others
  come from the three-anchor mapping.

## Provenance

D2 was developed in the local barrel registration project (Lana Smith, EPFL, October 2026). Four methods were compared
on 24 Allen-registered brains before it; see `SUMMARY_local_barrel_registration.md`. This text describes `run_d2.py` as
of 2026-10-30 11:22, which the `barrelreg` package reproduces, plus three later changes in the package: 10 µm /
Allen-registered input, depth below the closed surface, and the white-matter depth as the deeper of two estimates.
