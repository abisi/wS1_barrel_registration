# Methods

The method in one page, then the variants used to rescue difficult brains. The full derivations are in
[methods_explained.md](../methods_explained.md); the pipeline with figures in [D2_explained.md](../D2_explained.md).

## Why a local registration

A whole-brain registration (brainreg) aligns the cortex well overall but places individual barrels with an error of
about half a barrel to 1.5 barrels (median ~100–250 µm; up to 1.2 mm in the AB / MH brains). Barrel identity matters at
the scale of one barrel (~200–300 µm). The barrels are visible in the cleared brain itself (layer 4 is brighter in the
barrels than in the septa), so the atlas barrel map can be fitted onto them directly.

## The D2 fit, per hemisphere

1. **Own flatmap.** Cortical depth from a Laplace equation between the registered pia and white matter; streamlines
   carry each L4 voxel to a flat 2D position, so columns follow the brain's own curvature.
2. **Template.** The atlas (Allen CCF with Blue Brain barrel annotation) L4 barrels in the Blue Brain flatmap: 33
   named barrels (A1–E8 and the straddlers Alpha, Beta, Gamma, Delta) on a 10 µm flat grid.
3. **L4 slab.** The L4/L5 boundary is detected per flat patch in the image; a thin slab at a fixed offset above it is
   flattened into a 2D L4 image. Several offsets are fitted (−350 … +150 µm, every 50 µm) and the one with the highest
   barrel contrast is kept.
4. **Tissue barrels.** Band-pass filter, local maxima, watershed → barrel centroids. Artefacts (and, with the lesion
   mask, lesions) are masked.
5. **Search.** Every similarity (scale 0.6–1.5, rotation ±30°, shift ±800 µm in the wide search) of the synthetic
   barrel pattern is correlated with the slab; the best placements are kept and polished.
6. **Matching.** Coherent Point Drift (affine, then non-rigid) pulls atlas barrels onto tissue barrels; the Hungarian
   algorithm pairs them one-to-one; leave-one-out cleaning removes pairs that disagree with the rest. Among near-tied
   candidates, the one whose matches an affine map explains best wins.
7. **Contrast refinement.** The placements are re-ranked by barrel contrast and the atlas landmarks moved to maximise
   it; kept only when it beats the matched fit by > 0.02.
8. **Warp.** A thin-plate spline through the matched pairs (smoothing chosen by leave-one-out) carries the whole flat
   plane; outside the field it blends into a near-affine map.
9. **Back to 3D.** Each voxel takes the barrel whose convex hull contains its corrected flat position (Blue Brain's
   column definition); depth is mapped piecewise linearly (pia → pia, L4 bottom → L4 bottom, white matter → white
   matter). The result is a barrel label per voxel and a corrected CCF position per voxel.

## Variants

Each variant is the same fit with one setting changed (`scripts/run_variants.sh`, environment switches of
`barrelreg`):

| Variant | Switch | When it helps |
|---|---|---|
| `wide_ch0`, `wide_ch1` | `BARRELREG_CH=0/1`, `BARRELREG_SEARCH=wide` | always (first pass). ch1 often shows the barrels more clearly than the autofluorescence ch0. |
| `*_lesion` | `BARRELREG_LESION=1` | large lesions (probe tracks, damage) in the field: lesions and their rims are masked in the slab, so search and matching use only the surrounding barrels |
| `sharp1_*`, `sharp2_*` | `BARRELREG_SHARPEN=1,60` / `2,60` | blurred L4 image: unsharp mask (Gaussian σ 60 µm, amount 1 or 2) on both channels before anything else |
| `seeded_*` | `BARRELREG_SEEDS=seeds.json` | the search cannot find the field (rotated beyond ±30°, hidden by damage or puncta, fits disagree by whole rows) |

Shared settings: `BARRELREG_PICK=contrast` (slab depth by contrast), `BARRELREG_OFFSETS_RANGE=-400,150`,
`BARRELREG_FIT_OFFSETS=-350,…,150`, `BARRELREG_CONTRAST_REFINE=1`, `BARRELREG_DEBUG=1` (debug dumps for the review
page and reports, ~250 MB per hemisphere).

### Seeded fit

The clicked barrel centres (flat grid, `[row, col]`) replace the search: the least-squares similarity (Umeyama:
scale, rotation, shift) from the atlas barrel centroids to the clicks gives the initial placement. Matching, warp and
slab choice then run unchanged; the contrast refinement is skipped. The clicks are treated as ground truth when
judging the result ([click agreement](Metrics.md#click-agreement-seeded-brains)).

The seed images on the review page are the flat L4 slabs of a previous debug run on the same grid, in six versions
(band, L4-specific, 25th percentile, septa, clean, raw), three channels (ch0, ch1, geometric mean), several depths
(fixed −150 … 0 µm, the mean of the L4 slabs, and an automatic per-region depth), and optionally sharpened (×1, ×2).

### Sharpening

`I' = max(0, I + a · (I − G_σ * I))` with σ = 60 µm (about a quarter barrel) and a = 1 or 2, applied to both image channels
before slab extraction. It boosts barrel-scale edges without changing the coarse intensity. On MH015 (blurred) it
raised matched barrels from 24 to 29 and lowered the leave-one-out error from 36–42 to 31–34 µm.

### Choosing among variants

`scripts/build_final.py --score contrast_both` picks automatically: the variant with the highest mean ch0 / ch1
contrast; a brain passes when that fit reaches 0.30 on at least one channel. The review then overrides this per brain.

## Limits

- Barrel names come from the matching; there is no independent ground truth except clicked seeds and physiology
  (whisker responses on the probe tracks).
- Only the barrel field and its surroundings get corrected CCF positions.
- Surface damage wider than ~0.8 mm is not bridged; columns there start lower.
- One hemisphere per run in the AB / MH dataset (left, `--hemisphere 1`).
