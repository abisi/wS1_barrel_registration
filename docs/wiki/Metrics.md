# Metrics

Every fit writes `results/h<hemi>/metrics.json`; `summarize` collects them into `summary.csv` (one row per
hemisphere). The review page shows five of them per fit. No single number proves a fit right: a fit shifted by a
whole row can score well. Use them to rank fits, then judge by eye ([Review page](Review-page.md#compare-tab)).

## On the review page

| Column | metrics.json | Meaning | Good |
|---|---|---|---|
| **Match** | `n_matched` | atlas barrels (of 33) paired one-to-one with a tissue barrel after cleaning. The small number after it: gain over the brain's best automatic fit. | more; ≥ 15 is well constrained |
| **Contr.** | `contrast_after` | barrel contrast after the fit, on the band-passed L4 slab: mean of the barrel cores (atlas barrels eroded by 3 px) minus mean of a thin band along the barrel borders (septa), in units of the field's standard deviation. With a lesion mask, measured on unmasked tissue. | ≥ 0.30 pass; good fits ≥ 0.6 |
| **LOO** | `err_tps_loo_um` | leave-one-out error: each matched barrel is left out, the warp refitted through the others, and the distance from where it lands to its tissue barrel measured; median (µm). Self-consistency, not correctness. | lower; 25–45 µm typical |
| **Seed** | `seed_residual_um` | seeded fits only: mean distance between the clicks and the average atlas placed rigidly (similarity) through them (µm). Measures how far this brain's shape is from the average atlas, not a click error. | informative only |
| **Probe** | — | barrel at each probe track's L4 crossing under this fit (wS1 tracks from the insertion table). Several fits naming the same barrel is reassuring. | agreement |
| Global row | `err_initial_um` | median distance between the global (brainreg) barrels and their matched tissue barrels (µm) | — |

Colours: best value per column in green, contrast below 0.30 in red.

## Click agreement (seeded brains)

Clicked barrel centres are taken as ground truth. `scripts/click_agreement.py` scores every fit and the global
registration against them:

| Score | Meaning |
|---|---|
| `inside` | number of clicks that lie inside the fitted barrel of the same name |
| `median_um` | median distance from each click to the centre of the fitted barrel of the same name |

On the AB / MH seeded brains the seeded fits had 6–13 of 8–15 clicks inside and a median of 26–78 µm; most automatic
fits had none inside and were 400–1200 µm off (whole rows or arcs shifted), and the global registration 68–762 µm.

## metrics.json, complete

### Fit quality

| Field | Meaning |
|---|---|
| `n_atlas`, `n_blobs`, `n_matched` | atlas barrels (33), tissue barrels detected in the slab, matched pairs |
| `contrast_initial`, `contrast_after` | barrel contrast under the global / the fitted barrels (see above) |
| `contrast_after_ch0`, `contrast_after_ch1`, `contrast_both` | the fitted barrels scored on each image channel, and their mean (used to choose between channels) |
| `err_initial_um` | median distance matched atlas barrel → tissue barrel before the warp (global placement) |
| `err_tps_loo_um` | leave-one-out error of the warp (see above) |
| `affine_residual_um` | residual of the best affine map through the matches; large = strongly non-affine (or wrong) naming |
| `dice_initial`, `dice_after` | mean Dice overlap of each matched tissue barrel with its atlas barrel, before / after |
| `naming_initial`, `naming_after` | fraction of matched tissue barrels whose largest overlap is the barrel they are matched to |
| `naming_ambiguous`, `margin_to_other_naming`, `n_candidates_agree` | another search candidate with a different naming scored within 0.02 (check by eye); its margin; how many candidates agree |
| `jac_min` | minimum Jacobian of the warp; > 0 = no folding |
| `disp_median_um` | median displacement of the barrel field by the warp |
| `tps_smoothing` | warp smoothing chosen by leave-one-out |

### Depth (slab) choice

| Field | Meaning |
|---|---|
| `l4_bottom_um`, `atlas_l4_bottom_um` | median depth of the L4/L5 boundary below the pia, image and atlas |
| `l4_bottom_found_frac` | fraction of 100 µm flat patches with their own L4 bottom (rest filled from neighbours) |
| `l4_offset_um` | the fitted slab's position relative to the L4 bottom (µm; negative = above) |
| `offset_candidates_um`, `depth_contrast`, `depth_pattern` | slab offsets considered and their band-pass contrast / pattern score |
| `depth_fits`, `depth_pick` | per fitted offset: matched barrels and contrast; the rule used to choose (`contrast`) |

### Surfaces and registration checks

| Field | Meaning |
|---|---|
| `field_check` | agreement of the brainreg deformation fields with the registered atlas; should be 1.000 |
| `surface_bridged_frac` | fraction of the L4 band whose depth moved > 30 µm when the surface was closed (400 µm ball). Intact 0.02–0.06; > 0.1 = notched or torn surface |
| `brain_wm_depth_um`, `wm_registered_below_pia_um`, `wm_registered_thickness_um`, `atlas_wm_depth_um` | white-matter depth used for the columns (the deeper of the two registration estimates) |
| `filled_frac` | fraction of the flat field filled by interpolation (no streamline) |
| `hull_overlap_frac` | voxels claimed by two barrel convex hulls (resolved to the nearer) |
| `ccf_label_agreement` | labelled voxels whose mapped CCF voxel carries the same barrel; tests the inverse lookup (0.90–0.94 typical), not the names |
| `ccf_c2_in_c2`, `ccf_defined_frac` | the same for C2; fraction of field voxels with a CCF position |

### Settings recorded

`image_channel`, `lesion_mask`, `sharpen` (`[amount, sigma_um]` or null), `seeded`, `seed_barrels`,
`seed_residual_um`, `contrast_refine` / `refined` / `refine_placement` (contrast refinement and whether it replaced the
matched fit), `search_best` (scale, rotation, shift of the chosen placement), `near_um`, `margin_um`, `seconds`.

## Thresholds used

| Check | Rule |
|---|---|
| failed fit | `contrast_after` < 0.30 (export marks cells `fit_ok = False`) |
| lesion variant needed | lesion mask removes ≥ 3 % of the barrel field (`lesion_fraction.py`) |
| naming to check by eye | `naming_ambiguous = True` |
| damaged surface | `surface_bridged_frac` > 0.1 |
| folding | `jac_min` ≤ 0 (never seen) |
