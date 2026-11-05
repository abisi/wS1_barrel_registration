# Questions and answers on D2 changes (2026-10-30)

Results are from all 48 Allen hemispheres (24 AB/MH brains) in `/raid0/lmsmith/local_barrel_reg/D2_allen_l4bottom`,
compared with the previous D2 run in `/raid0/lmsmith/local_barrel_reg/D2_allen`. The per-hemisphere table is
`compare_old_vs_new.csv` in `neuropixels/local_barrel_registration/D2_l4bottom/`.

## 1. L4 bottom detection as the baseline for the slab and calculations

Done, and run on all 48 Allen hemispheres.

The paper's `barrel_bottom` is now in [l4bottom.py](../barrelreg/l4bottom.py), with the parameters from the
brain-space depth config:

- 100 µm patches, pooled over 200 µm.
- Bins of 0.02 normalised depth.
- Peak searched at 0.25–0.6, trough within 0.3 below it, bottom at the half-way point.

It is used in three places:

- **Slab:** sampled at an offset from that bottom instead of at a fixed depth. Offsets run from −300 to +50 µm. The
  two candidates are the best contrast (offsets −150 to 0 µm) and the best pattern match (−150 to −25 µm).
- **Depth mapping:** image L4 bottom → atlas L4 bottom, per flat pixel. The atlas bottom is stored in the template;
  its median is 623 µm.
- **Metrics:** `l4_bottom_um`, `l4_bottom_norm`, `atlas_l4_bottom_um`, `l4_offset_um`, `l4_bottom_found_frac`.

### Results over 48 hemispheres

| | Previous D2 | With L4 bottom |
|---|---|---|
| Failed fits (contrast after < 0.3) | 6 | 5 |
| Matched barrels, median (mean) | 26 (25.4) | 28 (26.6) |
| Contrast after fit, median | 0.92 | 1.02 |
| Naming ambiguous | 10 | 10 |

- **Matched barrels:** more in 27 hemispheres, fewer in 11.
- **Rescued:** AB147 left (contrast 0.17 → 0.89). The other 5 failures failed in both versions: AB131 right, AB154
  right, AB159 right, MH008 left, MH030 right.
- **Clearly better** (contrast up by more than 0.2): 12 hemispheres. Examples: AB153 right (18 → 33 matched, contrast
  0.60 → 1.48), AB092 left (0.35 → 0.80), AB085 right (0.76 → 1.02).
- **Clearly worse:** 2 hemispheres, AB082 left (0.98 → 0.60) and AB147 right (1.11 → 0.73). In both, the slab sat
  100–125 µm above the bottom, chosen because it matched more barrels.

**The L4 bottom itself:**

- Normalised depth median 0.49 (IQR 0.48–0.50), as in the paper (0.48–0.53 in MS176 and MS211).
- In µm, median 498 µm below the image pia; the atlas L4 bottom is 623 µm.
- 99% of patches find their own peak (median per hemisphere).

**Where the slab works best:** the band-pass contrast peaked 25–50 µm above the bottom in 46 of 48 hemispheres. The
chosen slab was at −25 or −50 µm in 42 of 48.

**Offset ranges were set from the data.** In the first pass the ranges were wider (contrast −250 to +25 µm, pattern
−200 to −25 µm). In AB082 left this picked L2/3 texture at −200 µm, about 300 µm deep (contrast 0.98 → 0.35). The
ranges were narrowed to −150 to 0 and −150 to −25 µm, and the 5 affected brains (AB082, AB085, AB131, AB147, AB159)
were rerun. Their first-pass metrics are kept in `D2_allen_l4bottom/pass1_ranges_wide/`.

**Barrel names change between versions.** Among the 42 hemispheres where both fits are good, the median fraction of
voxels with the same barrel name is 0.93:

| Same name | Hemispheres |
|---|---|
| ≥ 0.9 | 25 |
| 0.5–0.9 | 10 |
| < 0.5 | 7 |

In most of the 7 hemispheres renamed by more than half, the new fit has clearly higher contrast (for example AB085
right, 0.06 the same, contrast 0.76 → 1.02), so the new names are more likely right. There is still no ground truth;
the Neuropixels whisker responses are the independent check.

Templates built before this change have no atlas L4 bottom, so `build-template` must be rerun (about 2 minutes).

## 2. Does it use a new Laplace for the imaged data?

No. The Laplace equation is solved on the **registered atlas labels**, on the 20 µm crop
([flatmap.py](../barrelreg/flatmap.py), `laplace_depth`):

- **Boundaries:** 0 outside the registered brain (the registered pia), 1 in registered tissue that is not isocortex
  (white matter).
- **Flatmap:** each voxel is followed along the gradient to the registered L4 level and laid flat on a PCA plane.

The image is used only for depth: the distance below the segmented, closed tissue surface, and now the L4 bottom. So
the column directions come from the registration, and the depths along the columns come from the image.

## 3. Interpolating 20 µm onto 10 µm, or downsampling 10 µm to 20 µm?

Keep the 10 µm grid, which is what the code already does.

- **Flat coordinates are a smooth field.** Linear interpolation from 20 µm is accurate to well under 1 µm
  (error ≈ h²/8 · |F''|).
- **Allen input is native 10 µm.** Binning to 20 µm would throw away real septum detail, since septa are about
  30–80 µm wide, near the 40 µm limit of a 20 µm grid.
- **Perens input is 20 µm.** Upsampling adds no information, but costs no accuracy either.
- **The template.** At 20 µm its thin septa would merge, and binning the flat coordinates would be twice as coarse
  (±10 instead of ±5 µm).

20 µm would only save time and memory. The Laplace solve stays at 20 µm because that field is smooth.

## 4. Methods explained like for a bachelor student

Done: [methods_explained.md](methods_explained.md). It covers, in order:

1. Transforms
2. The exhaustive correlation search
3. CPD affine, then non-rigid
4. Hungarian assignment with the distance and area gates
5. Leave-one-out cleaning
6. Thin-plate spline, with smoothing chosen by leave-one-out and the near-affine blend outside the field

The application details are at the end, with every parameter and where it is in the code.

## 5. Code that performs the linear transformation search

[search.py](../barrelreg/search.py), `brute_force`, called from [register.py](../barrelreg/register.py) in `fit_at`:

```python
def brute_force(fix, mov, mask, pix, centre, topk=5, max_t=400, step_t=40, max_r=20, step_r=2.5,
                scales=(0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4)):
    res = []
    for sc in scales:
        for ang in np.arange(-max_r, max_r + 1e-6, step_r):
            th = np.deg2rad(ang)
            M = np.linalg.inv(sc * np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]]))
            for dy in np.arange(-max_t, max_t + 1e-6, step_t):
                for dx in np.arange(-max_t, max_t + 1e-6, step_t):
                    d = np.array([dy, dx]) / pix
                    w = ndimage.affine_transform(mov, M, offset=centre - M @ (centre + d), order=1, cval=0)
                    res.append((masked_corr(fix, w, mask), sc, ang, dy, dx))
    res.sort(key=lambda r: -r[0])
    tops = []
    for r in res:  # top-k that differ by >120 um shift, >=7.5 deg or >=0.15 scale
        if all(np.hypot(r[3] - t[3], r[4] - t[4]) > 120 or abs(r[2] - t[2]) >= 7.5 or abs(r[1] - t[1]) >= 0.15
               for t in tops):
            tops.append(r)
        if len(tops) == topk:
            break
    return tops
```

The call:

```python
tops = brute_force(fix[::2, ::2], mov[::2, ::2], mask[::2, ::2], PIX * 2, centre / 2, max_t=MAX_T, scales=SCALES)
```

In the pipeline it tries scales 0.7–1.4, rotations ±20° in 2.5° steps and shifts ±600 µm in 40 µm steps, at 20 µm.
That is about 131,000 placements.

`affine_transform` reads the input at M·o + offset for each output pixel, which is why the matrix is inverted. The
result places template pixel i at c + d + sR(i − c): rotated and scaled about the field centre c, then shifted by d.

## 6. Convex-hull barrel columns (Blue Brain `annotations_to_positions.py`)

Done, and used in the 48-hemisphere run.

`hull_columns` in [flatmap.py](../barrelreg/flatmap.py) follows the same logic as Blue Brain's
`atlas-enhancement/barrel-annotations/annotations_to_positions.py`:

- **Hull:** the convex hull of each barrel's flat L4 footprint.
- **Column:** every voxel whose corrected flat coordinate lies inside that hull gets the barrel's name.

Two things differ from the Blue Brain code:

- The point-in-hull test is vectorised with scipy, instead of shapely's one-point-at-a-time loop.
- Where two hulls overlap, the voxel goes to the nearest barrel; Blue Brain's code assigns it to both.

Two smaller details:

- The hull goes round the pixel corners.
- Only each barrel's largest connected piece is used; 2 of the 33 template barrels have a stray pixel.

On the template, the hulls grow the column area by about 3%. In the brains, under 1% of column voxels are in two
hulls (0.3–0.6% per hemisphere).

CCF label agreement fell from 0.93 to 0.90 in every hemisphere. This is expected: hull edges cover voxels that the
pixel-wise CCF annotation calls septum. The agreement tests the inverse lookup, not the names.
