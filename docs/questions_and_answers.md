# Questions and answers on D2 changes (2026-10-30)

Answers 1 and 6 are partly pending: the comparison over all 48 Allen hemispheres (AB/MH) is still running in
`/raid0/lmsmith/local_barrel_reg/D2_allen_l4bottom`.

## 1. L4 bottom detection as the baseline for the slab and calculations

Done; the validation over all hemispheres is pending the rerun.

The paper's `barrel_bottom` is now in [l4bottom.py](../barrelreg/l4bottom.py), with the parameters from the
brain-space depth config:

- 100 µm patches, pooled over 200 µm.
- Bins of 0.02 normalised depth.
- Peak searched at 0.25–0.6, trough within 0.3 below it, bottom at the half-way point.

It is used in three places:

- **Slab:** sampled at an offset from that bottom instead of at a fixed depth. Offsets run from −300 to +50 µm. The
  two candidates are the best contrast (offsets −250 to +25 µm) and the best pattern match (−200 to −25 µm).
- **Depth mapping:** image L4 bottom → atlas L4 bottom, per flat pixel. The atlas bottom is stored in the template;
  its median is 623 µm.
- **Metrics:** `l4_bottom_um`, `l4_bottom_norm`, `atlas_l4_bottom_um`, `l4_offset_um`, `l4_bottom_found_frac`.

First test, AB117 left:

| | Before | Now |
|---|---|---|
| Matched barrels | 31/33 | 33/33 |
| Naming margin | 0.049 | 0.169 |
| Normalised bottom | – | 0.52 (paper: 0.48–0.53) |
| Patches with their own peak | – | 99% |

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

Done; the effect on all brains is pending the rerun.

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

On the template, the hulls grow the column area by about 3%. In AB117, CCF label agreement fell from 0.930 to 0.902.
This is expected: hull edges cover voxels that the pixel-wise CCF annotation calls septum.
