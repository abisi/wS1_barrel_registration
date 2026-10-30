# The matching methods, explained from scratch

This note explains the mathematics behind the barrel registration (method D2, package `barrelreg`). It assumes
linear algebra and a little probability. Each idea is first explained on its own, with a small example. The last
section gives the exact settings used in the pipeline and where they are in the code.

The task in one sentence: we have a **template** (the atlas barrel map, 33 named barrels) and an **image** (a flat
picture of layer 4 of one brain, with bright unnamed blobs). We want to know which blob is which barrel, and then a
smooth map that carries every template point onto the image.

This is done in four stages, each one finer than the one before:

| Stage | What it finds | Tool |
|---|---|---|
| 1. Rough placement | one rotation, scale and shift of the whole template | exhaustive correlation search |
| 2. Point matching | template barrels pulled onto image blobs | CPD, affine then non-rigid |
| 3. Names | which blob is which barrel (one to one) | Hungarian algorithm + leave-one-out cleaning |
| 4. Warp | a smooth map of the whole plane | thin-plate spline |

---

## 1. Transformations of the plane

A point in the flat picture is a vector $p = (u, v)$. A **linear transformation** multiplies it by a 2 × 2 matrix; an
**affine** transformation also adds a shift $t$:

$$p' = A\,p + t .$$

Special cases:

- **Rotation** by an angle $\theta$:
  $R(\theta) = \begin{pmatrix}\cos\theta & -\sin\theta\\ \sin\theta & \cos\theta\end{pmatrix}$.
  It keeps lengths and angles.
- **Uniform scaling** by $s$: $A = s\,I$. It makes everything bigger or smaller, keeping the shape.
- **Similarity**: $A = s\,R(\theta)$. Shape is kept, size and orientation change. It has 4 numbers: $s$, $\theta$, and
  two shift components.
- **General affine**: any $A$, 6 numbers. It can also stretch more in one direction than another (anisotropy) and shear
  (turn rectangles into parallelograms).

Rotating "about the centre" $c$ instead of about the origin means: move $c$ to the origin, rotate, move back:
$p' = c + sR\,(p - c) + d$, where $d$ is an extra shift.

**Why affine is not enough.** One brain's barrel field is not an exact affine copy of the atlas. Rows bend a little
differently, and some barrels are bigger. A **non-rigid** map lets every region move a little differently, but it has
to be kept smooth, or it could map any point anywhere. Stages 2 and 4 are about keeping that freedom under control.

---

## 2. Rough placement: try every similarity and keep the best

### Measuring how well two pictures agree

Take the image $f$ and the moved template $g$ (both pictures on the same pixel grid). Their **Pearson correlation**
inside a mask $\Omega$ is

$$r = \frac{\sum_{\Omega}(f - \bar f)(g - \bar g)}{\sqrt{\sum_{\Omega}(f - \bar f)^2 \,\sum_{\Omega}(g - \bar g)^2}} \in [-1, 1].$$

It is 1 when $g$ is a brighter or dimmer copy of $f$, and near 0 when they are unrelated. It does not care about
overall brightness or contrast, which differ between brains.

The template is a list of labels, not a picture. So it is first drawn as a **synthetic image**: barrels = 1,
septa = 0, then blurred. Both pictures are **band-passed**: a small Gaussian blur (removes pixel noise) minus a large
Gaussian blur (removes slow brightness changes across the field). What remains is the barrel-sized pattern of bright
patches and dark septa.

### Why search exhaustively?

The usual alternative is gradient descent: start somewhere and walk downhill. Here that fails. The barrel field is a
nearly **periodic** pattern (rows and arcs of similar blobs), so the correlation has many local maxima. Shifting the
template by exactly one row gives a placement that also looks good. Gradient descent would stop at whichever maximum
is nearest the start.

With only 4 unknowns we can simply try every combination on a grid and keep the best few:

- 8 scales (0.7–1.4) × 17 rotations (−20° to +20° in 2.5° steps) × 31 × 31 shifts (±600 µm in 40 µm steps)
- about 131 000 placements, each scored by masked correlation, on a 20 µm grid.

The best 5 *distinct* placements are kept. Distinct means they differ by more than 120 µm in shift, or 7.5° in
rotation, or 0.15 in scale. Each of them is followed through stages 2–3, and the best end result wins (section 6).

### The code

`barrelreg/search.py`, lines 30–52:

```python
def brute_force(fix, mov, mask, pix, centre, topk=5, max_t=400, step_t=40, max_r=20, step_r=2.5,
                scales=(0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4)):
    """Every scale x rotation x shift of `mov` scored by masked correlation with `fix`.
    Returns the top-k distinct (corr, scale, rot_deg, dy_um, dx_um)."""
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
    for r in res:
        if all(np.hypot(r[3] - t[3], r[4] - t[4]) > 120 or abs(r[2] - t[2]) >= 7.5 or abs(r[1] - t[1]) >= 0.15
               for t in tops):
            tops.append(r)
        if len(tops) == topk:
            break
    return tops
```

It is called in `barrelreg/register.py`, line 283, on the slab image and template at half resolution (20 µm pixels):

```python
tops = brute_force(fix[::2, ::2], mov[::2, ::2], mask[::2, ::2], PIX * 2, centre / 2, max_t=MAX_T, scales=SCALES)
# MAX_T = 600 um, SCALES = (0.7, 0.8, ..., 1.4); max_r = 20 deg, step_r = 2.5 deg, step_t = 40 um (defaults)
```

**Why the inverse matrix?** `ndimage.affine_transform(mov, M, offset)` fills each *output* pixel $o$ by reading the
input at $M o + \text{offset}$. This is called **pull-back** (or backward) sampling, and it leaves no holes in the
output. With $M = (sR)^{-1}$ and offset $= c - M(c + d)$, output pixel $o$ reads input pixel
$i = c + M(o - c - d)$. Solving for $o$ gives

$$o = c + d + sR\,(i - c),$$

so the template is rotated and scaled about the field centre $c$ and then shifted by $d$, as intended.

Once the best placement is known, the same similarity is applied to the template barrel **centres** (not to the
picture) by `similarity_points` (`search.py`, line 55), which uses the forward formula above.

---

## 3. From pictures to points

Stages 2–4 work with **points**, not pictures:

- **Template points.** The 33 atlas barrel centres (centroids) and their areas.
- **Image points.** The slab image is band-passed, local maxima are taken as seeds, and a **watershed** grows each seed
  into a blob until it meets its neighbours. This is like flooding a landscape from its lowest points (here: the
  brightest ones), with walls where floods meet. The blob centroids and areas are the image points.

There are usually more image blobs (40–75) than template barrels (33), because the slab also holds vessels and
fragments, and some barrels are missing (damage, poor contrast). So the method must tolerate **outliers** on both sides.

---

## 4. Coherent Point Drift (CPD)

### The idea: points as a cloud of Gaussians

CPD (Myronenko & Song, 2010) turns point matching into fitting a probability model. Each moved template point
$T(y_m)$ is the centre of a Gaussian blob of width $\sigma$. The image points $x_n$ are treated as samples drawn from
the mixture of these Gaussians, plus a flat "outlier" component (weight $w$) that can produce a point anywhere.

We look for the transformation $T$ under which the image points are most probable. We do not know which template
point produced which image point, so CPD alternates two steps (the **EM algorithm**).

**E-step: soft assignments.** For every pair, the probability that template point $m$ produced image point $n$:

$$P_{mn} = \frac{\exp\!\left(-\frac{\lVert x_n - T(y_m)\rVert^2}{2\sigma^2}\right)}
{\sum_{k} \exp\!\left(-\frac{\lVert x_n - T(y_k)\rVert^2}{2\sigma^2}\right) + c},
\qquad c = (2\pi\sigma^2)^{D/2}\,\frac{w}{1-w}\,\frac{M}{N}.$$

The constant $c$ is the outlier term. An image point far from every template point gets small $P_{mn}$ for all $m$
and is effectively ignored.

**M-step: best transformation for those assignments.** Minimise the weighted squared distances
$\sum_{m,n} P_{mn}\lVert x_n - T(y_m)\rVert^2$, then update $\sigma^2$ to the weighted mean squared residual.

Repeat. As the fit improves, $\sigma$ shrinks by itself. Early on every template point "sees" many image points (a
wide, forgiving match); later each sees only its nearest one. This built-in coarse-to-fine behaviour makes CPD robust.

### Affine CPD

With $T(y) = By + t$, the M-step is a weighted least-squares problem with a closed-form solution. Let $\mu_x$ and
$\mu_y$ be the $P$-weighted means, and $\hat X$, $\hat Y$ the centred points. Then

$$B = \left(\hat X^\top P^\top \hat Y\right)\left(\hat Y^\top \mathrm{diag}(P\mathbf 1)\,\hat Y\right)^{-1},
\qquad t = \mu_x - B\mu_y .$$

This is the same "normal equations" solution as ordinary linear regression, with $P$ as weights. In the code it is
`cpd_affine`, `pointset.py` lines 32–52.

### Non-rigid CPD

Now every template point may move by its own displacement: $T(y_m) = y_m + v(y_m)$. To keep the motion smooth
(**motion coherence**: nearby points move together), the displacement field is built from Gaussian bumps centred on
the template points:

$$v(y) = \sum_k W_k\, \exp\!\left(-\frac{\lVert y - y_k\rVert^2}{2\beta^2}\right), \qquad\text{i.e. } V = GW .$$

- $\beta$ is the distance over which points move together. Large $\beta$ means the field is almost a rigid shift;
  small $\beta$ means each point moves on its own.
- A penalty $\lambda$ on wiggliness keeps the field smooth.

The M-step becomes one linear system for the coefficients $W$:

$$\left(\mathrm{diag}(P\mathbf 1)\,G + \lambda\sigma^2 I\right) W = P X - \mathrm{diag}(P\mathbf 1)\,Y .$$

In the code it is `cpd_nonrigid`, `pointset.py` lines 55–71.

**Why affine first, then non-rigid?** The affine step removes the global error (size, stretch, shear) that the search
grid could not capture exactly. Starting from there, the non-rigid step only has to correct small local differences.
If non-rigid CPD started from far away, it could bend the template into a wrong but locally well-fitting shape.

---

## 5. Hungarian algorithm: one barrel ↔ one blob

CPD gives *soft* assignments; we need *hard*, one-to-one names. This is the classic **assignment problem**. Given a
cost matrix $C_{mn}$ (here the distance between moved template barrel $m$ and blob $n$), choose pairs so that:

- each barrel is used at most once,
- each blob is used at most once,
- the total cost is minimal.

The **Hungarian algorithm** (Kuhn, 1955) solves this exactly in polynomial time. In Python it is
`scipy.optimize.linear_sum_assignment`.

Matching each barrel to its nearest blob independently could give two barrels the same blob. The Hungarian solution
avoids that, because it chooses all pairs together.

**Gating.** Some pairs are forbidden by setting their cost to a huge number (`1e6`) before solving; any pair that ends
up with that cost is thrown away:

- **Distance gate:** pairs further apart than 70 µm after CPD.
- **Area gate:** the blob's area divided by the barrel's atlas area must lie between 0.12 and 1.6. A much larger blob
  is probably two barrels merged; a much smaller one is a fragment or a vessel.

This is `hungarian`, `pointset.py` lines 74–84.

**Re-pairing.** A thin-plate spline (section 7) through the current matches predicts where *every* template barrel
should be, including unmatched ones. Matching again from these predictions can pick up barrels that CPD left just
outside the gate. This repeats up to 3 times, until the pairs stop changing (`register.py`, lines 301–306).

---

## 6. Leave-one-out: does each match agree with the others?

**Leave-one-out (LOO)** is the smallest form of cross-validation. For each match $i$:

1. Fit the map (a thin-plate spline) through all matches *except* $i$.
2. Predict where barrel $i$ should land.
3. Measure the error between the prediction and the blob $i$ was actually matched to.

A correct match is predicted well by its neighbours. A wrong one (for example a vessel taken for a barrel) sticks out,
because the other matches "vote" for a different position. Here, matches with LOO error larger than
max(2.5 × the median error, 70 µm) are dropped (`register.py`, lines 307–309).

LOO is used a second time to choose how stiff the final warp should be (section 7).

### Choosing among the 5 search candidates

Each of the 5 rough placements goes through CPD, matching and cleaning. To compare them, the template is warped onto
the image through its matches, and the correlation with the image is computed again ("warped correlation").

There is a trap. Once the warp pins the template barrels onto the blobs, a naming that is off by one row can correlate
almost as well as the right one, because the warped picture still has bright patches on every blob. So among
candidates within 0.02 of the best warped correlation, the one whose matches an **affine** map explains best (lowest
median residual) wins, then the one with more matches. A wrong naming forces rows to slide against each other, which
an affine map cannot do, so its residual is high. (`register.py`, lines 324–327.)

---

## 7. Thin-plate spline (TPS): a smooth warp through the matches

### The idea

Imagine the plane as a thin metal sheet. You pin the template barrel centres $p_i$ and pull each one to its matched
blob $q_i$. The sheet bends as little as possible in between. A **thin-plate spline** is exactly the map $f$ that
passes through the pins with the least **bending energy**

$$E[f] = \iint \left(f_{uu}^2 + 2 f_{uv}^2 + f_{vv}^2\right)\,du\,dv .$$

An affine map has zero bending energy (its second derivatives are 0), so TPS = an affine part + the smallest possible
bend on top.

### The formula

The solution has a closed form:

$$f(p) = a_0 + A\,p + \sum_i w_i\,\phi(\lVert p - p_i\rVert), \qquad \phi(r) = r^2 \log r .$$

Finding $a_0$, $A$ and $w_i$ means solving one linear system, with one equation per match plus three for the affine
part. `scipy.interpolate.RBFInterpolator(kernel="thin_plate_spline", degree=1)` does this (`tps_fit`,
`pointset.py` line 87).

### Smoothing: interpolate or approximate?

With smoothing 0, the spline passes *exactly* through every match, including a centroid that is 20 µm off because a
vessel crossed the blob. With smoothing $s > 0$, the spline may miss the matches by a little in exchange for less
bending. As $s \to \infty$ it becomes the best affine fit (bending is no longer allowed at all).

Which $s$ is right? Use LOO error again. Try $s \in \{0, 10, 100, 10^3, 3\cdot10^3, 10^4, 3\cdot10^4, 10^5, 10^6\}$, and
for each compute the mean LOO prediction error over all matches. The $s$ with the smallest error generalises best to
points that were not used in the fit (`register.py`, lines 352–353).

### Outside the barrel field

A TPS can swing wildly far from its pins, where nothing constrains it. So the final map is a blend of two maps:

- the chosen TPS inside the barrel field;
- a near-affine TPS ($s = 10^6$) far away.

The weight goes from 1 to 0 with a cosine ramp over 900 µm outside the (slightly dilated) field
(`register.py`, lines 355–360). The Jacobian determinant of the final map is checked to be positive everywhere
(`jac_min > 0`), which means no folding: the map never turns the plane inside out.

**Direction.** The final TPS is fitted from image to template (blobs → barrels). For each image pixel it says where
to *read* in the template, the same pull-back idea as in section 2, so the warped barrel map has no holes.

---

## 8. Application details (D2 settings and code locations)

### Coordinates and grids

| | Value | Where |
|---|---|---|
| Flat pixel, working voxel | 10 µm | `register.py` `PIX` |
| Laplace / streamline grid | 20 µm (20 µm input as is; 10 µm input: every 2nd voxel) | `register.py` `WORK_UM`, `prep` |
| Flat coordinates on the 10 µm grid | linear interpolation of the 20 µm streamline result | `register.py` `prep` |
| ch0 on the 10 µm grid | native for 10 µm input; linear upsampling for 20 µm input | `register.py` `prep` |

### L4 bottom and slab (new)

- **L4 bottom.** The bottom of the barrel row is found in ch0 per column of the own flatmap, with the same detection
  and parameters as in the brain-space depth analysis of the paper (`l4bottom.py`, `barrel_bottom`).
  - Columns are grouped into 100 µm flat patches and pooled over 200 µm.
  - Each pooled profile is ch0 against normalised depth (bins of 0.02, normalised by the brain white-matter depth at
    that flat position), detrended (Gaussian 1 bin − 6 bins).
  - The peak is searched at 0.25–0.6 and the trough within 0.3 below it. The bottom is the half-way point between
    them.
  - Patches without a peak are filled from neighbours, then median-smoothed. Finally the map is converted to µm and
    smoothed with a 50 µm Gaussian.
- **Slab.** Voxels with depth below the image pia within ±30 µm of (bottom + offset). Offsets run from −300 to +50 µm
  in 25 µm steps, and each fit averages 3 neighbouring offsets (±25 µm).
- **Two candidate offsets:**
  - best band-pass contrast for offsets −250 to +25 µm;
  - best pattern correlation (coarse search at 40 µm) for offsets −200 to −25 µm.

  The full fit runs at both, and the one with more matched barrels wins.

### Search

| Parameter | Value | Where |
|---|---|---|
| Image band-pass for the search | Gaussian σ 1 px − σ 15 px (10 / 150 µm) | `search.py` `BP` |
| Scales | 0.7, 0.8, …, 1.4 | `register.py` `SCALES` |
| Rotations | −20° to +20°, step 2.5° | `search.py` defaults |
| Shifts | ±600 µm, step 40 µm, on a 20 µm grid | `register.py` `MAX_T` |
| Candidates kept | 5, distinct by > 120 µm or ≥ 7.5° or ≥ 0.15 scale | `search.py` |
| Mask | template footprint dilated by 150 µm, minus artefacts | `register.py` `foot15`, `v2` |

### CPD

| Parameter | Affine | Non-rigid | Where |
|---|---|---|---|
| Outlier weight $w$ | 0.3 | 0.3 | `register.py` line 296–297 |
| Starting $\sigma^2$ | 16 px² (σ = 40 µm) | 9 px² (σ = 30 µm) | same |
| Iterations | 100 | 150 | `pointset.py` |
| $\beta$ (coherence length) | – | 25 px = 250 µm (about one barrel spacing) | same |
| $\lambda$ (smoothness) | – | 3 | same |
| $\sigma^2$ floor | 0.5 px² | 0.5 px² | `pointset.py` |

### Matching and cleaning

| Parameter | Value | Where |
|---|---|---|
| Distance gate | 70 µm (`GATE_PX` = 7) | `pointset.py` |
| Area gate | blob / atlas area in [0.12, 1.6] | `pointset.py` `hungarian` |
| Re-pairing | up to 3 rounds, TPS smoothing 100 | `register.py` lines 301–306, `REMATCH_S` |
| LOO cleaning | drop if error > max(2.5 × median, 70 µm) | `register.py` lines 307–309 |
| Minimum matches | 6 | `register.py` |
| Candidate tie | within 0.02 warped correlation → lowest affine residual (5 µm steps), then most matches | `register.py` `TIE_CORR` |

### Warp

| Parameter | Value | Where |
|---|---|---|
| TPS smoothing grid | 0, 10, 100, 1e3, 3e3, 1e4, 3e4, 1e5, 1e6, chosen by mean LOO error | `pointset.py` `SMOOTH_GRID` |
| Far field | TPS with smoothing 1e6 (≈ affine) | `register.py` line 359 |
| Blend | cosine ramp over 900 µm (`TAPER_PX` = 90) outside the field dilated by 50 µm | `register.py` lines 355–360 |

### Back to 3D

- **Barrel columns (new).** A voxel belongs to the barrel whose **convex hull** in the flat plane contains its
  corrected flat coordinate (`flatmap.py`, `hull_columns`). This is the column definition of Blue Brain's barrel
  annotations (`atlas-enhancement/barrel-annotations/annotations_to_positions.py`):
  - **Blue Brain:** the convex hull of each barrel's flat-mapped voxels, then every SSp-bfd voxel whose flat position
    lies inside it.
  - **Here:** the hull of the barrel's flat L4 footprint in the template, taken round the pixel corners. Only the
    largest connected piece of a barrel is used (2 of the 33 barrels have a stray pixel).
  - **Overlaps:** where two hulls overlap (about 3% of column points), the voxel goes to the barrel with the nearest
    pixel.
- **Depth.** Mapped piecewise linearly per column with three anchors:
  - image pia → atlas pia;
  - image L4 bottom → atlas L4 bottom (both per flat pixel);
  - brain white matter → atlas white matter.

  The atlas L4 bottom is the deepest barrel L4 voxel in each template pixel (median 623 µm).
- **CCF.** (Corrected flat coordinate, mapped atlas depth) → inverse flatmap → CCF voxel.

## References

- Myronenko A, Song X (2010). Point set registration: Coherent Point Drift. *IEEE TPAMI* 32:2262–2275.
- Kuhn HW (1955). The Hungarian method for the assignment problem. *Naval Research Logistics Quarterly* 2:83–97.
- Bookstein FL (1989). Principal warps: thin-plate splines and the decomposition of deformations. *IEEE TPAMI*
  11:567–585.
- Blue Brain Project, atlas-enhancement / barrel-annotations,
  https://github.com/BlueBrain/atlas-enhancement/tree/main/barrel-annotations
