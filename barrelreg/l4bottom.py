"""Bottom of the barrel row (the L4/L5 boundary) in ch0, per column of the brain's own flatmap.

The same detection as barrel_bottom() in the brain-space depth analysis used elsewhere in the paper
(brain_space_depth/columns.py), with the same parameters:

- Columns are grouped into 100 um patches. The patch profiles are pooled over 200 um (about one barrel plus septum, so
  every pooled profile contains barrel).
- Each pooled profile is mean ch0 against normalised depth (bins of 0.02). The slow pia-to-WM brightness trend is
  removed (Gaussian 1 bin minus Gaussian 6 bins).
- The barrel peak is searched at normalised depth 0.25-0.6, and the trough within 0.3 below it. The bottom is where
  the profile has fallen halfway from the peak to the trough (linear between bins).
- Patches without a peak are filled from their neighbours, and the estimate is median-smoothed over the surface.

Two things differ, because D2 has its own columns and its own cortex bottom:

- A column is a patch of the own flatmap (voxels along the Laplace streamlines). In the paper it was the voxels whose
  nearest pia point lies in a surface patch.
- Normalised depth is depth below the (closed) image pia divided by the brain's white-matter depth at that flat
  position (`bw` in register.py). In the paper it was d_pia / (d_pia + d_wm) to the registered cortex bottom.

The result is converted to um below the image pia per flat pixel (normalised bottom x white-matter depth) and smoothed
with a 50 um Gaussian to remove the steps between 100 um patches.
"""
import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.spatial import cKDTree

PATCH_UM = 100.0  # BARREL_PATCH_VOXELS (5) x 20 um
POOL_RADIUS_UM = 200.0  # BARREL_POOL_RADIUS_UM
PROFILE_BIN = 0.02  # BARREL_PROFILE_BIN (normalised depth)
PEAK_RANGE = (0.25, 0.6)  # BARREL_PEAK_RANGE (normalised depth)
MAX_NORM_DEPTH = 1.25
CH0_SIGMA_UM = 14.0  # ch0 smoothing of the paper: 0.7 voxels at 20 um
SMOOTH_UM = 50.0  # flat-plane smoothing of the per-pixel bottom


def barrel_bottom(ch0, depth, G, sel, flat, wm):
    """Barrel bottom per flat pixel of `flat`, in um below the image pia.

    ch0, depth (um below the image pia), G (own flat coordinate, um, per voxel): the 10 um working grid of the crop;
    sel: the voxels to use; wm: brain white-matter depth per flat pixel (um, filled). Returns the bottom map (um), the
    normalised bottom map, and a per-patch table for QC (raw = own peak, NaN where none was found)."""
    pix = flat.pix
    smooth = ndimage.gaussian_filter(ch0, CH0_SIGMA_UM / pix)[sel]
    F = G[sel]
    norm = depth[sel] / np.maximum(wm.ravel()[flat.bins(F)], 1.0)
    keep = norm < MAX_NORM_DEPTH
    F, norm, smooth = F[keep], norm[keep], smooth[keep]
    n_bins = int(np.ceil(MAX_NORM_DEPTH / PROFILE_BIN))
    centres = (np.arange(n_bins) + 0.5) * PROFILE_BIN
    which = np.minimum((norm / PROFILE_BIN).astype(int), n_bins - 1)

    patch_ij, patch = np.unique(np.floor((F - flat.lo) / PATCH_UM).astype(int), axis=0, return_inverse=True)
    patch = patch.ravel()
    n_patches = len(patch_ij)
    sums = np.zeros((n_patches, n_bins))
    counts = np.zeros((n_patches, n_bins))
    np.add.at(sums, (patch, which), smooth)
    np.add.at(counts, (patch, which), 1)

    tree = cKDTree(patch_ij.astype(float))
    neighbours = tree.query_ball_point(patch_ij.astype(float), POOL_RADIUS_UM / PATCH_UM)
    lo, hi = PEAK_RANGE
    span = (centres > 0.15) & (centres < 0.85)
    window = np.flatnonzero((centres > lo) & (centres < hi))

    raw = np.full(n_patches, np.nan)
    for i, near in enumerate(neighbours):
        n = counts[near].sum(0)
        if (n[span] > 0).mean() < 0.8:
            continue
        profile = np.where(n > 0, sums[near].sum(0) / np.maximum(n, 1), np.nan)
        profile = pd.Series(profile).interpolate(limit_direction="both").to_numpy()
        profile = ndimage.gaussian_filter1d(profile, 1) - ndimage.gaussian_filter1d(profile, 6, mode="nearest")
        peak = window[np.argmax(profile[window])]
        below = np.flatnonzero((centres > centres[peak]) & (centres < centres[peak] + 0.3))
        trough = below[np.argmin(profile[below])]
        half = (profile[peak] + profile[trough]) / 2
        segment = np.arange(peak, trough + 1)
        j = segment[np.argmax(profile[segment] < half)]
        if j == peak or profile[peak] <= profile[trough]:
            continue
        raw[i] = centres[j - 1] + (profile[j - 1] - half) / (profile[j - 1] - profile[j]) * PROFILE_BIN

    # fill, then smooth, over the surface
    smoothed = np.full(n_patches, np.nan)
    for i, near in enumerate(neighbours):
        local = raw[near]
        local = local[np.isfinite(local)]
        if len(local):
            smoothed[i] = np.median(local)
    if not np.isfinite(raw).any():
        raise RuntimeError("no L4 bottom found in any patch")
    smoothed[~np.isfinite(smoothed)] = np.nanmedian(raw)

    # per flat pixel: the nearest patch, x the white-matter depth, smoothed over the plane
    U, V = flat.shape
    gy, gx = np.mgrid[0:U, 0:V]
    centre_ij = (np.c_[gy.ravel(), gx.ravel()] + 0.5) * pix / PATCH_UM - 0.5
    _, nearest = tree.query(centre_ij)
    bottom_norm = smoothed[nearest].reshape(flat.shape)
    bottom_um = ndimage.gaussian_filter(bottom_norm * wm, SMOOTH_UM / pix, mode="nearest").astype(np.float32)
    patches = pd.DataFrame({"flat_u_um": flat.lo[0] + (patch_ij[:, 0] + 0.5) * PATCH_UM,
                            "flat_v_um": flat.lo[1] + (patch_ij[:, 1] + 0.5) * PATCH_UM,
                            "n_voxels": counts.sum(1).astype(int),
                            "l4_bottom_raw": raw, "l4_bottom": smoothed})
    return bottom_um, bottom_norm.astype(np.float32), patches
