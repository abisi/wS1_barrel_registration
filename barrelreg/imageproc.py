"""Image helpers: band-pass, contrast, tissue / artefact masks, barrel segmentation in a flat L4 image."""
import numpy as np
from scipy import ndimage
from skimage.feature import peak_local_max
from skimage.segmentation import watershed

from .atlas import ISO_IDS

PK_SIGMA = 3.5  # marker scale (px); ~ barrel radius / 2


def bandpass(x, valid, s1=1.2, s2=8):
    x = np.where(valid, x, np.median(x[valid]) if valid.any() else 0)
    return ndimage.gaussian_filter(x, s1) - ndimage.gaussian_filter(x, s2)


def contrast(bp, valid):
    """Spread of a band-passed image (clipped std); high when barrels and septa alternate."""
    if valid.sum() < 100:
        return 0.0
    lo, hi = np.percentile(bp[valid], [5, 95])
    return float(np.std(np.clip(bp[valid], lo, hi)))


def tissue_mask(ch0, atl):
    """Image-based brain tissue mask (for the pial boundary). Threshold halfway (in log intensity) between the
    background (registered outside, far from the brain) and cortex intensities, smoothed; holes filled.
    Its morphology wipes the array border: use tissue_mask_padded on crops."""
    sm = ndimage.gaussian_filter(ch0.astype(np.float32), 1.5)
    lg = np.log1p(sm)
    inside = atl > 0
    far_out = ~ndimage.binary_dilation(inside, iterations=15)
    ctx = np.isin(atl, ISO_IDS) & ndimage.binary_erosion(inside, iterations=15)
    if far_out.sum() < 1000:  # crop fully inside: fall back to the lowest percentile
        bg = np.percentile(lg, 2)
    else:
        bg = np.median(lg[far_out])
    fg = np.median(lg[ctx]) if ctx.any() else np.percentile(lg, 70)
    thr = bg + 0.4 * (fg - bg)
    m = lg > thr
    m = ndimage.binary_opening(m, iterations=1)
    lab, n = ndimage.label(m)
    if n > 1:
        sz = np.bincount(lab.ravel()); sz[0] = 0
        m = lab == sz.argmax()
    # fill holes slice-wise along each axis (vessels / dark septa inside tissue)
    m = ndimage.binary_closing(m, iterations=2)
    for ax in range(3):
        m = np.stack([ndimage.binary_fill_holes(s) for s in np.moveaxis(m, ax, 0)], ax)
    return m, float(np.expm1(thr))


def tissue_mask_padded(ch0, atl, pw=10):
    """tissue_mask on an edge-padded crop, so the border damage lands in the padding (otherwise every crop face
    looks like pia and depth below the pia is partly measured from the crop faces)."""
    return tissue_mask(np.pad(ch0, pw, mode="edge"), np.pad(atl, pw, mode="edge"))[0][pw:-pw, pw:-pw, pw:-pw]


def artefact_mask(img, valid, region, z=4.5, grow=3):
    """Lesions / blood vessels / bright debris in a flat image: |robust z| > z within the region, dilated."""
    v = img[region & valid]
    med = np.median(v)
    mad = 1.4826 * np.median(np.abs(v - med)) + 1e-9
    zz = (img - med) / mad
    m = (np.abs(zz) > z) & valid
    return ndimage.binary_dilation(m, iterations=grow)


def lesion_mask(img, valid, region, sigma=4, z=2.5, min_px=400, grow=8):
    """Large lesions (probe / injection damage, holes with bright rims) in a flat image: the background-flattened
    image smoothed at ~40 um, |robust z| > z within the region, components of >= min_px (~2 barrels) kept, dilated by
    grow px to take the rim. artefact_mask only catches their cores; the rims then dominate the band-pass and search."""
    w = valid.astype(np.float32)
    x = np.where(valid, np.nan_to_num(img), 0).astype(np.float32)
    bg = ndimage.gaussian_filter(x, 30) / np.maximum(ndimage.gaussian_filter(w, 30), 1e-3)
    s = ndimage.gaussian_filter(x, sigma) / np.maximum(ndimage.gaussian_filter(w, sigma), 1e-3) / np.maximum(bg, 1e-6)
    v = s[region & valid]
    med = np.median(v)
    zz = (s - med) / (1.4826 * np.median(np.abs(v - med)) + 1e-9)
    m = ndimage.binary_opening((np.abs(zz) > z) & valid, iterations=2)
    lab, n = ndimage.label(m)
    if n:
        area = np.bincount(lab.ravel())
        area[0] = 0
        m = (area >= min_px)[lab]
    return ndimage.binary_dilation(m, iterations=grow)


def l4_bandpass(l4, valid, field):
    x = l4.copy()
    x = np.minimum(x, np.percentile(x[valid & field], 97))  # bright puncta (cells, vessels)
    x = ndimage.median_filter(x, 3)
    return bandpass(x, valid, 1.0, 8)


def segment(bp, roi):
    """Barrels in a band-passed flat L4 image: local maxima + watershed. Returns labels, centroids, areas, strength."""
    sm = ndimage.gaussian_filter(bp, 1.2)
    pk_img = ndimage.gaussian_filter(bp, PK_SIGMA)
    thr = max(0.0, np.percentile(pk_img[roi], 50))
    pk = peak_local_max(np.where(roi, pk_img, pk_img.min()), min_distance=6, threshold_abs=thr, exclude_border=False)
    mk = np.zeros(bp.shape, int)
    mk[tuple(pk.T)] = np.arange(1, len(pk) + 1)
    lab = watershed(-sm, mk, mask=(sm > 0) & roi)
    area = np.bincount(lab.ravel())
    keep = (area >= 25) & (area <= 1200)
    keep[0] = False
    lab = np.where(keep[lab], lab, 0)
    u = np.unique(lab)[1:]
    lut = np.zeros(lab.max() + 1, int)
    lut[u] = np.arange(1, len(u) + 1)
    lab = lut[lab]
    n = lab.max()
    idx = np.arange(1, n + 1)
    cen = np.array(ndimage.center_of_mass(np.ones_like(lab), lab, idx)).reshape(-1, 2)
    area = ndimage.sum(np.ones_like(lab), lab, idx)
    strength = ndimage.mean(bp, lab, idx)
    return lab, cen, np.asarray(area), np.asarray(strength)
