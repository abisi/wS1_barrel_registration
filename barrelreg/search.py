"""Exhaustive similarity search of the synthetic atlas barrel pattern over a flat L4 image."""
import numpy as np
from scipy import ndimage

from .atlas import SEPTA

BP = (1.0, 15)  # band-pass sigmas in 10 um px


def synth(alab, sigma=1.0):
    """Synthetic image from barrel labels (barrels 1, septa / outside 0), band-passed like the image."""
    s = np.where((alab > 0) & (alab < SEPTA), 1.0, 0.0)
    s = ndimage.gaussian_filter(s, sigma)
    out = ndimage.gaussian_filter(s, BP[0]) - ndimage.gaussian_filter(s, BP[1])
    return out.astype(np.float32)


def znorm(x, m):
    lo, hi = np.percentile(x[m], [1, 99])
    x = np.clip(x, lo, hi)
    return ((x - x[m].mean()) / (x[m].std() + 1e-9)).astype(np.float32)


def masked_corr(a, b, m):
    a = a[m]; b = b[m]
    a = a - a.mean(); b = b - b.mean()
    return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum() + 1e-12))


def _score(fix, mov, mask, pix, centre, sc, ang, dy, dx):
    th = np.deg2rad(ang)
    M = np.linalg.inv(sc * np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]]))
    d = np.array([dy, dx]) / pix
    w = ndimage.affine_transform(mov, M, offset=centre - M @ (centre + d), order=1, cval=0)
    return masked_corr(fix, w, mask)


def _distinct(res, topk):
    tops = []
    for r in sorted(res, key=lambda r: -r[0]):
        if all(np.hypot(r[3] - t[3], r[4] - t[4]) > 120 or abs(r[2] - t[2]) >= 7.5 or abs(r[1] - t[1]) >= 0.15
               for t in tops):
            tops.append(r)
        if len(tops) == topk:
            break
    return tops


def brute_force(fix, mov, mask, pix, centre, topk=5, max_t=400, step_t=40, max_r=20, step_r=2.5,
                scales=(0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4), refine=False):
    """Every scale x rotation x shift of `mov` scored by masked correlation with `fix`.
    Returns the top-k distinct (corr, scale, rot_deg, dy_um, dx_um). With `refine`, each of the top-k is then
    polished on a fine local grid (scale +-0.05 by 0.025, rotation +-step_r by 1 deg, shift +-step_t by step_t / 4)."""
    res = []
    for sc in scales:
        for ang in np.arange(-max_r, max_r + 1e-6, step_r):
            for dy in np.arange(-max_t, max_t + 1e-6, step_t):
                for dx in np.arange(-max_t, max_t + 1e-6, step_t):
                    res.append((_score(fix, mov, mask, pix, centre, sc, ang, dy, dx), sc, ang, dy, dx))
    brute_force.last_all = np.array(res, np.float32)  # the coarse grid, every placement (read by the debug dump)
    tops = _distinct(res, topk)
    brute_force.last_coarse = list(tops)
    if refine:
        fine = []
        for _, sc0, a0, y0, x0 in tops:
            loc = [(_score(fix, mov, mask, pix, centre, sc, ang, dy, dx), sc, ang, dy, dx)
                   for sc in sc0 + np.arange(-0.05, 0.05 + 1e-6, 0.025)
                   for ang in a0 + np.arange(-step_r, step_r + 1e-6, 1.0)
                   for dy in y0 + np.arange(-step_t, step_t + 1e-6, step_t / 4)
                   for dx in x0 + np.arange(-step_t, step_t + 1e-6, step_t / 4)]
            fine.append(max(loc, key=lambda r: r[0]))
        tops = _distinct(fine, topk)
    return tops


def similarity_points(P, sc, ang, d_px, centre):
    th = np.deg2rad(ang)
    Rm = sc * np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    return (P - centre) @ Rm.T + centre + d_px
