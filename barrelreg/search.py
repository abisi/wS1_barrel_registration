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
    brute_force.last_all = np.array(res, np.float32)  # every placement scored (read by the debug dump)
    return tops


def similarity_points(P, sc, ang, d_px, centre):
    th = np.deg2rad(ang)
    Rm = sc * np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    return (P - centre) @ Rm.T + centre + d_px
