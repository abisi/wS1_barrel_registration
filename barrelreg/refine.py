"""Segmentation-free refinement of a D2 fit by barrel contrast (BARRELREG_REFINE=1).

The point matching depends on the image barrels found by local maxima + watershed; where those fragment or merge
(AB133: barrels clearly visible, 18 of 33 matched, field squeezed into the top) the fit follows few points. Here the
pattern search's placements are re-ranked by the barrel-core vs septum contrast of the similarity-placed atlas
(no segmentation), and from the best few distinct ones every atlas barrel landmark is moved by coordinate ascent
(8 directions, steps of 60/30/15 um), keeping a move only when the contrast of the TPS-warped atlas rises and the warp
does not fold. Landmarks stay within 300 um of their start. The caller keeps the refined fit only if it beats the
matched fit; the shapes stay the atlas barrels, only the landmark positions (barrel centres) change.
"""
import numpy as np
from scipy import ndimage

from . import pointset as ps
from .atlas import SEPTA
from .search import similarity_points

N_RANK = 1500  # search placements re-ranked by contrast
N_START = 3  # distinct placements refined
CAP_PX = 30.0  # max landmark move from its start (300 um)
JAC_MIN = 0.3
STEPS = (6.0, 3.0, 1.5)
PASSES = 3
_D = np.array([[1, 0], [-1, 0], [0, 1], [0, -1], [1, 1], [1, -1], [-1, 1], [-1, -1]], float)
DIRS = _D / np.linalg.norm(_D, axis=1)[:, None]


def similarity_cols(cols, sc, ang, dy, dx, centre, pix):
    th = np.deg2rad(ang)
    Mi = np.linalg.inv(sc * np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]]))
    return ndimage.affine_transform(cols, Mi, offset=centre - Mi @ (centre + np.array([dy, dx]) / pix), order=0)


class _Scorer:
    """Contrast of the atlas warped by a TPS through (image landmark -> atlas centroid), on a box around the field."""

    def __init__(self, alab, Ac, bp, valid, s, field):
        self.alab, self.Ac, self.bp, self.valid, self.s = alab, Ac, bp, valid, s
        ys, xs = np.where(field)
        U, V = alab.shape
        self.box = (max(ys.min() - 5, 0), min(ys.max() + 6, U), max(xs.min() - 5, 0), min(xs.max() + 6, V))
        y0, y1, x0, x1 = self.box
        self.gy, self.gx = np.mgrid[y0:y1, x0:x1]
        self.grid = np.c_[self.gy.ravel(), self.gx.ravel()].astype(float)
        self.n = 0

    def cols(self, Y):
        y0, y1, x0, x1 = self.box
        d = ps.tps_fit(Y, self.Ac, self.s)(self.grid).T.reshape(2, y1 - y0, x1 - x0)
        al = ndimage.map_coordinates(self.alab, [self.gy + d[0], self.gx + d[1]], order=0, cval=0)
        return np.where(al < SEPTA, al, 0), d

    def score(self, Y):
        self.n += 1
        c, d = self.cols(Y)
        if not (c > 0).any() or ps.jacobian_2d(d)[c > 0].min() < JAC_MIN:
            return -np.inf
        y0, y1, x0, x1 = self.box
        f = self.valid[y0:y1, x0:x1] & ndimage.binary_dilation(c > 0, iterations=5)
        return ps.sep_contrast(self.bp[y0:y1, x0:x1], c, f)


def _ascend(S, Y0):
    Y, best = Y0.copy(), S.score(Y0)
    for step in STEPS:
        for _ in range(PASSES):
            moved = False
            for k in range(len(Y)):
                for dv in DIRS:
                    Yt = Y.copy()
                    Yt[k] = Y[k] + step * dv
                    if np.linalg.norm(Yt[k] - Y0[k]) > CAP_PX:
                        continue
                    sc = S.score(Yt)
                    if sc > best + 1e-4:
                        best, Y, moved = sc, Yt, True
                        break
            if not moved:
                break
    return Y, best


def refine(alab, cols, Ac, bp, valid, s, centre, search_all, pix, log=print):
    """Returns (Y image landmarks of the 33 atlas barrels, contrast, placement) of the best refined start."""
    top = search_all[np.argsort(-search_all[:, 0])[:N_RANK]]
    rank = []
    for c0, sc, ang, dy, dx in top:
        cs = similarity_cols(cols, sc, ang, dy, dx, centre, pix)
        if (cs > 0).any():
            rank.append((ps.sep_contrast(bp, cs, valid & ndimage.binary_dilation(cs > 0, iterations=5)), sc, ang, dy, dx))
    rank.sort(key=lambda r: -r[0])
    starts = []
    for r in rank:  # distinct: >= 150 um, 6 deg or 0.1 scale apart
        if all(abs(r[3] - q[3]) + abs(r[4] - q[4]) >= 150 or abs(r[2] - q[2]) >= 6 or abs(r[1] - q[1]) >= 0.1
               for q in starts):
            starts.append(r)
        if len(starts) == N_START:
            break
    S = _Scorer(alab, Ac, bp, valid, s, ndimage.binary_dilation(cols > 0, iterations=90))
    best = (-np.inf, None, None)
    for con, sc, ang, dy, dx in starts:
        Y, c = _ascend(S, similarity_points(Ac, sc, ang, np.array([dy, dx]) / pix, centre))
        log(f"   contrast refinement from scale {sc:.2f}, rot {ang:.0f}, shift {dy:.0f},{dx:.0f} um: "
            f"{con:.3f} -> {c:.3f}")
        if c > best[0]:
            best = (c, Y, (sc, ang, dy, dx))
    return best[1], best[0], best[2]
