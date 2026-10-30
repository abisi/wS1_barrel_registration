"""Barrel point-set registration: CPD, one-to-one matching, thin-plate splines, and overlap metrics."""
import numpy as np
from scipy import ndimage
from scipy.interpolate import RBFInterpolator
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from skimage.segmentation import find_boundaries

from .atlas import BARREL_NAMES

GATE_PX = 7.0  # max residual (70 um) for accepting a correspondence after the non-rigid CPD / TPS
REMATCH_S = 100.0
TAPER_PX = 90  # displacement decays over 900 um outside the (dilated) barrel field
SMOOTH_GRID = [0.0, 10.0, 100.0, 1e3, 3e3, 1e4, 3e4, 1e5, 1e6]  # RBF smoothing (chosen by LOO); 1e6 ~ affine


def atlas_points(cols):
    ks = [k for k in np.unique(cols) if k > 0]
    cen = np.array(ndimage.center_of_mass(np.ones_like(cols), cols, ks))
    area = np.asarray(ndimage.sum(np.ones_like(cols), cols, ks))
    return np.array(ks), cen, area


def _estep(X, TY, s2, w):
    D = X.shape[1]
    d = cdist(TY, X, "sqeuclidean")
    num = np.exp(-d / (2 * s2))
    c = (2 * np.pi * s2) ** (D / 2) * w / (1 - w) * TY.shape[0] / X.shape[0]
    return num / (num.sum(0, keepdims=True) + c)


def cpd_affine(X, Y, w=0.3, iters=100, s2=None):
    D = 2
    B, t = np.eye(2), np.zeros(2)
    if s2 is None:
        s2 = cdist(Y, X, "sqeuclidean").mean() / D
    for _ in range(iters):
        TY = Y @ B.T + t
        P = _estep(X, TY, s2, w)
        Np = P.sum()
        if Np < 1e-6:
            break
        muX = X.T @ P.sum(0) / Np
        muY = Y.T @ P.sum(1) / Np
        Xh, Yh = X - muX, Y - muY
        Am = Xh.T @ P.T @ Yh
        YPY = Yh.T @ (P.sum(1)[:, None] * Yh)
        B = Am @ np.linalg.inv(YPY)
        t = muX - B @ muY
        s2n = (np.sum(P.sum(0) * (Xh ** 2).sum(1)) - np.trace(Am @ B.T)) / (Np * D)
        s2 = max(abs(s2n), 0.5)
    return Y @ B.T + t, B, t, s2


def cpd_nonrigid(X, Y, beta=25.0, lam=3.0, w=0.3, iters=150, s2=None):
    D = 2
    M = len(Y)
    G = np.exp(-cdist(Y, Y, "sqeuclidean") / (2 * beta ** 2))
    W = np.zeros((M, D))
    if s2 is None:
        s2 = cdist(Y, X, "sqeuclidean").mean() / D
    TY = Y.copy()
    for _ in range(iters):
        P = _estep(X, TY, s2, w)
        P1, Pt1 = P.sum(1), P.sum(0)
        Np = P1.sum()
        PX = P @ X
        W = np.linalg.solve(P1[:, None] * G + lam * s2 * np.eye(M), PX - P1[:, None] * Y)
        TY = Y + G @ W
        s2 = max((np.sum(Pt1 * (X ** 2).sum(1)) - 2 * np.sum(PX * TY) + np.sum(P1 * (TY ** 2).sum(1))) / (Np * D), 0.5)
    return TY, P


def hungarian(TA, B, gate, areaA=None, areaB=None):
    d = cdist(TA, B)
    cost = d.copy()
    bad = d > gate
    if areaA is not None:  # a blob much larger than its atlas barrel = merged barrels; much smaller = fragment
        ratio = areaB[None] / np.maximum(areaA[:, None], 1)
        bad |= (ratio > 1.6) | (ratio < 0.12)
    cost[bad] = 1e6
    r, c = linear_sum_assignment(cost)
    ok = cost[r, c] < 1e6
    return r[ok], c[ok]


def tps_fit(P, Q, s):
    """RBF thin-plate spline mapping points P -> displacement Q - P."""
    return RBFInterpolator(P, Q - P, kernel="thin_plate_spline", smoothing=s, degree=1)


def loo(P, Q, s):
    """Leave-one-out error of the TPS P -> Q (px)."""
    err = []
    for i in range(len(P)):
        m = np.ones(len(P), bool)
        m[i] = False
        f = tps_fit(P[m], Q[m], s)
        err.append(np.linalg.norm(P[i] + f(P[i:i + 1])[0] - Q[i]))
    return np.array(err)


def sep_contrast(bp, cols, field):
    """Barrel-core vs. atlas-septum contrast of the image (segmentation-free), in units of field std."""
    core = np.zeros(cols.shape, bool)
    for k in np.unique(cols)[1:]:
        core |= ndimage.binary_erosion(cols == k, iterations=3)
    bnd = find_boundaries(cols, mode="inner") & (cols > 0)
    bnd = ndimage.binary_dilation(bnd) & ndimage.binary_erosion(cols > 0, iterations=1)
    sd = bp[field].std()
    return float((bp[core].mean() - bp[bnd & ~core].mean()) / sd)


def overlap_metrics(cols, blobs, names_of_blob):
    """names_of_blob: {blob label: barrel idx} (final correspondences). Dice per barrel, naming by max overlap."""
    dice, cont, correct = [], [], []
    for j, k in names_of_blob.items():
        bm = blobs == j
        am = cols == k
        inter = (bm & am).sum()
        dice.append(2 * inter / (bm.sum() + am.sum()))
        cont.append(inter / bm.sum())
        ov = np.bincount(cols[bm], minlength=len(BARREL_NAMES) + 1)
        ov[0] = 0
        correct.append(int(ov.argmax() == k and ov.max() > 0))
    return float(np.mean(dice)), float(np.mean(cont)), float(np.mean(correct))


def jacobian_2d(disp):
    """Jacobian determinant of y -> y + disp(y), disp shape (2, U, V)."""
    du0 = np.gradient(disp[0])
    du1 = np.gradient(disp[1])
    return (1 + du0[0]) * (1 + du1[1]) - du0[1] * du1[0]
