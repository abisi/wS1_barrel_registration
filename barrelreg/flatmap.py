"""Cortical depth (Laplace), streamlines, the flat plane frame, flat-grid binning, barrel columns by convex hull, and
the Blue Brain flatmap crop."""
import zlib

import numpy as np
import tifffile
from scipy import ndimage, sparse
from scipy.sparse.linalg import cg
from scipy.spatial import ConvexHull, Delaunay

from .atlas import BFD_IDS, ISO_IDS


def flat_valid(F):
    return np.isfinite(F).all(-1) & (F >= 0).all(-1)


def convert_flatmap(nrrd, annotation, out_crop, out_lo, margin=120):
    """flatmap_float.nrrd (gzip, sizes 2 1320 800 1140, fastest first) -> atlas-space asr crop around SSp-bfd (.npy)."""
    raw = open(nrrd, "rb").read()
    hdr_end = raw.index(b"\n\n") + 2
    buf = zlib.decompress(raw[hdr_end:], 15 + 32)
    fm = np.frombuffer(buf, "<f4").reshape(1140, 800, 1320, 2)  # [lr, dv, ap, c] (nrrd: first size fastest)
    fm = fm.transpose(2, 1, 0, 3)  # -> [ap, dv, lr, c] = brainglobe asr
    ann = tifffile.imread(annotation)
    assert ann.shape == fm.shape[:3], (ann.shape, fm.shape)
    ys = np.argwhere(np.isin(ann[::4, ::4, ::4], BFD_IDS)) * 4
    lo = np.maximum(ys.min(0) - margin, 0)
    hi = np.minimum(ys.max(0) + margin, ann.shape)
    crop = np.ascontiguousarray(fm[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]])
    np.save(out_crop, crop)
    np.save(out_lo, lo)
    print(f"flatmap crop {lo}-{hi} {crop.shape}, valid fraction {flat_valid(crop).mean():.3f}", flush=True)


def laplace_depth(atl, tol=1e-6, maxiter=3000):
    """u = 0 at pia (outside brain), 1 at white matter / non-cortex; Neumann at crop edges."""
    iso = np.isin(atl, ISO_IDS)
    inside = atl > 0
    ctx = iso
    pia = ~inside
    wm = inside & ~ctx
    idx = -np.ones(atl.shape, np.int64)
    n = int(ctx.sum())
    idx[ctx] = np.arange(n)
    rows, cols, diag = [], [], np.zeros(n)
    rhs = np.zeros(n)
    for ax in range(3):
        for sh in (1, -1):
            nb_idx = np.roll(idx, -sh, axis=ax)
            nb_pia = np.roll(pia, -sh, axis=ax)
            nb_wm = np.roll(wm, -sh, axis=ax)
            valid = np.ones(atl.shape, bool)  # crop edge = Neumann
            sl = [slice(None)] * 3
            sl[ax] = slice(-1, None) if sh == 1 else slice(0, 1)
            valid[tuple(sl)] = False
            m = ctx & valid
            a = idx[m]
            b = nb_idx[m]
            inn = b >= 0
            rows.append(a[inn]); cols.append(b[inn])
            diag[a[inn]] += 1
            p = nb_pia[m]
            w = nb_wm[m] & ~inn
            diag[a[p]] += 1
            diag[a[w]] += 1
            rhs[a[w]] += 1.0
    r = np.concatenate(rows); c = np.concatenate(cols)
    A = sparse.csr_matrix((-np.ones(len(r)), (r, c)), shape=(n, n)) + sparse.diags(diag)
    # initial guess from distances
    dp = ndimage.distance_transform_edt(~pia)
    dw = ndimage.distance_transform_edt(~wm)
    x0 = (dp / np.maximum(dp + dw, 1e-6))[ctx]
    isolated = diag == 0
    diag[isolated] = 1
    M = sparse.diags(1.0 / diag)
    x, info = cg(A, rhs, x0=x0, rtol=tol, maxiter=maxiter, M=M)
    u = np.full(atl.shape, np.nan, np.float32)
    u[ctx] = np.clip(x, 0, 1)
    ue = u.copy()
    ue[pia] = 0
    ue[wm] = 1
    return u, ue, ctx, info


def trace_to_level(ue, u0, pts, h=0.4, nmax=400):
    """Follow +-grad(u) from each point until u == u0. Returns end points (voxel coords)."""
    ues = ndimage.gaussian_filter(ue, 1.0)
    g = np.stack(np.gradient(ues))
    X = pts.astype(np.float32).copy()
    uu = ndimage.map_coordinates(ues, X.T, order=1)
    s = np.sign(u0 - uu)
    active = s != 0
    for _ in range(nmax):
        if not active.any():
            break
        Xa = X[active]
        ga = np.stack([ndimage.map_coordinates(g[k], Xa.T, order=1) for k in range(3)], 1)
        nrm = np.linalg.norm(ga, axis=1, keepdims=True)
        ga = ga / np.maximum(nrm, 1e-9)
        Xn = Xa + h * s[active, None] * ga
        un = ndimage.map_coordinates(ues, Xn.T, order=1)
        ua = ndimage.map_coordinates(ues, Xa.T, order=1)
        crossed = np.sign(u0 - un) != s[active]
        stuck = nrm[:, 0] < 1e-7
        t = np.clip((u0 - ua) / np.where(np.abs(un - ua) > 1e-9, un - ua, 1e-9), 0, 1)
        Xn[crossed] = Xa[crossed] + t[crossed, None] * (Xn[crossed] - Xa[crossed])
        X[active] = Xn
        ai = np.where(active)[0]
        active[ai[crossed | stuck]] = False
    return X, active  # active == True -> did not converge


def flat_frame(P):
    """PCA plane through reference-surface points P (voxel coords, asr). e1 ~ anterior, e2 ~ dorsal."""
    c = P.mean(0)
    _, _, vt = np.linalg.svd(P - c, full_matrices=False)
    e1, e2, nrm = vt
    # e1: most AP-aligned in-plane direction
    cand = [e1, e2]
    k = int(np.argmax([abs(v[0]) for v in cand]))
    a = cand[k]
    if a[0] > 0:  # asr: axis0 increases posterior -> want anterior positive
        a = -a
    b = np.cross(nrm, a)
    if b[1] > 0:  # axis1 increases inferior -> want dorsal positive
        b = -b
    return c, a, b, nrm


class Flat:
    """A 2D flat grid (pixel `pix` um, origin `lo`) with binning of flat coordinates F (N, 2) in um."""

    def __init__(self, F, pix_um, pad=20):
        self.pix = pix_um
        self.lo = F.min(0) - pad * pix_um
        hi = F.max(0) + pad * pix_um
        self.shape = tuple(np.ceil((hi - self.lo) / pix_um).astype(int))

    @classmethod
    def from_grid(cls, lo, pix, shape):
        f = cls.__new__(cls)
        f.lo, f.pix, f.shape = lo, float(pix), tuple(int(s) for s in shape)
        return f

    def bins(self, F):
        ij = np.floor((F - self.lo) / self.pix).astype(int)
        ij = np.clip(ij, 0, np.array(self.shape) - 1)
        return ij[:, 0] * self.shape[1] + ij[:, 1]

    def mean(self, F, v, fill=True):
        b = self.bins(F)
        n = np.bincount(b, minlength=np.prod(self.shape)).reshape(self.shape).astype(np.float64)
        s = np.bincount(b, weights=v, minlength=np.prod(self.shape)).reshape(self.shape)
        if fill:  # normalised convolution to fill small holes
            ns = ndimage.gaussian_filter(n, 0.7)
            ss = ndimage.gaussian_filter(s, 0.7)
            img = np.where(n > 0, s / np.maximum(n, 1e-9), ss / np.maximum(ns, 1e-9))
            valid = (n > 0) | (ns > 0.05)
        else:
            img = s / np.maximum(n, 1e-9)
            valid = n > 0
        return img.astype(np.float32), valid, n

    def mode(self, F, lab, nlab):
        b = self.bins(F)
        h = np.bincount(b * nlab + lab, minlength=np.prod(self.shape) * nlab).reshape(-1, nlab)
        out = h.argmax(1).reshape(self.shape)
        cnt = h.sum(1).reshape(self.shape)
        return out.astype(np.int32), cnt


def hull_columns(cols, pts):
    """Barrel column of each flat point (continuous pixel coordinates of the barrel map `cols`, 0 = no barrel).

    As in Blue Brain's barrel annotations (atlas-enhancement/barrel-annotations/annotations_to_positions.py): the
    column of a barrel is every point whose flat position lies inside the convex hull of the barrel's flat L4
    footprint. The hull goes round the pixel corners and uses the barrel's largest connected piece (a stray pixel
    would stretch it). Where hulls of neighbouring barrels overlap, the point goes to the barrel with the nearest
    pixel. Returns the labels and the number of points that were inside more than one hull."""
    out = np.zeros(len(pts), np.int32)
    hits = np.zeros(len(pts), np.int16)
    corners = np.array([[-0.5, -0.5], [-0.5, 0.5], [0.5, -0.5], [0.5, 0.5]])
    for k in np.unique(cols[cols > 0]):
        lab, n = ndimage.label(cols == k)
        if n > 1:
            sz = np.bincount(lab.ravel())
            sz[0] = 0
            lab = lab == sz.argmax()
        P = (np.argwhere(lab)[:, None] + corners).reshape(-1, 2)
        hv = P[ConvexHull(P).vertices]
        box = np.flatnonzero((pts >= hv.min(0)).all(1) & (pts <= hv.max(0)).all(1))
        inside = box[Delaunay(hv).find_simplex(pts[box]) >= 0]
        out[inside] = k
        hits[inside] += 1
    multi = np.flatnonzero(hits > 1)
    if len(multi):
        _, ind = ndimage.distance_transform_edt(cols == 0, return_indices=True)
        r = tuple(np.clip(np.rint(pts[multi]).astype(int), 0, np.array(cols.shape) - 1).T)
        out[multi] = cols[ind[0][r], ind[1][r]]
    return out, len(multi)
