"""The fixed 2D Blue Brain barrel template and the inverse flatmap (flat, depth) -> CCF voxel. Built once.

The Blue Brain flatmap only covers the CCF hemisphere axis 2 >= midline (brainglobe hemisphere 1). The CCF annotation
is mirror-symmetric, so one template serves both hemispheres: for brain hemisphere 2 the CCF axis-2 voxel index is
mirrored (1139 - x) at the end (register.py).
1. CCF voxels of SSp-bfd L4 (barrels + SSp-bfd4) -> Blue Brain flat coordinates.
2. Template frame T (um): affine map flat units -> the PCA plane through the same L4 voxels (e1 anterior, e2 dorsal),
   origin at the SSp-bfd L4 centroid. The brain's own flatmap uses the same plane convention and origin (centroid of
   the registered SSp-bfd L4), so the identity is the starting alignment and left and right hemispheres look the same.
   The map is linear, so the template keeps the flatmap's curved columns.
3. Barrel map from barrel L4 only (alab, 10 um px), atlas cortical thickness per flat pixel, atlas L4 depth, and the
   atlas L4 bottom (L4/L5 boundary) per flat pixel: the deepest barrel L4 voxel in each flat pixel, filled from the
   nearest pixel outside the barrels and median-filtered. It is the atlas end of the L4 depth anchor (register.py).
4. Inverse flatmap: every isocortex voxel binned by (T, depth below the atlas pia) -> mean CCF voxel position;
   empty bins filled from the nearest filled bin.
"""
import numpy as np
import tifffile
from scipy import ndimage

from .atlas import BFD_IDS, ISO_IDS, L4_BFD_IDS, SEPTA, BARREL_NAMES, collapse, is_l4
from .flatmap import Flat, flat_frame, flat_valid

PIX = 10.0
DPIX = 20.0  # depth bin of the inverse flatmap (um)


def build(cfg, hv=1):
    FM = np.load(cfg.flatmap_crop, mmap_mode="r")
    FM_LO = np.load(cfg.flatmap_crop_lo)
    ann = tifffile.memmap(cfg.annotation)
    sl = tuple(slice(int(a), int(a) + n) for a, n in zip(FM_LO, FM.shape[:3]))
    A = np.asarray(ann[sl])
    Fa = np.asarray(FM).astype(np.float32)
    rr = np.arange(sl[2].start, sl[2].stop)[None, None]
    mid = ann.shape[2] // 2
    hemi = np.broadcast_to((rr >= mid) if hv == 1 else (rr < mid), A.shape)
    ok = flat_valid(Fa) & hemi
    bfd = np.isin(A, BFD_IDS)
    l4 = np.isin(A, L4_BFD_IDS) & ok  # all SSp-bfd L4 (barrels + SSp-bfd4), as Perens labels it: plane and origin
    l4b = is_l4(A) & bfd & ok  # barrel L4 only: barrel map and L4 depth
    X = np.argwhere(l4).astype(float)
    c, e1, e2, nrm = flat_frame(X)
    Pp = np.stack([(X - c) @ e1, (X - c) @ e2], 1) * PIX  # um in the L4 plane
    Fl4 = Fa[l4].astype(np.float64)
    Ph = np.c_[Fl4, np.ones(len(Fl4))]
    L, *_ = np.linalg.lstsq(Ph, Pp, rcond=None)  # (3, 2): T = [F, 1] @ L
    res = np.linalg.norm(Ph @ L - Pp, axis=1)
    T0 = (Ph @ L).mean(0)
    L[2] -= T0  # origin at the SSp-bfd L4 centroid

    def to_T(F):
        return np.c_[F.astype(np.float64), np.ones(len(F))] @ L

    Tl4 = to_T(Fa[l4b])
    flat = Flat(Tl4, PIX, pad=100)
    bar = collapse(A)
    labA = np.where(bar[l4b] > 0, bar[l4b], SEPTA)
    alab, cnt = flat.mode(Tl4, labA, SEPTA + 1)
    alab = np.where(cnt > 0, ndimage.median_filter(alab, 3), 0).astype(np.int32)
    cols = np.where(alab < SEPTA, alab, 0)
    adepth = ndimage.distance_transform_edt(A > 0) * PIX
    iso = np.isin(A, ISO_IDS) & ok
    Ti = to_T(Fa[iso])
    inG = (Ti >= flat.lo).all(1) & (Ti < flat.lo + np.array(flat.shape) * PIX).all(1)
    b = flat.bins(Ti[inG])
    di = adepth[iso][inG]
    thick = np.zeros(np.prod(flat.shape))
    np.maximum.at(thick, b, di)
    thick = ndimage.median_filter(thick.reshape(flat.shape), 5)
    atl_l4_depth = float(np.median(adepth[l4b]))
    l4bot = np.zeros(np.prod(flat.shape))
    np.maximum.at(l4bot, flat.bins(Tl4), adepth[l4b])
    l4bot = l4bot.reshape(flat.shape)
    _, ind = ndimage.distance_transform_edt(l4bot <= 0, return_indices=True)
    l4bot = ndimage.median_filter(l4bot[tuple(ind)], 5).astype(np.float32)
    # inverse flatmap: (flat bin, depth bin) -> mean CCF voxel (10 um CCF voxel index, full annotation frame)
    nd = int(np.ceil(adepth[iso].max() / DPIX)) + 1
    kd = np.minimum((di / DPIX).astype(int), nd - 1)
    key = b * nd + kd
    vox = np.argwhere(iso)[inG] + FM_LO
    n = np.bincount(key, minlength=np.prod(flat.shape) * nd)
    inv = np.stack([np.bincount(key, weights=vox[:, k], minlength=len(n)) for k in range(3)], -1) / np.maximum(n, 1)[:, None]
    inv = inv.reshape(flat.shape + (nd, 3)).astype(np.float32)
    have = n.reshape(flat.shape + (nd,)) > 0
    _, ind = ndimage.distance_transform_edt(~have, return_indices=True)
    inv = inv[tuple(ind)]
    print(f"hemisphere {hv}: {l4.sum()} SSp-bfd L4 voxels ({l4b.sum()} barrel L4), flat->plane affine residual median "
          f"{np.median(res):.0f} um (95th pct {np.percentile(res, 95):.0f}), grid {flat.shape}, "
          f"{len(np.unique(cols)) - 1} of {len(BARREL_NAMES)} barrels, atlas L4 depth {atl_l4_depth:.0f} um, "
          f"L4 bottom {np.median(l4bot[cols > 0]):.0f} um, "
          f"inverse table {inv.shape}, filled {have.mean():.2f}", flush=True)
    np.savez_compressed(cfg.template_file, L=L, lo=flat.lo, shape=flat.shape, pix=PIX, alab=alab, thick=thick,
                        atl_l4_depth=atl_l4_depth, l4_bottom=l4bot, inv=inv, inv_have=have, dpix=DPIX, plane_res_um=np.median(res),
                        frame=np.stack([c + FM_LO, e1, e2, nrm]))
