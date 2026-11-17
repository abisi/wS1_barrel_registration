"""D2 per brain: D's barrel matching on the brain's OWN flatmap, a coarse registration, and the fixed 2D Blue
Brain barrel template, ending in Allen CCF through the barrel matches.

1. coarse step: brainreg to perens_lsfm_mouse_20um or any other atlas with Allen region IDs (sample space asr, 20 or
   10 um). Used only for the hemisphere, the rough SSp-bfd area and the pia / white-matter surfaces.
2. own flatmap: Laplace between the registered pia and white matter on the 20 um crop -> streamlines to the registered
   L4 level -> PCA plane (e1 anterior, e2 dorsal), origin at the centroid of the registered SSp-bfd L4. Same plane
   convention and origin as the template, so the identity is the starting alignment. Flat coordinates are then
   interpolated to a 10 um grid (ch0 upsampled linearly).
3. image L4: depth is measured below the IMAGE pia, from the surface closed with a 400 um ball so that notches, dents
   and tears do not pull the slab and the layers into them. The bottom of the barrel row (L4/L5 boundary) is found in
   ch0 per column of the own flatmap, as in the brain-space depth analysis of the paper (l4bottom.py). The slab is
   sampled at a fixed offset above that bottom, so it follows the brain's own L4. Two candidate offsets (best
   band-pass contrast; best coarse match to the barrel pattern); the full fit runs at both and the offset with more
   matched barrels wins.
4. artefact mask, band-pass, local maxima + watershed -> image barrel centroids.
5. exhaustive search of the synthetic pattern (scale 0.7-1.4, rotation +-20 deg, shift +-600 um) -> top 5.
6. per candidate: CPD affine -> CPD non-rigid -> Hungarian -> re-pairing -> leave-one-out cleaning. Among candidates
   within 0.02 warped correlation of the best: lowest affine residual, then most matches.
7. TPS through the matches (smoothing by leave-one-out); outside the field it blends into a near-affine fit.
8. 3D: label(x) = the template barrel whose convex hull in the flat plane contains the corrected flat coordinate of x
   (the column definition of Blue Brain's barrel annotations). Depth is mapped piecewise linearly with three anchors
   per column: image pia -> atlas pia, image L4 bottom -> atlas L4 bottom (both per flat pixel), and the brain's white
   matter -> atlas white matter (template thickness). The brain's white matter per flat pixel of the own flatmap is the
   deeper of: the deepest registered isocortex below the image pia, and the registered cortical thickness (which does
   not move when the registered cortex is shifted against the image). Columns run from the image pia to that depth.
9. CCF: x -> (corrected template coordinate, mapped atlas depth) -> inverse flatmap -> CCF voxel. Brain hemisphere 2
   is mirrored onto CCF axis 2. Defined only in and around the barrel field; NaN elsewhere.
"""
import json
import os
import time
import traceback
import warnings

import numpy as np
import tifffile
from scipy import ndimage

from . import pointset as ps
from .atlas import BARREL_NAMES, BFD_IDS, C2, ISO_IDS, L4_BFD_IDS, SEPTA, collapse
from .flatmap import Flat, flat_frame, hull_columns, laplace_depth, trace_to_level
from .imageproc import artefact_mask, bandpass, contrast, l4_bandpass, segment, tissue_mask_padded
from .l4bottom import barrel_bottom
from .plotting import coronal_figure, flat_fit_figure
from .search import BP, brute_force, masked_corr, similarity_points, synth, znorm

PIX = 10.0  # flat pixel and working voxel, um
OFFSETS_UM = list(range(-300, 75, 25))  # slab centre relative to the image L4 bottom (negative = above it)
# rescue runs: BARRELREG_CH = image channel (default 0); BARRELREG_FIT_OFFSETS = comma list of L4 offsets (um) fitted
# instead of the contrast / pattern picks (the fit with most matched barrels wins, as for the two picks)
IMG_CH = int(os.environ.get("BARRELREG_CH", "0"))
FIT_OFFSETS = [int(v) for v in os.environ.get("BARRELREG_FIT_OFFSETS", "").split(",") if v.strip()]
# Offsets searched for the best band-pass contrast and for the best match to the barrel pattern. Over 36 AB/MH
# hemispheres the contrast peaked at -25 / -50 um in 33; the 2 peaks at -200 / -250 um were L2/3 texture (AB082 left
# then fitted a slab at 300 um depth, contrast after 0.98 -> 0.35). The pattern peaked at -25 to -100 um in 30 of 36.
CONTRAST_RANGE = (-150, 0)
PATTERN_RANGE = (-150, -25)
TIE_CORR = 0.02  # candidates within this warped correlation of the best are treated as tied
MAX_T = 600  # search shift range (um)
SCALES = (0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4)
MAX_R, STEP_R, STEP_T, TOPK, REFINE = 20, 2.5, 40, 5, False
if os.environ.get("BARRELREG_SEARCH", "") == "wide":
    # wider search: rotation +-30 deg (7/48 hemispheres sat on the +-20 bound), scale 0.6-1.5, shift +-800 um, 8
    # candidates, each polished on a fine local grid (scale +-0.05, rotation +-3 deg, shift +-50 um)
    MAX_T, SCALES = 800, tuple(np.round(np.arange(0.6, 1.5 + 1e-6, 0.1), 2))
    MAX_R, STEP_R, STEP_T, TOPK, REFINE = 30, 3.0, 50, 8, True
NEAR_UM = 1000  # flat coordinates are computed this far around the registered SSp-bfd
CCF_NX = 1140  # CCF 10 um, axis 2 (mirror: x -> CCF_NX - 1 - x)
WORK_UM = 20.0  # grid of the Laplace / streamline step
SURF_CLOSE_UM = 400  # surface damage narrower than ~2x this is bridged when measuring depth below the pia
SURF_OPEN_UM = 75  # structures thinner than ~2x this (detached membranes) are not surface
# BARRELREG_DEBUG=1: every intermediate step of a hemisphere is saved to h<hv>/debug.npz (2D) and debug3d.npz (crop
# volumes), for scripts/brain_report.py. Off by default; the fit itself is unchanged.
DEBUG = os.environ.get("BARRELREG_DEBUG", "") == "1"
DBG = {}


def dbg(prefix="", **kw):
    if DEBUG:
        for k, v in kw.items():
            v = np.asarray(v)
            if v.dtype == np.float64 and v.ndim == 3:
                v = v.astype(np.float32)
            DBG[prefix + k] = v


def dbg_save(od):
    if not DEBUG:
        return
    d2 = {k: v for k, v in DBG.items() if not k.startswith("v_")}  # v_ = crop volumes (float16 to keep it small)
    d3 = {k: (v.astype(np.float16) if v.dtype in (np.float32, np.float64) else v) for k, v in DBG.items()
          if k.startswith("v_")}
    np.savez_compressed(f"{od}/debug.npz", **d2)
    np.savez_compressed(f"{od}/debug3d.npz", **d3)
    DBG.clear()


def up(x, order, UP):
    """20 um crop -> 10 um grid (voxel i covers 10 um voxels UP*i .. UP*i + UP - 1)."""
    if x.dtype == bool:
        return ndimage.zoom(x.astype(np.uint8), UP, order=0, mode="nearest", grid_mode=True) > 0
    return ndimage.zoom(x, UP, order=order, mode="nearest", grid_mode=True)


def block_mode(lab, nlab, UP):
    """10 um labels -> 20 um (majority over UP^3 blocks)."""
    s = lab.shape
    b = lab.reshape(s[0] // UP, UP, s[1] // UP, UP, s[2] // UP, UP)
    cnt = np.stack([(b == k).sum((1, 3, 5)) for k in range(nlab)], -1)
    return cnt.argmax(-1).astype(np.uint8)


def block_mean(x, UP):
    s = x.shape
    b = x.reshape(s[0] // UP, UP, s[1] // UP, UP, s[2] // UP, UP)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(b, (1, 3, 5)).astype(np.float32)


# ----------------------------------------------------------------------------- geometry
def crop20(reg, hv, vox, dn, margin_um=700):
    """Box around the largest registered SSp-bfd component of hemisphere hv (sample voxels of size vox); the box
    lengths are multiples of dn so it splits into 20 um blocks."""
    atl = tifffile.memmap(f"{reg}/registered_atlas.tiff", mode="r")
    hem = tifffile.memmap(f"{reg}/registered_hemispheres.tiff", mode="r")
    keep = np.isin(np.asarray(atl[::4, ::4, ::4]), BFD_IDS) & (np.asarray(hem[::4, ::4, ::4]) == hv)
    lab, n = ndimage.label(keep)
    sz = np.bincount(lab.ravel())
    sz[0] = 0
    pts = np.argwhere(lab == sz.argmax()) * 4
    margin = int(round(margin_um / vox))
    shape = np.array(atl.shape)
    lo = np.maximum(pts.min(0) - margin, 0)
    hi = np.minimum(pts.max(0) + 4 + margin, shape)
    n = -(-(hi - lo) // dn) * dn
    lo = np.maximum(np.minimum(lo, shape - n), 0)
    hi = lo + (np.minimum(n, shape) // dn) * dn
    return tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))


def pia_depth(tissue):
    """Depth below the image pia (um), measured from a cleaned surface.

    1. Opening with a ball of radius SURF_OPEN_UM, largest component: removes thin structures stuck to the brain
       (detached meninges, membranes), which the closing would otherwise bridge to and take as the surface.
    2. Closing with a ball of radius SURF_CLOSE_UM: under a notch, dent or tear the depth is measured from where the
       surface would be, not from the damage (the nearest-edge distance makes the L4 slab and the layers bow into it).
    The convex cortex surface is unchanged by both. Returns the depth and the plain nearest-edge depth."""
    ro, rc = SURF_OPEN_UM / PIX, SURF_CLOSE_UM / PIX
    pw = int(rc) + 2
    t = np.pad(tissue, pw, mode="edge")
    raw = ndimage.distance_transform_edt(t).astype(np.float32)
    opened = ndimage.distance_transform_edt(~(raw > ro)) <= ro
    lab, n = ndimage.label(opened)
    if n > 1:
        sz = np.bincount(lab.ravel())
        sz[0] = 0
        opened = lab == sz.argmax()
    closed = ndimage.distance_transform_edt(ndimage.distance_transform_edt(~opened) <= rc) > rc
    dep = ndimage.distance_transform_edt(closed | (opened & t)).astype(np.float32)
    un = (slice(pw, -pw),) * 3
    return np.where(tissue, dep[un], 0) * PIX, raw[un] * PIX


def prep(cfg, bid, hv):
    """Crop around SSp-bfd, own flatmap on a 20 um grid, everything else on the 10 um working grid. Works for brainreg
    output at 20 um (upsampled to 10 um) and at 10 um (used as is; the flatmap step takes every second voxel)."""
    t0 = time.time()
    VOX = cfg.voxel_um
    if VOX not in (10.0, 20.0):
        raise ValueError(f"voxel_um must be 10 or 20, not {VOX}")
    DN = int(WORK_UM / VOX)  # sample voxels per 20 um flatmap voxel
    UP = int(WORK_UM / PIX)  # 10 um voxels per 20 um flatmap voxel
    reg = cfg.reg_dir(bid)
    sl = crop20(reg, hv, VOX, DN)
    A = np.asarray(tifffile.memmap(f"{reg}/registered_atlas.tiff", mode="r")[sl]).astype(np.int64)
    H = np.asarray(tifffile.memmap(f"{reg}/registered_hemispheres.tiff", mode="r")[sl])
    I = np.asarray(tifffile.memmap(cfg.ds_file(reg, IMG_CH), mode="r")[sl]).astype(np.float32)
    A20, H20 = A[::DN, ::DN, ::DN], H[::DN, ::DN, ::DN]
    # deformation-field check: field (mm) -> registration-atlas voxel -> annotation == registered atlas
    ann = cfg.registration_atlas()
    sub = (slice(None, None, 3),) * 3
    fld = np.stack([np.asarray(tifffile.memmap(f"{reg}/deformation_field_{i}.tiff", mode="r")[sl][sub]) for i in range(3)], -1)
    pv = np.rint(fld * 1000 / cfg.registration_atlas_um).astype(int)
    inb = (pv >= 0).all(-1) & (pv < np.array(ann.shape)).all(-1) & (A[sub] > 0)
    chk = float(np.mean(np.asarray(ann[tuple(pv[inb].T)]) == A[sub][inb]))
    # own flatmap on the 20 um crop: Laplace between the registered pia and WM, streamlines to the L4 level
    bfd20 = np.isin(A20, BFD_IDS) & (H20 == hv)
    l4_20 = np.isin(A20, L4_BFD_IDS) & (H20 == hv)
    u, ue, ctx, info = laplace_depth(A20)
    u0_reg = float(np.nanmedian(u[l4_20 & ctx]))
    u0 = float(np.clip(u0_reg, 0.3, 0.5))
    near20 = ndimage.distance_transform_edt(~bfd20) * WORK_UM <= NEAR_UM
    pts = np.argwhere(ctx & near20 & (H20 == hv))
    P, bad = trace_to_level(ue, u0, pts)
    inl4 = l4_20[tuple(pts.T)] & ~bad
    c, e1, e2, nrm = flat_frame(P[inl4])
    F = np.stack([(P - c) @ e1, (P - c) @ e2], 1) * WORK_UM
    F -= F[inl4].mean(0)  # origin: centroid of the registered SSp-bfd L4 (as the template)
    F20 = np.full(A20.shape + (2,), np.nan, np.float32)
    ok = ~bad
    F20[tuple(pts[ok].T)] = F[ok]
    have = np.isfinite(F20[..., 0])
    _, ind = ndimage.distance_transform_edt(~have, return_indices=True)
    F20f = F20[tuple(ind)]  # nearest-filled so the linear upsampling does not mix in NaN
    print(f"  h{hv}: crop {[(s.start, s.stop) for s in sl]}, field check {chk:.3f}, Laplace info {info}, "
          f"u0 {u0:.3f} (registered L4 {u0_reg:.3f}), {len(pts)} streamlines ({int(bad.sum())} not converged)  "
          f"({time.time() - t0:.0f}s)", flush=True)
    dbg(sl=[[s.start, s.stop] for s in sl], field_check=chk, u0=u0, u0_reg=u0_reg, stream_start=pts[::25],
        stream_end=P[::25], stream_bad=bad[::25], v_A20=A20.astype(np.int32), v_u20=u, v_l4_20=l4_20, v_F20=F20)
    # 10 um working grid
    if DN == 1:
        atl, ch0, hem = up(A20, 0, UP), up(I, 1, UP), up(H20, 0, UP)
    else:
        atl, ch0, hem = A, I, H
    hemi = hem == hv
    Fu = np.stack([up(F20f[..., k], 1, UP) for k in range(2)], -1)
    Fu[~(up(have, 0, UP) & up(near20, 0, UP))] = np.nan
    tissue = tissue_mask_padded(ch0, atl)
    depth, depth_raw = pia_depth(tissue)  # um below the image pia (damage bridged)
    pw = 10  # um below the registered surface (edge padding: the crop faces are not surface)
    dreg = (ndimage.distance_transform_edt(np.pad(atl > 0, pw, mode="edge"))[pw:-pw, pw:-pw, pw:-pw] * PIX).astype(np.float32)
    l4band = tissue & hemi & np.isfinite(Fu[..., 0]) & (np.abs(depth - 450) <= 75)
    bridged = float(np.mean(depth[l4band] - depth_raw[l4band] > 30)) if l4band.any() else 0.0
    dbg(v_atl=atl.astype(np.int32), v_ch0=ch0, v_depth=depth, v_depth_raw=depth_raw, v_tissue=tissue, v_F=Fu,
        v_hemi=hemi, v_dreg=dreg, surface_bridged_frac=bridged)
    return dict(sl=sl, atl=atl, ch0=ch0, hemi=hemi, F=Fu, tissue=tissue, depth=depth, dreg=dreg, near=up(near20, 0, UP),
                surface_bridged_frac=bridged,
                ctx=np.isin(atl, ISO_IDS) & hemi,
                field_check=chk, u0=u0, laplace_info=int(info), n_nonconv=int(bad.sum()))


# ----------------------------------------------------------------------------- per hemisphere
def run_hemi(cfg, bid, hv, od):
    t0 = time.time()
    os.makedirs(od, exist_ok=True)
    TPL = cfg.template()
    if "l4_bottom" not in TPL:
        raise RuntimeError(f"{cfg.template_file} has no atlas L4 bottom (built before it was added): rerun build-template")
    P = prep(cfg, bid, hv)
    flat = Flat.from_grid(TPL["lo"], TPL["pix"], TPL["shape"])
    alab = TPL["alab"].astype(np.int32)
    cols = np.where(alab < SEPTA, alab, 0)
    foot = alab > 0
    thick = TPL["thick"]
    atl_l4_depth = float(TPL["atl_l4_depth"])

    # sample side: flat coords of tissue voxels near the registered barrel field; fill gaps from the nearest
    F, tissue, depth = P["F"], P["tissue"], P["depth"]
    region = tissue & P["hemi"] & P["near"] & (depth < 2200)
    good = np.isfinite(F).all(-1) & region
    if (region & ~good).any():
        _, ind = ndimage.distance_transform_edt(~good, return_indices=True)
        Ff = F[tuple(ind)]
    else:
        Ff = F
    G = np.full(F.shape, np.nan, np.float32)
    G[region] = Ff[region]
    filled_frac = float((region & ~good).sum() / max(region.sum(), 1))

    # brain white matter per flat pixel (own flatmap, uncorrected), two estimates from the coarse registration: the
    # deepest registered isocortex below the image pia, and the registered cortical thickness (its depth below the
    # registered surface). The first inherits any shift of the registered cortex against the image (MH064 left: 300 um
    # too shallow), the second does not. The deeper of the two is used, so a shifted registration cannot cut the
    # columns short; the overshoot is at most the surface offset.
    mw = region & P["ctx"]
    bmw = flat.bins(G[mw])

    def flat_max(v):
        b = np.zeros(np.prod(flat.shape))
        np.maximum.at(b, bmw, v)
        b = ndimage.median_filter(b.reshape(flat.shape), 5)
        if (b > 0).any():
            _, ind = ndimage.distance_transform_edt(b <= 0, return_indices=True)
            b = b[tuple(ind)]
        return b

    bw_pia, bw_thick = flat_max(depth[mw]), flat_max(P["dreg"][mw])
    bw = np.maximum(bw_pia, bw_thick)
    dbg(bw_pia=bw_pia, bw_thick=bw_thick, bw=bw, flat_lo=flat.lo, flat_shape=flat.shape)

    # image L4 bottom (L4/L5 boundary) per flat pixel, and every voxel's depth relative to it
    bot, bot_norm, bot_patches = barrel_bottom(P["ch0"], depth, G, good, flat, bw)
    bot_patches.to_csv(f"{od}/l4_bottom_patches.csv", index=False)
    fi = (G[region] - flat.lo) / PIX - 0.5  # uncorrected flat coordinate of each region voxel (pixels)
    b_img = ndimage.map_coordinates(bot, fi.T, order=1, mode="nearest")
    rel = np.full(depth.shape, np.nan, np.float32)
    rel[region] = depth[region] - b_img
    dbg(bot=bot, bot_norm=bot_norm, v_rel=rel, v_region=region, v_good=good)

    # image L4 slab at offsets from that bottom: two candidate offsets, full fit at both, more matched barrels wins
    fieldd = ndimage.binary_dilation(foot, iterations=10)
    foot15 = ndimage.binary_dilation(foot, iterations=15)
    mov = znorm(synth(alab), foot15)
    centre = np.argwhere(foot).mean(0)
    stack, con, pat = {}, {}, {}
    for d in OFFSETS_UM:
        m = good & (np.abs(rel - d) <= 30)
        img, valid, _ = flat.mean(G[m], P["ch0"][m])
        stack[d] = (img, valid)
        if CONTRAST_RANGE[0] <= d <= CONTRAST_RANGE[1]:
            con[d] = contrast(bandpass(img, valid), valid & fieldd)
        if PATTERN_RANGE[0] <= d <= PATTERN_RANGE[1]:
            art_d = artefact_mask(img, valid, foot15)
            v_d = valid & ~art_d
            fx = znorm(bandpass(np.where(art_d, np.median(img[v_d]), img), valid, *BP), foot15 & v_d)
            fx[~v_d] = 0
            pat[d] = brute_force(fx[::4, ::4], mov[::4, ::4], (foot15 & v_d)[::4, ::4], PIX * 4, centre / 4, topk=1,
                                 max_t=MAX_T, step_t=80, step_r=5, max_r=10, scales=(0.8, 0.9, 1.0, 1.1))[0][0]
    dbg(offsets=OFFSETS_UM, stack_img=np.stack([stack[d][0] for d in OFFSETS_UM]),
        stack_valid=np.stack([stack[d][1] for d in OFFSETS_UM]), con_d=list(con), con_v=list(con.values()),
        pat_d=list(pat), pat_v=list(pat.values()), alab=alab, cols=cols, foot=foot, mov=mov, centre=centre)
    ksA, Ac, areaA = ps.atlas_points(cols)
    U, V = cols.shape
    gy, gx = np.mgrid[0:U, 0:V]
    grid = np.c_[gy.ravel(), gx.ravel()].astype(float)

    def fit_at(d4):
        ks = [d for d in OFFSETS_UM if abs(d - d4) <= 25]
        l4 = np.mean([stack[d][0] for d in ks], 0)
        valid = np.all([stack[d][1] for d in ks], 0)
        art = artefact_mask(l4, valid, foot15)
        l4f = np.where(art, np.median(l4[valid & ~art]), l4)
        v2 = valid & ~art
        bpA = l4_bandpass(l4f, valid, fieldd)
        mask = foot15 & v2
        fix = znorm(bandpass(l4f, valid, *BP), mask)
        fix[~v2] = 0
        tops = brute_force(fix[::2, ::2], mov[::2, ::2], mask[::2, ::2], PIX * 2, centre / 2, max_t=MAX_T, scales=SCALES,
                           max_r=MAX_R, step_r=STEP_R, step_t=STEP_T, topk=TOPK, refine=REFINE)
        # segment around the atlas field placed by the best search candidate (the identity can be ~600 um off)
        c0, sc, ang, dy, dx = tops[0]
        th = np.deg2rad(ang)
        Mi = np.linalg.inv(sc * np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]]))
        cols_s = ndimage.affine_transform(cols, Mi, offset=centre - Mi @ (centre + np.array([dy, dx]) / PIX), order=0)
        roi = v2 & ndimage.binary_dilation((cols > 0) | (cols_s > 0), iterations=25)
        blobs, Bc, areaB, _ = segment(bpA, roi)
        p = f"o{d4}_"
        dbg(p, l4=l4, valid=valid, art=art, bpA=bpA, fix=fix, mask=mask, search_all=brute_force.last_all,
            tops=np.array(tops), cols_s=cols_s, roi=roi, blobs=blobs, Bc=Bc, areaB=areaB)
        print(f"  h{hv} L4 slab {d4:+d} um from the L4 bottom, {len(Bc)} blobs, {len(Ac)} atlas barrels, "
              f"top search corr {[round(t[0], 3) for t in tops]}  ({time.time() - t0:.0f}s)", flush=True)
        cands = []
        for j, (c0, sc, ang, dy, dx) in enumerate(tops):
            T_sim = similarity_points(Ac, sc, ang, np.array([dy, dx]) / PIX, centre)
            T_aff, *_ = ps.cpd_affine(Bc, T_sim, w=0.3, s2=16.0)
            T_nr, _ = ps.cpd_nonrigid(Bc, T_aff, beta=25.0, lam=3.0, w=0.3, s2=9.0)
            ia, ib = ps.hungarian(T_nr, Bc, ps.GATE_PX, areaA, areaB)
            dbg(f"{p}c{j}_", T_sim=T_sim, T_aff=T_aff, T_nr=T_nr, ia_h=ia, ib_h=ib)
            if len(ia) < 6:
                continue
            for _ in range(3):
                f = ps.tps_fit(Ac[ia], Bc[ib], ps.REMATCH_S)
                ia2, ib2 = ps.hungarian(Ac + f(Ac), Bc, ps.GATE_PX, areaA, areaB)
                if set(zip(ia2, ib2)) == set(zip(ia, ib)) or len(ia2) < 6:
                    break
                ia, ib = ia2, ib2
            e = ps.loo(Ac[ia], Bc[ib], ps.REMATCH_S)
            keep = e <= max(2.5 * np.median(e), ps.GATE_PX)
            ia, ib = ia[keep], ib[keep]
            fb = ps.tps_fit(Bc[ib], Ac[ia], ps.REMATCH_S)
            d_ = fb(grid).T.reshape(2, U, V)
            wl = ndimage.map_coordinates(alab, [gy + d_[0], gx + d_[1]], order=0, cval=0)
            corr = masked_corr(fix, znorm(synth(wl), ndimage.binary_dilation(wl > 0, iterations=15)), mask)
            Ph = np.c_[Ac[ia], np.ones(len(ia))]
            Xa, *_ = np.linalg.lstsq(Ph, Bc[ib], rcond=None)
            affres = float(np.median(np.linalg.norm(Ph @ Xa - Bc[ib], axis=1)) * PIX)
            cands.append(dict(corr=corr, ia=ia, ib=ib, wl=wl, affres=affres, search=(c0, sc, ang, dy, dx)))
            dbg(f"{p}c{j}_", ia=ia, ib=ib, corr=corr, affres=affres, wl=wl, search=(c0, sc, ang, dy, dx))
            print(f"   candidate search corr {c0:.3f} (scale {sc}, rot {ang}, shift {dy:.0f},{dx:.0f} um): "
                  f"{len(ia)} matched, affine residual {affres:.0f} um -> warped corr {corr:.3f}", flush=True)
        if not cands:
            raise RuntimeError(f"no candidate with >= 6 matched barrels at L4 offset {d4} um")
        # the warped-pattern correlation is blind to barrel identity once the TPS pins the atlas onto the blobs (a
        # partly mis-named grid can score the same); among near-tied candidates, a mis-naming shows up as matches
        # that an affine map cannot explain -> lowest affine residual, then most matches
        top = max(c["corr"] for c in cands)
        tied = [c for c in cands if c["corr"] >= top - TIE_CORR]
        best = min(tied, key=lambda c: (round(c["affres"] / 5), -len(c["ia"])))
        bar0 = (best["wl"] > 0) & (best["wl"] < SEPTA)
        same = [c for c in cands if c is not best and (c["wl"][bar0] == best["wl"][bar0]).mean() >= 0.7]
        other = [c["corr"] for c in cands if c is not best and not any(c is s for s in same)]
        return dict(d4=d4, l4=l4, valid=valid, art=art, v2=v2, bpA=bpA, blobs=blobs, Bc=Bc, best=best,
                    n_agree=1 + len(same), n_cands=len(cands), margin=best["corr"] - max(other) if other else np.nan,
                    search_top=[round(t[0], 3) for t in tops], cand_corr=[round(c["corr"], 3) for c in cands])

    d_con, d_pat = max(con, key=con.get), max(pat, key=pat.get)
    fits = []
    for d in sorted(FIT_OFFSETS or {d_con, d_pat}):
        try:
            fits.append(fit_at(d))
        except RuntimeError as e:
            print(f"  h{hv}: {e}", flush=True)
    dbg(fit_offsets=sorted(FIT_OFFSETS or {d_con, d_pat}), img_ch=IMG_CH,
        search_params=[MAX_T, STEP_T, MAX_R, STEP_R, min(SCALES), max(SCALES), TOPK, int(REFINE)])
    if not fits:
        raise RuntimeError("no L4 offset gave a fit")
    W = max(fits, key=lambda w: (len(w["best"]["ia"]), w["best"]["corr"]))
    print(f"  h{hv} L4 offset candidates: contrast {d_con} um, pattern {d_pat} um -> using {W['d4']} um "
          f"(L4 bottom median {np.median(bot[foot]):.0f} um below the image pia)", flush=True)
    d_off, v2, art, bpA, blobs, Bc, best = (W[k] for k in ("d4", "v2", "art", "bpA", "blobs", "Bc", "best"))
    n_agree, margin = W["n_agree"], W["margin"]

    # final TPS: smoothing by leave-one-out, backward field tapered outside the field
    ia, ib = best["ia"], best["ib"]
    errs = {s: np.mean(ps.loo(Ac[ia], Bc[ib], s)) for s in ps.SMOOTH_GRID}
    s_best = min(errs, key=errs.get)
    disp = ps.tps_fit(Bc[ib], Ac[ia], s_best)(grid).T.reshape(2, U, V)
    # taper outside the field as placed in the image (the correction is not small at the edge of the template field)
    placed = ndimage.map_coordinates(foot.astype(np.uint8), [gy + disp[0], gx + disp[1]], order=0, cval=0) > 0
    dist = ndimage.distance_transform_edt(~ndimage.binary_dilation(foot | placed, iterations=5))
    wt = np.where(dist >= ps.TAPER_PX, 0.0, 0.5 * (1 + np.cos(np.pi * np.minimum(dist, ps.TAPER_PX) / ps.TAPER_PX)))
    disp_far = ps.tps_fit(Bc[ib], Ac[ia], 1e6)(grid).T.reshape(2, U, V)  # ~affine outside the field
    dbg(disp_tps=disp, disp_far=disp_far, wt=wt, placed=placed)
    disp = wt[None] * disp + (1 - wt[None]) * disp_far
    jac = ps.jacobian_2d(disp)
    alab_new = ndimage.map_coordinates(alab, [gy + disp[0], gx + disp[1]], order=0, cval=0)
    cols_new = np.where(alab_new < SEPTA, alab_new, 0)
    e_before = np.linalg.norm(Ac[ia] - Bc[ib], axis=1) * PIX
    e_loo = ps.loo(Ac[ia], Bc[ib], s_best) * PIX
    dbg(chosen_offset=d_off, d_con=d_con, d_pat=d_pat, Ac=Ac, ksA=ksA, ia=ia, ib=ib, smooth_s=list(errs),
        smooth_err=list(errs.values()), s_best=s_best, disp=disp, jac=jac, alab_new=alab_new, cols_new=cols_new,
        e_before=e_before, e_loo=e_loo)

    # 3D: corrected flat coordinate -> template barrel column (convex hull); depth: pia -> pia, image L4 bottom ->
    # atlas L4 bottom, WM -> atlas WM
    dsp = np.stack([ndimage.map_coordinates(disp[k], fi.T, order=1, mode="nearest") for k in range(2)], 1)
    fc = fi + dsp
    lab, n_hull_overlap = hull_columns(cols, fc)
    th = ndimage.map_coordinates(thick, fc.T, order=1, mode="nearest")  # atlas thickness (WM depth)
    b_atl = ndimage.map_coordinates(TPL["l4_bottom"], fc.T, order=1, mode="nearest")  # atlas L4 bottom
    tb = np.maximum(ndimage.map_coordinates(bw, fi.T, order=1, mode="nearest"), b_img + 100)  # brain WM depth
    dep = depth[region]
    adep = np.where(dep <= b_img, dep * b_atl / b_img,
                    b_atl + (dep - b_img) * (th - b_atl) / (tb - b_img))  # atlas depth (um)
    incol = dep <= tb  # the registered labels do not cut the column (they are shifted with the registration)
    lab[~incol] = 0
    out = np.zeros(P["atl"].shape, np.uint8)
    out[region] = lab
    if DEBUG:
        vols = {k: np.full(depth.shape, np.nan, np.float32) for k in ("adep", "bimg", "batl", "tb", "th")}
        for k, v in zip(vols, (adep, b_img, b_atl, tb, th)):
            vols[k][region] = v
        dbg(v_out=out, **{f"v_{k}": v for k, v in vols.items()})

    # CCF: (corrected template coordinate, mapped atlas depth) -> inverse flatmap -> CCF voxel (10 um)
    inv, inv_have = TPL["inv"], TPL["inv_have"]
    kd = adep / float(TPL["dpix"]) - 0.5
    have2d = inv_have.any(-1)
    in_grid = (fc >= 0).all(1) & (fc <= np.array(flat.shape) - 1).all(1)
    ok = in_grid & incol & (ndimage.map_coordinates(have2d.astype(np.uint8), fc.T, order=0, mode="constant") > 0)
    ccf = np.full((len(fc), 3), np.nan, np.float32)
    crd = np.c_[fc[ok], kd[ok]].T
    for k in range(3):
        ccf[ok, k] = ndimage.map_coordinates(inv[..., k], crd, order=1, mode="nearest")
    if hv == 2:
        ccf[:, 2] = CCF_NX - 1 - ccf[:, 2]
    ccf_vol = np.full(P["atl"].shape + (3,), np.nan, np.float32)
    ccf_vol[region] = ccf
    # check: labelled voxels -> CCF annotation at the mapped voxel (collapsed) == their label
    ann = tifffile.memmap(cfg.annotation, mode="r")
    hit = ok & (lab > 0)
    cv = np.clip(np.rint(ccf[hit]).astype(int), 0, np.array(ann.shape) - 1)
    ccf_lab = collapse(np.asarray(ann[tuple(cv.T)]))
    ccf_agree = float(np.mean(ccf_lab == lab[hit])) if hit.any() else np.nan
    ccf_c2 = float(np.mean(ccf_lab[lab[hit] == C2] == C2)) if (lab[hit] == C2).any() else np.nan

    Gc = np.full(G.shape, np.nan, np.float32)
    Gc[region] = (fc + 0.5) * PIX + flat.lo
    np.savez_compressed(f"{od}/flatcoords.npz", crop20_lo=[s.start for s in P["sl"]], up=int(cfg.voxel_um / PIX),
                        flat_lo=flat.lo, G=G, G_corrected=Gc, l4_bottom_um=bot, l4_bottom_norm=bot_norm,
                        l4_offset_um=d_off, atlas_l4_bottom_um=TPL["l4_bottom"], atlas_l4_depth_um=atl_l4_depth,
                        brain_wm_depth_um=bw)

    # metrics: initial placement (identity: Perens SSp-bfd centroid on the CCF one) vs corrected
    fv = v2 & ndimage.binary_dilation((cols > 0) | (cols_new > 0), iterations=5)
    names_blob = {int(ib[n]) + 1: int(ksA[ia[n]]) for n in range(len(ia))}
    d_b, _, n_b = ps.overlap_metrics(cols, blobs, names_blob)
    d_a, _, n_a = ps.overlap_metrics(cols_new, blobs, names_blob)
    sc_ = best["search"]
    met = dict(brain=bid, hemisphere=hv, registration=os.path.basename(cfg.reg_dir(bid)), field_check=P["field_check"],
               laplace_info=P["laplace_info"], u0=P["u0"], streamlines_not_converged=P["n_nonconv"],
               filled_frac=filled_frac, surface_bridged_frac=P["surface_bridged_frac"],
               l4_depth_um=float(np.median(bot[foot]) + d_off), atlas_l4_depth_um=atl_l4_depth,
               l4_bottom_um=float(np.median(bot[foot])), l4_bottom_norm=float(np.median(bot_norm[foot])),
               atlas_l4_bottom_um=float(np.median(TPL["l4_bottom"][foot])), l4_offset_um=d_off,
               l4_bottom_found_frac=float(np.isfinite(bot_patches["l4_bottom_raw"]).mean()),
               hull_overlap_frac=n_hull_overlap / max(len(lab), 1),
               brain_wm_depth_um=float(np.median(bw[foot])), atlas_wm_depth_um=float(np.median(thick[foot])),
               wm_registered_below_pia_um=float(np.median(bw_pia[foot])),
               wm_registered_thickness_um=float(np.median(bw_thick[foot])),
               n_blobs=len(Bc), n_atlas=len(Ac), n_matched=len(ia), search_top=W["search_top"], cand_corr=W["cand_corr"],
               search_best=dict(corr=sc_[0], scale=sc_[1], rot_deg=sc_[2], shift_um=[sc_[3], sc_[4]]),
               offset_candidates_um=sorted({d_con, d_pat}), n_candidates_agree=n_agree, margin_to_other_naming=margin,
               tps_smoothing=s_best, affine_residual_um=best["affres"],
               depth_contrast={str(k): round(v, 3) for k, v in con.items()},
               depth_pattern={str(k): round(v, 3) for k, v in pat.items()},
               naming_ambiguous=bool(np.isfinite(margin) and margin < TIE_CORR),
               err_initial_um=float(np.median(e_before)), err_tps_loo_um=float(np.median(e_loo)),
               contrast_initial=ps.sep_contrast(bpA, cols, fv), contrast_after=ps.sep_contrast(bpA, cols_new, fv),
               dice_initial=d_b, dice_after=d_a, naming_initial=n_b, naming_after=n_a,
               jac_min=float(jac[foot | placed].min()),
               disp_median_um=float(np.median(np.hypot(*disp[:, cols > 0])) * PIX),
               ccf_label_agreement=ccf_agree, ccf_c2_in_c2=ccf_c2, ccf_defined_frac=float(ok.mean()),
               seconds=round(time.time() - t0))
    json.dump(met, open(f"{od}/metrics.json", "w"), indent=1, default=float)
    flat_fit_figure(f"{bid}_h{hv}", od, bpA, v2, art, cols, cols_new, blobs, Ac, Bc, ia, ib, met)
    dbg_save(od)
    print(f"{bid} h{hv}: {len(ia)}/{len(Ac)} matched | err {met['err_initial_um']:.0f} -> LOO "
          f"{met['err_tps_loo_um']:.0f} um | contrast {met['contrast_initial']:.2f} -> {met['contrast_after']:.2f} | "
          f"Dice {d_b:.2f} -> {d_a:.2f} | {n_agree}/{W['n_cands']} candidates agree, margin {margin:.3f} | "
          f"jac min {met['jac_min']:.2f} | CCF label agreement {ccf_agree:.3f}, C2 {ccf_c2:.3f} ({met['seconds']}s)",
          flush=True)
    return P["sl"], out, ccf_vol, met


def run(cfg, bid, hemispheres=(1, 2)):
    """Both hemispheres of one brain -> barrels_corrected_full.tif, ccf_field_{0,1,2}.tif, h<hv>/, coronal figure."""
    t0 = time.time()
    od = cfg.brain_dir(bid)
    os.makedirs(od, exist_ok=True)
    names = f"{cfg.work_dir}/barrel_index.csv"
    if not os.path.exists(names):
        with open(names, "w") as f:
            f.write("index,barrel\n" + "".join(f"{k + 1},{n}\n" for k, n in enumerate(BARREL_NAMES)))
    UP = int(cfg.voxel_um / PIX)
    reg = cfg.reg_dir(bid)
    shape = tifffile.memmap(f"{reg}/registered_atlas.tiff", mode="r").shape
    full = np.zeros(shape, np.uint8)
    ccf_full = np.full(shape + (3,), np.nan, np.float32)
    print(f"{bid}: {os.path.basename(reg)}", flush=True)
    mets = []
    for hv in hemispheres:
        try:
            sl, out, ccf_vol, met = run_hemi(cfg, bid, hv, f"{od}/h{hv}")
        except Exception as e:  # keep the other hemisphere
            traceback.print_exc()
            print(f"{bid} h{hv}: FAILED {e}", flush=True)
            continue
        o20 = block_mode(out, SEPTA, UP)
        cur = full[sl]
        full[sl] = np.where(o20 > 0, o20, cur)
        c20 = np.stack([block_mean(ccf_vol[..., k], UP) for k in range(3)], -1)
        cc = ccf_full[sl]
        put = np.isfinite(c20).all(-1) & ~np.isfinite(cc).all(-1)
        cc[put] = c20[put]
        ccf_full[sl] = cc
        mets.append(met)
    tifffile.imwrite(f"{od}/barrels_corrected_full.tif", full, compression="zlib")
    for k in range(3):
        tifffile.imwrite(f"{od}/ccf_field_{k}.tif", ccf_full[..., k] / 100.0, compression="zlib")  # mm
    coronal_figure(bid, tifffile.memmap(cfg.ds_file(reg, 0), mode="r"), full, f"{od}/{bid}_coronal.png")
    print(f"{bid}: done, {len(mets)}/{len(hemispheres)} hemispheres ({time.time() - t0:.0f}s)", flush=True)
    return mets
