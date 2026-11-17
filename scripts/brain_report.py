"""One PDF per brain documenting every D2 step on that brain's own data, plus its Neuropixels track(s).

Needs a run with BARRELREG_DEBUG=1 (h<hv>/debug.npz + debug3d.npz). Pages:
  0 summary                      6 step 5: pattern search
  1 inputs: crop, registered     7 step 6: point matching, per candidate
  2 step 1: own flatmap          8 step 7: warp
  3 step 2: template             9 step 8: back to 3D (coronal columns, depth mapping)
  4 step 3: depth and L4 bottom  10 Neuropixels track(s): brainreg vs D2 barrels along the probe
  5 step 3-4: L4 slab and image barrels

usage: python scripts/brain_report.py -c config.toml BRAIN [BRAIN ...] [--hemisphere 1] [--tracks tracks_summary.csv]
--tracks: table with columns brain, track (path below segmentation/sample_space/tracks/), table_target_area,
um_in_d2_barrels, um_in_brainreg_ssp_bfd (Lana Smith's D2_neuropixels/tracks/tracks_summary.csv); tracks with target
wS1 are drawn, and for brains without a probe-table entry (MH) every track running >= 100 um through SSp-bfd.
"""
import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root, if not pip-installed

import h5py
import numpy as np
import pandas as pd
import tifffile
from scipy import ndimage
from skimage.segmentation import find_boundaries
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import patheffects
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D

from barrelreg.atlas import BARREL_NAMES, BFD_IDS, C2, ISO_IDS, S, SEPTA, collapse, row_of
from barrelreg.config import Config
from barrelreg.imageproc import bandpass
from barrelreg.register import CONTRAST_RANGE, PATTERN_RANGE, PIX

plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "figure.facecolor": "white"})
FIG = (16.5, 10)
ACR = dict(zip(S.id.astype(int), S.acronym))
STEPS = ["1 own flatmap (Laplace between registered pia and WM -> streamlines -> flat L4 plane)",
         "2 fixed template (atlas L4 barrels in the Blue Brain flatmap, same plane convention)",
         "3 L4 bottom per column from ch0; L4 slab at two candidate offsets above it",
         "4 image barrels: artefact mask, band-pass, local maxima + watershed",
         "5 exhaustive search: scale x rotation x shift of the synthetic barrel pattern -> top 5",
         "6 per candidate: CPD affine -> CPD non-rigid -> Hungarian -> re-pairing -> LOO cleaning; tie-break",
         "7 thin-plate spline through the matches, near-affine outside the field",
         "8 3D: convex-hull barrel columns, piecewise depth mapping, inverse flatmap -> CCF"]


def name(k):
    return BARREL_NAMES[int(k) - 1] if 0 < int(k) <= len(BARREL_NAMES) else ""


# row palette of the report (bright on a grey slab, row order as a hue sweep, straddlers white)
PAL = {"A": "#FF6F6F", "B": "#FFB547", "C": "#B5E655", "D": "#4DD4E8", "E": "#A98BFF", "S": "#F2F2F2"}


def rowcol(k):
    return PAL[row_of(name(k))]


ROW_LABEL = {"A": "row A", "B": "row B", "C": "row C", "D": "row D", "E": "row E", "S": "straddlers"}


def draw(ax, cols, lw=1.0, names=True, fs=8, ls="-", alpha=1.0):
    """Barrel outlines (smoothed contour of each barrel mask) coloured by row, names at the centroids."""
    for k in np.unique(cols):
        if k <= 0 or k >= SEPTA:
            continue
        m = ndimage.gaussian_filter((cols == k).astype(np.float32), 1.0)
        ax.contour(m, [0.5], colors=[rowcol(k)], linewidths=lw, linestyles=ls, alpha=alpha)
        if names:
            cy, cx = np.argwhere(cols == k).mean(0)
            ax.text(cx, cy, name(k), color=rowcol(k), fontsize=fs, ha="center", va="center", weight="bold",
                    path_effects=[patheffects.withStroke(linewidth=1.4, foreground="black")])


def flatten(img, valid, sigma=30, fine=0.8):
    """Slab with the slow brightness gradient removed: ratio to a normalised-convolution Gaussian (300 um)."""
    w = valid.astype(np.float32)
    x = np.where(valid, np.nan_to_num(img), 0).astype(np.float32)
    bg = ndimage.gaussian_filter(x, sigma) / np.maximum(ndimage.gaussian_filter(w, sigma), 1e-3)
    return ndimage.gaussian_filter(x, fine) / np.maximum(ndimage.gaussian_filter(w, fine), 1e-3) / np.maximum(bg, 1e-6)


def square_box(mask, pad=25):
    """Square (y0, y1, x0, x1) around a mask, pad in flat px (10 um)."""
    ys, xs = np.where(mask)
    cy, cx = (ys.min() + ys.max()) / 2, (xs.min() + xs.max()) / 2
    h = max(ys.max() - ys.min(), xs.max() - xs.min()) / 2 + pad
    return cy - h, cy + h, cx - h, cx + h


def flat_panel(ax, img, valid, box, cols=None, cols_ghost=None, signed=False, title="", fs=7, bar=True, names=True):
    """One square flat-L4 view: grey image (contrast from the visible valid pixels), barrel outlines, 500 um bar."""
    y0, y1, x0, x1 = box
    yy, xx = np.mgrid[0:img.shape[0], 0:img.shape[1]]
    inbox = valid & (yy >= y0) & (yy <= y1) & (xx >= x0) & (xx <= x1)
    v = img[inbox] if inbox.any() else img[valid]
    if signed:
        s_ = np.percentile(np.abs(v), 99)
        lo, hi = -s_, s_
    else:
        lo, hi = np.percentile(v, (1, 99.5))
    ax.set_facecolor("black")
    ax.imshow(np.where(valid, img, np.nan), cmap="gray", vmin=lo, vmax=hi, interpolation="bilinear")
    if cols_ghost is not None:
        draw(ax, cols_ghost, lw=0.8, names=False, ls="dashed", alpha=0.8)
    if cols is not None:
        draw(ax, cols, lw=1.3, fs=fs, names=names)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y1, y0)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    if bar:
        L = 500 / PIX
        bx, by = x0 + 0.05 * (x1 - x0), y1 - 0.05 * (y1 - y0)
        ax.plot([bx, bx + L], [by, by], color="w", lw=2.5, solid_capstyle="butt")
        ax.text(bx + L / 2, by - 0.025 * (y1 - y0), "500 um", color="w", ha="center", va="bottom", fontsize=fs)
    if title:
        ax.set_title(title, fontsize=fs + 2)


def row_legend(ax, fs=8):
    h = [Line2D([], [], color=PAL[r], lw=2, label=ROW_LABEL[r]) for r in "ABCDES"]
    ax.legend(handles=h, loc="lower right", fontsize=fs, frameon=True, facecolor="black", edgecolor="none",
              labelcolor="w", framealpha=0.6, handlelength=1.2)


def flat_views(D, p, cols, cols_new, out_dir, tag):
    """Publication panels, one square file each (png 300 dpi, pdf, svg): flat L4 slab with brainreg and with the
    refined (D2) barrels, raw and band-passed, plus both side by side (brainreg left, refined right)."""
    os.makedirs(out_dir, exist_ok=True)
    valid, bp = D[p + "valid"], D[p + "bpA"]
    l4 = flatten(D[p + "l4"], valid)
    v2 = valid & ~D[p + "art"]
    box = square_box((cols > 0) | (cols_new > 0))
    views = {
        "slab": dict(img=l4, valid=valid, title="flat L4 slab"),
        "slab_brainreg": dict(img=l4, valid=valid, cols=cols, title="brainreg"),
        "slab_refined": dict(img=l4, valid=valid, cols=cols_new, title="refined (D2)"),
        "bandpass_brainreg": dict(img=bp, valid=v2, cols=cols, signed=True, title="brainreg, band-passed"),
        "bandpass_refined": dict(img=bp, valid=v2, cols=cols_new, signed=True, title="refined (D2), band-passed"),
    }
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"], "svg.fonttype": "none",
                         "pdf.fonttype": 42})
    for key, kw in views.items():
        fig = plt.figure(figsize=(4, 4))
        ax = fig.add_axes([0, 0, 1, 0.93])
        flat_panel(ax, box=box, **kw)
        if key == "slab_refined":
            row_legend(ax, 7)
        for ext in ("png", "pdf", "svg"):
            fig.savefig(f"{out_dir}/{tag}_{key}.{ext}", dpi=300, facecolor="white")
        plt.close(fig)
    # side by side: brainreg (global) left, refined right, same crop, outlines only
    fig = plt.figure(figsize=(8, 4.2))
    for i, (lab, t) in enumerate(((cols, "brainreg (global)"), (cols_new, "refined (local)"))):
        ax = fig.add_axes([i * 0.5 + 0.005, 0, 0.49, 0.93])
        flat_panel(ax, l4, valid, box, cols=lab, names=False, title=t, bar=i == 0)
        if i == 1:
            row_legend(ax, 7)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(f"{out_dir}/{tag}_side_by_side.{ext}", dpi=300, facecolor="white")
    plt.close(fig)
    for ext in ("png", "pdf", "svg"):  # replaced by the side-by-side pair
        if os.path.exists(f"{out_dir}/{tag}_overlay.{ext}"):
            os.remove(f"{out_dir}/{tag}_overlay.{ext}")
    plt.rcParams.update({"svg.fonttype": "path", "pdf.fonttype": 3})
    return box, l4, valid, v2, bp


def page(pdf, title, sub="", h=FIG[1]):
    """New page; h < 10 for one-row pages (less white space)."""
    fig = plt.figure(figsize=(FIG[0], h))
    fig.suptitle(title, x=0.01, ha="left", fontsize=15, weight="bold", y=0.985)
    if sub:
        fig.text(0.01, 1 - 0.55 / h, sub, fontsize=10, color="0.3", va="top", wrap=True)
    if h < FIG[1]:
        fig.subplots_adjust(left=0.03, right=0.97, top=1 - 1.3 / h, bottom=0.04, wspace=0.25)
    return fig


def done(pdf, fig):
    pdf.savefig(fig)
    plt.close(fig)


def grey(ax, img, valid=None, pct=(1, 99), title=""):
    v = img[valid] if valid is not None and valid.any() else img[np.isfinite(img)]
    lo, hi = np.nanpercentile(v, pct) if v.size else (0, 1)
    ax.imshow(np.where(valid, img, np.nan) if valid is not None else img, cmap="gray", vmin=lo, vmax=hi,
              interpolation="nearest")
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])


def outline_labels(ax, lab, names=True, lw=0.6, only=None):
    """Barrel labels (index 1..33) drawn as row-coloured outlines (2D slice or flat map)."""
    for k in np.unique(lab):
        if k <= 0 or k >= SEPTA or (only is not None and k not in only):
            continue
        m = lab == k
        yy, xx = np.where(find_boundaries(m, mode="inner"))
        ax.scatter(xx, yy, s=lw, c=rowcol(k), lw=0)
        if names:
            cy, cx = np.argwhere(m).mean(0)
            ax.text(cx, cy, name(k), color=rowcol(k), fontsize=7, ha="center", va="center", weight="bold",
                    path_effects=[patheffects.withStroke(linewidth=1.5, foreground="white")])


def zoom_to(ax, mask, pad=15):
    ys, xs = np.where(mask)
    if len(ys):
        ax.set_xlim(xs.min() - pad, xs.max() + pad)
        ax.set_ylim(ys.max() + pad, ys.min() - pad)


def similarity_cols(cols, sc, ang, dy, dx, centre):
    th = np.deg2rad(ang)
    Mi = np.linalg.inv(sc * np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]]))
    return ndimage.affine_transform(cols, Mi, offset=centre - Mi @ (centre + np.array([dy, dx]) / PIX), order=0)


# ----------------------------------------------------------------------------- tracks
def read_points(f):
    with h5py.File(f, "r") as h:
        return h["df/block0_values"][:].astype(np.float64)


def track_samples(P, depth, lo):
    """Clicked points (sample voxels) -> crop voxels every 10 um along the main axis, shallow end first."""
    P = P - lo
    c = P.mean(0)
    u = np.linalg.svd(P - c)[2][0]
    P = P[np.argsort((P - c) @ u)]
    t = np.r_[0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
    tt = np.arange(0, t[-1] + 1e-9, 1.0)
    X = np.stack([np.interp(tt, t, P[:, k]) for k in range(3)], 1)
    q = np.clip(np.rint(X).astype(int), 0, np.array(depth.shape) - 1)
    dq = depth[tuple(q.T)].astype(float)
    if np.nanmean(dq[: len(dq) // 4]) > np.nanmean(dq[-len(dq) // 4:]):
        X, q = X[::-1], q[::-1]
    return X, q


# ----------------------------------------------------------------------------- report
def report(cfg, bid, hv, tracks, out_pdf):
    od = f"{cfg.brain_dir(bid)}/h{hv}"
    D = dict(np.load(f"{od}/debug.npz"))
    V = np.load(f"{od}/debug3d.npz")
    met = json.load(open(f"{od}/metrics.json"))
    reg = cfg.reg_dir(bid)
    lo = np.array([s[0] for s in D["sl"]])
    d_off = int(D["chosen_offset"])
    p = f"o{d_off}_"
    cols, cols_new, alab = D["cols"], D["cols_new"], D["alab"]
    atl = V["v_atl"]
    ch0 = V["v_ch0"].astype(np.float32)
    depth = V["v_depth"].astype(np.float32)
    out = V["v_out"]
    region = V["v_region"]
    lab_reg = collapse(atl) * (V["v_hemi"] > 0)
    flat_lo, PIXf = D["flat_lo"], PIX

    # tracks of this brain
    trk = []
    if tracks is not None:
        # wS1 target in the probe table; brains without a table entry (MH): tracks >= 100 um in a barrel column
        t = tracks[tracks.brain == bid]
        no_tab = t.table_target_area.isna() | (t.table_target_area.astype(str).str.strip() == "")
        in_bfd = (t.um_in_d2_barrels.fillna(0) >= 100) | (t.um_in_brainreg_ssp_bfd.fillna(0) >= 100)
        for r in t[(t.table_target_area == "wS1") | (no_tab & in_bfd)].drop_duplicates("track").itertuples():
            f = f"{reg}/segmentation/sample_space/tracks/{r.track}"
            if os.path.exists(f):
                X, q = track_samples(read_points(f), depth, lo)
                if (V["v_hemi"][tuple(q.T)] > 0).mean() > 0.5:
                    trk.append((r.track.replace(".points", ""), X, q))
    # coronal slice: through the track's L4 point, else through C2 of the D2 columns
    l4pts = [q[np.argmin(np.abs(depth[tuple(q.T)] - (V["v_bimg"][tuple(q.T)].astype(float) - 60)))] for _, _, q in trk]
    if l4pts:
        z = int(l4pts[0][0])
    else:
        z = int(np.median(np.argwhere(out == C2)[:, 0])) if (out == C2).any() else out.shape[0] // 2

    with PdfPages(out_pdf) as pdf:
        # ---------------------------------------------------------------- 0 summary
        fig = page(pdf, f"{bid}, {'left' if hv == 1 else 'right'} hemisphere: local barrel-field registration (D2)",
                   f"registration: {reg} | one page per step, all panels from this brain's own "
                   f"intermediate data")
        ax = fig.add_axes([0.02, 0.05, 0.46, 0.85])
        ax.axis("off")
        keys = [("n_blobs", "image blobs"), ("n_atlas", "atlas barrels"), ("n_matched", "matched barrels"),
                ("contrast_initial", "contrast, brainreg"), ("contrast_after", "contrast, D2  (< 0.3 = failed)"),
                ("dice_initial", "Dice vs image blobs, brainreg"), ("dice_after", "Dice, D2"),
                ("err_initial_um", "landmark error before (um)"), ("err_tps_loo_um", "landmark error after, LOO (um)"),
                ("affine_residual_um", "affine residual of the matches (um)"),
                ("naming_ambiguous", "naming ambiguous"), ("margin_to_other_naming", "margin to another naming"),
                ("l4_bottom_um", "image L4 bottom, median (um below pia)"),
                ("atlas_l4_bottom_um", "atlas L4 bottom, median (um)"), ("l4_offset_um", "slab offset from L4 bottom (um)"),
                ("surface_bridged_frac", "surface bridged fraction"), ("field_check", "deformation-field check"),
                ("tps_smoothing", "TPS smoothing (LOO)"), ("disp_median_um", "median correction (um)"),
                ("jac_min", "min Jacobian (> 0: no folding)"), ("ccf_label_agreement", "CCF label agreement"),
                ("seconds", "run time (s)")]
        lines = [f"{lab:<44s} {met[k]:.3g}" if isinstance(met[k], float) else f"{lab:<44s} {met[k]}"
                 for k, lab in keys if k in met]
        sb = met["search_best"]
        lines += ["", f"best search placement: scale {sb['scale']}, rotation {sb['rot_deg']} deg, "
                      f"shift {sb['shift_um'][0]:.0f}, {sb['shift_um'][1]:.0f} um (corr {sb['corr']:.3f})"]
        ax.text(0, 1, "\n".join(lines), family="monospace", fontsize=9.5, va="top")
        ax = fig.add_axes([0.5, 0.05, 0.48, 0.85])
        ax.axis("off")
        txt = ["Pipeline steps (pages 2-9):"] + [f"  {s}" for s in STEPS] + ["", "Neuropixels wS1 track(s) (page 10):"]
        for nm, X, q in trk:
            b_reg = lab_reg[tuple(q.T)]
            b_d2 = out[tuple(q.T)]
            m4 = np.abs(V["v_rel"][tuple(q.T)].astype(float) + 80) < 80  # L4 band (up to 160 um above the bottom)
            def top(b):
                u, n = np.unique(b[m4 & (b > 0) & (b < SEPTA)], return_counts=True)
                return ", ".join(f"{name(k)} {c * 10} um" for k, c in sorted(zip(u, n), key=lambda x: -x[1])[:3]) or "-"
            txt += [f"  {nm}:", f"     L4 barrels, brainreg: {top(b_reg)}", f"     L4 barrels, D2:       {top(b_d2)}"]
        if not trk:
            txt += ["  none found in this hemisphere"]
        ax.text(0, 1, "\n".join(txt), fontsize=10, va="top")
        done(pdf, fig)

        # ---------------------------------------------------------------- 1 inputs
        fig = page(pdf, "Inputs: autofluorescence (ch0) and the coarse whole-brain registration",
                   "brainreg to allen_mouse_bluebrain_barrels_10um: used for the hemisphere, the rough SSp-bfd position "
                   "and the pia / white-matter surfaces only. Box = crop around the registered SSp-bfd (+700 um).", h=6.5)
        full = tifffile.memmap(cfg.ds_file(reg, 0), mode="r")
        zf = lo[0] + z
        ax = fig.add_subplot(1, 3, 1)
        img = np.asarray(full[zf])
        grey(ax, img, pct=(1, 99.5), title=f"coronal slice {zf} of the whole brain (10 um)")
        ax.add_patch(plt.Rectangle((D["sl"][2][0], D["sl"][1][0]), D["sl"][2][1] - D["sl"][2][0],
                                   D["sl"][1][1] - D["sl"][1][0], fill=False, ec="c", lw=1.5))
        ax = fig.add_subplot(1, 3, 2)
        grey(ax, ch0[z], title="crop: ch0")
        bfd = np.isin(atl[z], BFD_IDS)
        ax.contour(bfd, [0.5], colors="y", linewidths=1)
        ax.contour(np.isin(atl[z], ISO_IDS), [0.5], colors="c", linewidths=0.6)
        ax.legend(handles=[Line2D([], [], color="y", label="registered SSp-bfd"),
                           Line2D([], [], color="c", label="registered isocortex")], loc="lower left", fontsize=8)
        ax = fig.add_subplot(1, 3, 3)
        grey(ax, ch0[z], title="crop: brainreg barrel columns (registered atlas)")
        outline_labels(ax, lab_reg[z], names=True)
        done(pdf, fig)

        # ---------------------------------------------------------------- 2 own flatmap
        fig = page(pdf, "Step 1. The brain's own flatmap",
                   f"Laplace u between the registered pia (u=0) and white matter (u=1) on a 20 um grid; streamlines "
                   f"follow grad u to the L4 level u0 = {float(D['u0']):.2f} (registered L4 {float(D['u0_reg']):.2f}); "
                   f"end points are projected on a PCA plane (e1 anterior, e2 dorsal), origin = registered SSp-bfd L4 "
                   f"centroid. Columns therefore follow the registered cortex curvature.", h=6.5)
        u20 = V["v_u20"].astype(np.float32)
        z20 = min(z // 2, u20.shape[0] - 1)
        ax = fig.add_subplot(1, 3, 1)
        im = ax.imshow(u20[z20], cmap="viridis", vmin=0, vmax=1)
        ax.contour(np.ma.masked_invalid(u20[z20]), [float(D["u0"])], colors="w", linewidths=1.2)
        st, en, bad = D["stream_start"], D["stream_end"], D["stream_bad"]
        sel = (np.abs(st[:, 0] - z20) < 1) & ~bad
        for a, b in zip(st[sel][::3], en[sel][::3]):
            ax.plot([a[2], b[2]], [a[1], b[1]], color="r", lw=0.4)
        ax.set_title("Laplace u (coronal, 20 um)\nwhite = L4 level, red = streamlines")
        plt.colorbar(im, ax=ax, fraction=0.04)
        ax.set_xticks([]); ax.set_yticks([])
        F = V["v_F"].astype(np.float32)
        for k, lab in ((0, "e1 (anterior)"), (1, "e2 (dorsal)")):
            ax = fig.add_subplot(1, 3, 2 + k)
            im = ax.imshow(np.where(region[z], F[z, ..., k], np.nan), cmap="twilight", interpolation="nearest")
            ax.contour(np.ma.masked_invalid(np.where(region[z], F[z, ..., k], np.nan)), np.arange(-3000, 3001, 200), colors="k", linewidths=0.3)
            ax.set_title(f"flat coordinate {lab} (um)\niso-lines every 200 um")
            plt.colorbar(im, ax=ax, fraction=0.04)
            ax.set_xticks([]); ax.set_yticks([])
        done(pdf, fig)

        # ---------------------------------------------------------------- 3 template
        fig = page(pdf, "Step 2. Fixed template: atlas L4 barrels in the Blue Brain flatmap",
                   "Built once from the CCF annotation: barrel map (33 barrels), cortical thickness and atlas L4 "
                   "bottom per flat pixel, and an inverse lookup (flat position, depth) -> CCF voxel. The search moves "
                   "this map, not the brain.", h=6.5)
        ax = fig.add_subplot(1, 3, 1)
        ax.imshow(np.where(alab > 0, 1, np.nan), cmap="Greys", vmin=0, vmax=3)
        draw(ax, cols)
        zoom_to(ax, alab > 0)
        ax.set_title("template barrels (outline colour = row)")
        ax.set_xticks([]); ax.set_yticks([])
        ax = fig.add_subplot(1, 3, 2)
        ax.imshow(D["mov"], cmap="gray")
        zoom_to(ax, alab > 0)
        ax.set_title("synthetic pattern used by the search (band-passed)")
        ax.set_xticks([]); ax.set_yticks([])
        TPL = cfg.template()
        ax = fig.add_subplot(1, 3, 3)
        im = ax.imshow(np.where(alab > 0, TPL["l4_bottom"], np.nan), cmap="magma")
        plt.colorbar(im, ax=ax, fraction=0.04, label="atlas L4 bottom (um below pia)")
        zoom_to(ax, alab > 0)
        ax.set_title("atlas L4 bottom (the atlas end of the depth anchor)")
        ax.set_xticks([]); ax.set_yticks([])
        done(pdf, fig)

        # ---------------------------------------------------------------- 4 depth and L4 bottom
        fig = page(pdf, "Step 3. Depth below the image pia and the L4 bottom (L4/L5 boundary) from ch0",
                   f"Depth = distance below the segmented tissue surface after closing with a 400 um ball (bridges "
                   f"notches and tears; bridged fraction {met['surface_bridged_frac']:.2f}). The L4 bottom is where the "
                   f"ch0 barrel peak has fallen halfway to the trough below it (100 um patches pooled over 200 um). "
                   f"The slab is sampled {d_off:+d} um from that bottom.")
        rel = V["v_rel"].astype(np.float32)
        ax = fig.add_subplot(2, 3, 1)
        grey(ax, ch0[z], title="ch0: closed-surface depth (cyan, every 200 um), raw edge (orange)")
        ax.contour(depth[z], np.arange(200, 2001, 200), colors="c", linewidths=0.5)
        ax.contour(V["v_depth_raw"][z].astype(np.float32), [5], colors="orange", linewidths=0.7)
        ax = fig.add_subplot(2, 3, 2)
        grey(ax, ch0[z], title=f"L4 bottom (red) and slab {d_off:+d} +- 30 um (yellow)")
        rz = np.where(region[z], rel[z], np.nan)
        ax.contour(np.ma.masked_invalid(rz), [0], colors="r", linewidths=1)
        ax.contour(np.ma.masked_invalid(rz), [d_off - 30, d_off + 30], colors="y", linewidths=0.8)
        ax = fig.add_subplot(2, 3, 3)
        grey(ax, ch0[z], title="brain WM depth used for the columns (magenta)")
        tb = V["v_tb"][z].astype(np.float32)
        ax.contour(np.ma.masked_invalid(depth[z] - tb), [0], colors="m", linewidths=1)
        # mean ch0 depth profile over the D2 barrel columns
        ax = fig.add_subplot(2, 3, 4)
        m = region & (out > 0) & np.isfinite(rel)
        r_, c_ = rel[m], ch0[m]
        bins = np.arange(-600, 801, 10)
        k = np.digitize(r_, bins)
        n_ = np.bincount(k, minlength=len(bins) + 1)[1:len(bins)]
        prof = np.bincount(k, weights=c_, minlength=len(bins) + 1)[1:len(bins)] / np.maximum(n_, 1)
        prof[n_ == 0] = np.nan
        ax.plot(bins[:-1] + 5, prof, color="k")
        ax.axvline(0, color="r", label="L4 bottom")
        ax.axvspan(d_off - 30, d_off + 30, color="y", alpha=0.3, label="slab")
        ax.set_xlabel("depth relative to the image L4 bottom (um, negative = above)")
        ax.set_ylabel("mean ch0 in the barrel columns")
        ax.legend(fontsize=8)
        ax.set_title("ch0 profile across the barrel field")
        for j, (key, lab, cm) in enumerate((("bot", "image L4 bottom (um below pia)", "magma"),
                                            ("bw", "brain WM depth (um)", "viridis"))):
            ax = fig.add_subplot(2, 3, 5 + j)
            im = ax.imshow(np.where(alab > 0, D[key], np.nan), cmap=cm)
            plt.colorbar(im, ax=ax, fraction=0.04)
            zoom_to(ax, alab > 0)
            ax.set_title(f"{lab}, flat map (template field)")
            ax.set_xticks([]); ax.set_yticks([])
        done(pdf, fig)

        # ---------------------------------------------------------------- 5 slab and image barrels
        offs = list(D["offsets"])
        show = [o for o in offs if -150 <= o <= 0]
        fig = page(pdf, "Steps 3-4. L4 slab choice and image barrels",
                   f"Top: ch0 in the flat plane at each offset from the L4 bottom (band-passed). Candidates: best "
                   f"contrast ({int(D['d_con'])} um, range {CONTRAST_RANGE}) and best coarse pattern match "
                   f"({int(D['d_pat'])} um, range {PATTERN_RANGE}); the full fit ran at both and kept {d_off:+d} um "
                   f"(more matched barrels). Bottom: slab at {d_off:+d} um, artefact mask, image barrels.")
        gs = fig.add_gridspec(2, len(show) + 1, top=0.88, hspace=0.25)
        fld = ndimage.binary_dilation(D["foot"], iterations=10)
        for j, o in enumerate(show):
            ax = fig.add_subplot(gs[0, j])
            i = offs.index(o)
            v = D["stack_valid"][i]
            grey(ax, bandpass(D["stack_img"][i], v), v & fld, title=f"{o:+d} um" + (" (used)" if o == d_off else ""))
            zoom_to(ax, D["foot"], 20)
        ax = fig.add_subplot(gs[0, -1])
        ax.plot(D["con_d"], D["con_v"], "o-", label="contrast")
        ax2 = ax.twinx()
        ax2.plot(D["pat_d"], D["pat_v"], "s-", color="C1", label="pattern corr")
        ax.axvline(d_off, color="k", ls=":")
        ax.set_xlabel("offset (um)")
        ax.set_title("slab criteria")
        ax.legend(handles=[Line2D([], [], color="C0", marker="o", label="contrast"),
                           Line2D([], [], color="C1", marker="s", label="pattern corr")], fontsize=7)
        bpA, v2, art = D[p + "bpA"], D[p + "valid"] & ~D[p + "art"], D[p + "art"]
        sub = fig.add_gridspec(2, 4, top=0.88, hspace=0.25)
        ax = fig.add_subplot(sub[1, 0])
        grey(ax, D[p + "l4"], D[p + "valid"], title="slab (mean of offsets +-25 um)")
        zoom_to(ax, D["foot"], 25)
        ax = fig.add_subplot(sub[1, 1])
        grey(ax, D[p + "l4"], D[p + "valid"], title="artefacts masked (red: |robust z| > 4.5)")
        ax.contour(art, [0.5], colors="r", linewidths=0.8)
        zoom_to(ax, D["foot"], 25)
        ax = fig.add_subplot(sub[1, 2])
        grey(ax, bpA, v2, title="band-passed (sigma 10 / 80 um)")
        zoom_to(ax, D["foot"], 25)
        ax = fig.add_subplot(sub[1, 3])
        grey(ax, bpA, v2, title=f"image barrels: {len(D[p + 'Bc'])} blobs (watershed) + centroids")
        yy, xx = np.where(find_boundaries(D[p + "blobs"], mode="inner"))
        ax.scatter(xx, yy, s=0.5, c="c", lw=0)
        ax.scatter(D[p + "Bc"][:, 1], D[p + "Bc"][:, 0], s=8, c="r")
        zoom_to(ax, D["foot"], 25)
        done(pdf, fig)

        # ---------------------------------------------------------------- 6 search
        sa = D[p + "search_all"]
        tops = D[p + "tops"]
        fig = page(pdf, "Step 5. Exhaustive search of the barrel pattern",
                   f"{len(sa):,} placements (scale x rotation x shift, 20 um grid) scored by masked correlation of the "
                   f"synthetic pattern with the band-passed slab. Brainreg = identity (scale 1, rotation 0, shift 0). "
                   f"The 5 best distinct placements go to the point matching.")
        ax = fig.add_subplot(2, 3, 1)
        scs, angs = np.unique(sa[:, 1]), np.unique(sa[:, 2])
        H = np.full((len(scs), len(angs)), np.nan)
        for i, s_ in enumerate(scs):
            for j, a_ in enumerate(angs):
                mm = (sa[:, 1] == s_) & (sa[:, 2] == a_)
                H[i, j] = sa[mm, 0].max()
        im = ax.imshow(H, aspect="auto", cmap="viridis", origin="lower",
                       extent=[angs[0] - 1.25, angs[-1] + 1.25, -0.5, len(scs) - 0.5])
        ax.set_yticks(range(len(scs)), [f"{s_:.1f}" for s_ in scs])
        ax.set_xlabel("rotation (deg)")
        ax.set_ylabel("scale")
        ax.set_title("best correlation over shifts")
        plt.colorbar(im, ax=ax, fraction=0.04)
        for t in tops:
            ax.plot(t[2], int(np.argmin(np.abs(scs - t[1]))), "r+", ms=10)
        ax = fig.add_subplot(2, 3, 4)
        b = tops[0]
        mm = np.isclose(sa[:, 1], b[1], atol=1e-3) & np.isclose(sa[:, 2], b[2], atol=1e-3)
        dys, dxs = np.unique(sa[mm, 3]), np.unique(sa[mm, 4])
        G = np.full((len(dys), len(dxs)), np.nan)
        for r in sa[mm]:
            G[list(dys).index(r[3]), list(dxs).index(r[4])] = r[0]
        im = ax.imshow(G, cmap="viridis", origin="upper", extent=[dxs[0], dxs[-1], dys[-1], dys[0]])
        ax.plot(b[4], b[3], "r+", ms=12)
        ax.plot(0, 0, "wo", ms=5)
        ax.set_xlabel("shift, flat axis 2 (um)")
        ax.set_ylabel("shift, flat axis 1 (um)")
        ax.set_title(f"correlation over shifts at scale {b[1]:.1f}, rotation {b[2]:.1f}")
        plt.colorbar(im, ax=ax, fraction=0.04)
        fix = D[p + "fix"]
        for j, t in enumerate(tops[:4]):
            ax = fig.add_subplot(2, 3, [2, 3, 5, 6][j])
            grey(ax, fix, D[p + "mask"], title=f"#{j + 1}: corr {t[0]:.3f}, scale {t[1]:.1f}, rot {t[2]:.1f}, "
                                               f"shift {t[3]:.0f},{t[4]:.0f} um")
            outline_labels(ax, similarity_cols(cols, *t[1:], D["centre"]), names=True, lw=0.5)
            zoom_to(ax, D["foot"], 40)
        done(pdf, fig)

        # ---------------------------------------------------------------- 7 matching
        cand = sorted({int(k.split("_")[1][1:]) for k in D if k.startswith(p + "c") and k.endswith("_T_sim")})
        Ac, Bc = D["Ac"], D[p + "Bc"]
        fig = page(pdf, "Step 6. Point matching per candidate placement",
                   "Grey = image blob centroids; blue = atlas barrel centres at the search placement; arrows = final "
                   "one-to-one matches, drawn from each atlas barrel at the brainreg (identity) position to its blob (CPD affine -> CPD non-rigid -> Hungarian within 70 um and plausible size -> "
                   "TPS re-pairing -> leave-one-out cleaning). Among candidates within 0.02 warped correlation of the "
                   "best, the one whose matches an affine map explains best wins (a one-row naming shift cannot be "
                   "affine).")
        for j, c in enumerate(cand[:5]):
            ax = fig.add_subplot(2, 3, j + 1)
            grey(ax, bpA, v2)
            T = D[f"{p}c{c}_T_sim"]
            ax.scatter(Bc[:, 1], Bc[:, 0], s=10, c="0.5")
            ax.scatter(T[:, 1], T[:, 0], s=10, c="b")
            if f"{p}c{c}_ia" in D:
                ia_, ib_ = D[f"{p}c{c}_ia"], D[f"{p}c{c}_ib"]
                for a_, b_ in zip(ia_, ib_):
                    ax.annotate("", Bc[b_, ::-1], Ac[a_, ::-1], arrowprops=dict(arrowstyle="->", color="y", lw=0.9))
                chosen = len(ia_) == len(D["ia"]) and np.array_equal(np.sort(ia_), np.sort(D["ia"])) and \
                    np.array_equal(D[f"{p}c{c}_ib"][np.argsort(ia_)], D["ib"][np.argsort(D["ia"])])
                ax.set_title(f"candidate {c + 1}: {len(ia_)} matched, warped corr {float(D[f'{p}c{c}_corr']):.3f}, "
                             f"affine res {float(D[f'{p}c{c}_affres']):.0f} um" + ("  CHOSEN" if chosen else ""),
                             color="r" if chosen else "k", fontsize=9)
            else:
                ax.set_title(f"candidate {c + 1}: < 6 matches, dropped", fontsize=9)
            zoom_to(ax, D["foot"], 40)
        ax = fig.add_subplot(2, 3, 6)
        ax.axis("off")
        ax.text(0, 1, f"chosen: {len(D['ia'])}/{len(Ac)} atlas barrels matched\n"
                      f"naming ambiguous: {met['naming_ambiguous']}\nmargin to another naming: "
                      f"{met['margin_to_other_naming']:.3f}\ncandidates agreeing on the names: "
                      f"{met['n_candidates_agree']}\n\nunmatched atlas barrels:\n  " +
                ", ".join(name(k) for i, k in enumerate(D["ksA"]) if i not in set(D["ia"])), va="top", fontsize=10)
        done(pdf, fig)

        # ---------------------------------------------------------------- 8 warp
        disp = D["disp"]
        fig = page(pdf, "Step 7. Thin-plate-spline warp through the matched barrels",
                   f"Smoothing chosen by leave-one-out ({D['s_best']:g}); outside the field the warp blends into a "
                   f"near-affine fit (900 um taper). Landmark error {met['err_initial_um']:.0f} um before -> "
                   f"{met['err_tps_loo_um']:.0f} um leave-one-out after; min Jacobian {met['jac_min']:.2f}.")
        ax = fig.add_subplot(2, 3, 1)
        grey(ax, bpA, v2, title="brainreg (identity) placement")
        draw(ax, cols)
        zoom_to(ax, (cols > 0) | (cols_new > 0), 30)
        ax = fig.add_subplot(2, 3, 2)
        grey(ax, bpA, v2, title="D2 corrected barrels")
        draw(ax, cols_new)
        zoom_to(ax, (cols > 0) | (cols_new > 0), 30)
        ax = fig.add_subplot(2, 3, 3)
        fig.subplots_adjust(wspace=0.35)
        U, Vv = disp.shape[1:]
        gy, gx = np.mgrid[0:U:12, 0:Vv:12]
        mag = np.hypot(*disp) * PIX
        im = ax.imshow(mag, cmap="magma")
        ax.quiver(gx, gy, disp[1][::12, ::12], -disp[0][::12, ::12], color="c", scale=1, scale_units="xy", angles="xy")
        ax.contour(alab > 0, [0.5], colors="w", linewidths=0.6)
        plt.colorbar(im, ax=ax, fraction=0.04, label="|displacement| (um)")
        zoom_to(ax, (cols > 0) | (cols_new > 0), 60)
        ax.set_title("displacement (template -> image)")
        ax.set_xticks([]); ax.set_yticks([])
        ax = fig.add_subplot(2, 3, 4)
        im = ax.imshow(D["jac"], cmap="coolwarm", vmin=0, vmax=2)
        plt.colorbar(im, ax=ax, fraction=0.04)
        zoom_to(ax, (cols > 0) | (cols_new > 0), 60)
        ax.set_title("Jacobian determinant (1 = area kept, < 0 = fold)")
        ax.set_xticks([]); ax.set_yticks([])
        ax = fig.add_subplot(2, 3, 5)
        ax.semilogx(np.maximum(D["smooth_s"], 1), np.array(D["smooth_err"]) * PIX, "o-")
        ax.axvline(max(float(D["s_best"]), 1), color="r", ls=":")
        ax.set_xlabel("TPS smoothing (0 plotted at 1)")
        ax.set_ylabel("mean leave-one-out error (um)")
        ax.set_title("smoothing choice")
        ax = fig.add_subplot(2, 3, 6)
        ax.hist(D["e_before"], bins=20, alpha=0.6, label="before (brainreg -> image)")
        ax.hist(D["e_loo"], bins=20, alpha=0.6, label="after (leave-one-out)")
        ax.set_xlabel("barrel centre error (um)")
        ax.legend(fontsize=8)
        ax.set_title("per-barrel landmark error")
        done(pdf, fig)

        # ---------------------------------------------------------------- flat L4 views (publication panels)
        fv_dir = os.path.join(os.path.dirname(out_pdf), "flat_views")
        box, l4s, valid, v2f, bpf = flat_views(D, p, cols, cols_new, fv_dir, f"{bid}_h{hv}")
        fig = page(pdf, "Flat L4 view: brainreg vs refined registration",
                   f"Square crops of the flat L4 slab (offset {d_off} um from the L4 bottom; background flattened by the ratio to a 300 um Gaussian), barrel outlines coloured "
                   f"by row. Top: brainreg (global) vs refined (local) on the slab; bottom: the same on the band-passed slab "
                   f"(artefacts masked), with barrel names. Each panel is also saved "
                   f"as png / pdf / svg in {fv_dir}.")
        fig.subplots_adjust(left=0.02, right=0.98, top=0.86, bottom=0.02, wspace=0.04, hspace=0.12)
        for i, (img, vv, lab, sg, t) in enumerate((
                (l4s, valid, cols, False, "brainreg (global)"),
                (l4s, valid, cols_new, False, "refined (local)"),
                (bpf, v2f, cols, True, "brainreg (global), band-passed"),
                (bpf, v2f, cols_new, True, "refined (local), band-passed"))):
            ax = fig.add_subplot(2, 2, i + 1)
            flat_panel(ax, img, vv, box, cols=lab, signed=sg, title=t, fs=7, names=i >= 2)
            if i == 1:
                row_legend(ax, 8)
        done(pdf, fig)

        # ---------------------------------------------------------------- 9 3D
        fig = page(pdf, "Step 8. Back to 3D: barrel columns and depth mapping",
                   "Each voxel takes the barrel whose convex hull (flat plane) contains its corrected flat coordinate; "
                   "columns run from the image pia to the brain WM depth. Depth is mapped piecewise linearly: image "
                   "pia -> atlas pia, image L4 bottom -> atlas L4 bottom, brain WM -> atlas WM; then the inverse "
                   "flatmap gives the CCF position.")
        zs = [z] + [int(v) for v in np.percentile(np.argwhere(out > 0)[:, 0], [20, 80])]
        for j, zz in enumerate(zs):
            for i, (lab, t) in enumerate(((lab_reg, "brainreg"), (out, "D2"))):
                ax = fig.add_subplot(2, 4, 1 + j + (4 if i else 0) if j < 3 else 4)
                grey(ax, ch0[zz], title=f"{t}, coronal {lo[0] + zz}")
                outline_labels(ax, lab[zz], names=True, lw=0.5)
                zoom_to(ax, (out[zz] > 0) | (lab_reg[zz] > 0), 30)
        ax = fig.add_subplot(2, 4, 4)
        k_show = C2 if (out == C2).any() else int(np.bincount(out[out > 0]).argmax())
        mm = (out == k_show) & np.isfinite(V["v_adep"].astype(np.float32))
        dd, aa = depth[mm], V["v_adep"][mm].astype(np.float32)
        o = np.argsort(dd)[:: max(1, mm.sum() // 4000)]
        ax.scatter(dd[o], aa[o], s=1, c="k")
        lim = [0, max(dd.max(), aa.max())]
        ax.plot(lim, lim, color="0.6", ls=":")
        ax.axvline(np.nanmedian(V["v_bimg"][mm].astype(np.float32)), color="r", lw=0.8, label="image L4 bottom")
        ax.axhline(np.nanmedian(V["v_batl"][mm].astype(np.float32)), color="b", lw=0.8, label="atlas L4 bottom")
        ax.set_xlabel("depth below image pia (um)")
        ax.set_ylabel("mapped atlas depth (um)")
        ax.set_title(f"depth mapping in the {name(k_show)} column")
        ax.legend(fontsize=8)
        ax = fig.add_subplot(2, 4, 8)
        ax.axis("off")
        ax.text(0, 1, f"CCF label agreement {met['ccf_label_agreement']:.3f}\n(labelled voxels -> CCF annotation "
                      f"at the mapped voxel;\ntests the inverse lookup, not the names)\n\nCCF defined fraction "
                      f"{met['ccf_defined_frac']:.2f}\nC2 voxels mapped into CCF C2: {met['ccf_c2_in_c2']:.3f}",
                va="top", fontsize=10)
        done(pdf, fig)

        # ---------------------------------------------------------------- 10 tracks
        for nm, X, q in trk:
            fig = page(pdf, f"Neuropixels track {nm}: barrels along the probe, brainreg vs D2",
                       "Track = brainreg-segment points in sample space, sampled every 10 um from the shallow end. "
                       "Left: coronal slice at the track's L4 crossing. Middle: the track's L4 crossing in the flat "
                       "view. Right: labels along the probe.")
            zt = int(np.median(q[:, 0]))
            for i, (lab, t) in enumerate(((lab_reg, "brainreg"), (out, "D2"))):
                ax = fig.add_subplot(2, 3, 1 + 3 * i)
                grey(ax, ch0[zt], title=f"{t} barrel columns, coronal {lo[0] + zt}")
                outline_labels(ax, lab[zt], names=True, lw=0.5)
                near = np.abs(X[:, 0] - zt) < 15
                ax.plot(X[:, 2], X[:, 1], color="m", lw=0.8, alpha=0.4)
                ax.plot(X[near, 2], X[near, 1], color="m", lw=2)
                zoom_to(ax, ((out[zt] > 0) | (lab_reg[zt] > 0)), 40)
            # flat view: L4 crossing of the track
            Fv = F[tuple(q.T)]
            relq = rel[tuple(q.T)]
            l4 = np.isfinite(Fv).all(1) & (np.abs(relq - d_off) < 60)
            fi = (Fv[l4] - flat_lo) / PIXf - 0.5
            dsp = np.stack([ndimage.map_coordinates(disp[k], fi.T, order=1, mode="nearest") for k in range(2)], 1)
            for i, (lab, t, pts) in enumerate(((cols, "brainreg (identity)", fi), (cols_new, "D2", fi))):
                ax = fig.add_subplot(2, 3, 2 + 3 * i)
                grey(ax, bpA, v2, title=f"flat L4 view, {t}: track L4 crossing (magenta)")
                draw(ax, lab)
                ax.scatter(pts[:, 1], pts[:, 0], s=12, c="m")
                zoom_to(ax, (cols > 0) | (cols_new > 0), 30)
            ax = fig.add_subplot(1, 3, 3)
            dq = depth[tuple(q.T)]
            for i, (lab, t) in enumerate(((lab_reg, "brainreg"), (out, "D2"))):
                b = lab[tuple(q.T)]
                for kk in np.unique(b[b > 0]):
                    if kk >= SEPTA:
                        continue
                    yy = dq[b == kk]
                    ax.scatter(np.full(len(yy), i), yy, c=rowcol(kk), s=12, marker="s")
                    ax.text(i + 0.12, np.median(yy), name(kk), color=rowcol(kk), fontsize=9, va="center", weight="bold")
            acr = [ACR.get(int(a), "") for a in atl[tuple(q.T)]]
            lay = [m.group(1) if (m := re.search(r"(1|2/3|4|5|6a|6b)$", a)) and a.startswith("SSp") else ""
                   for a in acr]
            for i in range(0, len(lay), 5):
                if lay[i] and (i == 0 or lay[i] != lay[max(i - 5, 0)]):
                    ax.text(-0.6, dq[i], f"L{lay[i]}", fontsize=8, color="0.4")
            bq = V["v_bimg"][tuple(q.T)].astype(np.float32)
            ax.axhline(np.nanmedian(bq), color="r", lw=0.8, ls=":", label="image L4 bottom")
            ax.set_xlim(-0.8, 1.8)
            ax.set_xticks([0, 1], ["brainreg", "D2"])
            ax.invert_yaxis()
            ax.set_ylabel("depth below image pia (um)")
            ax.set_title("barrel along the probe (left: brainreg layer)")
            ax.legend(fontsize=8, loc="lower right")
            done(pdf, fig)
    print("wrote", out_pdf, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", default="config.toml")
    ap.add_argument("brains", nargs="+")
    ap.add_argument("--hemisphere", type=int, default=1)
    ap.add_argument("--tracks")
    ap.add_argument("--out")
    a = ap.parse_args()
    cfg = Config(a.config)
    tracks = pd.read_csv(a.tracks) if a.tracks else None
    out = a.out or f"{cfg.work_dir}/reports"
    os.makedirs(out, exist_ok=True)
    for b in a.brains:
        try:
            report(cfg, b, a.hemisphere, tracks, f"{out}/{b}_h{a.hemisphere}_D2_report.pdf")
        except Exception:
            import traceback
            traceback.print_exc()
            print(b, "FAILED", flush=True)


if __name__ == "__main__":
    main()
