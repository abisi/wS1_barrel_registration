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
import matplotlib.patches
import matplotlib.pyplot as plt
from matplotlib import patheffects
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D

from barrelreg.atlas import BARREL_NAMES, BFD_IDS, C2, ISO_IDS, S, SEPTA, collapse, row_of
from barrelreg.config import Config, results_dir
from barrelreg.imageproc import bandpass
from barrelreg.register import CONTRAST_RANGE, PATTERN_RANGE, PIX

plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "figure.facecolor": "white"})
FIG = (16.5, 10)
ACR = dict(zip(S.id.astype(int), S.acronym))
STEPS = ["1 own flatmap (Laplace between registered pia and WM -> streamlines -> flat L4 plane)",
         "2 fixed template (atlas L4 barrels in the Blue Brain flatmap, same plane convention)",
         "3 L4 bottom per column from ch0; L4 slab at candidate offsets above it, best fit kept",
         "4 image barrels: artefact mask, band-pass, local maxima + watershed",
         "5 exhaustive search: scale x rotation x shift of the synthetic barrel pattern -> top candidates",
         "6 per candidate: CPD affine -> CPD non-rigid -> Hungarian -> re-pairing -> LOO cleaning; tie-break",
         "7 thin-plate spline through the matches, near-affine outside the field",
         "8 3D: convex-hull barrel columns, piecewise depth mapping, inverse flatmap -> CCF"]


def name(k):
    return BARREL_NAMES[int(k) - 1] if 0 < int(k) <= len(BARREL_NAMES) else ""


# row palette of the report (bright on a grey slab, row order as a hue sweep, straddlers magenta)
PAL = {"A": "#FF6F6F", "B": "#FFB547", "C": "#B5E655", "D": "#4DD4E8", "E": "#A98BFF", "S": "#F77FE0"}


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
            ax.text(cx, cy, name(k), color=rowcol(k), fontsize=fs, ha="center", va="center", weight="bold")


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


REF_COL = "white"


def draw_ref(ax, blobs, lw=0.7):
    """Reference: outline of each image barrel (watershed blob) the atlas barrels were matched to."""
    for k in np.unique(blobs):
        if k <= 0:
            continue
        m = ndimage.gaussian_filter((blobs == k).astype(np.float32), 1.0)
        ax.contour(m, [0.5], colors=[REF_COL], linewidths=lw, alpha=0.9)


def flat_panel(ax, img, valid, box, cols=None, cols_ghost=None, signed=False, title="", fs=7, bar=True, names=True,
               ref=None, ls="-", sig=None, cross=()):
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
    if sig is not None:
        draw_signal(ax, sig)
    if ref is not None:
        draw_ref(ax, ref)
    if cols_ghost is not None:
        draw(ax, cols_ghost, lw=0.8, names=False, ls="dashed", alpha=0.8)
    if cols is not None:
        draw(ax, cols, lw=1.3, fs=fs, names=names, ls=ls)
    for r_, c_ in cross:
        ax.scatter([c_], [r_], s=70, facecolors="none", edgecolors="w", linewidths=1.6, zorder=5)
        ax.scatter([c_], [r_], s=70, facecolors="none", edgecolors=SIG_COL, linewidths=0.8, zorder=6)
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


def row_legend(ax, fs=8, ref=False, outside=False, sig=False, cross=False):
    h = [Line2D([], [], color=PAL[r], lw=2, label=ROW_LABEL[r]) for r in "ABCDES"]
    if ref:
        k = "0.3" if outside else "w"  # line-style entries: dark on the white page, white on the image
        h += [Line2D([], [], color=k, lw=1.3, ls="--", label="brainreg (global)"),
              Line2D([], [], color=k, lw=1.3, label="refined (local)"),
              Line2D([], [], color="0.75" if outside else REF_COL, lw=0.9, label="image barrels (reference)",
                     path_effects=[patheffects.withStroke(linewidth=1.8, foreground="0.3")] if outside else None)]
    if sig:
        h.append(matplotlib.patches.Patch(color=SIG_COL, alpha=0.7, label="signal channel (probe dye)"))
    if cross:
        h.append(Line2D([], [], ls="", marker="o", mfc="none", mec=SIG_COL, mew=1.2, ms=7, label="track L4 crossing"))
    if outside:  # to the right of the axes, on the white page
        ax.legend(handles=h, loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=fs, frameon=False,
                  handlelength=2.2)
        return
    ax.legend(handles=h, loc="lower right", fontsize=fs, frameon=True, facecolor="black", edgecolor="none",
              labelcolor="w", framealpha=0.6, handlelength=1.2)


SIG_COL = "#FF3B30"  # signal channel (probe dye) overlay


def signal_flat(sig3, F, region, flat_lo, shape):
    """Signal channel projected onto the flat map: per flat pixel the 99th percentile over the whole column (pia to
    WM), background-subtracted and scaled to 0..1 (median -> 0, 99.9th percentile of the field -> 1)."""
    m = region & np.isfinite(F).all(-1)
    ij = np.rint((F[m] - flat_lo) / PIX - 0.5).astype(int)
    ok = (ij >= 0).all(1) & (ij[:, 0] < shape[0]) & (ij[:, 1] < shape[1])
    ij, v = ij[ok], sig3[m][ok].astype(np.float32)
    lin = ij[:, 0] * shape[1] + ij[:, 1]
    o = np.lexsort((v, lin))
    lin, v = lin[o], v[o]
    starts = np.r_[0, np.flatnonzero(np.diff(lin)) + 1]
    ends = np.r_[starts[1:], len(lin)]
    out = np.full(shape[0] * shape[1], np.nan, np.float32)
    out[lin[starts]] = v[starts + ((ends - starts - 1) * 0.99).astype(int)]
    out = out.reshape(shape)
    have = np.isfinite(out)
    out = ndimage.gaussian_filter(np.where(have, out, 0), 1.0) / np.maximum(
        ndimage.gaussian_filter(have.astype(np.float32), 1.0), 1e-3)
    lo, hi = np.nanpercentile(out[have], (50, 99.9))
    return np.where(have, np.clip((out - lo) / max(hi - lo, 1e-6), 0, 1), 0)


def draw_signal(ax, sig, gamma=1.5, amax=0.85):
    rgba = np.zeros(sig.shape + (4,), np.float32)
    rgba[..., :3] = matplotlib.colors.to_rgb(SIG_COL)
    rgba[..., 3] = amax * sig ** gamma
    ax.imshow(rgba, interpolation="bilinear")


def flat_views(D, p, cols, cols_new, out_dir, tag, sig=None, cross=()):
    """Publication panels, one square file each (png 300 dpi, pdf, svg): flat L4 slab with brainreg and with the
    refined (D2) barrels, raw and band-passed, plus both side by side (brainreg left, refined right). With `sig`
    (signal channel on the flat map) a second side-by-side pair shows the probe dye in red, `cross` = flat (row, col)
    of each track's L4 crossing (ring markers)."""
    os.makedirs(out_dir, exist_ok=True)
    valid, bp = D[p + "valid"], D[p + "bpA"]
    l4 = flatten(D[p + "l4"], valid)
    v2 = valid & ~D[p + "art"]
    box = square_box((cols > 0) | (cols_new > 0))
    blobs = D[p + "blobs"]
    ref = np.where(np.isin(blobs, np.asarray(D["ib"]) + 1), blobs, 0)  # blob k <-> centroid Bc[k - 1]
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
    # side by side: brainreg (global) left, refined right, same crop; second version with the signal channel
    for suffix, sg in (("", None), ("_signal", sig)):
        if suffix and sg is None:
            continue
        fig = plt.figure(figsize=(9.6, 4.2))
        for i, (lab, t, ls) in enumerate(((cols, "brainreg (global)", "dashed"), (cols_new, "refined (local)", "-"))):
            ax = fig.add_axes([i * 0.415 + 0.004, 0, 0.405, 0.93])
            flat_panel(ax, l4, valid, box, cols=lab, names=True, title=t, bar=i == 0, ref=ref, ls=ls, sig=sg,
                       cross=cross if sg is not None else ())
            if i == 1:
                row_legend(ax, 7, ref=True, outside=True, sig=sg is not None, cross=len(cross) > 0 and sg is not None)
        for ext in ("png", "pdf", "svg"):
            fig.savefig(f"{out_dir}/{tag}_side_by_side{suffix}.{ext}", dpi=300, facecolor="white")
        plt.close(fig)
    for ext in ("png", "pdf", "svg"):  # replaced by the side-by-side pair
        if os.path.exists(f"{out_dir}/{tag}_overlay.{ext}"):
            os.remove(f"{out_dir}/{tag}_overlay.{ext}")
    plt.rcParams.update({"svg.fonttype": "path", "pdf.fonttype": 3})
    return box, l4, valid, v2, bp, ref


def page(pdf, title, sub="", h=FIG[1]):
    """New page; h < 10 for one-row pages (less white space). `sub` = explanation under the title; paragraphs are
    separated by blank lines, each wrapped to the page width. fig.top_ = top of the panel area below the text."""
    import textwrap
    paras = [textwrap.fill(" ".join(q.split()), 205) for q in sub.split("\n\n") if q.strip()] if sub else []
    n_lines = sum(q.count("\n") + 1 for q in paras) + max(len(paras) - 1, 0) * 0.4
    h = h + 0.17 * n_lines  # the page grows with its text, the panels keep their size
    fig = plt.figure(figsize=(FIG[0], h))
    fig.suptitle(title, x=0.01, ha="left", fontsize=15, weight="bold", y=1 - 0.15 / h)
    y = 1 - 0.5 / h
    for q in paras:
        fig.text(0.01, y, q, fontsize=10, color="0.25", va="top", linespacing=1.35)
        y -= (q.count("\n") + 1.4) * 0.17 / h
    fig.top_ = y - 0.35 / h
    if h - 0.17 * n_lines < FIG[1]:
        fig.subplots_adjust(left=0.03, right=0.97, top=fig.top_, bottom=0.04, wspace=0.25)
    else:
        fig.subplots_adjust(top=fig.top_)
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


# ----------------------------------------------------------------------------- variants
def variant_grid(pdf, cfg, bid, hv):
    """Final runs (build_final.py): every candidate fit of this hemisphere side by side, the chosen one framed."""
    sf = f"{cfg.brain_dir(bid)}/source.json"
    if not os.path.exists(sf):
        return
    src = json.load(open(sf))
    dirs = src.get("candidate_dirs", {})
    V = []
    for lab, d in dirs.items():
        f = f"{results_dir(d, bid)}/h{hv}/debug.npz"
        if os.path.exists(f):
            V.append((lab, np.load(f), src["candidates"].get(lab, {})))
    if len(V) < 2:
        return
    D0 = V[0][1]
    box = square_box((D0["cols"] > 0) | np.any([D["cols_new"] > 0 for _, D, _ in V], 0))
    score = src.get("score", "contrast_after")
    nc = min(len(V), 4)
    nr = -(-len(V) // nc)
    fig = page(pdf, "Every variant of the fit, chosen one framed",
               f"Same hemisphere fitted with each combination of image channel, search and lesion mask (wide search, "
               f"slab depth chosen per fit by barrel contrast). Dashed: brainreg barrels; solid: fitted barrels; black: "
               f"masked pixels. Score = {score} (barrel-vs-septum contrast after the fit"
               f"{', mean of ch0 and ch1 images' if score == 'contrast_both' else ''}); the variant with the highest "
               f"score is chosen. With the lesion mask the contrast is measured on unmasked tissue only, so judge those "
               f"fits by eye.", h=4.6 * nr + 0.6)
    gs = fig.add_gridspec(nr, nc, left=0.02, right=0.98, top=fig.top_, bottom=0.02, wspace=0.05, hspace=0.12)
    for i, (lab, D, c) in enumerate(V):
        ax = fig.add_subplot(gs[i // nc, i % nc])
        p = f"o{int(D['chosen_offset'])}_"
        val = D[p + "valid"] & ~D[p + "art"]
        chosen = lab == src["chosen_run"]
        t = (f"{lab}{'  (CHOSEN)' if chosen else ''}\n{score} {c.get(score, float('nan')):.2f}, contrast "
             f"{c.get('contrast_after', float('nan')):.2f}, {c.get('n_matched', '-')} matched, slab "
             f"{c.get('l4_offset_um', 0):+d} um")
        flat_panel(ax, flatten(D[p + "l4"], D[p + "valid"]), val, box, cols=D["cols_new"], cols_ghost=D["cols"],
                   fs=5, bar=i == 0, title=t)
        ax.title.set_color("#1a7f37" if chosen else "0.2")
        if chosen:
            for sp in ax.spines.values():
                sp.set_visible(True)
                sp.set_color("#1a7f37")
                sp.set_linewidth(3)
    done(pdf, fig)


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
                   f"What this is: brainreg aligns the whole brain to the Allen / Blue Brain barrel atlas. It gets the "
                   f"hemisphere, the cortical surfaces and the rough position of the barrel field right, but single "
                   f"barrels can be off by one or more barrel widths (100-500 um). D2 keeps only these coarse pieces "
                   f"and finds the barrels again in this brain's own autofluorescence (ch0), where L4 barrels are visible "
                   f"as bright patches separated by darker septa. The atlas barrel map is then warped onto them and the "
                   f"result is carried back to 3D and to CCF coordinates.\n\n"
                   f"Left: quality numbers of this fit (contrast after < 0.3 = failed: the warped atlas barrels are not "
                   f"brighter than the septa). Right: the steps and, for Neuropixels tracks in this hemisphere, the L4 "
                   f"barrels they cross before and after. Every following page shows one step on this brain's own "
                   f"intermediate data. Registration: {reg}.")
        ax = fig.add_axes([0.02, 0.05, 0.46, fig.top_ - 0.05])
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
        ax = fig.add_axes([0.5, 0.05, 0.48, fig.top_ - 0.05])
        ax.axis("off")
        txt = ["Pipeline steps (one page each):"] + [f"  {s}" for s in STEPS] + ["", "Neuropixels wS1 track(s) (last pages):"]
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

        # ---------------------------------------------------------------- 0b every variant (final runs)
        variant_grid(pdf, cfg, bid, hv)

        # ---------------------------------------------------------------- 1 inputs
        fig = page(pdf, "Inputs: autofluorescence (ch0) and the coarse whole-brain registration",
                   "What: the cleared brain imaged in the autofluorescence channel (ch0, downsampled to 10 um) and "
                   "brainreg's registration of the atlas (allen_mouse_bluebrain_barrels_10um) to it.\n\n"
                   "Why only coarse: a whole-brain registration is driven by large structures (outline, ventricles, "
                   "white matter); the barrels are small and only visible in a thin layer, so they barely pull on it. "
                   "D2 therefore uses brainreg for the hemisphere, the pia and white-matter surfaces and the rough "
                   "SSp-bfd position only.\n\n"
                   "How to read: coronal ch0 with the registered atlas outlines; box = the crop around the registered "
                   "SSp-bfd (+700 um) in which everything else is computed.", h=6.5)
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
                   f"Why: barrels are columns perpendicular to the curved cortical sheet. To see the barrel pattern, L4 "
                   f"has to be sampled along that curvature and laid flat, exactly as the atlas flatmap does for the "
                   f"template (step 2).\n\n"
                   f"How: Laplace's equation is solved between the registered pia (u = 0) and white matter (u = 1) on a "
                   f"20 um grid; u rises smoothly through the cortex and its gradient points along the columns. From each "
                   f"voxel a streamline follows the gradient to the L4 level u0 = {float(D['u0']):.2f} (registered L4 "
                   f"{float(D['u0_reg']):.2f}). The end points form a curved sheet, projected on its best-fit (PCA) "
                   f"plane: e1 anterior, e2 dorsal, origin = centroid of the registered SSp-bfd L4. Every voxel thus "
                   f"gets a flat (e1, e2) position shared by its whole column.\n\n"
                   f"How to read: left, u on a coronal slice with the L4 level (white) and some streamlines (red); "
                   f"middle and right, the two flat coordinates with iso-lines every 200 um.", h=6.5)
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
                   "What: the reference that is fitted to every brain, built once from the CCF annotation and the Blue "
                   "Brain flatmap: the map of the 33 L4 barrels (rows A-E and the 4 straddlers), the cortical thickness "
                   "and atlas L4 bottom per flat pixel, and an inverse lookup (flat position, depth) -> CCF voxel.\n\n"
                   "Why fixed: the template is the same for all brains, so the fit is comparable across brains, and the "
                   "inverse lookup brings every result back to the same CCF space. The search and the warp move this "
                   "map onto the brain, never the brain onto the map.", h=6.5)
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
                   f"Why: barrels are only visible within L4 (about 150-200 um thick), and how deep L4 lies varies across "
                   f"the field and between brains (shrinkage, cutting angle, registration error). The slab is therefore "
                   f"placed relative to the L4 seen in the image, not the atlas L4.\n\n"
                   f"How: depth = distance below the segmented tissue surface, after closing it with a 400 um ball so "
                   f"that notches and tears do not count as surface (bridged fraction "
                   f"{met['surface_bridged_frac']:.2f}). In ch0, L4 is brighter than L5; in 100 um patches (pooled over "
                   f"200 um) the depth profile of ch0 has a peak in L4, and the L4 bottom is where it has fallen halfway "
                   f"to the trough below. The bottom is smoothed over 50 um in the flat plane. The slab used for the fit "
                   f"lies {d_off:+d} um from that bottom (negative = above).\n\n"
                   f"How to read: coronal ch0 with the closed-surface depth (cyan, every 200 um) and the raw tissue edge "
                   f"(orange); the L4 bottom (red) and the slab (yellow); the white-matter depth used for the columns "
                   f"(magenta); the ch0 depth profile across the barrel field; flat maps of these quantities.")
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
        fo = [int(v) for v in np.atleast_1d(D["fit_offsets"])] if "fit_offsets" in D else [int(D["d_con"]), int(D["d_pat"])]
        fig = page(pdf, "Steps 3-4. L4 slab choice and image barrels",
                   f"Slab choice: ch0 is averaged in 60 um thick slabs at a series of offsets from the L4 bottom and "
                   f"projected on the flat plane (top row, band-passed). The barrel pattern is sharpest somewhere inside "
                   f"L4, but where exactly differs between brains. Two criteria score each slab: barrel / septum "
                   f"contrast (best {int(D['d_con'])} um) and a coarse match to the atlas pattern (best "
                   f"{int(D['d_pat'])} um). The full fit (steps 4-7) ran at {', '.join(f'{o:+d}' for o in fo)} um and "
                   f"kept {d_off:+d} um, the one with most matched barrels.\n\n"
                   f"Image barrels: the chosen slab is band-passed (difference of Gaussians, about 10-150 um) to remove "
                   f"slow brightness changes and pixel noise. Artefacts (bubbles, tears, dye; |robust z| > 4.5) are "
                   f"masked. Local maxima seed a watershed, giving one blob per putative barrel; their centroids are the "
                   f"image landmarks for the matching.\n\n"
                   f"How to read: bottom row, slab, artefact mask (red), band-passed slab with the blobs and their "
                   f"centroids.")
        gs = fig.add_gridspec(2, len(show) + 1, top=fig.top_, hspace=0.25)
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
        sub = fig.add_gridspec(2, 4, top=fig.top_, hspace=0.25)
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
        sp = np.asarray(D["search_params"], float) if "search_params" in D else None
        fig = page(pdf, "Step 5. Exhaustive search of the barrel pattern",
                   f"Why exhaustive: brainreg can be one or more barrels off, rotated or scaled. A local optimiser started "
                   f"there would lock onto the nearest barrel-sized pattern, which on a periodic grid is often the wrong "
                   f"one. So every placement on a grid is tried.\n\n"
                   f"How: the atlas barrels are drawn as a synthetic image (barrels 1, septa 0, band-passed like the "
                   f"slab) and correlated with the slab, inside the field, for {len(sa):,} placements: "
                   + (f"scale {sp[4]:.2f}-{sp[5]:.2f}, rotation +-{sp[2]:g} deg by {sp[3]:g}, shift +-{sp[0]:g} um by "
                      f"{sp[1]:g} um (20 um pixels)" if sp is not None else "scale x rotation x shift on a 20 um grid")
                   + f". Brainreg is the identity (scale 1, rotation 0, shift 0). The {len(tops)} best distinct "
                   f"placements"
                   + (" are polished on a fine local grid (scale +-0.05, rotation +-3 deg, shift +-50 um) and" if sp is not None and sp[7] else "")
                   + f" go to the point matching (step 6), which decides between them.\n\n"
                   f"How to read: best correlation for each scale and rotation (over all shifts), red + = candidates; "
                   f"correlation over shifts at the best scale and rotation; the slab with the atlas barrels at each "
                   f"candidate placement.")
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
        # nearest coarse-grid scale / rotation (a refined top lies between grid points)
        s0, a0 = scs[np.argmin(np.abs(scs - b[1]))], angs[np.argmin(np.abs(angs - b[2]))]
        mm = np.isclose(sa[:, 1], s0, atol=1e-3) & np.isclose(sa[:, 2], a0, atol=1e-3)
        dys, dxs = np.unique(sa[mm, 3]), np.unique(sa[mm, 4])
        G = np.full((len(dys), len(dxs)), np.nan)
        for r in sa[mm]:
            G[list(dys).index(r[3]), list(dxs).index(r[4])] = r[0]
        im = ax.imshow(G, cmap="viridis", origin="upper", extent=[dxs[0], dxs[-1], dys[-1], dys[0]])
        ax.plot(b[4], b[3], "r+", ms=12)
        ax.plot(0, 0, "wo", ms=5)
        ax.set_xlabel("shift, flat axis 2 (um)")
        ax.set_ylabel("shift, flat axis 1 (um)")
        ax.set_title(f"correlation over shifts at scale {s0:.2f}, rotation {a0:.1f}")
        plt.colorbar(im, ax=ax, fraction=0.04)
        fix = D[p + "fix"]
        for j, t in enumerate(tops[:4]):
            ax = fig.add_subplot(2, 3, [2, 3, 5, 6][j])
            grey(ax, fix, D[p + "mask"], title=f"#{j + 1}: corr {t[0]:.3f}, scale {t[1]:.2f}, rot {t[2]:.1f}, "
                                               f"shift {t[3]:.0f},{t[4]:.0f} um")
            outline_labels(ax, similarity_cols(cols, *t[1:], D["centre"]), names=True, lw=0.5)
            zoom_to(ax, D["foot"], 40)
        done(pdf, fig)

        # ---------------------------------------------------------------- 7 matching
        cand = sorted({int(k.split("_")[1][1:]) for k in D if k.startswith(p + "c") and k.endswith("_T_sim")})
        Ac, Bc = D["Ac"], D[p + "Bc"]
        fig = page(pdf, "Step 6. Point matching per candidate placement",
                   "Why: the search gives a global placement; to correct each barrel, every atlas barrel needs its own "
                   "image barrel. Blobs are not all barrels (septa, vessels, noise) and some barrels are not detected, "
                   "so the pairing must tolerate outliers and missing points.\n\n"
                   "How, per candidate placement: coherent point drift (CPD) aligns the atlas barrel centres to the blob "
                   "centroids, first affine then non-rigid; the Hungarian algorithm then pairs them one-to-one within "
                   "70 um and with a plausible blob size; pairs are re-made after a thin-plate-spline fit and pairs that "
                   "a leave-one-out fit cannot predict are dropped. Choosing between candidates: among those within 0.02 "
                   "warped correlation of the best, the one whose matches an affine map explains best wins, because a "
                   "naming shifted by one row or arc fits the image equally well after warping but is not affine.\n\n"
                   "How to read: grey = image blob centroids; blue = atlas barrel centres at the search placement; "
                   "arrows = final matches, from each atlas barrel at the brainreg (identity) position to its blob.")
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
                   f"Why: barrel fields differ between mice and are distorted by tissue processing, not uniformly, so "
                   f"a single scale / rotation is not enough. A smooth deformation through the matched centres moves each "
                   f"atlas barrel onto its image barrel and carries the unmatched ones along with their neighbours.\n\n"
                   f"How: thin-plate spline from the matched atlas centres to the blob centroids. Its smoothing "
                   f"({D['s_best']:g}) is chosen by leave-one-out: each barrel is predicted from all the others, so the "
                   f"warp cannot just memorise the matches. Outside the matched field it fades into a near-affine fit "
                   f"over 900 um. Landmark error {met['err_initial_um']:.0f} um before -> {met['err_tps_loo_um']:.0f} um "
                   f"leave-one-out after; min Jacobian {met['jac_min']:.2f} (> 0: no folding).\n\n"
                   f"How to read: band-passed slab with brainreg and corrected barrels; size and direction of the "
                   f"displacement; local area change (Jacobian); smoothing choice; per-barrel errors.")
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
        fv_dir = os.path.dirname(out_pdf)
        sig, cross = None, []
        try:
            sl_ = tuple(slice(int(a), int(b)) for a, b in D["sl"])
            sig3 = np.asarray(tifffile.memmap(cfg.ds_file(reg, 1), mode="r")[sl_])
            Fs = V["v_F"].astype(np.float32)
            sig = signal_flat(sig3, Fs, region & (V["v_hemi"] > 0), flat_lo, D[p + "l4"].shape)
            relv = V["v_rel"].astype(np.float32)
            for _, _, q in trk:
                Fq, rq = Fs[tuple(q.T)], relv[tuple(q.T)]
                k_ = np.isfinite(Fq).all(1) & (np.abs(rq - d_off) < 60)
                if k_.any():
                    cross.append(tuple((Fq[k_] - flat_lo).mean(0) / PIX - 0.5))
        except (IndexError, OSError) as e:
            print(f"  no signal channel overlay: {e}", flush=True)
        box, l4s, valid, v2f, bpf, ref = flat_views(D, p, cols, cols_new, fv_dir, f"{bid}_h{hv}", sig=sig,
                                                    cross=cross)
        fig = page(pdf, "Flat L4 view: brainreg vs refined registration",
                   f"Summary of the fit on the flat L4 slab ({d_off:+d} um from the L4 bottom; the slow brightness "
                   f"gradient is removed by dividing by a 300 um Gaussian). Barrel outlines are coloured by row: "
                   f"brainreg (global) dashed, refined (local) solid; thin white = the image barrels (watershed blobs) "
                   f"matched to atlas barrels, i.e. what the refinement aims at. A good fit puts each solid outline on "
                   f"one white blob.\n\n"
                   f"Top: on the slab; bottom: on the band-passed slab (artefacts masked). All panels are saved as "
                   f"square png / pdf / svg (editable text) in {fv_dir}; *_side_by_side_signal adds the signal channel "
                   f"(probe dye, red, 99th percentile through each column) and a ring where each wS1 track crosses L4.")
        fig.subplots_adjust(left=0.12, right=0.75, top=fig.top_, bottom=0.02, wspace=0.03, hspace=0.12)
        for i, (img, vv, lab, sg, t) in enumerate((
                (l4s, valid, cols, False, "brainreg (global)"),
                (l4s, valid, cols_new, False, "refined (local)"),
                (bpf, v2f, cols, True, "brainreg (global), band-passed"),
                (bpf, v2f, cols_new, True, "refined (local), band-passed"))):
            ax = fig.add_subplot(2, 2, i + 1)
            flat_panel(ax, img, vv, box, cols=lab, signed=sg, title=t, fs=7, names=True,
                       ref=ref if i < 2 else None, ls="dashed" if i % 2 == 0 else "-")
            if i == 1:
                row_legend(ax, 8, ref=True, outside=True)
        done(pdf, fig)

        # ---------------------------------------------------------------- 9 3D
        fig = page(pdf, "Step 8. Back to 3D: barrel columns and depth mapping",
                   "Why: units and probe tracks live in 3D, so each voxel of the barrel cortex needs a barrel label and a "
                   "CCF position consistent with the corrected flat map.\n\n"
                   "How: every voxel has a flat position (step 1); the warp (step 7) gives its corrected position in the "
                   "template, and the voxel takes the barrel whose convex hull contains it, so barrels become columns "
                   "from the image pia to the white matter. Depth is mapped piecewise linearly with three anchors (image "
                   "pia -> atlas pia, image L4 bottom -> atlas L4 bottom, white matter -> atlas white matter), so layers "
                   "line up even where this brain's cortex is thinner or thicker; the template's inverse lookup then "
                   "gives the CCF voxel.\n\n"
                   "How to read: coronal slices with brainreg and D2 barrel columns; depth mapping in one column; "
                   "checks of the CCF lookup.")
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
                       "Why: to assign each recording site to a barrel, the probe track (brainreg-segment points of the "
                       "dye track, in sample space) is read out in the brainreg and in the D2 barrel columns.\n\n"
                       "How to read: the track is sampled every 10 um from the shallow end. Left: coronal slice at the "
                       "track's L4 crossing, brainreg (top) and D2 (bottom) columns. Middle: the L4 crossing in the flat "
                       "view. Right: barrel and layer labels along the probe.")
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
    ap.add_argument("--hemisphere", type=int, choices=[1, 2], help="default: every hemisphere with a debug dump")
    ap.add_argument("--tracks")
    ap.add_argument("--out", help="default: <work_dir>/<brain>/report")
    a = ap.parse_args()
    cfg = Config(a.config)
    tracks = pd.read_csv(a.tracks) if a.tracks else None
    for b in a.brains:
        hvs = [a.hemisphere] if a.hemisphere else [h for h in (1, 2)
                                                   if os.path.exists(f"{cfg.brain_dir(b)}/h{h}/debug.npz")]
        rd = f"{a.out}/{b}" if a.out else cfg.report_dir(b)
        os.makedirs(rd, exist_ok=True)  # one folder per brain: report + flat-view figures
        for hv in hvs:
            try:
                report(cfg, b, hv, tracks, f"{rd}/{b}_h{hv}_D2_report.pdf")
            except Exception:
                import traceback
                traceback.print_exc()
                print(b, f"h{hv}", "FAILED", flush=True)


if __name__ == "__main__":
    main()
