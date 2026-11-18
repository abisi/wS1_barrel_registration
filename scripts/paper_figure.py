"""Paper figure for one brain and one Neuropixels track (2 x 3 grid, panels a-f):
  a  flat L4 view, brainreg (global) barrels, signal channel (probe dye) and the track's L4 crossing
  b  empty (placeholder)
  c  flat L4 view, refined (D2) barrels, same crop
  d  coronal slice through the track's L4 crossing, centred on the track: ch0 + probe dye, D2 barrel columns (solid)
     and brainreg columns (dashed)
  e  barrel along the probe vs depth below the pia, brainreg and D2, with the cortical layers
  f  empty (placeholder)
Needs a run made with BARRELREG_DEBUG=1 (results/h<hv>/debug.npz + debug3d.npz).

usage: python scripts/paper_figure.py -c RUN/config.toml AB164 --tracks tracks_summary.csv [--track imec2] [--out DIR]
Writes <out>/<ID>_h<hv>_<track>_paper_figure.{png,pdf,svg} (default out: the brain's report folder).
"""
import argparse
import json
import os
import re
import sys

import numpy as np
import pandas as pd
import tifffile
from scipy import ndimage
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from barrelreg.atlas import SEPTA, collapse
from barrelreg.config import Config
from barrelreg.register import PIX
from brain_report import (ACR, PAL, ROW_LABEL, SIG_COL, flat_panel, flatten, name, read_points, rowcol, signal_flat,
                          square_box, track_samples)

VOX = 10.0  # um, downsampled sample space
LAYERS = ("1", "2/3", "4", "5", "6a", "6b")


def load_tracks(cfg, bid, hv, tracks, V, depth, lo):
    """wS1 tracks of this hemisphere (same selection as brain_report.py)."""
    t = tracks[tracks.brain == bid]
    no_tab = t.table_target_area.isna() | (t.table_target_area.astype(str).str.strip() == "")
    in_bfd = (t.um_in_d2_barrels.fillna(0) >= 100) | (t.um_in_brainreg_ssp_bfd.fillna(0) >= 100)
    out = []
    for r in t[(t.table_target_area == "wS1") | (no_tab & in_bfd)].drop_duplicates("track").itertuples():
        f = f"{cfg.reg_dir(bid)}/segmentation/sample_space/tracks/{r.track}"
        if os.path.exists(f):
            X, q = track_samples(read_points(f), depth, lo)
            if (V["v_hemi"][tuple(q.T)] > 0).mean() > 0.5:
                out.append((os.path.basename(r.track).replace(".points", ""), X, q))
    return out


def outlines(ax, lab, ls="-", lw=1.0, names=True, fs=6, alpha=1.0):
    """Smoothed outline of each barrel label in a 2D slice, row colours, names without edge."""
    for k in np.unique(lab):
        if k <= 0 or k >= SEPTA:
            continue
        m = ndimage.gaussian_filter((lab == k).astype(np.float32), 0.8)
        ax.contour(m, [0.5], colors=[rowcol(k)], linewidths=lw, linestyles=ls, alpha=alpha)
        if names and (lab == k).sum() >= 300:  # no names for barrels barely inside the crop
            cy, cx = np.argwhere(lab == k).mean(0)
            ax.text(cx, cy, name(k), color=rowcol(k), fontsize=fs, ha="center", va="center", weight="bold")


def runs(v):
    """(start, stop, value) of the constant runs of a 1D array."""
    v = np.asarray(v)
    cut = np.flatnonzero(v[1:] != v[:-1]) + 1
    st, en = np.r_[0, cut], np.r_[cut, len(v)]
    return [(s, e, v[s]) for s, e in zip(st, en)]


def letter(fig, ax, s):
    bb = ax.get_position()
    fig.text(bb.x0 - 0.01, bb.y1 + 0.012, s, fontsize=11, weight="bold", ha="right", va="bottom")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", required=True)
    ap.add_argument("brain")
    ap.add_argument("--hemisphere", type=int, default=1)
    ap.add_argument("--tracks", required=True)
    ap.add_argument("--track", help="track name (default: the first wS1 track)")
    ap.add_argument("--out")
    ap.add_argument("--half-width-um", type=float, default=900, help="coronal crop half width (d)")
    a = ap.parse_args()
    cfg, bid, hv = Config(a.config), a.brain, a.hemisphere
    od = f"{cfg.brain_dir(bid)}/h{hv}"
    D = dict(np.load(f"{od}/debug.npz"))
    V = np.load(f"{od}/debug3d.npz")
    met = json.load(open(f"{od}/metrics.json"))
    reg = cfg.reg_dir(bid)
    lo = np.array([s[0] for s in D["sl"]])
    d_off = int(D["chosen_offset"])
    p = f"o{d_off}_"
    cols, cols_new, flat_lo = D["cols"], D["cols_new"], D["flat_lo"]
    atl, out, region = V["v_atl"], V["v_out"], V["v_region"]
    depth = V["v_depth"].astype(np.float32)
    ch0 = V["v_ch0"].astype(np.float32)
    lab_reg = collapse(atl) * (V["v_hemi"] > 0)
    F, rel = V["v_F"].astype(np.float32), V["v_rel"].astype(np.float32)
    sl_ = tuple(slice(int(s0), int(s1)) for s0, s1 in D["sl"])
    sig3 = np.asarray(tifffile.memmap(cfg.ds_file(reg, 1), mode="r")[sl_]).astype(np.float32)

    trk = load_tracks(cfg, bid, hv, pd.read_csv(a.tracks), V, depth, lo)
    if a.track:
        trk = [t for t in trk if t[0] == a.track]
    if not trk:
        sys.exit(f"{bid} h{hv}: no wS1 track")
    tname, X, q = trk[0]

    # L4 crossing of the track: flat position (a, c) and voxel (d)
    Fq, rq = F[tuple(q.T)], rel[tuple(q.T)]
    k4 = np.isfinite(Fq).all(1) & (np.abs(rq - d_off) < 60)
    cross = [tuple((Fq[k4] - flat_lo).mean(0) / PIX - 0.5)] if k4.any() else []
    i4 = int(np.flatnonzero(k4)[len(np.flatnonzero(k4)) // 2]) if k4.any() else len(q) // 2
    z4, y4, x4 = q[i4]

    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"], "font.size": 7,
                         "svg.fonttype": "none", "pdf.fonttype": 42, "axes.linewidth": 0.6})
    W, H = 7.2, 5.3  # in, 2 x 3 square panels + legend strip at the bottom
    fig = plt.figure(figsize=(W, H))
    side, gap, left = 2.0, 0.42, 0.33  # in
    pw, ph = side / W, side / H
    xs = [(left + i * (side + gap)) / W for i in range(3)]
    ys = [(H - 0.25 - side) / H, (H - 0.25 - 2 * side - 0.38) / H]
    ax = {k: fig.add_axes([xs[i % 3], ys[i // 3], pw, ph]) for i, k in enumerate("abcdef")}

    # ---------------------------------------------------------------- a, c: flat L4 views
    sig = signal_flat(sig3, F, region & (V["v_hemi"] > 0), flat_lo, D[p + "l4"].shape)
    valid = D[p + "valid"]
    l4 = flatten(D[p + "l4"], valid)
    box = square_box((cols > 0) | (cols_new > 0))
    blobs = D[p + "blobs"]
    ref = np.where(np.isin(blobs, np.asarray(D["ib"]) + 1), blobs, 0)
    for k, lab, t, ls in (("a", cols, "brainreg (global)", "dashed"), ("c", cols_new, "refined (local)", "-")):
        flat_panel(ax[k], l4, valid, box, cols=lab, names=True, title=t, fs=5.5, bar=k == "a", ref=ref, ls=ls,
                   sig=sig, cross=cross)
        ax[k].title.set_fontsize(8)
    for k in "bf":
        ax[k].axis("off")

    # ---------------------------------------------------------------- d: coronal slice centred on the track
    # centre: probe-dye centroid (ch1, top 0.5 %) within 400 um of the track's L4 crossing, else the crossing
    hw = int(a.half_width_um / VOX)
    r_ = int(400 / VOX)
    ya, xa = max(y4 - r_, 0), max(x4 - r_, 0)
    win = sig3[z4, ya:y4 + r_, xa:x4 + r_]
    yc, xc = y4, x4
    if win.size:
        cy_, cx_ = np.argwhere(win >= np.percentile(win, 99.5)).mean(0)
        yc, xc = int(round(ya + cy_)), int(round(xa + cx_))
    y0, y1 = max(yc - hw, 0), min(yc + hw, ch0.shape[1])
    x0, x1 = max(xc - hw, 0), min(xc + hw, ch0.shape[2])
    img = ch0[z4, y0:y1, x0:x1]
    s1 = sig3[z4, y0:y1, x0:x1]
    vlo, vhi = np.nanpercentile(img, (1, 99.5))
    axd = ax["d"]
    axd.imshow(img, cmap="gray", vmin=vlo, vmax=vhi, interpolation="bilinear")
    slo, shi = np.nanpercentile(s1, (50, 99.9))
    s = np.clip((s1 - slo) / max(shi - slo, 1e-6), 0, 1)
    rgba = np.zeros(s.shape + (4,), np.float32)
    rgba[..., :3] = matplotlib.colors.to_rgb(SIG_COL)
    rgba[..., 3] = 0.85 * s ** 1.5
    axd.imshow(rgba, interpolation="bilinear")
    outlines(axd, lab_reg[z4, y0:y1, x0:x1], ls="dashed", lw=0.6, names=False, alpha=0.7)
    outlines(axd, out[z4, y0:y1, x0:x1], ls="-", lw=1.1, names=True, fs=6)
    axd.scatter([x4 - x0], [y4 - y0], s=40, facecolors="none", edgecolors="w", linewidths=1.4, zorder=5)
    axd.scatter([x4 - x0], [y4 - y0], s=40, facecolors="none", edgecolors=SIG_COL, linewidths=0.7, zorder=6)
    L = 500 / VOX
    axd.plot([0.05 * (x1 - x0), 0.05 * (x1 - x0) + L], [0.95 * (y1 - y0)] * 2, color="w", lw=2, solid_capstyle="butt")
    axd.text(0.05 * (x1 - x0) + L / 2, 0.93 * (y1 - y0), "500 um", color="w", ha="center", va="bottom", fontsize=6)
    axd.set_xticks([])
    axd.set_yticks([])
    for sp_ in axd.spines.values():
        sp_.set_visible(False)
    axd.set_title(f"coronal, track {tname} at L4", fontsize=8)

    # ---------------------------------------------------------------- e: barrel along the probe
    axe = ax["e"]
    dq = depth[tuple(q.T)].astype(float)
    acr = [ACR.get(int(v), "") for v in atl[tuple(q.T)]]
    lay = np.array([m.group(1) if (m := re.search(r"(1|2/3|4|5|6a|6b)$", s_)) and s_.startswith("SSp") else ""
                    for s_ in acr])
    inc = np.flatnonzero(lay != "")
    if len(inc) == 0:
        inc = np.arange(len(q))
    i0, i1 = inc[0], inc[-1] + 1
    dmax = np.nanmax(dq[i0:i1]) + 20
    # layer bands (atlas layers along the probe, brainreg)
    for s_, e_, v in runs(lay[i0:i1]):
        if not v:
            continue
        a_, b_ = dq[i0 + s_], dq[i0 + e_ - 1] + VOX
        shade = "0.93" if LAYERS.index(v) % 2 == 0 else "0.86"
        axe.add_patch(Rectangle((-0.5, a_), 3.0, b_ - a_, color=shade, lw=0, zorder=0))
        axe.text(-0.45, (a_ + b_) / 2, f"L{v}", fontsize=6.5, va="center", ha="left", color="0.3")
    # barrels along the probe
    for i, (lab, ttl) in enumerate(((lab_reg, "brainreg"), (out, "refined"))):
        b = lab[tuple(q.T)][i0:i1].astype(int)
        b = np.where(b >= SEPTA, 0, b)
        # fill gaps of <= 3 samples (30 um, septum / boundary voxels) between runs of the same barrel
        rr = runs(b)
        for j in range(1, len(rr) - 1):
            s0, e0, v0 = rr[j]
            if v0 == 0 and e0 - s0 <= 3 and rr[j - 1][2] == rr[j + 1][2] > 0:
                b[s0:e0] = rr[j - 1][2]
        xc = 0.75 + i * 1.0
        for s_, e_, v in runs(b):
            if v <= 0:
                continue
            a_, b_ = dq[i0 + s_], dq[i0 + e_ - 1] + VOX
            axe.add_patch(Rectangle((xc - 0.3, a_), 0.6, b_ - a_, color=rowcol(v), lw=0, zorder=2,
                                    ls="--" if i == 0 else "-"))
            if b_ - a_ >= 40:
                axe.text(xc, (a_ + b_) / 2, name(v), fontsize=7, ha="center", va="center", weight="bold",
                         color="0.1", zorder=3)
    bimg = np.nanmedian(V["v_bimg"][tuple(q.T)].astype(float))
    axe.axhline(bimg, color="0.2", lw=0.7, ls=":", zorder=4)
    axe.text(1.25, bimg - 6, "image L4 bottom", fontsize=5, va="bottom", ha="center", color="0.2", zorder=4)
    axe.set_xlim(-0.5, 2.5)
    axe.set_ylim(dmax, max(np.nanmin(dq[i0:i1]) - 20, 0))
    axe.set_xticks([0.75, 1.75], ["brainreg", "refined"])
    axe.tick_params(axis="x", length=0)
    axe.set_ylabel("depth below pia (um)")
    for sp_ in ("top", "right"):
        axe.spines[sp_].set_visible(False)
    axe.set_title("barrel along the probe", fontsize=8)

    # legend: one strip under the grid (b and f stay empty)
    h = [Line2D([], [], color=PAL[r], lw=1.5, label=ROW_LABEL[r]) for r in "ABCDES"]
    h += [Line2D([], [], color="0.3", lw=1.0, ls="--", label="brainreg"),
          Line2D([], [], color="0.3", lw=1.0, label="refined"),
          Line2D([], [], color="0.7", lw=0.8, label="image barrels"),
          Patch(color=SIG_COL, alpha=0.7, label="probe dye (ch1)"),
          Line2D([], [], ls="", marker="o", mfc="none", mec=SIG_COL, mew=1.0, ms=5, label="track L4 crossing")]
    fig.legend(handles=h, loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=6, fontsize=6, frameon=False,
               handlelength=1.6, columnspacing=1.2)
    for k in "abcdef":
        letter(fig, ax[k], k)
    rd = a.out or cfg.report_dir(bid)
    os.makedirs(rd, exist_ok=True)
    stem = f"{rd}/{bid}_h{hv}_{tname}_paper_figure"
    for ext in ("png", "pdf", "svg"):
        fig.savefig(f"{stem}.{ext}", dpi=300, facecolor="white")
    print("wrote", stem, f"(contrast {met['contrast_after']:.2f}, {met['n_matched']} matched)")


if __name__ == "__main__":
    main()
