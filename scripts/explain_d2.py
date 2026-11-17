"""Explanatory figures for D2, drawn from real outputs of one brain (default MS261, left hemisphere).

  D2_0 pipeline                 D2_2 fixed template + inverse lookup
  D2_1 the brain's own flatmap  D2_3 three-anchor depth mapping
The matching steps (L4 slab, image barrels, search, candidate choice, warp) are the same as in method D; their
figures are D3-D7 in D_explained/.

usage: python scripts/explain_d2.py -c config.toml [--brain MS261] [--hemisphere 1] [--out DIR]
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root, if not pip-installed

import numpy as np
import tifffile
from scipy import ndimage
from skimage.segmentation import find_boundaries
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Patch

from barrelreg.atlas import BFD_IDS, C2, ISO_IDS, ROW_COL, SEPTA
from barrelreg.config import Config
from barrelreg.plotting import draw

BG = "black"
plt.rcParams.update({"figure.facecolor": BG, "axes.facecolor": BG, "text.color": "w", "axes.labelcolor": "w",
                     "xtick.color": "w", "ytick.color": "w", "axes.edgecolor": "#888888", "font.size": 12})


def save(fig, out, name, title, subtitle):
    fig.suptitle(title, fontsize=17, weight="bold", y=1.03)
    fig.text(0.5, 0.985, subtitle, ha="center", va="top", fontsize=12, color="#CCCCCC")
    fig.savefig(f"{out}/{name}.png", dpi=90, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print("wrote", name, flush=True)


def legend_rows(fig, extra=()):
    h = [Line2D([], [], color=c, lw=3, label=f"row {r}" if r != "S" else "straddlers") for r, c in ROW_COL.items()]
    h += [Patch(facecolor="w", alpha=0.5, label="C2")] + list(extra)
    fig.legend(handles=h, loc="lower center", ncol=len(h), frameon=False, fontsize=11, bbox_to_anchor=(0.5, -0.04))


# ------------------------------------------------------------------ 0 pipeline
def fig0(out):
    steps = [
        ("1  Coarse registration", "brainreg to the Perens LSFM atlas\nonly: hemisphere, rough SSp-bfd,\npia + white-matter surfaces", "perens"),
        ("2  Own flatmap", "Laplace depth between registered\npia and WM -> streamlines ->\nflat L4 plane of THIS brain", "C"),
        ("3  Fixed template", "Blue Brain flatmap of the CCF\n-> barrel map in the same plane,\nbuilt once (+ inverse lookup)", "new"),
        ("4  L4 slab", "L4 bottom found in ch0 per column;\nslab at 2 candidate offsets above it\nfit both, keep more matches", "new"),
        ("5  Image barrels", "artefact mask, band-pass,\nlocal maxima + watershed\n-> barrel centroids", "D"),
        ("6  Search + match", "scale 0.7-1.4, rot +-20 deg,\nshift +-600 um -> top 5 ->\nCPD matching, affine tie-break", "D"),
        ("7  Warp", "thin-plate spline through the\nmatches; near-affine outside\nthe field", "D"),
        ("8  Back to 3D", "label = barrel whose convex hull\nholds the corrected flat coordinate;\ndepth: 3 anchors per column", "new"),
        ("9  Into CCF", "(flat position, atlas depth)\n-> inverse flatmap -> CCF voxel;\nright hemisphere mirrored", "new"),
    ]
    col = {"perens": "#BBBBBB", "C": "#009E73", "D": "#E69F00", "new": "#CC79A7"}
    src = {"perens": "coarse step", "C": "from C", "D": "as in D", "new": "new in D2"}
    fig, ax = plt.subplots(figsize=(24, 10))
    ax.set_xlim(0, 3)
    ax.set_ylim(0, 3.25)
    ax.axis("off")
    for i, (t, d, s) in enumerate(steps):
        r, c = divmod(i, 3)
        x, y = c + 0.05, 2.25 - r * 1.1
        ax.add_patch(FancyBboxPatch((x, y), 0.85, 0.85, boxstyle="round,pad=0.02", fc="#1b1b1b", ec=col[s], lw=2.5))
        ax.text(x + 0.425, y + 0.72, t, ha="center", fontsize=15, weight="bold")
        ax.text(x + 0.425, y + 0.38, d, ha="center", va="center", fontsize=12, color="#DDDDDD")
        ax.text(x + 0.425, y + 0.07, src[s], ha="center", fontsize=11, color=col[s])
        if c < 2:
            ax.annotate("", (x + 0.95, y + 0.425), (x + 0.87, y + 0.425), arrowprops=dict(arrowstyle="->", color="w", lw=2))
        elif r < 2:
            ax.plot([2.475, 2.475, 0.475], [y - 0.02, y - 0.12, y - 0.12], color="w", lw=2)
            ax.annotate("", (0.475, y - 0.23), (0.475, y - 0.12), arrowprops=dict(arrowstyle="->", color="w", lw=2))
    save(fig, out, "D2_0_pipeline", "Method D2: barrel matching on the brain's own flatmap, ending in Allen CCF",
         "box colour = where the step comes from; no Allen registration of the brain is needed")


# ------------------------------------------------------------------ 1 own flatmap
def fig1(cfg, bid, hv, out):
    reg = cfg.reg_dir(bid)
    fz = np.load(f"{cfg.brain_dir(bid)}/h{hv}/flatcoords.npz")
    lo, UP = fz["crop20_lo"], int(fz["up"])
    G = fz["G"]
    lab = tifffile.imread(f"{cfg.brain_dir(bid)}/barrels_corrected_full.tif")
    hem = tifffile.memmap(f"{reg}/registered_hemispheres.tiff", mode="r")
    c2 = np.argwhere((lab == C2) & (np.asarray(hem) == hv))
    z20 = int(np.median(c2[:, 0]))
    zw = (z20 - lo[0]) * UP + UP // 2  # working-grid (10 um) plane inside the crop
    n1, n2 = G.shape[1] // UP, G.shape[2] // UP
    sl = (z20, slice(lo[1], lo[1] + n1), slice(lo[2], lo[2] + n2))
    img = ndimage.zoom(np.asarray(tifffile.memmap(cfg.ds_file(reg, 0), mode="r")[sl]).astype(np.float32), UP, order=1)
    atl = ndimage.zoom(np.asarray(tifffile.memmap(f"{reg}/registered_atlas.tiff", mode="r")[sl]), UP, order=0)
    L = ndimage.zoom(np.where(np.asarray(hem[sl]) == hv, lab[sl], 0), UP, order=0)
    g = G[zw]
    m = (L > 0) | np.isfinite(g[..., 0])
    ys, xs = np.where(ndimage.binary_dilation(L > 0, iterations=40))
    fig, axs = plt.subplots(1, 2, figsize=(20, 9))
    for ax in axs:
        ax.imshow(img, cmap="gray", vmin=np.percentile(img[m], 1), vmax=np.percentile(img[m], 99.5), interpolation="lanczos")
        ax.contour(np.isin(atl, ISO_IDS), [0.5], colors=["#00E5FF"], linewidths=1.2, linestyles="--")
        ax.set_xlim(xs.min(), xs.max())
        ax.set_ylim(ys.max(), ys.min())
        ax.axis("off")
    ok = np.isfinite(g[..., 0])
    for k, cc in enumerate(["#56B4E9", "#E69F00"]):
        gk = np.where(ok, g[..., k], np.nan)
        axs[0].contour(gk, np.arange(np.nanmin(gk) // 100 * 100, np.nanmax(gk), 100), colors=cc, linewidths=0.8,
                       negative_linestyles="solid")
    axs[0].contour(np.isin(atl, BFD_IDS), [0.5], colors=["#CC79A7"], linewidths=1.5)
    axs[0].set_title("the brain's own flat coordinates (lines every 100 um) follow the columns\n"
                     "cyan dashed = registered isocortex (pia / white matter), pink = registered SSp-bfd", fontsize=13)
    draw(axs[1], L.astype(int))
    axs[1].set_title("D2 barrel columns: from the image pia down to the brain white matter", fontsize=13)
    x0, y1 = xs.min() + 10, ys.max() - 10
    axs[0].plot([x0, x0 + 50], [y1, y1], color="w", lw=3)
    axs[0].text(x0 + 25, y1 - 6, "500 um", color="w", ha="center")
    legend_rows(fig)
    save(fig, out, "D2_1_own_flatmap", f"Step 2 + 8: the brain's own flatmap ({bid}, {'left' if hv == 1 else 'right'}, coronal slice {z20})",
         "built from the registered pia and white matter of this brain (coarse registration)")


# ------------------------------------------------------------------ 2 template
def fig2(cfg, out):
    T = cfg.template()
    alab, thick = T["alab"], T["thick"]
    cols = np.where(alab < SEPTA, alab, 0)
    foot = alab > 0
    ys, xs = np.where(ndimage.binary_dilation(foot, iterations=10))
    fig, axs = plt.subplots(1, 2, figsize=(20, 9))
    axs[0].imshow(np.zeros_like(cols), cmap="gray", vmin=0, vmax=1)
    draw(axs[0], cols)
    axs[0].set_title("template: atlas L4 barrels in the Blue Brain flatmap, mapped to the L4 plane\n"
                     "(same plane convention and origin as each brain's own flatmap)", fontsize=13)
    im = axs[1].imshow(np.where(ndimage.binary_dilation(foot, iterations=10), thick, np.nan), cmap="viridis")
    yy, xx = np.where(find_boundaries(cols, mode="inner") & (cols > 0))
    axs[1].scatter(xx, yy, s=0.6, c="w", lw=0)
    plt.colorbar(im, ax=axs[1], fraction=0.046, label="atlas cortical thickness (um)")
    for ax in axs:
        ax.set_xlim(xs.min(), xs.max())
        ax.set_ylim(ys.max(), ys.min())
        ax.axis("off")
    axs[0].plot([xs.min() + 10, xs.min() + 60], [ys.max() - 10] * 2, color="w", lw=3)
    axs[0].text(xs.min() + 35, ys.max() - 16, "500 um", color="w", ha="center")
    axs[1].set_title(f"atlas thickness per flat pixel (white-matter anchor); atlas L4 bottom {float(np.median(T['l4_bottom'][cols > 0])):.0f} um\n"
                     "inverse lookup: every (flat pixel, atlas depth) -> mean CCF voxel", fontsize=13)
    legend_rows(fig)
    save(fig, out, "D2_2_template", "Step 3: the fixed template (built once from the CCF)",
         f"the flatmap -> plane map is linear (median residual {float(T['plane_res_um']):.0f} um), so the template keeps "
         "the flatmap's curved columns; one template serves both hemispheres (right mirrored)")


# ------------------------------------------------------------------ 3 depth mapping
def fig3(cfg, bid, hv, out):
    m = json.load(open(f"{cfg.brain_dir(bid)}/h{hv}/metrics.json"))
    fz = np.load(f"{cfg.brain_dir(bid)}/h{hv}/flatcoords.npz")
    T = cfg.template()
    d4, a4 = m["l4_bottom_um"], m["atlas_l4_bottom_um"]
    bw, aw = m["brain_wm_depth_um"], m["atlas_wm_depth_um"]
    fig, axs = plt.subplots(1, 2, figsize=(22, 9))
    ax = axs[0]
    x = np.linspace(0, bw * 1.05, 200)
    ax.plot(x, x * a4 / d4, color="#888888", ls="--", lw=2, label=f"single factor (method D): x {a4 / d4:.2f}")
    y = np.where(x <= d4, x * a4 / d4, a4 + (x - d4) * (aw - a4) / (bw - d4))
    ax.plot(x, y, color="#F0E442", lw=3, label="three anchors (D2)")
    for (px, py, t, dx, dy, ha) in [(0, 0, "pia", 40, 50, "left"), (d4, a4, "L4 bottom", 60, -40, "left"),
                                    (bw, aw, "white matter", -40, 90, "right")]:
        ax.plot(px, py, "o", color="w", ms=9)
        ax.annotate(f"{t}: image {px:.0f} um -> atlas {py:.0f} um", (px, py), (px + dx, py + dy), color="w",
                    fontsize=11, va="center", ha=ha)
    ax.axvline(bw, color="#00E5FF", ls=":", lw=1.5)
    ax.text(bw - 15, aw * 0.15, "column ends here\n(brain WM)", color="#00E5FF", fontsize=11, ha="right")
    ax.set_xlabel("depth below the image pia (um)")
    ax.set_ylabel("atlas depth (um) used for the CCF lookup")
    ax.invert_yaxis()
    ax.legend(frameon=False, loc="lower left")
    ax.set_title(f"{bid} {'left' if hv == 1 else 'right'}, median over the barrel field", fontsize=13)
    diff = fz["brain_wm_depth_um"] - T["thick"]
    foot = T["alab"] > 0
    ys, xs = np.where(ndimage.binary_dilation(foot, iterations=10))
    s = np.nanpercentile(np.abs(diff[foot]), 98)
    im = axs[1].imshow(np.where(ndimage.binary_dilation(foot, iterations=10), diff, np.nan), cmap="coolwarm", vmin=-s, vmax=s)
    plt.colorbar(im, ax=axs[1], fraction=0.046, label="brain WM depth - atlas WM depth (um)")
    axs[1].contour(foot, [0.5], colors="w", linewidths=1)
    axs[1].set_xlim(xs.min(), xs.max())
    axs[1].set_ylim(ys.max(), ys.min())
    axs[1].axis("off")
    axs[1].set_title("where this brain's cortex is thicker (red) or thinner (blue) than the atlas\n"
                     "(own flatmap, uncorrected flat position)", fontsize=13)
    save(fig, out, "D2_3_depth_mapping", "Step 8: mapping depth with three anchors per column",
         "image pia -> atlas pia, image L4 bottom (ch0) -> atlas L4 bottom, brain white matter -> atlas white matter; "
         "D's single factor stopped columns short where the cortex below L4 is thicker than the atlas")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", default="config.toml")
    ap.add_argument("--brain", default="MS261")
    ap.add_argument("--hemisphere", type=int, default=1)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    cfg = Config(a.config)
    out = a.out or f"{cfg.work_dir}/D2_explained"
    os.makedirs(out, exist_ok=True)
    fig0(out)
    fig1(cfg, a.brain, a.hemisphere, out)
    fig2(cfg, out)
    fig3(cfg, a.brain, a.hemisphere, out)
