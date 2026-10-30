"""Figures: flat-fit per hemisphere, coronal slices through C2, per-mouse overview, contact sheet.
Colours: outline = barrel row (A orange, B blue, C green, D yellow, E red, straddlers pink), C2 white."""
import os

import numpy as np
import tifffile
from scipy import ndimage
from skimage.segmentation import find_boundaries
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import patheffects

from .atlas import BARREL_NAMES, C2, ISO_IDS, ROW_COL, row_of

HEMI = {1: "left", 2: "right"}


def row_colour(k):
    return "w" if k == C2 else ROW_COL[row_of(BARREL_NAMES[k - 1])]


def draw(ax, cols):
    """Barrel outlines coloured by row, names at the centroids, C2 filled."""
    for k in np.unique(cols)[1:]:
        n = BARREL_NAMES[k - 1]
        m = cols == k
        col = ROW_COL[row_of(n)]
        if k == C2:
            ax.imshow(np.ma.masked_where(~m, m), cmap=matplotlib.colors.ListedColormap(["w"]), alpha=0.35,
                      interpolation="nearest")
        yy, xx = np.where(find_boundaries(m, mode="inner"))
        ax.scatter(xx, yy, s=3.2 if k == C2 else 1.2, c="w" if k == C2 else col, lw=0)
        cy, cx = np.argwhere(m).mean(0)
        ax.text(cx, cy, n, color=col, fontsize=8, ha="center", va="center", weight="bold",
                path_effects=[patheffects.withStroke(linewidth=2, foreground="black")])


def flat_fit_figure(name, od, bp, valid, art, cols_before, cols_new, blobs, Ac, Bc, ia, ib, met):
    ys, xs = np.where(ndimage.binary_dilation((cols_new > 0) | (cols_before > 0), iterations=20))
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    s = np.percentile(np.abs(bp[valid]), 98)
    fig, ax = plt.subplots(1, 3, figsize=(27, 9.5), facecolor="k")
    for a in ax:
        a.imshow(bp, cmap="gray", vmin=-s, vmax=s)
        a.contour(art, [0.5], colors="r", linewidths=0.6)
        a.set_xlim(x0, x1)
        a.set_ylim(y1, y0)
        a.axis("off")
    ax[0].set_title("template at the initial placement (registered SSp-bfd centroid)", color="w", fontsize=13)
    draw(ax[0], cols_before)
    ax[1].set_title("image barrels (grey) + matches; red = masked artefacts", color="w", fontsize=13)
    yy, xx = np.where(find_boundaries(blobs, mode="inner"))
    ax[1].scatter(xx, yy, s=0.8, c="#BBBBBB", lw=0)
    for a_, b_ in zip(ia, ib):
        ax[1].annotate("", Bc[b_, ::-1], Ac[a_, ::-1], arrowprops=dict(arrowstyle="->", color="y", lw=1.2))
    ax[2].set_title("D2: corrected barrels", color="w", fontsize=13)
    draw(ax[2], cols_new)
    ax[0].plot([x0 + 10, x0 + 60], [y1 - 10] * 2, color="w", lw=3)
    ax[0].text(x0 + 35, y1 - 16, "500 um", color="w", ha="center")
    fig.suptitle(f"{name}  D2 (own flatmap, coarse-registration start)  L4 {met['l4_depth_um']} um below image pia | matched "
                 f"{met['n_matched']}/{met['n_atlas']} | landmark err {met['err_initial_um']:.0f} -> "
                 f"{met['err_tps_loo_um']:.0f} um (LOO) | contrast {met['contrast_initial']:.2f} -> "
                 f"{met['contrast_after']:.2f} | Dice {met['dice_initial']:.2f} -> {met['dice_after']:.2f} | "
                 f"candidates agree {met['n_candidates_agree']}", color="w", fontsize=12)
    plt.tight_layout(rect=(0, 0, 1, 0.95))
    plt.savefig(f"{od}/{name}_flat_fit.png", dpi=90, facecolor="k")
    plt.close(fig)


def coronal_figure(bid, ch0, lab, path):
    """Coronal slice through C2 of each hemisphere: ch0 with the corrected barrel outlines."""
    zs = [int(np.median(np.argwhere(lab[..., h] == C2)[:, 0])) for h in (slice(None, lab.shape[2] // 2),
                                                                          slice(lab.shape[2] // 2, None))
          if (lab[..., h] == C2).any()]
    if not zs:
        return
    fig, ax = plt.subplots(1, len(zs), figsize=(10 * len(zs), 7), facecolor="k", squeeze=False)
    for a, z in zip(ax[0], zs):
        im = np.asarray(ch0[z]).astype(np.float32)
        a.imshow(im, cmap="gray", vmin=np.percentile(im, 1), vmax=np.percentile(im, 99.7))
        L = lab[z]
        for k in np.unique(L)[1:]:
            a.contour(L == k, [0.5], colors=[row_colour(k)], linewidths=1.2 if k == C2 else 0.6)
        ys, xs = np.where(L > 0)
        a.set_xlim(xs.min() - 40, xs.max() + 40)
        a.set_ylim(ys.max() + 40, ys.min() - 40)
        a.set_title(f"{bid} coronal slice {z} (through C2, white)", color="w")
        a.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=90, facecolor="k")
    plt.close(fig)


def overview_figure(cfg, bid, met):
    """Left / right flat maps (from the flat-fit figures) and coronal ch0 slices through C2, raw and with the
    corrected barrel outlines."""
    reg = cfg.reg_dir(bid)
    od = cfg.brain_dir(bid)
    lab = tifffile.imread(f"{od}/barrels_corrected_full.tif")
    hem = tifffile.memmap(f"{reg}/registered_hemispheres.tiff")
    ch0 = tifffile.memmap(cfg.ds_file(reg, 0))
    fig, ax = plt.subplots(3, 2, figsize=(16, 20), facecolor="k", gridspec_kw=dict(height_ratios=[1.15, 1, 1]))
    for a in ax.ravel():
        a.axis("off")
    for j, hv in enumerate((1, 2)):
        m = met.get(hv)
        status = "no fit" if m is None else ("FIT FAILED - do not use" if not m["fit_ok"] else
                                              "naming ambiguous - check" if m["naming_ambiguous"] else "")
        png = f"{od}/h{hv}/{bid}_h{hv}_flat_fit.png"
        if os.path.exists(png):
            im = plt.imread(png)
            ax[0, j].imshow(im[int(im.shape[0] * 0.09):, int(im.shape[1] * 0.67):])
        head = f"{HEMI[hv]} barrel field (flat L4 view)"
        if m is not None:
            head += f"\n{m['n_matched']}/33 matched, Dice {m['dice_after']:.2f}, L4 {m['l4_depth_um']} um"
        ax[0, j].set_title(head + (f"\n{status}" if status else ""), color="r" if "FAIL" in status else "w",
                           fontsize=13)
        # coronal slice through C2 of this hemisphere
        hz = np.asarray(hem[::2, ::2, ::2]) == hv
        c2 = np.argwhere((lab[::2, ::2, ::2] == C2) & hz)
        if len(c2) == 0:
            continue
        z = int(np.median(c2[:, 0])) * 2
        L = np.where(np.asarray(hem[z]) == hv, lab[z], 0)
        img = np.asarray(ch0[z]).astype(np.float32)
        ys, xs = np.where(L > 0)
        cy, cx = np.argwhere(L == C2).mean(0) if (L == C2).any() else (ys.mean(), xs.mean())
        v = cfg.voxel_um
        h = int(1100 / v)  # 1.1 mm half-window around C2
        y0, y1 = int(max(cy - h, 0)), int(min(cy + h, img.shape[0]))
        x0, x1 = int(max(cx - h, 0)), int(min(cx + h, img.shape[1]))
        ctx = ndimage.binary_dilation(L > 0, iterations=int(200 / v))[y0:y1, x0:x1]  # window on the cortex, not background
        vmin, vmax = np.percentile(img[y0:y1, x0:x1][ctx], [1, 99.5])
        for i in (1, 2):
            ax[i, j].imshow(img, cmap="gray", vmin=vmin, vmax=vmax, interpolation="nearest")
            ax[i, j].set_xlim(x0, x1)
            ax[i, j].set_ylim(y1, y0)
        iso = np.isin(np.asarray(tifffile.memmap(f"{reg}/registered_atlas.tiff")[z]), ISO_IDS)
        ax[2, j].contour(ndimage.binary_opening(iso, iterations=1), [0.5], colors=["#00E5FF"], linewidths=1.3,
                         linestyles="--")  # registered isocortex outline (pia / white matter)
        for k in np.unique(L[y0:y1, x0:x1])[1:]:
            mk = L == k
            ax[2, j].contour(mk, [0.5], colors=[row_colour(k)], linewidths=2.0 if k == C2 else 1.0)
            yy, xx = np.argwhere(mk).mean(0)
            if mk.sum() >= 6000 / v ** 2 and y0 < yy < y1 and x0 < xx < x1:
                ax[2, j].text(xx, yy, BARREL_NAMES[k - 1], color=row_colour(k), fontsize=10, ha="center",
                              va="center", weight="bold",
                              path_effects=[patheffects.withStroke(linewidth=2, foreground="black")])
        ax[1, j].set_title(f"{HEMI[hv]}: coronal slice {z} ({v:.0f} um ch0) through C2", color="w", fontsize=13)
        ax[2, j].set_title(f"{HEMI[hv]}: same slice + D2 barrel columns (C2 white),\n"
                           f"cyan dashed = registered isocortex (pia / white matter)", color="w", fontsize=13)
        sb = 500 / v
        ax[1, j].plot([x0 + 80 / v, x0 + 80 / v + sb], [y1 - 80 / v] * 2, color="w", lw=3)
        ax[1, j].text(x0 + 80 / v + sb / 2, y1 - 140 / v, "500 um", color="w", ha="center")
    p = cfg.params.loc[bid]
    title = f"{bid}  D2 barrel registration"
    if {"bending_energy_weight", "grid_spacing"} <= set(p.index):
        title += f"  (brainreg be{p.bending_energy_weight} / gs{p.grid_spacing})"
    fig.suptitle(title, color="w", fontsize=16)
    plt.tight_layout(rect=(0, 0, 1, 0.97))
    out = f"{od}/{bid}_D2_overview.png"
    plt.savefig(out, dpi=80, facecolor="k")
    plt.close(fig)
    return out


def contact_sheet(pngs, labels, path, ncol=6, width=2400):
    """Grid of the 'corrected barrels' panel of each flat-fit figure."""
    from PIL import Image, ImageDraw
    tiles = []
    for p in pngs:
        im = Image.open(p).convert("RGB")
        w, h = im.size
        tiles.append(im.crop((int(w * 0.67), int(h * 0.09), w, h)))
    tw = width // ncol
    tiles = [t.resize((tw, round(t.height * tw / t.width))) for t in tiles]
    th = max(t.height for t in tiles) + 24
    nrow = int(np.ceil(len(tiles) / ncol))
    sheet = Image.new("RGB", (width, th * nrow), "black")
    d = ImageDraw.Draw(sheet)
    for i, (t, lab) in enumerate(zip(tiles, labels)):
        x, y = (i % ncol) * tw, (i // ncol) * th
        sheet.paste(t, (x, y + 24))
        d.text((x + 6, y + 4), lab, fill="white")
    sheet.save(path)
