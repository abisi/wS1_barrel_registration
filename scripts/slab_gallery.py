"""Every L4 slab depth of a hemisphere on one page, to see at which depth (and where in the field) the barrels show.

Per brain: the flat image at each slab offset from the image L4 bottom (-300..+50 um, 25 um steps, 60 um thick),
background-flattened, with brainreg's barrels (dashed); then a per-pixel best-slab composite (band-passed slabs
weighted by their local barrel-scale energy, so a region where the barrels sit deeper or shallower than the L4 bottom
estimate contributes its own best depth), the depth of that best slab per pixel, and the final fit on the composite.
Needs a run with BARRELREG_DEBUG=1.

usage: python scripts/slab_gallery.py -c config.toml BRAIN [BRAIN ...] [--hemisphere 1] [--out PDF]
"""
import argparse
import os
import sys

import numpy as np
from scipy import ndimage
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from barrelreg.config import Config
from barrelreg.imageproc import bandpass
from brain_report import flat_panel, flatten, square_box

COMP_RANGE = (-250, 50)  # slabs entering the composite
ENERGY_SIGMA = 20  # px (200 um): neighbourhood over which a slab's barrel-scale energy is compared
SHARP = 4  # weight = energy ** SHARP (close to a per-pixel argmax, without seams)


def composite(stack, valid, offsets, field):
    """Per-pixel best-slab composite of band-passed slabs; returns the image, the joint valid mask and the best depth."""
    keep = [i for i, d in enumerate(offsets) if COMP_RANGE[0] <= d <= COMP_RANGE[1]]
    vj = np.all(valid[keep], 0)
    w0 = ndimage.gaussian_filter(vj.astype(np.float32), ENERGY_SIGMA)
    B, E = [], []
    for i in keep:
        b = bandpass(stack[i], valid[i])
        m = vj & field
        b = (b - np.median(b[m])) / (1.4826 * np.median(np.abs(b[m] - np.median(b[m]))) + 1e-9)
        b = np.clip(b, -4, 4)
        B.append(b)
        E.append(ndimage.gaussian_filter(np.where(vj, b ** 2, 0), ENERGY_SIGMA) / np.maximum(w0, 1e-3))
    B, E = np.array(B), np.array(E)
    W = E ** SHARP
    W /= W.sum(0, keepdims=True) + 1e-12
    best = np.asarray(offsets)[keep][np.argmax(E, 0)].astype(float)
    return (W * B).sum(0), vj, np.where(vj, best, np.nan)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", required=True)
    ap.add_argument("brains", nargs="+")
    ap.add_argument("--hemisphere", type=int, default=1)
    ap.add_argument("--out")
    a = ap.parse_args()
    cfg = Config(a.config)
    hv = a.hemisphere
    out = a.out or f"{cfg.work_dir}/slab_gallery.pdf"
    with PdfPages(out) as pdf:
        for b in a.brains:
            D = np.load(f"{cfg.brain_dir(b)}/h{hv}/debug.npz")
            offs, S, V = list(D["offsets"]), D["stack_img"], D["stack_valid"]
            cols, cols_new, foot = D["cols"], D["cols_new"], D["foot"]
            field = ndimage.binary_dilation(foot, iterations=10)
            box = square_box((cols > 0) | (cols_new > 0))
            comp, vj, best = composite(S, V, offs, field)
            ch = int(D["img_ch"]) if "img_ch" in D.files else 0
            n = len(offs) + 3
            nc = 6
            nr = int(np.ceil(n / nc))
            fig, axs = plt.subplots(nr, nc, figsize=(16.5, 2.85 * nr + 0.6))
            fig.suptitle(f"{b} left hemisphere, ch{ch}: L4 slabs by depth from the image L4 bottom (dashed: brainreg "
                         f"barrels); chosen slab {int(D['chosen_offset']):+d} um", fontsize=12, x=0.01, ha="left")
            axs = axs.ravel()
            for i, d in enumerate(offs):
                flat_panel(axs[i], flatten(S[i], V[i]), V[i], box, cols=cols, ls="dashed", names=False, fs=6,
                           bar=i == 0, title=f"{d:+d} um" + ("  (chosen)" if d == int(D["chosen_offset"]) else ""))
            k = len(offs)
            flat_panel(axs[k], comp, vj, box, cols=cols, ls="dashed", names=False, fs=6, bar=False, signed=True,
                       title="best-slab composite")
            axs[k + 1].set_facecolor("black")
            im = axs[k + 1].imshow(best, cmap="viridis", vmin=COMP_RANGE[0], vmax=COMP_RANGE[1])
            axs[k + 1].contour(foot.astype(float), [0.5], colors="w", linewidths=0.6)
            axs[k + 1].set_xlim(box[2], box[3])
            axs[k + 1].set_ylim(box[1], box[0])
            axs[k + 1].set_xticks([])
            axs[k + 1].set_yticks([])
            axs[k + 1].set_title("best slab per pixel (um)", fontsize=8)
            plt.colorbar(im, ax=axs[k + 1], fraction=0.046, pad=0.02).ax.tick_params(labelsize=6)
            flat_panel(axs[k + 2], comp, vj, box, cols=cols_new, names=True, fs=5, bar=False, signed=True,
                       title="composite + final fit (failed)")
            for ax in axs[k + 3:]:
                ax.axis("off")
            fig.tight_layout(rect=(0, 0, 1, 0.97))
            pdf.savefig(fig, dpi=110)
            plt.close(fig)
            print(f"{b}: page", flush=True)
    print("wrote", out)


if __name__ == "__main__":
    main()
