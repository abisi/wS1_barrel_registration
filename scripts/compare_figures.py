"""Figures comparing a reference D2 run (e.g. Lana Smith's D2_neuropixels) with our runs, for the problematic
hemispheres: failed in the reference, or passed in the reference but failed in the wide run.

One row of flat-L4 panels per hemisphere: brainreg (global) | reference refined | each of our runs
(default or wide search, 7 slab depths, channel 0 or 1). The reference run saves no debug dump, so its refined barrels are
rebuilt from its flatcoords.npz: per flat pixel, the mean corrected flat coordinate (G_corrected) of the voxels on it,
then the template barrel at that coordinate (as register.py does with the TPS field). The rebuild is checked on the
wide run itself (its own flatcoords vs its saved cols_new). Each panel lists the run's metrics and the contrast of its
barrels recomputed on the image shown, so the reference is scored on the same image as ours.

usage: python scripts/compare_figures.py --ref DIR --wide DIR --runs LABEL=DIR ... --csv compare_runs.csv --out DIR
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import ndimage
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from barrelreg import pointset as ps
from barrelreg.atlas import SEPTA
from barrelreg.config import results_dir
from barrelreg.register import PIX
from brain_report import flat_panel, flatten, row_legend, square_box

FAIL = 0.3


def rebuild_cols(fc_npz, alab, flat_lo):
    """Refined flat barrel labels of a run from its flatcoords.npz, on the flat grid of `alab` (origin flat_lo)."""
    z = np.load(fc_npz)
    G, Gc = z["G"].reshape(-1, 2), z["G_corrected"].reshape(-1, 2)
    ok = np.isfinite(G).all(1) & np.isfinite(Gc).all(1)
    fi = (G[ok] - flat_lo) / PIX - 0.5
    fc = (Gc[ok] - flat_lo) / PIX - 0.5
    ij = np.rint(fi).astype(int)
    inb = (ij >= 0).all(1) & (ij[:, 0] < alab.shape[0]) & (ij[:, 1] < alab.shape[1])
    lin = ij[inb, 0] * alab.shape[1] + ij[inb, 1]
    n = np.bincount(lin, minlength=alab.size)
    have = (n > 0).reshape(alab.shape)
    m = np.stack([np.bincount(lin, fc[inb, k], minlength=alab.size) for k in range(2)], 0) / np.maximum(n, 1)
    m = m.reshape(2, *alab.shape)
    lab = ndimage.map_coordinates(alab, m, order=0, cval=0)
    lab = np.where(have & (lab < SEPTA), lab, 0)
    return lab, have, int(z["l4_offset_um"])


def nearest_prefix(D, d):
    offs = sorted({int(k.split("_")[0][1:]) for k in D if k.startswith("o") and k.endswith("_l4")})
    o = min(offs, key=lambda v: abs(v - d))
    return f"o{o}_", o


def contrast_on(D, p, cols):
    v2 = D[p + "valid"] & ~D[p + "art"]
    fv = v2 & ndimage.binary_dilation(cols > 0, iterations=5)
    return ps.sep_contrast(D[p + "bpA"], cols, fv) if (cols > 0).any() else np.nan


def placement(m):
    sb = m.get("search_best") or {}
    if not sb:
        return ""
    sh = np.max(np.abs(sb["shift_um"]))
    return f"scale {sb['scale']:.2f}, rot {sb['rot_deg']:+.0f} deg, shift {sh:.0f} um"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--wide", required=True)
    ap.add_argument("--runs", nargs="+", required=True, help="our runs as label=dir, one panel each, in order")
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ref-name", default="Lana Smith (reference)")
    a = ap.parse_args()
    a.runs = [tuple(v.split("=", 1)) for v in a.runs]
    os.makedirs(a.out, exist_ok=True)
    T = pd.read_csv(a.csv)
    T = T[T.hemisphere == 1]
    bad = T[(T.ok_ref == False) | ((T.ok_ref == True) & (T.ok_wide == False))].copy()

    def best_of(b):
        v = [json.load(open(f"{results_dir(d, b)}/h1/metrics.json"))["contrast_after"] for _, d in a.runs
             if os.path.exists(f"{results_dir(d, b)}/h1/metrics.json")]
        return max(v) if v else np.nan

    best_ours = bad.brain.map(best_of)
    bad["group"] = np.where(bad.ok_ref == True, "2 passed in reference, failed in wide run",
                            np.where(best_ours >= FAIL, "0 failed in reference, passes in one of our runs",
                                     "1 failed in reference and in all our runs"))
    bad = bad.sort_values(["group", "brain"])
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
                         "svg.fonttype": "none", "pdf.fonttype": 42})
    checks, rows = [], []
    with PdfPages(f"{a.out}/reference_vs_ours_problematic_left.pdf") as pdf:
        for r in bad.itertuples():
            b, hv = r.brain, 1
            od = f"{results_dir(a.wide, b)}/h{hv}"
            D = dict(np.load(f"{od}/debug.npz"))
            mw = json.load(open(f"{od}/metrics.json"))
            rf = f"{results_dir(a.ref, b)}/h{hv}"
            mr = json.load(open(f"{rf}/metrics.json"))
            alab, lo = D["alab"], D["flat_lo"]
            cols, cols_w = D["cols"], D["cols_new"]
            # check of the rebuild on the wide run itself
            cw2, _, _ = rebuild_cols(f"{od}/flatcoords.npz", alab, lo)
            k = (cols_w > 0) | (cw2 > 0)
            checks.append(float(np.mean(cw2[k] == cols_w[k])))
            cols_r, _, off_r = rebuild_cols(f"{rf}/flatcoords.npz", alab, lo)
            pr, o_r = nearest_prefix(D, off_r)
            pw = f"o{int(D['chosen_offset'])}_"
            resc = None  # our best run (highest contrast after)
            panels = [
                dict(D=D, p=pr, cols=cols, ls="dashed", ref=None, ok=None,
                     title="brainreg (global)", lines=[f"slab {o_r:+d} um (as reference), ch0",
                                                      f"contrast here {contrast_on(D, pr, cols):.2f}"]),
                dict(D=D, p=pr, cols=cols_r, ls="-", ref=None, title=f"{a.ref_name}, ch0",
                     ok=mr["contrast_after"] >= FAIL,
                     lines=[f"contrast {mr['contrast_after']:.2f}, {mr['n_matched']} matched, slab {off_r:+d} um",
                            placement(mr), f"contrast here {contrast_on(D, pr, cols_r):.2f}"
                            + ("" if o_r == off_r else f" (slab {o_r:+d} um)")])]
            # our runs, one panel each (missing run -> empty panel)
            for lab_, rd in a.runs:
                f = f"{results_dir(rd, b)}/h{hv}"
                if not os.path.exists(f"{f}/debug.npz"):
                    panels.append(dict(D=None, title=lab_, lines=["not run"]))
                    continue
                Dr = D if rd == a.wide else dict(np.load(f"{f}/debug.npz"))
                m = json.load(open(f"{f}/metrics.json"))
                p_ = f"o{int(Dr['chosen_offset'])}_"
                panels.append(dict(D=Dr, p=p_, cols=Dr["cols_new"], ls="-", ref=(Dr[p_ + "blobs"], Dr["ib"]),
                                   title=lab_, ok=m["contrast_after"] >= FAIL, lines=[
                        f"contrast {m['contrast_after']:.2f}, {m['n_matched']} matched, slab {m['l4_offset_um']:+d} um",
                        placement(m)]))
                if resc is None or m["contrast_after"] > resc[1]["contrast_after"]:
                    resc = (f, m, lab_)
            box = square_box((cols > 0) | (cols_r > 0) | (cols_w > 0))
            n = len(panels)
            fig = plt.figure(figsize=(3.2 * n + 1.6, 4.6))
            w = 3.2 / (3.2 * n + 1.6)
            for i, P in enumerate(panels):
                ax = fig.add_axes([i * w + 0.003, 0.17, w - 0.006, 0.66])
                if P["D"] is None:
                    ax.axis("off")
                    ax.set_title(P["title"], fontsize=8.5)
                    ax.text(0.5, 0.5, "not run", ha="center", va="center", color="0.5", transform=ax.transAxes)
                    continue
                Dp, p = P["D"], P["p"]
                ref = None
                if P["ref"] is not None:
                    bl, ib = P["ref"]
                    ref = np.where(np.isin(bl, np.asarray(ib) + 1), bl, 0)
                flat_panel(ax, flatten(Dp[p + "l4"], Dp[p + "valid"]), Dp[p + "valid"], box, cols=P["cols"],
                           names=True, title=P["title"], fs=6.5, bar=i == 0, ref=ref, ls=P["ls"])
                if P["ok"] is not None:  # pass / fail of the run (contrast after >= 0.3)
                    ax.title.set_color("#1a7f37" if P["ok"] else "#cf222e")
                    ax.set_title(("PASS  " if P["ok"] else "FAIL  ") + P["title"], fontsize=8.5,
                                 color=ax.title.get_color())
                ax.text(0.5, -0.03, "\n".join(l for l in P["lines"] if l), transform=ax.transAxes, ha="center",
                        va="top", fontsize=7.5)
                if i == n - 1:
                    row_legend(ax, 7, ref=True, outside=True)
            ok_r = "passed" if r.ok_ref else "failed"
            fig.suptitle(f"{b} left hemisphere: reference {ok_r} (contrast {r.contrast_after_ref:.2f}); "
                         f"{r.group[2:]}", fontsize=10, y=0.96)
            bd = f"{a.wide}/{b}/report"  # per-brain copy next to the brain's report
            os.makedirs(bd, exist_ok=True)
            for ext in ("png", "pdf", "svg"):
                fig.savefig(f"{bd}/{b}_h{hv}_reference_vs_ours.{ext}", dpi=200, facecolor="white")
            pdf.savefig(fig, facecolor="white")
            plt.close(fig)
            rows.append(dict(brain=b, hemisphere=hv, group=r.group[2:],
                             contrast_ref_reported=mr["contrast_after"],
                             contrast_ref_rebuilt_same_slab=contrast_on(D, pr, cols_r),
                             rebuild_slab_um=o_r, ref_slab_um=off_r, contrast_wide=mw["contrast_after"],
                             contrast_best_ours=resc[1]["contrast_after"] if resc else np.nan,
                             best_run=resc[2] if resc else ""))
            print(f"{b}: rebuild check {checks[-1]:.3f}", flush=True)
    pd.DataFrame(rows).to_csv(f"{a.out}/reference_vs_ours_problematic_left.csv", index=False)
    print(f"rebuild check (wide run, own flatcoords vs cols_new): median {np.median(checks):.3f}, "
          f"min {np.min(checks):.3f}")


if __name__ == "__main__":
    main()
