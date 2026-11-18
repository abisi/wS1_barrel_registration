"""Rescue report of a final run (build_final.py): one PDF to check by eye every hemisphere that the reference run
failed and the final run passes.

Page 1: selection rule and a table of every hemisphere (reference vs final contrast, chosen run, status).
Then one page per rescued hemisphere: flat L4 view with brainreg (global), the reference fit (rerun of the reference
settings with the debug dump, so drawn on its own slab) and the final fit, with the probe dye (ch1) and each wS1
track's L4 crossing; under the panels, every candidate run's contrast / matched barrels / slab depth and the barrel
each track crosses in L4 under the reference and the final fit.

usage: python scripts/rescue_report.py -c FINAL/config.toml --ref-debug DIR --tracks tracks_summary.csv [--out PDF]
"""
import argparse
import json
import os
import sys
import textwrap

import numpy as np
import pandas as pd
import tifffile
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from barrelreg.config import Config, results_dir
from barrelreg.register import PIX
from brain_report import flat_panel, flatten, name, row_legend, signal_flat, square_box
from paper_figure import load_tracks

FAIL = 0.3
RULE = ("Selection per left hemisphere: the primary run (default search, slab fitted at 7 depths plus the automatic "
        "picks, depth chosen by barrel contrast, ch0) if its contrast after warping is >= 0.3; otherwise the passing "
        "run with the highest contrast among wide search ch0, default search ch1 and wide search ch1 (same depths and "
        "depth rule); otherwise the primary run, flagged failed. Reference = Lana Smith's settings (default search, "
        "automatic slab picks, ch0), rerun with the debug dump. Contrast is the pass criterion and the selection score, "
        "so each rescue below needs a visual check: the refined outlines should sit on the image barrels and the "
        "names should follow the rows.")


def slab(D):
    p = f"o{int(D['chosen_offset'])}_"
    return flatten(D[p + "l4"], D[p + "valid"]), D[p + "valid"], p


def matched_blobs(D, p):
    bl = D[p + "blobs"]
    return np.where(np.isin(bl, np.asarray(D["ib"]) + 1), bl, 0)


def label_at(cols, rc):
    r, c = int(round(rc[0])), int(round(rc[1]))
    if 0 <= r < cols.shape[0] and 0 <= c < cols.shape[1]:
        return name(cols[r, c]) or "septum / outside"
    return "outside"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", required=True)
    ap.add_argument("--ref-debug", required=True, help="rerun of the reference settings with BARRELREG_DEBUG=1")
    ap.add_argument("--tracks", required=True)
    ap.add_argument("--out")
    a = ap.parse_args()
    cfg = Config(a.config)
    S = pd.read_csv(f"{cfg.work_dir}/selection.csv")
    tracks = pd.read_csv(a.tracks)
    out = a.out or f"{cfg.work_dir}/rescue_report.pdf"
    resc = S[(S.reference_passed == False) & (S.passed == True)].sort_values("brain")
    lost = S[(S.reference_passed == True) & (S.passed == False)]
    still = S[(S.reference_passed == False) & (S.passed == False)]
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"], "pdf.fonttype": 42})
    with PdfPages(out) as pdf:
        # ---------------------------------------------------------------- summary page
        fig = plt.figure(figsize=(11.7, 8.3))
        fig.text(0.04, 0.95, "D2 barrel registration, left hemispheres: rescue report", fontsize=15, weight="bold")
        fig.text(0.04, 0.915, "\n".join(textwrap.wrap(RULE, 190)), fontsize=8.5, va="top")
        fig.text(0.04, 0.82, f"{len(S)} hemispheres; reference passes {int(S.reference_passed.sum())}; final passes "
                             f"{int(S.passed.sum())}; rescued {len(resc)} (one page each below); lost {len(lost)}"
                             + (f" ({', '.join(lost.brain)})" if len(lost) else "") + f"; failing in every run "
                             f"{len(still)}: {', '.join(still.brain)}", fontsize=9, va="top", wrap=True)
        T = S.sort_values("brain")
        cols_ = ["brain", "reference_contrast", "contrast_after", "n_matched", "l4_offset_um", "chosen_run", "status"]
        T = T.assign(status=np.where(T.passed & ~T.reference_passed.astype(bool), "rescued",
                                     np.where(T.passed, "pass", np.where(T.reference_passed.astype(bool), "LOST",
                                                                          "fail"))))
        T = T[cols_].round(2)
        half = (len(T) + 1) // 2
        for i, part in enumerate((T.iloc[:half], T.iloc[half:])):
            ax = fig.add_axes([0.03 + i * 0.49, 0.03, 0.46, 0.74])
            ax.axis("off")
            tb = ax.table(cellText=part.values, colLabels=["brain", "ref", "final", "matched", "slab um", "run",
                                                             "status"], loc="upper center", cellLoc="center")
            tb.auto_set_font_size(False)
            tb.set_fontsize(6.5)
            tb.scale(1, 0.92)
            for (r, c), cell in tb.get_celld().items():
                cell.set_linewidth(0.3)
                if r > 0 and c == 6:
                    cell.set_text_props(color={"rescued": "#1a7f37", "fail": "#cf222e", "LOST": "#cf222e"}
                                        .get(part.iloc[r - 1, 6], "black"))
        pdf.savefig(fig)
        plt.close(fig)

        # ---------------------------------------------------------------- one page per rescued hemisphere
        for r in resc.itertuples():
            b, hv = r.brain, int(r.hemisphere)
            od = f"{cfg.brain_dir(b)}/h{hv}"
            D = dict(np.load(f"{od}/debug.npz"))
            V = np.load(f"{od}/debug3d.npz")
            src = json.load(open(f"{cfg.brain_dir(b)}/source.json"))
            Dr = dict(np.load(f"{results_dir(a.ref_debug, b)}/h{hv}/debug.npz"))
            if not (np.allclose(Dr["flat_lo"], D["flat_lo"]) and Dr["cols"].shape == D["cols"].shape):
                print(f"{b}: reference and final flat grids differ, skipped", flush=True)
                continue
            l4, valid, p = slab(D)
            l4r, validr, pr = slab(Dr)
            cols, cols_f, cols_r = D["cols"], D["cols_new"], Dr["cols_new"]
            flat_lo = D["flat_lo"]
            sig, cross = None, []
            try:
                sl_ = tuple(slice(int(s0), int(s1)) for s0, s1 in D["sl"])
                sig3 = np.asarray(tifffile.memmap(cfg.ds_file(cfg.reg_dir(b), 1), mode="r")[sl_]).astype(np.float32)
                F, rel = V["v_F"].astype(np.float32), V["v_rel"].astype(np.float32)
                sig = signal_flat(sig3, F, V["v_region"] & (V["v_hemi"] > 0), flat_lo, D[p + "l4"].shape)
                depth = V["v_depth"].astype(np.float32)
                lo = np.array([s[0] for s in D["sl"]])
                for tn, X, q in load_tracks(cfg, b, hv, tracks, V, depth, lo):
                    Fq, rq = F[tuple(q.T)], rel[tuple(q.T)]
                    k = np.isfinite(Fq).all(1) & (np.abs(rq - int(D["chosen_offset"])) < 60)
                    if k.any():
                        cross.append((tn, tuple((Fq[k] - flat_lo).mean(0) / PIX - 0.5)))
            except (IndexError, OSError) as e:
                print(f"{b}: no signal overlay ({e})", flush=True)
            box = square_box((cols > 0) | (cols_f > 0) | (cols_r > 0))
            fig = plt.figure(figsize=(11.7, 6.2))
            ok_ref = r.reference_contrast
            fig.text(0.02, 0.96, f"{b} left hemisphere: reference {ok_ref:.2f} (failed) -> final {r.contrast_after:.2f} "
                                 f"(pass), chosen run {r.chosen_run}", fontsize=12, weight="bold")
            panels = ((l4r, validr, cols, "dashed", "brainreg (global)", None),
                      (l4r, validr, cols_r, "-", f"reference fit (Lana settings), {ok_ref:.2f}", matched_blobs(Dr, pr)),
                      (l4, valid, cols_f, "-", f"final fit: {r.chosen_run}, {r.contrast_after:.2f}",
                       matched_blobs(D, p)))
            for i, (img, vv, lab, ls, t, ref) in enumerate(panels):
                ax = fig.add_axes([0.005 + i * 0.28, 0.17, 0.275, 0.75])
                flat_panel(ax, img, vv, box, cols=lab, names=True, title=t, fs=6, bar=i == 0, ref=ref, ls=ls,
                           sig=sig if i != 1 else None, cross=[c for _, c in cross] if i != 1 else ())
                if i == 2:
                    row_legend(ax, 7, ref=True, outside=True, sig=sig is not None, cross=len(cross) > 0)
            cand = "; ".join(f"{k} {v['contrast_after']:.2f} ({v['n_matched']} matched, slab {v['l4_offset_um']:+d} um)"
                             for k, v in src["candidates"].items())
            trk = "; ".join(f"{tn}: {label_at(cols, c)} (brainreg) / {label_at(cols_r, c)} (reference) / "
                            f"{label_at(cols_f, c)} (final)" for tn, c in cross) or "no wS1 track"
            fig.text(0.02, 0.15, "\n".join(textwrap.wrap("Candidate runs: " + cand, 200)), fontsize=8, va="top")
            fig.text(0.02, 0.07, "\n".join(textwrap.wrap("Track L4 crossing: " + trk, 200)), fontsize=8, va="top")
            pdf.savefig(fig)
            plt.close(fig)
            print(f"{b}: page", flush=True)
    print("wrote", out)


if __name__ == "__main__":
    main()
