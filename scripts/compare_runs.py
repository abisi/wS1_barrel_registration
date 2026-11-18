"""Compare D2 runs hemisphere by hemisphere and explain why a fit that failed in one run succeeds in another.

The runs differ in one setting at a time, so the first run in the chain where a failed hemisphere passes names the
change that rescued it:
  reference  e.g. Lana Smith's D2_neuropixels: default search, slab fitted at the contrast / pattern picks only
  offsets    same search, slab fitted at 7 offsets (BARRELREG_FIT_OFFSETS)
  channel    as `offsets` on image channel 1 (BARRELREG_CH=1)
  wide       wide search (BARRELREG_SEARCH=wide) + 7 offsets
Failed = contrast after < 0.3 (as in export.py).

usage: python scripts/compare_runs.py --ref DIR [--offsets DIR] [--channel DIR] --wide DIR --out DIR
Writes compare_runs.csv (one row per hemisphere, all runs side by side), compare_runs.md and compare_runs.png.
"""
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

FAIL = 0.3
OLD = dict(rot=20.0, scale=(0.7, 1.4), shift=600.0)  # bounds of the default search grid (register.py)
KEEP = ["contrast_initial", "contrast_after", "n_matched", "n_blobs", "dice_after", "l4_offset_um", "l4_bottom_um",
        "l4_bottom_found_frac", "surface_bridged_frac", "err_initial_um", "err_tps_loo_um", "affine_residual_um",
        "naming_ambiguous", "margin_to_other_naming", "n_candidates_agree", "jac_min", "disp_median_um"]


def load(run):
    rows = []
    # <run>/<brain>/results/h*/ (new layout) or <run>/<brain>/h*/ (older runs, e.g. D2_neuropixels)
    for f in glob.glob(f"{run}/*/results/h[12]/metrics.json") + glob.glob(f"{run}/*/h[12]/metrics.json"):
        m = json.load(open(f))
        r = {k: m.get(k) for k in KEEP}
        sb = m.get("search_best") or {}
        r.update(brain=m["brain"], hemisphere=m["hemisphere"], scale=sb.get("scale"), rot=sb.get("rot_deg"),
                 search_corr=sb.get("corr"),
                 shift=float(np.max(np.abs(sb["shift_um"]))) if sb.get("shift_um") is not None else np.nan,
                 offsets_fitted=",".join(str(int(v)) for v in (m.get("offset_candidates_um") or [])))
        rows.append(r)
    return pd.DataFrame(rows).set_index(["brain", "hemisphere"]) if rows else pd.DataFrame()


def beyond_old(r, p):
    """Which bounds of the default grid the best placement of run p lies on or beyond."""
    out = []
    if abs(r[f"rot_{p}"]) >= OLD["rot"]:
        out.append(f"rotation {r[f'rot_{p}']:+.1f} deg")
    if r[f"scale_{p}"] <= OLD["scale"][0] or r[f"scale_{p}"] >= OLD["scale"][1]:
        out.append(f"scale {r[f'scale_{p}']:.2f}")
    if r[f"shift_{p}"] >= OLD["shift"]:
        out.append(f"shift {r[f'shift_{p}']:.0f} um")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--offsets")
    ap.add_argument("--channel")
    ap.add_argument("--wide", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    runs = {k: getattr(a, k) for k in ("ref", "offsets", "channel", "wide") if getattr(a, k)}
    D = None
    for k, d in runs.items():
        t = load(d).add_suffix(f"_{k}")
        D = t if D is None else D.join(t, how="outer")
    for k in runs:
        D[f"ok_{k}"] = D[f"contrast_after_{k}"] >= FAIL
    D = D[D["contrast_after_ref"].notna()].copy()  # hemispheres of the reference run

    # why the reference failed / why a later run passed
    why = []
    for idx, r in D.iterrows():
        notes = []
        if not r["ok_ref"]:
            b = beyond_old(r, "ref")
            if b:
                notes.append("ref best placement on the search bound (" + ", ".join(b) + ")")
            if "offsets" in runs and r.get("ok_offsets") == True:
                notes.append(f"rescued by the slab depth: {r['l4_offset_um_offsets']:+.0f} um (ref fitted "
                             f"{r['offsets_fitted_ref']} um)")
            elif "channel" in runs and r.get("ok_channel") == True and r.get("ok_wide") != True:
                notes.append("rescued only on image channel 1")
            elif r.get("ok_wide") == True:
                bw = beyond_old(r, "wide")
                notes.append("rescued by the wide search: " + (
                    "best placement beyond the old grid (" + ", ".join(bw) + ")" if bw else
                    "within the old grid, i.e. by the local refinement / 8 candidates"))
            else:
                notes.append("still failed")
        elif r.get("ok_wide") is False:
            notes.append("passed in ref, FAILED in wide")
        why.append("; ".join(notes))
    D["explanation"] = why
    D.to_csv(f"{a.out}/compare_runs.csv")

    # ---------------------------------------------------------------- summary text
    fr = D[~D["ok_ref"]]
    lines = [f"# D2 run comparison", "",
             f"Runs: " + ", ".join(f"{k} = `{v}`" for k, v in runs.items()), "",
             f"Hemispheres in the reference: {len(D)}; failed in ref: {len(fr)} (contrast after < {FAIL}).", ""]
    for k in runs:
        if k != "ref" and f"ok_{k}" in D:
            lines.append(f"- {k}: passes {int(D[f'ok_{k}'].sum())}/{len(D)}; rescues "
                         f"{int(fr[f'ok_{k}'].fillna(False).sum())}/{len(fr)} of the ref failures; "
                         f"breaks {int((D['ok_ref'] & D[f'contrast_after_{k}'].notna() & ~D[f'ok_{k}']).sum())} ref passes")
    nb = sum(bool(beyond_old(r, "ref")) for _, r in fr.iterrows())
    nb_ok = sum(bool(beyond_old(r, "ref")) for _, r in D[D["ok_ref"]].iterrows())
    lines += ["", f"Ref best placement on the old search bound: {nb}/{len(fr)} failed vs {nb_ok}/{int(D['ok_ref'].sum())} "
                  f"passed hemispheres.", ""]
    # what distinguishes rescued from still-failing hemispheres (ref quantities)
    if "ok_wide" in D:
        resc, still = fr[fr["ok_wide"] == True], fr[fr["ok_wide"] != True]
        lines += [f"Rescued by the final run: {len(resc)}; still failing: {len(still)}.", "",
                  "| ref quantity (median) | passed in ref | rescued | still failing |", "|---|---|---|---|"]
        for c in ["contrast_initial_ref", "search_corr_ref", "n_blobs_ref", "n_matched_ref", "l4_bottom_found_frac_ref",
                  "surface_bridged_frac_ref", "err_initial_um_ref"]:
            if c in D:
                lines.append(f"| {c[:-4]} | {D[D['ok_ref']][c].median():.3g} | {resc[c].median():.3g} | "
                             f"{still[c].median():.3g} |")
    lines += ["", "| brain | h | contrast ref -> wide | matched ref -> wide | explanation |", "|---|---|---|---|---|"]
    for (b, h), r in fr.iterrows():
        lines.append(f"| {b} | {h} | {r['contrast_after_ref']:.2f} -> {r.get('contrast_after_wide', np.nan):.2f} | "
                     f"{r['n_matched_ref']:.0f} -> {r.get('n_matched_wide', np.nan):.0f} | {r['explanation']} |")
    open(f"{a.out}/compare_runs.md", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))

    # ---------------------------------------------------------------- figure
    fig, ax = plt.subplots(1, 4, figsize=(20, 4.8))
    col = np.where(D["ok_ref"], "0.6", np.where(D.get("ok_wide", False) == True, "C2", "C3"))
    lab = {"0.6": "passed in ref", "C2": "failed in ref, rescued", "C3": "failed in both"}
    for c in lab:
        m = col == c
        ax[0].scatter(D["contrast_after_ref"][m], D["contrast_after_wide"][m], c=c, s=22, label=lab[c])
    ax[0].axhline(FAIL, color="k", ls=":", lw=0.8)
    ax[0].axvline(FAIL, color="k", ls=":", lw=0.8)
    ax[0].plot([-0.2, 1.4], [-0.2, 1.4], color="0.8", lw=0.8)
    ax[0].set_xlabel("contrast after, ref")
    ax[0].set_ylabel("contrast after, wide")
    ax[0].legend(fontsize=8)
    for i, (q, bnd, t) in enumerate((("rot", (-OLD["rot"], OLD["rot"]), "rotation (deg)"),
                                     ("scale", OLD["scale"], "scale"), ("shift", (OLD["shift"],), "max |shift| (um)"))):
        x = ax[i + 1]
        for c in lab:
            m = col == c
            x.scatter(D[f"{q}_ref"][m], D[f"{q}_wide"][m], c=c, s=22)
        for v in bnd:
            x.axvline(v, color="k", ls=":", lw=0.8)
            x.axhline(v, color="k", ls=":", lw=0.8)
        x.set_xlabel(f"{t}, ref best placement")
        x.set_ylabel(f"{t}, wide best placement")
        x.set_title("dotted = bound of the default grid")
    fig.tight_layout()
    fig.savefig(f"{a.out}/compare_runs.png", dpi=150)


if __name__ == "__main__":
    main()
