"""summary.csv (one row per hemisphere, from h<hv>/metrics.json) and a contact sheet of the corrected flat maps."""
import json
import os

import pandas as pd

from .plotting import contact_sheet

COLUMNS = ["brain", "hemisphere", "l4_depth_um", "n_matched", "err_initial_um", "err_tps_loo_um", "contrast_initial",
           "contrast_after", "dice_initial", "dice_after", "n_candidates_agree", "margin_to_other_naming",
           "naming_ambiguous", "jac_min", "ccf_label_agreement", "ccf_c2_in_c2"]


def summarize(cfg, ids, fail_contrast=0.3):
    rows, pngs, labels = [], [], []
    for b in ids:
        for hv in (1, 2):
            f = f"{cfg.brain_dir(b)}/h{hv}/metrics.json"
            if not os.path.exists(f):
                continue
            m = json.load(open(f))
            rows.append({k: m.get(k) for k in COLUMNS})
            png = cfg.flat_fit_png(b, hv)
            if os.path.exists(png):
                pngs.append(png)
                tag = " FAILED" if m["contrast_after"] < fail_contrast else " ambiguous" if m["naming_ambiguous"] else ""
                labels.append(f"{b} h{hv}{tag}")
    df = pd.DataFrame(rows, columns=COLUMNS)
    df.to_csv(f"{cfg.work_dir}/summary.csv", index=False)
    if pngs:
        contact_sheet(pngs, labels, f"{cfg.work_dir}/contact_sheet_D2.png")
    ok = df.contrast_after >= fail_contrast
    print(f"{len(df)} hemispheres, {int((~ok).sum())} failed (contrast after < {fail_contrast}); medians over the rest: "
          f"{df[ok].n_matched.median():.1f} matched, LOO {df[ok].err_tps_loo_um.median():.0f} um, "
          f"contrast {df[ok].contrast_after.median():.2f}, Dice {df[ok].dice_after.median():.2f}", flush=True)
    return df
