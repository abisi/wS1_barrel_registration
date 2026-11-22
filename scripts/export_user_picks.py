"""Export the user-picked D2 fits (picker decisions) to a final project folder and to each mouse's Anatomy folders.

usage: python scripts/export_user_picks.py --run DIR --picks CSV --out DIR --data-root DIR [--variants DIR ...] [--dry-run]

--variants: extra variant dirs (<dir>/<variant>/<brain>, e.g. refit runs), searched after <run>/variants; later wins.

--picks: user_picks_<date>.csv (brain, variant, decision use_local / use_global / redo, note).
Project folder (--out), one sub-folder per brain (all brains, incl. several brains of one mouse):
  use_local  -> results of the picked variant (barrels_corrected_full.tif, ccf_field_{0,1,2}.tif, h1/metrics.json,
                h1/flatcoords.npz, h1/l4_bottom_patches.csv) + report pngs + registration_info.json.
                debug*.npz are not copied (they stay in <run>/variants/<variant>/<brain>/results/h1).
  use_global / redo -> registration_info.json only.
Anatomy (--data-root/<mouse>/<session>/Anatomy/wS1_barrel_registration/, every session folder of the mouse that has an
Anatomy folder): one brain per mouse. Among the mouse's brains: a use_local brain (if several: --prefer, then the one
with annotated probe tracks, then the highest contrast),
else registration_info.json saying use_global (or redo pending). Files: barrels_corrected_full.tif, ccf_field_*.tif,
barrel_index.csv, metrics.json, l4_bottom_patches.csv, report pngs, registration_info.json, README.txt.
"""
import argparse
import glob
import json
import os
import re
import shutil
from datetime import date

import pandas as pd

README = """wS1 barrel-field local registration (D2, repo abisi/wS1_barrel_registration), user-picked fit.
registration_info.json: brain, picked variant, decision, fit metrics, source folder.
decision use_local:
  barrels_corrected_full.tif  uint8 barrel index per voxel, brainreg downsampled sample space (same grid as
                              brainreg downsampled ch0); names in barrel_index.csv (1-29 A1..E8, then Alpha Beta Delta Gamma)
  ccf_field_{0,1,2}.tif       corrected Allen CCF position (mm) of each sample voxel, brainreg deformation-field
                              convention (10 um CCF voxel = value * 100); NaN outside the barrel-field columns
  metrics.json                fit quality (contrast_after >= 0.3 pass; err_tps_loo_um; n_matched; l4_offset_um)
decision use_global: the local fit was rejected by eye; use the global brainreg registration for this mouse.
decision redo: fit pending.
Left hemisphere only (brainglobe hemisphere value 1).
"""


def mouse_of(brain):
    return re.match(r"^((AB|MH)\d+)", brain).group(1)


def cp(src, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copyfile(src, dst)  # NAS refuses chmod / utime


def tracks_counter(params_csv):
    """brain -> number of annotated probe tracks (<reg_dir>/segmentation/sample_space/tracks/**/*.points)"""
    reg = {r["brain"]: r.get("reg_dir") for r in pd.read_csv(params_csv).to_dict("records")}
    return lambda b: len(glob.glob(f"{reg.get(b)}/segmentation/sample_space/tracks/**/*.points", recursive=True)) if reg.get(b) else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--picks", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--variants", nargs="*", default=[])
    ap.add_argument("--prefer", nargs="*", default=[], help="brains preferred for their mouse's Anatomy folders")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    picks = pd.read_csv(a.picks, keep_default_na=False)
    n_tracks = tracks_counter(f"{a.run}/registration_params.csv")
    sel = pd.read_csv(f"{a.run}/selection.csv").set_index("brain")
    today = date.today().isoformat()
    infos = {}
    for r in picks.itertuples():
        b, v = r.brain, r.variant
        info = dict(brain=b, mouse_id=mouse_of(b), hemisphere=1, decision=r.decision, note=r.note,
                    picked_variant=None if r.decision != "use_local" else v,
                    auto_variant=sel.chosen_run.get(b), picked_at=r.picked_at, exported=today,
                    run=os.path.basename(a.run.rstrip("/")))
        od = f"{a.out}/{b}"
        if r.decision == "use_local":
            src = [d for d in [f"{a.run}/variants/{v}/{b}"] + [f"{vd}/{v}/{b}" for vd in a.variants]
                   if os.path.exists(f"{d}/results/h1/metrics.json")][-1]
            m = json.load(open(f"{src}/results/h1/metrics.json"))
            info.update(source_dir=src, contrast_after=m["contrast_after"], contrast_both=m.get("contrast_both"),
                        n_matched=m["n_matched"], err_tps_loo_um=m["err_tps_loo_um"], err_initial_um=m["err_initial_um"],
                        l4_offset_um=m["l4_offset_um"], naming_ambiguous=m["naming_ambiguous"],
                        ccf_label_agreement=m["ccf_label_agreement"], image_channel=m.get("image_channel"),
                        lesion_mask=m.get("lesion_mask"))
            files = [f"results/{f}" for f in ("barrels_corrected_full.tif", "ccf_field_0.tif", "ccf_field_1.tif",
                                              "ccf_field_2.tif", "h1/metrics.json", "h1/flatcoords.npz",
                                              "h1/l4_bottom_patches.csv")]
            files += [os.path.relpath(p, src) for p in glob.glob(f"{src}/report/*.png")]
            if not a.dry_run:
                for f in files:
                    cp(f"{src}/{f}", f"{od}/{f}")
                cp(f"{a.run}/barrel_index.csv", f"{od}/results/barrel_index.csv")
        infos[b] = info
        if not a.dry_run:
            os.makedirs(od, exist_ok=True)
            json.dump(info, open(f"{od}/registration_info.json", "w"), indent=1, default=str)
    if not a.dry_run:
        cp(a.picks, f"{a.out}/{os.path.basename(a.picks)}")
        pd.DataFrame(infos.values()).to_csv(f"{a.out}/final_registration_table.csv", index=False)
        open(f"{a.out}/README.txt", "w").write(README)

    # Anatomy: one brain per mouse
    rows = []
    for mouse, brains in pd.Series(list(infos)).groupby(pd.Series([mouse_of(b) for b in infos])):
        cand = [infos[b] for b in brains]
        loc = [c for c in cand if c["decision"] == "use_local"]
        # several use_local brains of one mouse: --prefer, then the brain with annotated probe tracks (the units are
        # linked through them), then the highest contrast
        chosen = max(loc, key=lambda c: (c["brain"] in a.prefer, n_tracks(c["brain"]), c["contrast_after"])) if loc else (
            next((c for c in cand if c["decision"] == "redo"), None) or cand[0])
        sessions = sorted(glob.glob(f"{a.data_root}/{mouse}/{mouse}_*/Anatomy"))
        for an in sessions:
            td = f"{an}/wS1_barrel_registration"
            rows.append(dict(mouse_id=mouse, session_id=os.path.basename(os.path.dirname(an)), brain=chosen["brain"],
                             decision=chosen["decision"], other_brains=";".join(c["brain"] for c in cand if c is not chosen),
                             target=td))
            if a.dry_run:
                continue
            if chosen["decision"] == "use_local":
                sd = f"{a.out}/{chosen['brain']}/results"
                for f in ("barrels_corrected_full.tif", "ccf_field_0.tif", "ccf_field_1.tif", "ccf_field_2.tif",
                          "barrel_index.csv", "h1/metrics.json", "h1/l4_bottom_patches.csv"):
                    cp(f"{sd}/{f}", f"{td}/{os.path.basename(f)}")
                for p in glob.glob(f"{a.out}/{chosen['brain']}/report/*.png"):
                    cp(p, f"{td}/{os.path.basename(p)}")
            os.makedirs(td, exist_ok=True)
            json.dump(dict(chosen, other_brains=rows[-1]["other_brains"], project_dir=f"{a.out}/{chosen['brain']}"),
                      open(f"{td}/registration_info.json", "w"), indent=1, default=str)
            open(f"{td}/README.txt", "w").write(README)
        if not sessions:
            rows.append(dict(mouse_id=mouse, session_id="", brain=chosen["brain"], decision=chosen["decision"],
                             other_brains="", target="NO ANATOMY FOLDER"))
    t = pd.DataFrame(rows)
    if not a.dry_run:
        t.to_csv(f"{a.out}/anatomy_export.csv", index=False)
    print(t.groupby("decision").agg(mice=("mouse_id", "nunique"), session_folders=("session_id", "count")))
    print(t[t.target == "NO ANATOMY FOLDER"])
    print(t[t.other_brains != ""].drop_duplicates("mouse_id"))


if __name__ == "__main__":
    main()
