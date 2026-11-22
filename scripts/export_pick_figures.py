"""Copy the D2 report PDF and the publication figure(s) (<ID>_h1_imec<N>_paper_figure.{pdf,png,svg}, brains with a
wS1 track only) of each user-picked fit to the final project folder and to the mouse's Anatomy folders.

Source: <run>/<ID>/report when the pick is the automatic variant, else <run>/variants/<variant>/<ID>/report (regenerated
for the picked variant). Targets: <final>/<ID>/report/ and every Anatomy target of that brain in
<final>/anatomy_export.csv.

usage: python scripts/export_pick_figures.py --run DIR --final DIR
"""
import argparse
import glob
import os
import shutil

import pandas as pd

PATS = ("*_D2_report.pdf", "*_paper_figure.pdf", "*_paper_figure.png", "*_paper_figure.svg")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--final", required=True)
    a = ap.parse_args()
    picks = pd.read_csv(f"{a.final}/user_picks_20261120.csv", keep_default_na=False)
    anat = pd.read_csv(f"{a.final}/anatomy_export.csv", keep_default_na=False)
    rows = []
    for r in picks[picks.decision == "use_local"].itertuples():
        diff = str(r.picked_differs_from_auto) == "True"
        src = f"{a.run}/variants/{r.variant}/{r.brain}/report" if diff else f"{a.run}/{r.brain}/report"
        files = sorted(f for p in PATS for f in glob.glob(f"{src}/{p}"))
        targets = [f"{a.final}/{r.brain}/report"] + list(anat[(anat.brain == r.brain) & (anat.decision == "use_local")].target)
        for t in targets:
            os.makedirs(t, exist_ok=True)
            for f in files:
                shutil.copyfile(f, f"{t}/{os.path.basename(f)}")
        rows.append(dict(brain=r.brain, variant=r.variant, regenerated=diff, n_files=len(files),
                         has_report=any(f.endswith("_D2_report.pdf") for f in files),
                         n_paper_figures=sum(f.endswith("_paper_figure.pdf") for f in files), n_targets=len(targets)))
    t = pd.DataFrame(rows)
    t.to_csv(f"{a.final}/figures_export.csv", index=False)
    print(t[["has_report", "n_paper_figures", "n_targets"]].describe().loc[["count", "mean", "min", "max"]])
    print("missing report:", list(t.brain[~t.has_report]))


if __name__ == "__main__":
    main()
