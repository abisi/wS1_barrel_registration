"""Replace one hemisphere's fit in a final run (build_final.py) by a fit accepted on visual check, e.g. a lesion-masked
fit whose contrast is below the pass threshold (contrast with the lesion masked is measured on the remaining tissue
only, so it is not comparable with the unmasked runs).

usage: python scripts/accept_fit.py --final DIR --brain ID --run LABEL=DIR [--note TEXT] [--hemisphere 1]
Copies <run>/<ID>/results/ over <final>/<ID>/results/, keeps the previous choice in source.json ("replaced"), sets
chosen_run, passed = True, accepted_visual = True, and updates selection.csv. Then rerun summarize / reports for ID.
"""
import argparse
import json
import os
import shutil
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from barrelreg.config import results_dir
from build_final import copy_tree


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--final", required=True)
    ap.add_argument("--brain", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--note", default="accepted on visual check")
    ap.add_argument("--hemisphere", type=int, default=1)
    a = ap.parse_args()
    lab, src = a.run.split("=", 1)
    b, hv = a.brain, a.hemisphere
    dst = f"{a.final}/{b}/results"
    old = json.load(open(f"{dst}/source.json"))
    m = json.load(open(f"{results_dir(src, b)}/h{hv}/metrics.json"))
    shutil.rmtree(dst)
    copy_tree(results_dir(src, b), dst)
    info = dict(old, chosen_run=lab, chosen_dir=src, passed=True, accepted_visual=True, note=a.note,
                contrast_after=m["contrast_after"], n_matched=m["n_matched"], l4_offset_um=m["l4_offset_um"],
                replaced={k: old[k] for k in ("chosen_run", "chosen_dir", "contrast_after", "n_matched", "l4_offset_um",
                                              "passed")})
    info["candidates"] = dict(old.get("candidates", {}), **{lab: dict(contrast_after=round(m["contrast_after"], 3),
                                                                      n_matched=m["n_matched"],
                                                                      l4_offset_um=m["l4_offset_um"])})
    json.dump(info, open(f"{dst}/source.json", "w"), indent=1, default=float)
    S = pd.read_csv(f"{a.final}/selection.csv")
    if "accepted_visual" not in S:
        S["accepted_visual"] = False
    i = S.index[S.brain == b]
    for k in ("chosen_run", "chosen_dir", "passed", "contrast_after", "n_matched", "l4_offset_um", "accepted_visual"):
        S.loc[i, k] = info[k]
    S.to_csv(f"{a.final}/selection.csv", index=False)
    print(f"{b}: {old['chosen_run']} ({old['contrast_after']:.2f}) -> {lab} ({m['contrast_after']:.2f}), accepted")


if __name__ == "__main__":
    main()
