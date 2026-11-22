"""Which hemispheres need the lesion-mask variant: fraction of the registered barrel field that imageproc.lesion_mask
would remove from the chosen slab of an unmasked run (debug dump). Prints the brains at or above --min (default 3 %).

usage: python scripts/lesion_fraction.py RUN_DIR [RUN_DIR ...] [--min 0.03] [--csv OUT]
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd
from scipy import ndimage

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from barrelreg.imageproc import lesion_mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--min", type=float, default=0.03)
    ap.add_argument("--hemisphere", type=int, default=1)
    ap.add_argument("--csv")
    a = ap.parse_args()
    rows = []
    for run in a.runs:
        for f in sorted(glob.glob(f"{run}/*/results/h{a.hemisphere}/debug.npz")):
            b = f.split("/")[-4]
            D = np.load(f)
            offs = [int(v) for v in D["offsets"]]
            d4 = int(D["chosen_offset"])
            ks = [i for i, d in enumerate(offs) if abs(d - d4) <= 25]
            l4, valid = D["stack_img"][ks].mean(0), D["stack_valid"][ks].all(0)
            foot = D["foot"]
            les = lesion_mask(l4, valid, ndimage.binary_dilation(foot, iterations=15))
            rows.append(dict(brain=b, run=os.path.basename(run.rstrip("/")), lesion_frac=float((les & foot).sum() / foot.sum())))
    T = pd.DataFrame(rows)
    g = T.groupby("brain").lesion_frac.max()
    if a.csv:
        T.to_csv(a.csv, index=False)
    print(" ".join(sorted(g.index[g >= a.min])))


if __name__ == "__main__":
    main()
