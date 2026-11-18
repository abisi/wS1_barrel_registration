"""Compare the outputs of two runs for the same brains (e.g. this repo vs an older run_d2.py run).

usage: python scripts/compare_outputs.py <work_dir_a> <work_dir_b> MS261 [MS262 ...]
Reports, per brain: label volume agreement, ccf_field max difference, and metrics.json fields that differ.
"""
import json
import os
import sys

import numpy as np
import tifffile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from barrelreg.config import results_dir


def main(a, b, ids):
    for bid in ids:
        la = tifffile.imread(f"{results_dir(a, bid)}/barrels_corrected_full.tif")
        lb = tifffile.imread(f"{results_dir(b, bid)}/barrels_corrected_full.tif")
        lab = (la > 0) | (lb > 0)
        print(f"{bid}: labels identical {np.array_equal(la, lb)}; agreement over labelled voxels "
              f"{np.mean(la[lab] == lb[lab]):.4f}")
        for k in range(3):
            fa = tifffile.imread(f"{results_dir(a, bid)}/ccf_field_{k}.tif")
            fb = tifffile.imread(f"{results_dir(b, bid)}/ccf_field_{k}.tif")
            both = np.isfinite(fa) & np.isfinite(fb)
            print(f"  ccf_field_{k}: same NaN pattern {np.array_equal(np.isfinite(fa), np.isfinite(fb))}, "
                  f"max |diff| {np.abs(fa[both] - fb[both]).max() if both.any() else 0:.3g} mm")
        for hv in (1, 2):
            fa, fb = f"{results_dir(a, bid)}/h{hv}/metrics.json", f"{results_dir(b, bid)}/h{hv}/metrics.json"
            if not (os.path.exists(fa) and os.path.exists(fb)):
                print(f"  h{hv}: metrics missing in {'a' if not os.path.exists(fa) else 'b'}")
                continue
            ma, mb = json.load(open(fa)), json.load(open(fb))
            skip = {"seconds", "registration"}
            diff = [k for k in sorted(set(ma) | set(mb)) if k not in skip and ma.get(k) != mb.get(k)]
            print(f"  h{hv}: metrics {'identical' if not diff else 'differ in ' + ', '.join(diff)}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3:])
