"""Score every fit of a seeded brain against the clicked barrel centres (the clicks are taken as ground truth).

usage: python scripts/click_agreement.py --seeds seeds.json --variants DIR [DIR ...] [--hemisphere 1] [--out scores.csv]

--seeds     {brain: {"points_flat_rc": {barrel: [row, col]}}} (scripts/db_to_files.py --seeds-json)
--variants  variant folders (<dir>/<variant>/<brain>/results/h<hemi>/debug.npz; fits need BARRELREG_DEBUG=1)
Per fit (and the global registration): `inside` = clicks lying in the barrel of the same name in the fitted map,
`median_um` = median distance from each click to the centre of that fitted barrel. More inside and a smaller
distance = closer to the clicks. Printed per brain, best first; --out writes a table.
"""
import argparse
import csv
import glob
import json
import os

import numpy as np

from barrelreg.atlas import BARREL_NAMES

PIX_UM = 10.0  # template flat grid


def score(lab, pts):
    inside, dist = 0, []
    for name, (r, c) in pts.items():
        k = BARREL_NAMES.index(name) + 1
        ri, ci = int(round(r)), int(round(c))
        if 0 <= ri < lab.shape[0] and 0 <= ci < lab.shape[1] and lab[ri, ci] == k:
            inside += 1
        ys, xs = np.where(lab == k)
        dist.append(np.hypot(ys.mean() - r, xs.mean() - c) * PIX_UM if len(ys) else np.nan)
    return inside, float(np.nanmedian(dist))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", required=True)
    ap.add_argument("--variants", nargs="+", required=True)
    ap.add_argument("--hemisphere", type=int, default=1)
    ap.add_argument("--out")
    a = ap.parse_args()
    rows = []
    for b, s in sorted(json.load(open(a.seeds)).items()):
        pts = s["points_flat_rc"]
        fits = {}
        for vd in a.variants:
            for d in sorted(glob.glob(f"{vd}/*/{b}/results/h{a.hemisphere}/debug.npz")):
                fits[d.replace("\\", "/").split("/")[-5]] = d  # <dir>/<variant>/<brain>/results/h<n>/debug.npz
        res = []
        for vid, f in sorted(fits.items()):
            D = np.load(f)
            if not any(r[0] == "global" for r in res):
                res.append(("global", *score(D["cols"], pts)))
            res.append((vid, *score(D["cols_new"], pts)))
        print(f"== {b}  ({len(pts)} clicks)")
        for vid, n_in, med in sorted(res, key=lambda r: (r[2], -r[1])):
            print(f"  {vid:24s} inside {n_in:2d}/{len(pts)}   median {med:5.0f} um")
            rows.append(dict(brain=b, fit=vid, n_clicks=len(pts), inside=n_in, median_um=round(med, 1)))
    if a.out and rows:
        with open(a.out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)


if __name__ == "__main__":
    main()
