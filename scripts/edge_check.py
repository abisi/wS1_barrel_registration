"""Which hemispheres may need a larger extraction volume (BARRELREG_MARGIN_UM / NEAR_UM / OFFSETS_RANGE): signs that
the image barrels lie at the edge of what was extracted around the registered barrel field.

Per hemisphere of a run with the debug dump:
- depth: band-pass contrast of every slab (-300..+50 um from the image L4 bottom) in the field; flagged when the
  peak is at the deepest or shallowest slab (barrels may sit outside the slab range);
- shift: best search placement on the shift bound (the field may be further than the search reaches);
- coverage: fraction of a 600 um ring around the registered field with valid flat image (low = the field sits near
  the edge of the extracted tissue);
- displacement: median shift of the barrels by the fit (large = the global registration is off).

usage: python scripts/edge_check.py -c FINAL/config.toml [--out CSV]
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import ndimage

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from barrelreg.config import Config
from barrelreg.imageproc import bandpass, contrast

RING_PX = 60  # 600 um


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", required=True)
    ap.add_argument("--hemisphere", type=int, default=1)
    ap.add_argument("--out")
    a = ap.parse_args()
    cfg = Config(a.config)
    hv = a.hemisphere
    brains = pd.read_csv(cfg.registration_params).iloc[:, 0].astype(str)
    rows = []
    for b in brains:
        d = f"{cfg.brain_dir(b)}/h{hv}"
        if not os.path.exists(f"{d}/debug.npz"):
            continue
        D = np.load(f"{d}/debug.npz")
        m = json.load(open(f"{d}/metrics.json"))
        src = json.load(open(f"{cfg.brain_dir(b)}/source.json")) if os.path.exists(f"{cfg.brain_dir(b)}/source.json") else {}
        foot = D["foot"]
        field = ndimage.binary_dilation(foot, iterations=10)
        offs = [int(v) for v in D["offsets"]]
        con = [contrast(bandpass(D["stack_img"][i], D["stack_valid"][i]), D["stack_valid"][i] & field)
               for i in range(len(offs))]
        k = int(np.argmax(con))
        ring = ndimage.binary_dilation(foot, iterations=RING_PX) & ~foot
        cov = float(D["stack_valid"][offs.index(min(offs, key=abs))][ring].mean())
        sp = D["search_params"]
        sb = m["search_best"]
        shift = float(np.hypot(*sb["shift_um"]))
        rows.append(dict(brain=b, passed=src.get("passed"), accepted_visual=src.get("accepted_visual", False),
                         contrast_after=m["contrast_after"], depth_peak_um=offs[k],
                         depth_peak_at_edge=k in (0, len(offs) - 1),
                         depth_peak_vs_edge=round(con[k] / max(con[1:-1]), 3) if k in (0, len(offs) - 1) else np.nan,
                         search_shift_um=round(shift), shift_on_bound=max(abs(v) for v in sb["shift_um"]) >= sp[0] - 1,
                         ring_valid_frac=round(cov, 2), fit_disp_median_um=round(m["disp_median_um"]),
                         err_initial_um=round(m["err_initial_um"])))
    T = pd.DataFrame(rows)
    T["flag"] = T.depth_peak_at_edge | T.shift_on_bound | (T.ring_valid_frac < 0.6)
    out = a.out or f"{cfg.work_dir}/edge_check.csv"
    T.to_csv(out, index=False)
    with pd.option_context("display.width", 200, "display.max_rows", 100):
        print(T.sort_values(["passed", "flag"]).to_string(index=False))
    print(f"\nflagged {int(T.flag.sum())}/{len(T)}; among failures {int(T[~T.passed.astype(bool)].flag.sum())}/"
          f"{int((~T.passed.astype(bool)).sum())}; wrote {out}")


if __name__ == "__main__":
    main()
