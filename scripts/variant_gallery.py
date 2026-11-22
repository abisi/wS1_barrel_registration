"""Images + manifest for the variant browser (one claude.ai page to pick the best fit per hemisphere).

For each brain of the final run: a plain flat-L4 image (no outlines) and one image per available variant (flat L4 at
the variant's chosen slab, brainreg barrels dashed, fitted barrels solid). Variants are the final choice, plus any run
given with --runs that has the brain, plus every candidate a final run records in source.json (candidate_dirs). Writes <out>/img/<ID>__<variant>.jpg and <out>/manifest.json.

usage: python scripts/variant_gallery.py -c FINAL/config.toml --runs LABEL=DIR [...] --out DIR [--hemisphere 1]
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from barrelreg.config import Config, results_dir
from brain_report import flat_panel, flatten, square_box

FAIL = 0.3


def load(d, hv):
    p = f"{d}/h{hv}"
    if not (os.path.exists(f"{p}/debug.npz") and os.path.exists(f"{p}/metrics.json")):
        return None
    return np.load(f"{p}/debug.npz"), json.load(open(f"{p}/metrics.json"))


def save_panel(path, img, valid, box, cols=None, ghost=None):
    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_axes([0, 0, 1, 1])
    flat_panel(ax, img, valid, box, cols=cols, cols_ghost=ghost, fs=6, bar=True, names=cols is not None)
    fig.savefig(path, dpi=150, pil_kwargs=dict(quality=82))
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", required=True)
    ap.add_argument("--runs", nargs="*", default=[])
    ap.add_argument("--out", required=True)
    ap.add_argument("--hemisphere", type=int, default=1)
    ap.add_argument("--brains", nargs="*", help="only these brains (default: all in selection.csv)")
    a = ap.parse_args()
    cfg = Config(a.config)
    hv = a.hemisphere
    os.makedirs(f"{a.out}/img", exist_ok=True)
    S = pd.read_csv(f"{cfg.work_dir}/selection.csv").set_index("brain")
    runs = [tuple(v.split("=", 1)) for v in a.runs]
    brains = []
    for b in (a.brains or S.index):
        src = json.load(open(f"{cfg.brain_dir(b)}/source.json"))
        fin = load(cfg.brain_dir(b), hv)
        cand = [(src["chosen_run"], fin, True)]
        for lab, d in list(src.get("candidate_dirs", {}).items()) + runs:
            if lab == src["chosen_run"] or any(c[0] == lab for c in cand):
                continue
            r = load(results_dir(d, b), hv)
            if r is not None:
                cand.append((lab, r, False))
        D0 = fin[0]
        box = square_box((D0["cols"] > 0) | np.any([c[1][0]["cols_new"] > 0 for c in cand], 0))
        p0 = f"o{int(D0['chosen_offset'])}_"
        save_panel(f"{a.out}/img/{b}__image.jpg", flatten(D0[p0 + "l4"], D0[p0 + "valid"]), D0[p0 + "valid"], box)
        vs = []
        for lab, (D, m), is_final in cand:
            p = f"o{int(D['chosen_offset'])}_"
            img, val = flatten(D[p + "l4"], D[p + "valid"]), D[p + "valid"] & ~D[p + "art"]
            save_panel(f"{a.out}/img/{b}__{lab}.jpg", img, val, box, cols=D["cols_new"], ghost=D["cols"])
            vs.append(dict(id=lab, img=f"img/{b}__{lab}.jpg", final=is_final, contrast=round(m["contrast_after"], 3),
                           matched=int(m["n_matched"]), slab_um=int(m["l4_offset_um"]),
                           channel=int(D["img_ch"]) if "img_ch" in D.files else 0,
                           lesion_mask=bool(m.get("lesion_mask", False)),
                           search="wide" if float(D["search_params"][0]) > 600 else "default",
                           contrast_both=round(m["contrast_both"], 3) if "contrast_both" in m else None,
                           err_brainreg_um=round(m["err_initial_um"]), loo_um=round(m["err_tps_loo_um"])))
        key = "contrast_both" if all(v["contrast_both"] is not None for v in vs) else "contrast"
        best = max(vs, key=lambda v: v[key])["id"]
        brains.append(dict(brain=b, image=f"img/{b}__image.jpg", variants=vs, best_contrast=best, best_key=key,
                           final=src["chosen_run"], final_passed=bool(src["passed"]),
                           accepted_visual=bool(src.get("accepted_visual", False)),
                           reference_contrast=src.get("reference_contrast")))
        print(f"{b}: {len(vs)} variants", flush=True)
    json.dump(dict(fail=FAIL, brains=brains), open(f"{a.out}/manifest.json", "w"), indent=1, default=float)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
