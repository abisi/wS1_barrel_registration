"""Assemble one final D2 run from several runs: per hemisphere, the primary run if it passes (contrast after >= 0.3),
otherwise the passing run with the highest contrast, otherwise the primary run (kept, flagged failed).

usage: python scripts/build_final.py --runs LABEL=DIR [LABEL=DIR ...] --out DIR [--ref DIR] [--hemisphere 1]
The first run is the primary. Each chosen <run>/<ID>/results/ is copied to <out>/<ID>/results/ with source.json
(chosen run, every candidate's contrast / matched barrels, reference contrast). Writes <out>/selection.csv and the
run-level files (config.toml with work_dir = out, template.npz, barrel_index.csv, registration_params.csv).
"""
import argparse
import json
import os
import shutil
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from barrelreg.config import results_dir

FAIL = 0.3


def copy_tree(src, dst):
    """Contents only (the NAS refuses chmod / utime, so no shutil.copytree)."""
    for d, _, files in os.walk(src):
        t = os.path.join(dst, os.path.relpath(d, src))
        os.makedirs(t, exist_ok=True)
        for f in files:
            shutil.copyfile(os.path.join(d, f), os.path.join(t, f))


def metrics(run, b, hv):
    f = f"{results_dir(run, b)}/h{hv}/metrics.json"
    return json.load(open(f)) if os.path.exists(f) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ref", help="reference run for comparison only (e.g. Lana Smith's D2_neuropixels)")
    ap.add_argument("--hemisphere", type=int, default=1)
    a = ap.parse_args()
    runs = [tuple(v.split("=", 1)) for v in a.runs]
    prim = runs[0][1]
    hv = a.hemisphere
    os.makedirs(a.out, exist_ok=True)
    for f in ("template.npz", "barrel_index.csv", "registration_params.csv"):
        shutil.copyfile(f"{prim}/{f}", f"{a.out}/{f}")  # NAS: no chmod
    cfg = open(f"{prim}/config.toml").read().replace(prim.rstrip("/"), a.out.rstrip("/"))
    open(f"{a.out}/config.toml", "w").write(cfg)
    brains = pd.read_csv(f"{prim}/registration_params.csv").iloc[:, 0].astype(str)
    rows = []
    for b in brains:
        cand = {lab: metrics(d, b, hv) for lab, d in runs}
        cand = {k: v for k, v in cand.items() if v is not None}
        if not cand:
            continue
        p0 = runs[0][0]
        ok = {k: v["contrast_after"] >= FAIL for k, v in cand.items()}
        if ok.get(p0):
            pick = p0
        elif any(ok.values()):
            pick = max((k for k in cand if ok[k]), key=lambda k: cand[k]["contrast_after"])
        else:
            pick = p0 if p0 in cand else max(cand, key=lambda k: cand[k]["contrast_after"])
        src = dict(runs)[pick]
        dst = f"{a.out}/{b}/results"
        if os.path.exists(dst):
            shutil.rmtree(dst)
        copy_tree(results_dir(src, b), dst)
        mr = metrics(a.ref, b, hv) if a.ref else None
        info = dict(brain=b, hemisphere=hv, chosen_run=pick, chosen_dir=src, passed=bool(ok[pick]),
                    contrast_after=cand[pick]["contrast_after"], n_matched=cand[pick]["n_matched"],
                    l4_offset_um=cand[pick]["l4_offset_um"],
                    reference_contrast=mr["contrast_after"] if mr else None,
                    reference_passed=bool(mr["contrast_after"] >= FAIL) if mr else None,
                    candidates={k: dict(contrast_after=round(v["contrast_after"], 3), n_matched=v["n_matched"],
                                        l4_offset_um=v["l4_offset_um"]) for k, v in cand.items()})
        json.dump(info, open(f"{dst}/source.json", "w"), indent=1, default=float)
        rows.append({k: v for k, v in info.items() if k != "candidates"}
                    | {f"contrast_{k}": v["contrast_after"] for k, v in cand.items()})
        print(f"{b}: {pick} (contrast {cand[pick]['contrast_after']:.2f}{'' if ok[pick] else ', FAILED'})", flush=True)
    S = pd.DataFrame(rows)
    S.to_csv(f"{a.out}/selection.csv", index=False)
    print(f"{len(S)} hemispheres: {int(S.passed.sum())} pass; chosen run: {S.chosen_run.value_counts().to_dict()}")
    if a.ref:
        r = S.reference_passed.fillna(False).astype(bool)
        print(f"reference passes {int(r.sum())}; rescued {int((~r & S.passed).sum())}; lost {int((r & ~S.passed).sum())}")


if __name__ == "__main__":
    main()
