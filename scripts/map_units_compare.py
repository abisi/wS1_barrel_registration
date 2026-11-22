"""Global (NWB ccf_atlas_*) vs refined (D2 barrel fit) localisation of every unit in the barrel-field columns, over all
user-accepted brains: does the area or the layer change, besides the barrel?

Units: NWB units table, bc_label good / mua, global CCF = ccf_atlas_{ap,dv,ml} (um, 10 um CCF grid). Mapping as in
map_channels_example.py (k-d tree on brainreg global positions of the sample voxels -> D2 ccf_field). Area = acronym
without barrel and layer (SSp-bfd-C2-4 -> SSp-bfd, SSp-un4 -> SSp-un); global area from the NWB acronym (Allen 2017),
refined from the barrel atlas at the refined voxel. Only units whose global position falls in the refined field.

Writes <out>/units_global_vs_refined.csv (keys mouse_id, session_id, electrode_group, cluster_id) and summary tables.
usage: python scripts/map_units_compare.py -c CONFIG --final DIR --nwb-dir DIR --out DIR
"""
import argparse
import os
import re
import sys

import h5py
import numpy as np
import pandas as pd
import tifffile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from barrelreg.atlas import BARREL_NAMES
from barrelreg.config import Config
from map_channels_example import label, lookup, structure

LAYER = re.compile(r"(1|2/3|4|5|6a|6b)$")


def area_of(a):
    a = re.sub(r"^SSp-bfd-[A-Za-z0-9]+-", "SSp-bfd", a)
    return LAYER.sub("", a) if a.startswith(("SS", "MO", "VIS", "AUD", "RSP", "PTL", "TEa", "ECT", "PERI")) else a


def layer_of(a):
    a = re.sub(r"^SSp-bfd-[A-Za-z0-9]+-", "SSp-bfd", a)
    m = LAYER.search(a)
    return m.group(1) if m and area_of(a) != a else ""


def units(f):
    with h5py.File(f, "r") as h:
        if "units" not in h or "ccf_atlas_ap" not in h["units"]:
            return None
        u = h["units"]
        d = pd.DataFrame({k: u[k][:] for k in ("cluster_id", "bc_label", "ccf_atlas_ap", "ccf_atlas_dv", "ccf_atlas_ml",
                                               "ccf_atlas_acronym")})
        d["electrode_group"] = [h[r].name.split("/")[-1] for r in u["electrode_group"][:]]
    for c in ("bc_label", "ccf_atlas_acronym", "ccf_atlas_ap", "ccf_atlas_dv", "ccf_atlas_ml"):
        if d[c].dtype == object:
            d[c] = [x.decode("utf-8") if isinstance(x, bytes) else x for x in d[c]]
    for c in ("ccf_atlas_ap", "ccf_atlas_dv", "ccf_atlas_ml"):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d[d.bc_label.isin(["good", "mua"]) & d[["ccf_atlas_ap", "ccf_atlas_dv", "ccf_atlas_ml"]].notna().all(1)]
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", required=True)
    ap.add_argument("--final", required=True)
    ap.add_argument("--nwb-dir", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cfg = Config(a.config)
    os.makedirs(a.out, exist_ok=True)
    ann = tifffile.memmap(cfg.annotation, mode="r")
    anat = pd.read_csv(f"{a.final}/anatomy_export.csv", keep_default_na=False)
    anat = anat[anat.decision == "use_local"]
    rows = []
    for b, q in anat.groupby("brain"):
        lk = None
        for s in q.session_id:
            f = f"{a.nwb_dir}/{s}.nwb"
            if not os.path.exists(f):
                continue
            d = units(f)
            if d is None or d.empty:
                continue
            lk = lk or lookup(cfg, a.final, b)
            tree, C, LAB, _ = lk
            v = d[["ccf_atlas_ap", "ccf_atlas_dv", "ccf_atlas_ml"]].values / 10.0
            dist, j = tree.query(v)
            Cr = C[j]
            inside = np.isfinite(Cr).all(1) & (dist < 3)
            d = d[inside].copy()
            if d.empty:
                continue
            Cr, v, j = Cr[inside], v[inside], j[inside]
            acr_r = [label(i) for i in structure(ann, Cr)]
            d = d.assign(mouse_id=re.match(r"^((AB|MH)\d+)", b).group(1), brain=b, session_id=s,
                         ccf_ref_ap=Cr[:, 0] * 10, ccf_ref_dv=Cr[:, 1] * 10, ccf_ref_ml=Cr[:, 2] * 10,
                         shift_um=np.linalg.norm(Cr - v, axis=1) * 10, acronym_refined=acr_r,
                         barrel_refined=[BARREL_NAMES[k - 1] if 0 < k <= len(BARREL_NAMES) else "" for k in LAB[j]],
                         area_global=[area_of(x) for x in d.ccf_atlas_acronym], area_refined=[area_of(x) for x in acr_r],
                         layer_global=[layer_of(x) for x in d.ccf_atlas_acronym], layer_refined=[layer_of(x) for x in acr_r])
            rows.append(d)
        print(b, sum(len(r) for r in rows if r.brain.iloc[0] == b), "units in field", flush=True)
    t = pd.concat(rows, ignore_index=True)
    t.to_csv(f"{a.out}/units_global_vs_refined.csv", index=False)
    t["area_changed"] = t.area_global != t.area_refined
    ctx = (t.layer_global != "") & (t.layer_refined != "")
    t["layer_changed"] = ctx & (t.layer_global != t.layer_refined)
    print(f"\n{len(t)} good/mua units in the refined field, {t.session_id.nunique()} sessions, {t.brain.nunique()} brains")
    print(f"area changed: {t.area_changed.mean():.1%}  ({t.area_changed.sum()})")
    print(f"layer changed (cortex both): {t.layer_changed[ctx].mean():.1%} of {ctx.sum()}")
    print(f"shift median {t.shift_um.median():.0f} um, IQR {t.shift_um.quantile(.25):.0f}-{t.shift_um.quantile(.75):.0f}")
    tr = pd.crosstab(t.area_global, t.area_refined)
    tr.to_csv(f"{a.out}/area_transitions.csv")
    big = tr.loc[tr.sum(1) >= 20, tr.sum(0) >= 20]
    print("\narea transitions (rows global, cols refined; areas with >= 20 units):\n", big.to_string())
    bfd = t[(t.area_global == "SSp-bfd") | (t.area_refined == "SSp-bfd")]
    lt = pd.crosstab(bfd.layer_global, bfd.layer_refined)
    lt.to_csv(f"{a.out}/layer_transitions_bfd.csv")
    print("\nlayer transitions, units in SSp-bfd under either:\n", lt.to_string())
    per = t.groupby("brain").agg(units=("cluster_id", "size"), area_changed=("area_changed", "mean"),
                                 layer_changed=("layer_changed", "mean"), shift_um=("shift_um", "median"))
    per.to_csv(f"{a.out}/per_brain.csv")
    print("\nper brain (fraction changed):\n", per.round(2).sort_values("area_changed").to_string())


if __name__ == "__main__":
    main()
