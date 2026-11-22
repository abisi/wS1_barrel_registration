"""Example: map Neuropixels channels (NWB electrodes, global CCF after ephys alignment) through the user-picked D2
barrel fit, and plot global vs refined positions and labels.

Mapping (global CCF -> refined CCF), per brain:
  every sample voxel around the barrel field has a global CCF position (brainreg deformation_field_{0,1,2}, x100 =
  10 um CCF voxel) and a refined one (D2 ccf_field_{0,1,2}, same convention). A k-d tree on the global positions finds
  each channel's sample voxel; there we read the refined CCF position, the D2 barrel (barrels_corrected_full.tif) and
  the structure at the refined CCF voxel (barrel + layer). Outside the barrel columns ccf_field is NaN -> global kept.
NWB electrodes ccf_atlas_{ap,dv,ml} are in um on the 10 um CCF grid (axis 0 AP, 1 DV, 2 ML; voxel = um / 10).

usage: python scripts/map_channels_example.py -c CONFIG --final DIR --nwb-dir DIR --out DIR [--examples B:SESSION:PROBE ...]
Without --examples: the 3 use_local brains whose NWB probe has the most channels in global SSp-bfd.
"""
import argparse
import glob
import os
import re
import sys

import h5py
import numpy as np
import pandas as pd
import tifffile
from scipy import ndimage
from scipy.spatial import cKDTree

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from barrelreg.atlas import BARREL_NAMES, BFD_IDS, L4_BFD_IDS, S, collapse
from barrelreg.config import Config

S_ID = S.set_index("id")
ACR = S_ID.acronym.to_dict()


def nwb_channels(f):
    with h5py.File(f, "r") as h:
        if "general/extracellular_ephys/electrodes" not in h:
            return None
        e = h["general/extracellular_ephys/electrodes"]
        d = pd.DataFrame({k: e[k][:] for k in ("group_name", "ccf_atlas_ap", "ccf_atlas_dv", "ccf_atlas_ml",
                                               "ccf_atlas_acronym", "rel_y")})
    for c in ("group_name", "ccf_atlas_acronym"):
        d[c] = d[c].str.decode("utf-8")
    return d


def lookup(cfg, final, brain):
    """k-d tree on the global CCF position of every sample voxel in the barrel crop + the refined fields there."""
    reg = cfg.reg_dir(brain)
    cf = [tifffile.imread(f"{final}/{brain}/results/ccf_field_{k}.tif") for k in range(3)]
    ok = np.isfinite(cf[0])
    lo = np.maximum(np.argwhere(ok).min(0) - 40, 0)
    hi = np.argwhere(ok).max(0) + 41
    sl = tuple(slice(a, b) for a, b in zip(lo, hi))
    G = np.stack([np.asarray(tifffile.memmap(f"{reg}/deformation_field_{k}.tiff", mode="r")[sl]) for k in range(3)], -1)
    G = G * 1000.0 / cfg.registration_atlas_um  # mm -> registration-atlas voxel (10 um CCF)
    C = np.stack([c[sl] for c in cf], -1) * 100.0
    lab = tifffile.imread(f"{final}/{brain}/results/barrels_corrected_full.tif")[sl]
    good = np.isfinite(G).all(-1) & (G != 0).any(-1)
    idx = np.argwhere(good)
    return cKDTree(G[good]), C[good], lab[good], idx + lo


def structure(ann, v):
    vi = np.clip(np.rint(v).astype(int), 0, np.array(ann.shape) - 1)
    return np.asarray(ann[tuple(vi.T)])


def label(i):
    a = ACR.get(int(i), "")
    return a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", required=True)
    ap.add_argument("--final", required=True)
    ap.add_argument("--nwb-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--examples", nargs="*")
    ap.add_argument("--n", type=int, default=3)
    a = ap.parse_args()
    cfg = Config(a.config)
    os.makedirs(a.out, exist_ok=True)
    ann = tifffile.memmap(cfg.annotation, mode="r")
    anat = pd.read_csv(f"{a.final}/anatomy_export.csv", keep_default_na=False)
    anat = anat[anat.decision == "use_local"]

    if a.examples:
        ex = [tuple(x.split(":")) for x in a.examples]
    else:  # candidates: probes with most channels in global SSp-bfd
        cand = []
        for r in anat.itertuples():
            f = f"{a.nwb_dir}/{r.session_id}.nwb"
            if not os.path.exists(f):
                continue
            d = nwb_channels(f)
            if d is None:
                continue
            for g, q in d.groupby("group_name"):
                n = q.ccf_atlas_acronym.str.startswith("SSp-bfd").sum()
                if n >= 40:
                    cand.append((n, r.brain, r.session_id, g))
        cand.sort(reverse=True)
        seen, ex = set(), []
        for n, b, s, g in cand:
            if b not in seen:
                ex.append((b, s, g))
                seen.add(b)
            if len(ex) == a.n:
                break
        print("examples:", ex, flush=True)

    rows = []
    for b, s, g in ex:
        d = nwb_channels(f"{a.nwb_dir}/{s}.nwb")
        d = d[d.group_name == g].reset_index(drop=True)
        v = d[["ccf_atlas_ap", "ccf_atlas_dv", "ccf_atlas_ml"]].values / 10.0
        # sanity: NWB acronym vs this annotation at the NWB voxel
        sid_g = structure(ann, v)
        # the NWB uses Allen 2017 names (SSp-bfd4), the barrel atlas splits columns (SSp-bfd-D2-4): compare without barrel
        nob = lambda s_: re.sub(r"^SSp-bfd-[A-Za-z0-9]+-", "SSp-bfd", s_)
        agree = np.mean([nob(label(i)) == x for i, x in zip(sid_g, d.ccf_atlas_acronym)])
        tree, C, LAB, IDX = lookup(cfg, a.final, b)
        dist, j = tree.query(v)
        Cr = C[j]
        inside = np.isfinite(Cr).all(1) & (dist < 3)
        Cr = np.where(inside[:, None], Cr, v)
        sid_r = structure(ann, Cr)
        bar_d2 = LAB[j]
        out = d.assign(brain=b, session_id=s, probe=g, lookup_dist_um=dist * 10,
                       ccf_ref_ap=Cr[:, 0] * 10, ccf_ref_dv=Cr[:, 1] * 10, ccf_ref_ml=Cr[:, 2] * 10,
                       shift_um=np.linalg.norm(Cr - v, axis=1) * 10, refined=inside,
                       acronym_global=[label(i) for i in sid_g], acronym_refined=[label(i) for i in sid_r],
                       barrel_refined=[BARREL_NAMES[k - 1] if 0 < k <= len(BARREL_NAMES) else "" for k in bar_d2])
        rows.append(out)
        print(f"{b} {s} {g}: {len(d)} ch, NWB-vs-annotation acronym agreement {agree:.2f}, lookup dist median "
              f"{np.median(dist) * 10:.0f} um, refined {inside.sum()} ch, shift median "
              f"{np.median(out.shift_um[inside]) if inside.any() else np.nan:.0f} um", flush=True)
        figure(out, ann, v, Cr, inside, f"{a.out}/{b}_{s}_{g}_channel_mapping")
    pd.concat(rows).to_csv(f"{a.out}/example_channel_mapping.csv", index=False)


def short(a):
    return a.replace("SSp-bfd-", "").replace("SSp-bfd", "bfd ")


def figure(t, ann, v, Cr, inside, path):
    fig, ax = plt.subplots(1, 3, figsize=(15, 6.2), gridspec_kw=dict(width_ratios=[1.25, 1, 0.9]))
    # A: top view (AP x ML) of the layer-4 barrels around the probe, global (grey) -> refined (colour) channels
    P_ = np.vstack([v[inside], Cr[inside]]) if inside.any() else v
    lo_, hi_ = P_.min(0) - 25, P_.max(0) + 25
    a0, a1 = int(lo_[0]), int(hi_[0])
    m0, m1 = int(lo_[2]), int(hi_[2])
    sub = np.asarray(ann[a0:a1, :, m0:m1])
    l4 = np.isin(sub, L4_BFD_IDS)
    bar = np.where(l4, collapse(sub), 0).max(1)  # AP x ML
    A = ax[0]
    A.imshow(np.where(bar > 0, 1, np.nan), cmap="Greys", vmin=0, vmax=3, origin="upper",
             extent=[m0 * 10, m1 * 10, a1 * 10, a0 * 10])
    for k in np.unique(bar):
        if k <= 0 or k > len(BARREL_NAMES):
            continue
        mk = ndimage.binary_fill_holes(bar == k).astype(float)
        A.contour(np.arange(m0, m1) * 10, np.arange(a0, a1) * 10, mk, [0.5], colors="0.35", linewidths=0.8)
        cy, cx = ndimage.center_of_mass(mk)
        A.text((m0 + cx) * 10, (a0 + cy) * 10, BARREL_NAMES[k - 1], fontsize=7, ha="center", va="center", color="0.25")
    bfd = np.isin(t.acronym_refined, [ACR[i] for i in BFD_IDS])
    A.plot(v[:, 2] * 10, v[:, 0] * 10, ".", color="0.6", ms=3, label="global (NWB)")
    for p, q in zip(v[inside][::8], Cr[inside][::8]):
        A.annotate("", xy=(q[2] * 10, q[0] * 10), xytext=(p[2] * 10, p[0] * 10),
                   arrowprops=dict(arrowstyle="->", color="#d97706", lw=0.7))
    A.plot(Cr[inside, 2] * 10, Cr[inside, 0] * 10, ".", color="#d97706", ms=3, label="refined (D2)")
    A.set_xlabel("ML (um, CCF)")
    A.set_ylabel("AP (um, CCF)")
    A.set_aspect("equal")
    A.legend(loc="lower left", fontsize=8, frameon=False)
    A.set_title("Top view: L4 barrels (Allen/Blue Brain) and channels\narrows: every 8th channel, global -> refined",
                fontsize=9, loc="left")
    # B: labels along the probe (depth from tip = rel_y)
    B = ax[1]
    y = t.rel_y.values
    labs = sorted(set(t.acronym_global) | set(t.acronym_refined), key=lambda s: (not s.startswith("SSp-bfd"), s))
    cmap = plt.get_cmap("tab20")
    col = {l: cmap(i % 20) for i, l in enumerate(labs)}
    for xi, key in ((0, "acronym_global"), (1, "acronym_refined")):
        for yy, l in zip(y, t[key]):
            B.add_patch(plt.Rectangle((xi - 0.35, yy - 10), 0.7, 20, color=col[l], lw=0))
        # label runs
        run, start = None, None
        for yy, l in list(zip(y, t[key])) + [(None, None)]:
            if l != run:
                if run is not None and abs(prev - start) >= 60:
                    B.text(xi, (start + prev) / 2, short(run), ha="center", va="center", fontsize=6.5)
                run, start = l, yy
            prev = yy if yy is not None else prev
    B.set_xlim(-0.6, 1.6)
    B.set_ylim(np.nanmin(y) - 20, np.nanmax(y) + 20)
    B.set_xticks([0, 1], ["global\n(NWB ccf_atlas)", "refined\n(D2 barrel fit)"])
    B.set_ylabel("distance from tip (um)")
    B.set_title("Structure per channel", fontsize=9, loc="left")
    # C: shift along the probe
    Cx = ax[2]
    Cx.plot(t.shift_um.where(t.refined), y, color="#d97706")
    Cx.set_xlabel("global -> refined shift (um)")
    Cx.set_title("Shift per channel\n(blank = outside barrel columns, global kept)", fontsize=9, loc="left")
    Cx.set_ylim(B.get_ylim())
    r = t.iloc[0]
    agree = (t.acronym_global == t.acronym_refined)[t.refined].mean()
    fig.suptitle(f"{r.brain} {r.session_id} {r.probe}: {int(t.refined.sum())}/{len(t)} channels in the refined field, "
                 f"label unchanged for {agree:.0%} of them; barrel of the L4 channels: "
                 f"{', '.join(sorted(set(t.barrel_refined[t.acronym_refined.str.endswith('4') & t.refined]) - {''}))}",
                 fontsize=10, x=0.01, ha="left")
    for x in ax:
        x.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path + ".png", dpi=150)
    fig.savefig(path + ".pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
