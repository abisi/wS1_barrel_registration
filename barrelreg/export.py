"""Per mouse: overview figure, detected cells -> CCF (kept if in SSp-bfd), optional copy of the results.

Cells: <cells_dir>/<ID>/downsampled.points (brainreg downsampled space, asr, same for every registration parameter set)
-> D2 ccf_field (nearest voxel, as brainreg does) -> CCF 10 um voxel. A cell is kept as wS1 when its CCF voxel lies in
SSp-bfd (barrels, their layers, and SSp-bfd1-6b between them). fit_ok = False for hemispheres whose fit failed
(contrast after < FAIL_CONTRAST); those cells are kept but should not be used.
Hemispheres: brainglobe value 1 = left (high axis 2), 2 = right.
"""
import glob
import json
import os
import shutil

import h5py
import numpy as np
import pandas as pd
import tifffile

from .atlas import BARREL_NAMES, BFD_IDS, S, collapse
from .plotting import HEMI, overview_figure

FAIL_CONTRAST = 0.3
S_ID = S.set_index("id")


def metrics(cfg, bid):
    m = {}
    for hv in (1, 2):
        f = f"{cfg.brain_dir(bid)}/h{hv}/metrics.json"
        if os.path.exists(f):
            m[hv] = json.load(open(f))
            m[hv]["fit_ok"] = m[hv]["contrast_after"] >= FAIL_CONTRAST
    return m


def read_points(f):
    with h5py.File(f, "r") as h:
        return h["df/block0_values"][:].astype(np.float64)


def downsampled_points(cfg, bid):
    """downsampled.points if present, else ch1.csv (raw z, y, x) mapped like brainreg (brainglobe-space, orientation
    and voxel sizes from brainreg.json)."""
    f = f"{cfg.cells_dir}/{bid}/downsampled.points"
    if os.path.exists(f):
        return read_points(f)
    import brainglobe_space as bgs
    reg = cfg.reg_dir(bid)
    js = json.load(open(f"{reg}/brainreg.json"))
    src = bgs.AnatomicalSpace(js["orientation"], shape=tifffile.TiffFile(cfg.raw_ch0(bid)).series[0].shape,
                              resolution=[float(v) for v in js["voxel_sizes"]])
    dst = bgs.AnatomicalSpace("asr", shape=tifffile.memmap(cfg.ds_file(reg, 0), mode="r").shape, resolution=(cfg.voxel_um,) * 3)
    print(f"{bid}: no downsampled.points, mapping ch1.csv from raw space", flush=True)
    return src.map_points_to(dst, pd.read_csv(f"{cfg.cells_dir}/{bid}/ch1.csv")[["z", "y", "x"]].values)


def structure_info(ids):
    acr = np.array([S_ID.acronym.get(int(i), "") for i in ids], object)
    name = np.array([S_ID.name.get(int(i), "") for i in ids], object)
    layer = np.array([a.rsplit("-", 1)[1] if a.startswith("SSp-bfd-") else a[len("SSp-bfd"):] for a in acr], object)
    return acr, name, layer


def cells(cfg, bid, met):
    reg = cfg.reg_dir(bid)
    od = cfg.brain_dir(bid)
    P = downsampled_points(cfg, bid)
    v = np.rint(P).astype(int)
    shape = np.array(tifffile.memmap(f"{reg}/registered_atlas.tiff", mode="r").shape)
    inb = (v >= 0).all(1) & (v < shape).all(1)
    fld = [tifffile.imread(f"{od}/ccf_field_{k}.tif") for k in range(3)]
    ccf = np.full((len(P), 3), np.nan)
    for k in range(3):
        ccf[inb, k] = fld[k][tuple(v[inb].T)] * 100.0  # mm -> 10 um CCF voxel
    ok = np.isfinite(ccf).all(1)
    ann = tifffile.memmap(cfg.annotation, mode="r")
    cv = np.clip(np.rint(ccf[ok]).astype(int), 0, np.array(ann.shape) - 1)
    sid = np.zeros(len(P), np.int64)
    sid[ok] = np.asarray(ann[tuple(cv.T)])
    ws1 = ok & np.isin(sid, BFD_IDS)
    hem = tifffile.memmap(f"{reg}/registered_hemispheres.tiff", mode="r")
    hv = np.zeros(len(P), int)
    hv[inb] = np.asarray(hem[tuple(v[inb].T)]) if inb.any() else 0
    lab = tifffile.imread(f"{od}/barrels_corrected_full.tif")
    d2 = np.zeros(len(P), int)
    d2[inb] = lab[tuple(v[inb].T)]
    i = np.where(ws1)[0]
    acr, name, layer = structure_info(sid[i])
    bar = collapse(sid[i])
    df = pd.DataFrame(dict(
        brain=bid, cell_index=i, hemisphere=[HEMI.get(h, "") for h in hv[i]],
        downsampled_axis_0=P[i, 0], downsampled_axis_1=P[i, 1], downsampled_axis_2=P[i, 2],
        ccf_axis_0=ccf[i, 0], ccf_axis_1=ccf[i, 1], ccf_axis_2=ccf[i, 2],
        ccf_um_0=ccf[i, 0] * 10, ccf_um_1=ccf[i, 1] * 10, ccf_um_2=ccf[i, 2] * 10,
        structure_id=sid[i], structure_acronym=acr, structure_name=name,
        barrel=[BARREL_NAMES[b - 1] if b > 0 else "septa/other" for b in bar], layer=layer,
        d2_sample_space_barrel=[BARREL_NAMES[b - 1] if b > 0 else "" for b in d2[i]],
        fit_ok=[bool(met.get(h, {}).get("fit_ok", False)) for h in hv[i]]))
    # for comparison: cells in SSp-bfd under the plain coarse registration (atlas.points; -1 if absent)
    try:
        ap = read_points(f"{cfg.cells_dir}/{bid}/atlas.points").astype(int)
        pann = cfg.registration_atlas()
        okp = (ap >= 0).all(1) & (ap < np.array(pann.shape)).all(1)
        n_coarse = int(np.isin(np.asarray(pann[tuple(ap[okp].T)]), BFD_IDS).sum())
    except Exception:
        n_coarse = -1
    out = f"{od}/{bid}_wS1_cells_ccf.csv"
    df.to_csv(out, index=False)
    summ = dict(brain=bid, n_cells=len(P), n_ccf_defined=int(ok.sum()), n_wS1=len(df),
                n_wS1_left=int((df.hemisphere == "left").sum()), n_wS1_right=int((df.hemisphere == "right").sum()),
                n_wS1_fit_ok=int(df.fit_ok.sum()), n_SSp_bfd_coarse_registration=n_coarse)
    print(summ, flush=True)
    return df, summ, out


def copy(src, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copyfile(src, dst)  # network shares may not allow setting permissions


def export(cfg, bid, do_cells=True, copy_to=None):
    met = metrics(cfg, bid)
    od = cfg.brain_dir(bid)
    files = [overview_figure(cfg, bid, met), f"{od}/barrels_corrected_full.tif"] + \
            [f"{od}/ccf_field_{k}.tif" for k in range(3)]
    summ = None
    if do_cells:
        _, summ, csv = cells(cfg, bid, met)
        files.append(csv)
    if copy_to:
        for f in files:
            copy(f, f"{copy_to}/{bid}/{os.path.basename(f)}")
        for hv in met:
            for f in glob.glob(f"{od}/h{hv}/*.json") + [cfg.flat_fit_png(bid, hv)]:
                copy(f, f"{copy_to}/{bid}/h{hv}_{HEMI[hv]}/{os.path.basename(f)}")
    return summ


def combine(cfg, ids, summs):
    """wS1_cell_counts.csv (merged with earlier counts) and all_mice_wS1_cells_ccf.csv over every exported mouse."""
    cnt = f"{cfg.work_dir}/wS1_cell_counts.csv"
    table = pd.read_csv(cnt).set_index("brain") if os.path.exists(cnt) else pd.DataFrame()
    for s in summs:
        if s:
            table.loc[s["brain"], list(s)[1:]] = list(s.values())[1:]
    if len(table):
        table.index.name = "brain"
        table.sort_index().astype(int).to_csv(cnt)
    have = [b for b in cfg.params.index if os.path.exists(f"{cfg.brain_dir(b)}/{b}_wS1_cells_ccf.csv")]
    if have:
        pd.concat([pd.read_csv(f"{cfg.brain_dir(b)}/{b}_wS1_cells_ccf.csv", keep_default_na=False) for b in have]
                  ).to_csv(f"{cfg.work_dir}/all_mice_wS1_cells_ccf.csv", index=False)
