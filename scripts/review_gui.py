"""Build the barrel review page: one web page (served by scripts/review_server.py, or published on claude.ai) to compare the D2 fits of every brain, pick one (or global / redo),
and seed brains whose automatic fit failed (replaces the separate variant picker and seed page).

Per brain it writes
  img/<ID>.jpg            one grey tile sheet: the backgrounds (ch0 / ch1 auto-slab, band + clean) and each fit's own
                          L4 slab image, on the template flat grid (10 um px); tile origins in brains/<ID>.json
  img/<ID>__seed_ch<c>[_s<a>].jpg  seed images (seed_images.py tile sheets, rows = modes, cols = slabs), --seed-brains
                          only; _s1 / _s2 = sharpened slabs (unsharp mask 60 um, amount 1 / 2), --sharp-seed-brains
  brains/<ID>.json        fits (metrics, barrel outlines as polygons), brainreg (global) outlines, atlas barrel centroids
                          (for the page's live seed check: same Umeyama similarity as register.py seed_placement), and
                          every Neuropixels track crossing L4 of the barrel field with the barrel it lands in under each fit
  manifest.json           brain list + per-fit summary (for filters and the brain index)
  index.html              the page (scripts/review_gui.html)

Fits = every <label>/<ID>/results/h1 under each --variants dir (later dirs win on a label clash). Picks, seeds, refit
requests and refit status live in the page's database (collections picks, seeds, requests, status): <out>/db/ with
review_server.py, or the claude.ai page database. See docs/wiki/Review-page.md.

usage: python scripts/review_gui.py --variants D2_all_brains/variants D2_redo_*/variants --out DIR
           [--brains ...] [--seed-brains ...] [--tracks-table probe_insertion_info.csv] [-j 16]
"""
import argparse
import glob
import json
import os
import shutil
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage
from skimage import measure

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from barrelreg.atlas import BARREL_NAMES, SEPTA
from barrelreg.config import Config
from barrelreg.register import PIX
from brain_report import flatten, read_points, square_box, track_samples
import seed_images as si

BASE = ("wide_ch0", "wide_ch1")  # runs whose depth stack gives the backgrounds and seed images
INSERTIONS = "/raid0/lmsmith/local_barrel_reg/D2_neuropixels/probe_insertion_info_from_axel.csv"


def label(vid):
    """readable fit name from the variant id"""
    p = vid.split("_")
    ch = next((x for x in p if x.startswith("ch")), "")
    parts = []
    if p[0] == "wide":
        parts.append("Wide search")
    elif p[0].startswith("seededGD"):
        parts.append("Seeded (Gamma/Delta fixed)")
    elif p[0] == "seeded":
        sh = next((x for x in p if x.startswith("sharp")), None)
        parts.append(f"Seeded + sharpened x{sh[5:]}" if sh else "Seeded")
    elif p[0].startswith("sharp"):
        parts.append(f"Sharpened x{p[0][5:]}")
    else:
        parts.append(p[0])
    parts.append(ch)
    if "lesion" in p:
        parts.append("lesion mask")
    return " · ".join(x for x in parts if x)


def kind(vid):
    return "seeded" if vid.startswith("seeded") else "sharpened" if vid.startswith("sharp") else "auto"


def polys(lab):
    """{barrel name: [[r, c], ...]} outline polygons (pixel-index coords, 0.1 px) of a flat label image"""
    out = {}
    for k in range(1, SEPTA):
        m = lab == k
        if not m.any():
            continue
        cs = measure.find_contours(np.pad(m, 1).astype(float), 0.5)
        c = max(cs, key=len) - 1
        c = measure.approximate_polygon(c, 0.6)
        out[BARREL_NAMES[k - 1]] = np.round(c, 1).tolist()
    return out


def barrel_at(lab, r, c, win=40):
    """barrel name at (r, c) of a flat label image, else the nearest barrel and its distance (um)"""
    ri, ci = int(round(r)), int(round(c))
    if 0 <= ri < lab.shape[0] and 0 <= ci < lab.shape[1] and 0 < lab[ri, ci] < SEPTA:
        return dict(name=BARREL_NAMES[lab[ri, ci] - 1], d_um=0)
    y0, x0 = max(ri - win, 0), max(ci - win, 0)
    sub = lab[y0:ri + win + 1, x0:ci + win + 1]
    ys, xs = np.where((sub > 0) & (sub < SEPTA))
    if not len(ys):
        return dict(name=None, d_um=None)
    d = np.hypot(ys + y0 - r, xs + x0 - c)
    i = int(d.argmin())
    return dict(name=BARREL_NAMES[sub[ys[i], xs[i]] - 1], d_um=int(round(d[i] * PIX)))


def u8(g, v):
    lo, hi = np.percentile(g[v], (1, 99.5))
    return np.where(v, np.clip((g - lo) / max(hi - lo, 1e-9), 0, 1) * 255, 0).astype(np.uint8)


def find_fits(variant_dirs, b):
    fits = {}
    for vd in variant_dirs:
        for d in sorted(glob.glob(f"{vd}/*/{b}/results/h1")):
            if os.path.exists(f"{d}/debug.npz") and os.path.exists(f"{d}/metrics.json"):
                fits[d.split("/")[-4]] = d
    return fits


def tracks_of(cfg, b, base_dir, ins):
    """[(track name, probe target, crop voxels q)] for every track file of the brain in this hemisphere"""
    reg = cfg.reg_dir(b)
    D = np.load(f"{base_dir}/debug.npz")
    V = np.load(f"{base_dir}/debug3d.npz")
    lo = np.array([s[0] for s in D["sl"]])
    depth, hemi = V["v_depth"].astype(np.float32), V["v_hemi"]
    out = []
    for f in sorted(glob.glob(f"{reg}/segmentation/sample_space/tracks/**/*.points", recursive=True)):
        name = f.split("tracks/")[1].replace(".points", "")
        try:
            X, q = track_samples(read_points(f), depth, lo)
        except Exception:
            continue
        if (hemi[tuple(q.T)] > 0).mean() <= 0.5:
            continue
        probe = os.path.basename(name).replace("imec", "")
        t = ins[(ins.mouse_name == b[:5]) & (ins.probe_id.astype(str) == probe)].target_area
        out.append((name, str(t.iloc[0]) if len(t) else "", q))
    return out, V["v_F"].astype(np.float32), V["v_rel"].astype(np.float32), D["flat_lo"]


def crossing(q, F, rel, flat_lo, d_off):
    Fq, rq = F[tuple(q.T)], rel[tuple(q.T)]
    k = np.isfinite(Fq).all(1) & (np.abs(rq - d_off) < 60)
    if not k.any():
        return None
    r, c = (Fq[k] - flat_lo).mean(0) / PIX - 0.5
    return float(r), float(c)


def one(args):
    b, A = args
    cfg = Config(A["config"])
    fits = find_fits(A["variants"], b)
    if not fits:
        return None
    base = {lab: fits[lab] for lab in BASE if lab in fits}
    D0 = np.load(f"{next(iter(base.values()) if base else next(iter(fits.values())))}/debug.npz")
    cols = D0["cols"]
    H, W = cols.shape
    tiles, sheet = [], []

    def add(key, lab, g):
        tiles.append(dict(key=key, label=lab, tx=len(sheet) * W, ty=0))
        sheet.append(g)

    # backgrounds: auto slab of each channel's depth stack, band + clean
    for lab, d in base.items():
        ch = int(lab[-1])
        _, offs, S, Vd = si.stack(f"{d}/debug.npz")
        a, va, _ = si.auto_slab(offs, S, Vd)
        r = si.render(a, va)
        for mode in ("band", "clean"):
            g, v = r[mode]
            add(f"ch{ch}_{mode}", f"ch{ch} · auto slab · {mode}", np.where(v, g * 255, 0).astype(np.uint8))
    ins = pd.read_csv(A["insertions"]) if os.path.exists(A["insertions"]) else pd.DataFrame(
        columns=["mouse_name", "probe_id", "target_area"])
    trk, F, rel, flat_lo = [], None, None, None
    tdir = next((d for d in list(base.values()) + list(fits.values()) if os.path.exists(f"{d}/debug3d.npz")), None)
    if tdir:
        try:
            trk, F, rel, flat_lo = tracks_of(cfg, b, tdir, ins)
        except Exception as e:
            print(b, "tracks failed:", e, flush=True)
    foot = ndimage.binary_dilation(D0["foot"], iterations=20)
    T = {n: dict(name=n, target=t, at={}) for n, t, _ in trk}
    variants, outl, union = [], {}, cols > 0
    for vid, d in sorted(fits.items(), key=lambda kv: (kind(kv[0]) != "auto", kv[0])):
        D = np.load(f"{d}/debug.npz")
        m = json.load(open(f"{d}/metrics.json"))
        off = int(D["chosen_offset"])
        p = f"o{off}_"
        val = D[p + "valid"] & ~D[p + "art"] if p + "art" in D.files else D[p + "valid"]
        add(f"fit_{vid}", f"{label(vid)} · its own slab ({off:+d} um)", u8(flatten(D[p + "l4"], D[p + "valid"]), val))
        cn = D["cols_new"]
        union |= cn > 0
        outl[vid] = polys(cn)
        for n, _, q in trk:
            x = crossing(q, F, rel, flat_lo, off)
            if x is not None and foot[int(np.clip(round(x[0]), 0, H - 1)), int(np.clip(round(x[1]), 0, W - 1))]:
                T[n]["at"][vid] = dict(rc=[round(x[0], 1), round(x[1], 1)], **barrel_at(cn, *x))
                T[n].setdefault("global", barrel_at(cols, *x))
                T[n].setdefault("rc", [round(x[0], 1), round(x[1], 1)])
        variants.append(dict(id=vid, label=label(vid), kind=kind(vid), channel=int(D["img_ch"]) if "img_ch" in D.files else 0,
                             matched=int(m["n_matched"]), contrast=round(float(m["contrast_after"]), 3),
                             loo_um=None if m.get("err_tps_loo_um") is None else round(float(m["err_tps_loo_um"])),
                             err_brainreg_um=None if m.get("err_initial_um") is None else round(float(m["err_initial_um"])),
                             slab_um=off, lesion_mask=bool(m.get("lesion_mask", False)),
                             seed_residual_um=None if m.get("seed_residual_um") is None else round(float(m["seed_residual_um"])),
                             sharpen=m.get("sharpen"), made=os.path.getmtime(f"{d}/metrics.json")))
    tracks, seen = [], set()  # the same track file is often saved twice (top level + session folder): keep one
    for t in sorted((t for t in T.values() if t["at"]), key=lambda t: ("/" not in t["name"], t["name"])):
        key = (round(t["rc"][0]), round(t["rc"][1]))
        if key not in seen:
            seen.add(key)
            tracks.append(t)
    os.makedirs(f"{A['out']}/img", exist_ok=True)
    Image.fromarray(np.concatenate(sheet, 1)).save(f"{A['out']}/img/{b}.jpg", quality=85)
    ks, Ac = D0["ksA"], D0["Ac"]
    detail = dict(brain=b, shape=[H, W], box=[round(float(x), 1) for x in square_box(union)], sheet=f"img/{b}.jpg",
                  tiles=tiles, global_outlines=polys(cols), outlines=outl, tracks=tracks,
                  atlas_centroids={BARREL_NAMES[int(k) - 1]: np.round(Ac[i], 2).tolist() for i, k in enumerate(ks)
                                   if 0 < int(k) < SEPTA})
    if b in A["seed_brains"]:
        detail["seed"] = seed_sheets(b, base, A["out"], (0, 1, 2) if b in A["sharp_brains"] else (0,))
    os.makedirs(f"{A['out']}/brains", exist_ok=True)
    json.dump(detail, open(f"{A['out']}/brains/{b}.json", "w"), separators=(",", ":"))
    print(b, len(variants), "fits,", len(tracks), "tracks", flush=True)
    return dict(brain=b, variants=variants, wS1=any(t["target"] == "wS1" for t in tracks),
                tracks=[dict(name=t["name"], target=t["target"]) for t in tracks], seed="seed" in detail)


def unsharp(S, Vd, amount, sigma_px=6.0):
    """unsharp mask per depth slab, I + amount * (I - Gaussian(I, 60 um)), the 2D analogue of BARRELREG_SHARPEN"""
    out = np.empty_like(S, dtype=np.float32)
    for i, (x, v) in enumerate(zip(S, Vd)):
        w = v.astype(np.float32)
        xx = np.where(v, x, 0).astype(np.float32)
        g = ndimage.gaussian_filter(xx, sigma_px) / np.maximum(ndimage.gaussian_filter(w, sigma_px), 1e-3)
        out[i] = np.where(v, np.maximum(x + amount * (x - g), 1e-3), x)
    return out


def seed_sheets(b, base, out, sharpen=(0,)):
    """seed images (seed_images.py) on one tile sheet per channel (0, 1, 2 = ch0+1) and sharpening amount"""
    slabs = ["mean", "auto"] + [str(o) for o in si.SLABS]
    images = []
    for a in sharpen:
        per = {}
        for lab, d in base.items():
            _, offs, S, Vd = si.stack(f"{d}/debug.npz")
            per[int(lab[-1])] = si.slabs_of(offs, unsharp(S, Vd, a) if a else S, Vd)
        if 0 in per and 1 in per:
            n = lambda x, v: np.log(np.maximum(x / np.median(x[v]), 1e-3))
            sl = {k: (np.exp(0.5 * (n(*per[0][0][k]) + n(*per[1][0][k]))), per[0][0][k][1] & per[1][0][k][1]) for k in per[0][0]}
            oth = None if per[0][1] is None or per[1][1] is None else 0.5 * (per[0][1] + per[1][1])
            per[2] = (sl, oth, None)
        for ch, (sl, other, _) in per.items():
            H, W = next(iter(sl.values()))[0].shape
            src = f"img/{b}__seed_ch{ch}" + (f"_s{a}" if a else "") + ".jpg"
            sh = np.zeros((H * len(si.MODES), W * len(slabs)), np.uint8)
            for j, s in enumerate(slabs):
                r = si.render(*sl[s], other)
                for i, mode in enumerate(si.MODES):
                    if mode in r:
                        g, v = r[mode]
                        sh[i * H:(i + 1) * H, j * W:(j + 1) * W] = np.where(v, g * 255, 0).astype(np.uint8)
                        images.append(dict(src=src, channel=ch, slab=s, mode=mode, sharpen=a, tx=j * W, ty=i * H))
            Image.fromarray(sh).save(f"{out}/{src}", quality=90)
    return dict(images=images, slabs=slabs, modes=si.MODES, sharpen=list(sharpen),
                channel_names={"0": "ch0", "1": "ch1", "2": "ch0+1"})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="+", required=True, help="variant dirs (<dir>/<label>/<ID>/results/h1)")
    ap.add_argument("--config", default=None, help="config.toml for reg_dir (default: <first variants dir>/../config.toml)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--brains", nargs="*")
    ap.add_argument("--seed-brains", nargs="*", default=[])
    ap.add_argument("--sharp-seed-brains", nargs="*", default=[], help="seed brains that also get sharpened seed images (x1, x2)")
    ap.add_argument("--insertions", default=INSERTIONS)
    ap.add_argument("-j", type=int, default=16)
    a = ap.parse_args()
    cfg = a.config or os.path.join(os.path.dirname(os.path.abspath(a.variants[0])), "config.toml")
    brains = a.brains or sorted({p.split("/")[-1] for vd in a.variants for p in glob.glob(f"{vd}/*/*")
                                 if os.path.isdir(f"{p}/results/h1")})
    A = dict(config=cfg, variants=a.variants, out=a.out, seed_brains=set(a.seed_brains) | set(a.sharp_seed_brains), sharp_brains=set(a.sharp_seed_brains), insertions=a.insertions)
    os.makedirs(a.out, exist_ok=True)
    with Pool(a.j) as p:
        res = [r for r in p.map(one, [(b, A) for b in brains]) if r]
    mf = f"{a.out}/manifest.json"
    old = {r["brain"]: r for r in json.load(open(mf))["brains"]} if a.brains and os.path.exists(mf) else {}
    old.update({r["brain"]: r for r in res})
    json.dump(dict(barrels=BARREL_NAMES, pix_um=PIX, brains=sorted(old.values(), key=lambda r: r["brain"])),
              open(mf, "w"), indent=0)
    shutil.copyfile(os.path.join(os.path.dirname(os.path.abspath(__file__)), "review_gui.html"), f"{a.out}/index.html")
    print("wrote", a.out, len(res), "brains")


if __name__ == "__main__":
    main()
