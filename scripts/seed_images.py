"""Images for the barrel seed page (fallback when the automatic D2 fit fails): flat L4 slabs of a brain on the full
template flat grid, so a click maps directly to a flat pixel (row, col) of the grid used by register.py
(BARRELREG_SEEDS). See README "Seeded fit (fallback)".

Per brain, per channel (the wide_ch0 / wide_ch1 debug.npz of a previous run; plus ch0+1 = geometric mean of both), per
slab (single depths SLABS of the 25 um stack; mean = average of the MEAN_RANGE slabs; auto = depth-adaptive: per region
the depth in AUTO_RANGE with the strongest barrel-scale texture, for brains whose L4 surface is off / tilted), renderings (all but raw: bright puncta < ~70 um removed by a
grey opening, log-flattened, darkest 2 % (holes / tears) masked black, locally equalised):
  band    band-passed at the barrel scale (Gaussian 20 um - 120 um): barrels bright, septa dark
  L4spec  band of (slab - mean of the L2/3 and L5 slabs), z-scored: keeps what is specific to L4 (barrels), removes
          vessels / puncta / damage running through all layers (AB139: the clearest lattice)
  pct25   band of a local 25th-percentile filter (80 um disk): stronger puncta suppression
  septa   black-hat (grey closing, 120 um - image), inverted: the dark septa network as dark lines
  clean   despeckled slab only
  raw     flattened slab (ratio to a 300 um Gaussian), 1-99.5 percentile
Compared on AB139 / AB145 (2026-11-20): Fourier ring (200-350 um), channel ratios and median-over-slabs did not help.
Coarser band-passes (40-250 um) were also tested (2026-11-20): barrels in the flat grid are ~150 um, the 20-120 um band
already matches them (AB080), coarser scales only blur. Plus a transparent overlay with the brainreg (identity) atlas
barrels. Writes one JPEG tile sheet per brain and channel <out>/img/<brain>__ch<c>.jpg (rows = MODES, cols = slabs, native
10 um px; manifest images[] give the tile origin tx, ty), <brain>__atlas.png and <out>/seed_manifest.json (page canvas
at SCALE x: row = y / SCALE, col = x / SCALE).

usage: python scripts/seed_images.py --variants DIR --brains AB139 [...] --out DIR [--append]
"""
import argparse
import json
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage
from skimage import exposure
from skimage.segmentation import find_boundaries

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from barrelreg.atlas import BARREL_NAMES, SEPTA
from brain_report import flatten

SCALE = 2
SLABS = (-150, -100, -50, 0)  # single slabs shown (um from the L4 bottom, 25 um depth stack of debug.npz)
OTHER = ((-400, -250), (50, 150))  # L2/3 and L5 slabs for L4spec
MEAN_RANGE = (-150, 0)
AUTO_RANGE = (-250, 50)  # depths the auto slab may pick from
MODES = ["band", "L4spec", "pct25", "septa", "clean", "raw"]
RUNS = {"wide_ch0": 0, "wide_ch1": 1}


def stretch(g, valid, lo_hi=(1, 99.5)):
    lo, hi = np.percentile(g[valid], lo_hi)
    return np.clip((np.where(valid, g, lo) - lo) / max(hi - lo, 1e-9), 0, 1)


def save(g, valid, f):
    g = np.where(valid, g * 255, 0).astype(np.uint8)
    Image.fromarray(g).resize((g.shape[1] * SCALE, g.shape[0] * SCALE), Image.BILINEAR).save(f)


def fl(x, v):
    return np.log(np.maximum(flatten(x, v), 1e-3))


def despeck(x, v):
    """bright puncta < ~70 um removed (grey opening), then a 3 px median"""
    x = np.where(v, x, np.median(x[v]))
    return ndimage.median_filter(ndimage.grey_opening(x, footprint=disk(3)), 3)


def holes(x, v):
    """conservative damage mask: the darkest 2 % of the despeckled, flattened slab, opened (no specks) and dilated"""
    h = v & (x < np.percentile(x[v], 2))
    return ndimage.binary_dilation(ndimage.binary_opening(h, iterations=2), iterations=3)


def band(x, v, s1=2, s2=12):
    w = v.astype(np.float32)
    xx = np.where(v, x, 0)
    return (ndimage.gaussian_filter(xx, s1) / np.maximum(ndimage.gaussian_filter(w, s1), 1e-3)
            - ndimage.gaussian_filter(xx, s2) / np.maximum(ndimage.gaussian_filter(w, s2), 1e-3))


def zs(x, v):
    return np.where(v, (x - x[v].mean()) / max(x[v].std(), 1e-9), 0)


def eq(x, v):
    return exposure.equalize_adapthist(stretch(x, v, (2, 98)), kernel_size=40, clip_limit=0.015)


def render(img, valid, other=None):
    """{mode: (image 0-1, valid)}. other: z-scored, despeckled, flattened mean of the L2/3 and L5 slabs (or None)."""
    out = {"raw": (stretch(flatten(img, valid), valid), valid)}
    x = fl(despeck(img, valid), valid)
    v = valid & ~holes(x, valid)
    out["clean"] = (eq(x, v), v)
    out["band"] = (eq(band(x, v), v), v)
    p25 = ndimage.percentile_filter(np.where(valid, img, np.median(img[valid])), 25, footprint=disk(4))
    out["pct25"] = (eq(band(fl(p25, valid), v), v), v)
    if other is not None:
        out["L4spec"] = (eq(band(zs(x, v) - other, v), v), v)
    out["septa"] = (eq(-ndimage.gaussian_filter(ndimage.grey_closing(x, footprint=disk(6)) - x, 1.5), v), v)
    return out


def disk(r):
    y, x = np.mgrid[-r:r + 1, -r:r + 1]
    return x * x + y * y <= r * r


def atlas_png(cols, f):
    up = np.kron(cols, np.ones((SCALE, SCALE), cols.dtype))
    b = find_boundaries(up, mode="inner") & (up > 0)
    rgba = np.zeros(up.shape + (4,), np.uint8)
    rgba[b] = (255, 220, 0, 200)
    im = Image.fromarray(rgba)
    d = ImageDraw.Draw(im)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 13)
    except OSError:
        font = ImageFont.load_default()
    for k in range(1, SEPTA):
        m = cols == k
        if m.any():
            cy, cx = ndimage.center_of_mass(m)
            d.text((cx * SCALE, cy * SCALE), BARREL_NAMES[k - 1], fill=(255, 220, 0, 230), font=font, anchor="mm")
    im.save(f)


def stack(f):
    d = np.load(f)
    return d, d["offsets"].astype(int), d["stack_img"], d["stack_valid"].astype(bool)


def auto_slab(offs, S, Vd):
    """depth-adaptive slab: per region, the depth (AUTO_RANGE) whose barrel-scale band energy (300 um window) is highest
    (soft-max over depths, energies normalised per depth), applied to the flattened slabs"""
    k = (offs >= AUTO_RANGE[0]) & (offs <= AUTO_RANGE[1])
    v = Vd[k].all(0)
    F, E = [], []
    for img in S[k]:
        x = fl(despeck(img, v), v)
        b = zs(band(x, v), v)
        E.append(np.sqrt(ndimage.gaussian_filter(b ** 2, 15)))
        F.append(flatten(img, v))
    E = np.array(E)
    E /= E[:, v].mean(1)[:, None, None]
    w = np.exp((E - E.max(0)) / 0.15)
    w /= w.sum(0)
    return (w * np.array(F)).sum(0), v, (w * offs[k][:, None, None]).sum(0)


def slabs_of(offs, S, Vd):
    """{slab name: (image, valid)} and the L2/3 + L5 reference for L4spec"""
    out = {}
    m = (offs >= MEAN_RANGE[0]) & (offs <= MEAN_RANGE[1])
    out["mean"] = (np.mean([flatten(S[i], Vd[m].all(0)) for i in np.where(m)[0]], 0), Vd[m].all(0))
    a, va, dep = auto_slab(offs, S, Vd)
    out["auto"] = (a, va)
    for o in SLABS:
        i = int(np.where(offs == o)[0][0])
        out[str(o)] = (S[i], Vd[i])
    oth = [i for i, o in enumerate(offs) if OTHER[0][0] <= o <= OTHER[0][1] or OTHER[1][0] <= o <= OTHER[1][1]]
    vo = Vd.all(0)
    other = np.mean([zs(fl(despeck(S[i], vo), vo), vo) for i in oth], 0) if oth else None
    return out, other, dep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", required=True)
    ap.add_argument("--brains", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--append", action="store_true", help="keep the other brains already in the manifest")
    a = ap.parse_args()
    os.makedirs(f"{a.out}/img", exist_ok=True)
    mf = f"{a.out}/seed_manifest.json"
    man = json.load(open(mf)) if a.append and os.path.exists(mf) else dict(brains=[])
    man.update(scale=SCALE, pix_um=10.0, barrels=BARREL_NAMES, modes=MODES, sprite=True,
               channel_names={"0": "ch0", "1": "ch1", "2": "ch0+1"})
    man["brains"] = [b for b in man["brains"] if b["brain"] not in a.brains]
    for b in a.brains:
        fs = {ch: f"{a.variants}/{run}/{b}/results/h1/debug.npz" for run, ch in RUNS.items()}
        fs = {ch: f for ch, f in fs.items() if os.path.exists(f)}
        if not fs:
            print(b, "no debug.npz", flush=True)
            continue
        per = {}
        for ch, f in fs.items():
            d, offs, S, Vd = stack(f)
            per[ch] = slabs_of(offs, S, Vd)
        cols = d["cols"]
        H, W = cols.shape
        entry = dict(brain=b, images=[], atlas=f"img/{b}__atlas.png", shape=[int(H), int(W)], tile=[int(H), int(W)])
        atlas_png(cols, f"{a.out}/{entry['atlas']}")
        if 0 in per and 1 in per:  # ch0+1: geometric mean of the flattened images, L4spec references averaged
            sl = {k: (np.exp(0.5 * (np.log(np.maximum(per[0][0][k][0] / np.median(per[0][0][k][0][per[0][0][k][1]]), 1e-3))
                                      + np.log(np.maximum(per[1][0][k][0] / np.median(per[1][0][k][0][per[1][0][k][1]]), 1e-3)))),
                      per[0][0][k][1] & per[1][0][k][1]) for k in per[0][0]}
            oth = None if per[0][1] is None or per[1][1] is None else 0.5 * (per[0][1] + per[1][1])
            per[2] = (sl, oth, None)
        slab_names = ["mean", "auto"] + [str(o) for o in SLABS]
        for ch, (sl, other, _) in per.items():
            sheet = np.zeros((H * len(MODES), W * len(slab_names)), np.uint8)
            for j, sname in enumerate(slab_names):
                img, valid = sl[sname]
                r = render(img, valid, other)
                for i, mode in enumerate(MODES):
                    if mode not in r:
                        continue
                    g, v = r[mode]
                    sheet[i * H:(i + 1) * H, j * W:(j + 1) * W] = np.where(v, g * 255, 0).astype(np.uint8)
                    entry["images"].append(dict(src=f"img/{b}__ch{ch}.jpg", channel=ch, slab=sname, mode=mode,
                                                tx=j * W, ty=i * H))
            Image.fromarray(sheet).save(f"{a.out}/img/{b}__ch{ch}.jpg", quality=92)
        entry["slabs"] = slab_names
        entry["mean_range_um"] = list(MEAN_RANGE)
        entry["auto_range_um"] = list(AUTO_RANGE)
        man["brains"].append(entry)
        print(b, len(entry["images"]), "tiles", flush=True)
    man["brains"].sort(key=lambda e: e["brain"])
    json.dump(man, open(mf, "w"), indent=1)


if __name__ == "__main__":
    main()
