# Workflow: from brainreg output to final barrel maps

The typical process for a dataset. It needs no agent: every step is a script in this repository. Commands are run
from the repository folder with the package's Python (`.venv/bin/python`, written `python` below).

```
 0 inputs        brainreg per brain + registration_params.csv
 1 setup         convert-flatmap, build-template              (once)
 2 first pass    wide ch0 / ch1 for every brain, lesion variants where needed, automatic selection
 3 review        review page: decide per brain  use_local (pick a fit) / use_global / redo
 4 refits        brains marked redo or global-but-fixable: sharpened fits, then seeded fits
 5 review again  pick among the new fits (repeat 3-5 until every brain has a decision)
 6 export        final project folder + each mouse's Anatomy folders
 7 analysis      barrel / layer / CCF per unit, cells, figures
```

Rule of thumb from the AB / MH dataset (101 brains, left hemisphere): the first pass gave a usable fit for about
three quarters of the brains. Of the rest, sharpening rescued blurred brains, seeding rescued brains whose barrel field
lies outside the search range or is hidden by damage, and about 20 brains stayed on the global registration.

---

## 0. Inputs

Per brain, a brainreg output folder (registered atlas, hemispheres, deformation fields, downsampled ch0 / ch1). The
list of brains is `registration_params.csv`: one row per brain, column `brain` plus whatever `dir_pattern` in the
config needs. A `reg_dir` column with the full path is the simplest (`run_variants.sh` then points `dir_pattern` at it).

```csv
brain,group,reg_dir,source
AB077,AB,/raid0/lmsmith/AB_brains/AB077/fused/registered,lana_raid0
```

See the README ("Inputs", "Configure") for the atlas files and config keys.

## 1. Setup (once per base folder)

A **base folder** holds `config.toml`, `template.npz`, `barrel_index.csv` and `registration_params.csv`; every
variant run copies them.

```bash
python -m barrelreg -c BASE/config.toml convert-flatmap    # ~5 min, ~10 GB RAM
python -m barrelreg -c BASE/config.toml build-template     # ~2 min -> template.npz, barrel_index.csv
```

The template does not depend on the brains: an existing `template.npz` can be copied.

## 2. First pass

```bash
ALL=$(tail -n +2 BASE/registration_params.csv | cut -d, -f1 | xargs)
JOBS=12 THREADS=2 scripts/run_variants.sh wide BASE RUN/variants $ALL          # wide_ch0, wide_ch1
python scripts/lesion_fraction.py RUN/variants/wide_ch0 RUN/variants/wide_ch1 --csv RUN/lesion_fraction.csv
scripts/run_variants.sh lesion BASE RUN/variants <brains it lists>              # wide_ch0_lesion, wide_ch1_lesion
python scripts/build_final.py --out RUN --score contrast_both \
    --runs wide_ch0=RUN/variants/wide_ch0 wide_ch1=RUN/variants/wide_ch1 \
           wide_ch0_lesion=RUN/variants/wide_ch0_lesion wide_ch1_lesion=RUN/variants/wide_ch1_lesion
python -m barrelreg -c RUN/config.toml summarize                               # summary.csv, contact sheet
```

- `wide`: the widened search (rotation ±30°, scale 0.6–1.5, shift ±800 µm), fitted once on each image channel
  (ch0 = autofluorescence, ch1 = the other channel, often clearer barrels).
- `lesion`: the same with large lesions (and their rims) masked, for brains where the mask removes ≥ 3 % of the
  barrel field (`lesion_fraction.py`).
- All variants share the settings in `run_variants.sh`: L4 slab fitted every 50 µm from −350 to +150 µm around the
  L4 bottom, the depth with the best barrel contrast kept, contrast refinement after the matched fit, debug dumps on
  (needed by the review page).
- `build_final.py` makes an **automatic** choice per brain (highest mean ch0/ch1 contrast). It is a starting point;
  the review decides.
- Time and memory: about 9 minutes per hemisphere per variant; 15–30 GB RAM per running brain. Choose `JOBS`
  accordingly (on a 500 GB machine, about 12 brains at once).

## 3. Review: build the page and decide

```bash
python scripts/review_gui.py --variants RUN/variants --config RUN/config.toml --out PAGE -j 4
python scripts/review_server.py PAGE            # then open http://localhost:8000
```

On a remote machine run both there and forward the port: `ssh -L 8000:localhost:8000 <server>`.
See [Review page](Review-page.md) for how to use it. For each brain:

- **Use A**: one fit is right (barrels sit on the bright L4 barrels, rows in order, the probe lands where expected).
- **Global is better**: no fit beats the global registration.
- **Redo**: worth another try; write in the note what you see (blurred, rotated, lesion…).

Decisions are saved as `PAGE/db/picks/<brain>.json`.

## 4. Refits

Two fallbacks, in this order.

**Sharpened fits** — for blurred L4 images (barrels visible but soft, fit loose). An unsharp mask (σ 60 µm,
amount 1 and 2) is applied to both channels before everything else:

```bash
scripts/run_variants.sh sharpen BASE REDO/variants MH015 MH036       # sharp1_ch0, sharp1_ch1, sharp2_ch0, sharp2_ch1
```

On MH015 this gave 29 vs 24 matched barrels and a lower leave-one-out error, and the fit was judged better by eye.

**Seeded fits** — when the automatic search cannot find the barrel field (rotated beyond ±30°, field partly hidden
by damage or puncta, or the fits disagree by whole rows). You click a few barrel centres; the search is replaced by
the similarity transform through your clicks, and matching and warping run as usual.

1. Build seed images for those brains (add `--sharp-seed-brains` for sharpened versions, off / ×1 / ×2):
   ```bash
   python scripts/review_gui.py --variants RUN/variants REDO/variants --config RUN/config.toml --out PAGE \
       --brains AB136 AB140 --seed-brains AB136 AB140 --sharp-seed-brains MH038
   ```
2. Click seeds in the page's **Seed** tab (5–15 barrels spread over the field; the straddlers Alpha–Delta and the
   arc-1 barrels first). Save.
3. Convert and fit:
   ```bash
   python scripts/db_to_files.py PAGE/db --seeds-json REDO/seeds_batch3.json --seed-brains AB136 AB140
   scripts/run_variants.sh seeded BASE REDO/variants --seeds REDO/seeds_batch3.json AB136 AB140
   ```
   Seven variants per brain: ch0 / ch1, with and without lesion mask, and sharpened ×1 / ×2.
4. Check how well each fit follows your clicks (clicks are ground truth):
   ```bash
   python scripts/click_agreement.py --seeds REDO/seeds_batch3.json --variants RUN/variants REDO/variants
   ```

Rule from the AB / MH dataset: seeded fits beat the automatic fits on every seeded brain (7 of 7, 6–13 of the
clicked barrels inside the right fitted barrel, median 26–78 µm from the clicks, versus 400–1200 µm for most
automatic fits).

## 5. Review again

Rebuild the page for the refit brains (`--brains` merges them into an existing page, keeping all other brains) and
decide. Brains with fits newer than their decision appear under **To decide** with a "new fits" tag.

```bash
python scripts/review_gui.py --variants RUN/variants REDO/variants --config RUN/config.toml --out PAGE \
    --brains AB136 AB140 --seed-brains AB136 AB140
```

## 6. Export

```bash
python scripts/db_to_files.py PAGE/db --picks-csv FINAL/user_picks_$(date +%Y%m%d).csv
python scripts/export_user_picks.py --run RUN --variants REDO/variants --picks FINAL/user_picks_<date>.csv \
    --out FINAL --data-root /mnt/lsens-analysis/Axel_Bisi/data --dry-run      # check the table first
python scripts/export_user_picks.py --run RUN --variants REDO/variants --picks FINAL/user_picks_<date>.csv \
    --out FINAL --data-root /mnt/lsens-analysis/Axel_Bisi/data
```

This writes the project folder `FINAL/<brain>/` for every brain and, for every mouse, one brain into each of its
session folders: `<data-root>/<mouse>/<session>/Anatomy/wS1_barrel_registration/`. Details in
[Outputs and export](Outputs-and-export.md).

## 7. Analysis

Per unit: look up its CCF / sample-space position in the exported `barrels_corrected_full.tif` (barrel name) and
`ccf_field_*.tif` (corrected CCF position). Brains on `use_global` keep the global (brainreg) labels;
`registration_info.json` says which applies. See [Outputs and export](Outputs-and-export.md#using-the-results).

---

## The AB / MH dataset (lab paths)

| What | Where |
|---|---|
| project folder | `/mnt/lsens-analysis/Axel_Bisi/combined_results_ks4/ws1-barrel-registration/` |
| first pass (RUN) | `D2_all_brains/` (variants in `D2_all_brains/variants/`) |
| refits (REDO) | `D2_redo_20261120/variants/`, `D2_redo_20261121/variants/` (seeds in `D2_redo_20261121/seeds_batch*.json`) |
| review page (PAGE) | `review_gui/` (published copy on claude.ai) |
| final export (FINAL) | `D2_final_user_picked/` |
| per-mouse copies | `/mnt/lsens-analysis/Axel_Bisi/data/<mouse>/<session>/Anatomy/wS1_barrel_registration/` |
| probe insertion table | `/raid0/lmsmith/local_barrel_reg/D2_neuropixels/probe_insertion_info_from_axel.csv` |
| compute | haas (500 GB RAM, root disk nearly full: outputs on the NAS) |
