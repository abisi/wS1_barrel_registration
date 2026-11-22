# Review page

One web page for every brain: compare all fits of a brain on one canvas, decide which to use, and click seeds for
brains that need a seeded fit. It replaces the earlier separate variant picker and seed page.

![Compare tab](images/01_compare_AB140.png)

## Opening the page

Build it (static files: `index.html`, `manifest.json`, `brains/`, `img/`), then open it in one of two ways.

```bash
python scripts/review_gui.py --variants RUN/variants [REDO/variants ...] --config RUN/config.toml --out PAGE \
    [--brains ...] [--seed-brains ...] [--sharp-seed-brains ...] [-j 4]
```

| Way | How | Where decisions and seeds are saved |
|---|---|---|
| **Local (no agent, no account)** | `python scripts/review_server.py PAGE` → open `http://localhost:8000`. On a server: run it there and `ssh -L 8000:localhost:8000 <server>`. | `PAGE/db/<collection>/<brain>.json` |
| **claude.ai page** | Claude publishes `PAGE` as a private claude.ai artifact; open the link signed in. | the page's own database on claude.ai (Claude reads it, or download it with the same one-file-per-brain layout) |

The page itself is the same in both. Opened as a plain file (`file://…`) it cannot load its data; opened from any
other web server it shows everything but cannot save (a red message says so).

- `review_server.py` needs only the Python standard library.
- `--brains` rebuilds only those brains and merges them into an existing page.
- `-j`: each worker loads a brain's `debug3d.npz` (several GB); keep it low when fits run on the same machine.

### Collections

| Collection | One document per brain | Written by |
|---|---|---|
| `picks` | `{brain, variant, decision, note, note_for, note_prev, final_at_pick, at}`; `variant` = fit id or `none`; `decision` = `use_local` / `use_global` / `redo` | Decision buttons |
| `seeds` | `{brain, points_flat_rc: {barrel: [row, col]}, n, note, image_channel, image_slab, image_mode, image_sharpen, live_mean_residual_um, at}` | Seed tab, Save |
| `requests` | `{brain, kind: seeded / sharpen, at}` | "Save and request fit", "Request sharpened refit" |
| `status` | `{brain, kind, state: fitting / ready, at}` | whoever runs the refit (shows a "fitting" tag) |

`scripts/db_to_files.py PAGE/db --picks-csv … --seeds-json …` turns `picks` and `seeds` into the inputs of the
export and the seeded fit.

---

## Layout

**Header**: progress (brains decided / all) and four filters:

![Brain list](images/03_brain_list.png)

| Filter | Shows |
|---|---|
| **To decide** | brains without a decision, or with fits newer than their decision ("new fits" tag) |
| **wS1 probe** | brains with a Neuropixels track targeted to wS1 (red dot) |
| **Seeding** | brains with seed images built (their Seed tab is enabled) |
| **All** | every brain |

Each row carries the decision (`local` / `global` / `redo`), "new fits", and "fitting" / "requested" while a refit is
pending.

**Probe banner** (top of a brain): for each track that crosses L4 inside the barrel field, the barrel it lands in under
the global registration and under each fit; "The fits disagree" when they differ.

## Compare tab

![Fits table](images/02_fits_table.png)

**Canvas.** Fit A in cyan (solid), fit B in orange (dashed), the global registration in white (dashed, "global"
checkbox), barrel names ("names"). Probe tracks are rings at their L4 crossing (red = wS1 target).
Scroll to zoom, drag to pan, "Reset view".

**Image** (background): the fit's **own slab** (the L4 depth that fit chose), or the automatic-depth images per
channel: `ch0 band`, `ch0 clean`, `ch1 band`, `ch1 clean` (band = band-passed, barrels bright; clean = background
flattened).

**Comparing two fits.** Choose A and B with the A/B buttons in the table (or keys `1`–`9` for A, `shift`+number for
B). "A + B" overlays both; "Flicker A/B" (key `f`) alternates them at the same position, which shows small shifts
better than an overlay; hold `space` to show B.

**Table.** One row per fit: Match, Contrast, LOO, Seed, Probe (see [Metrics](Metrics.md)). The best value per column
is green; failing contrast (< 0.30) is red; the small number after Match is the gain over the best automatic fit.
Tags: `seeded` / `sharpened` (refit kinds), `your pick`, `new` (made after your decision). The last row is the global
registration with its distance to the tissue barrels.

**Decision** (saved at once):

| Button | Key | Saves |
|---|---|---|
| Use A: *fit* | `p` | `use_local`, `variant` = fit A |
| Global is better | `g` | `use_global` |
| Redo | `r` | `redo` |

A note can be added for the decision. A note written for an earlier decision is shown struck through ("Earlier
note") and is not carried over to the new decision unless "Keep it" is clicked.

"Request sharpened refit" and "Seed this brain →" (when seed images exist) start the refit routes.

**How to judge a fit.** In order of weight:
1. The outlines sit on the bright barrels of the slab image, including at the edge of the field.
2. Rows are in order (A dorsal … E ventral, arcs 1… posterior to anterior) and the straddlers (Alpha–Delta) are behind
   arc 1. A fit shifted by one row can still have good numbers.
3. Several fits that differ in settings agree (same probe barrel).
4. The metrics: more matched barrels, contrast ≥ 0.3 (good fits ≥ 0.6), low LOO.
5. For seeded brains, the fit follows your clicks (`click_agreement.py`).

![Dark theme](images/08_compare_dark_AB077.png)

## Seed tab

![Seed tab](images/04_seed_MH038cmdii.png)

**Image controls** (the image is on the template flat grid, so a click is a flat-grid position):

| Control | Key | Options |
|---|---|---|
| Version | `1`–`6` | `band` (barrels bright), `L4spec` (L4 minus L2/3 and L5, for damaged brains), `pct25`, `septa` (dark septa lines), `clean`, `raw` |
| Channel | `c` | ch0, ch1, ch0+1 (geometric mean) |
| Slab | `[` `]` | `mean` (of the L4 slabs), `auto` (per region the depth with the strongest barrel texture), fixed −150 … 0 µm |
| Sharpen | `h` | off, ×1, ×2 (unsharp mask, σ 60 µm; brains built with `--sharp-seed-brains`) |

![Sharpened seed image](images/05_seed_sharpened.png)

**Clicking.** The panel names the next barrel to click (Alpha, Beta, Gamma, Delta, then A1, B1, C1, D1, E1, C2, D2, E2,
C3, E3). Click its centre. Any other barrel can be chosen from the list below; `s` skips, `u` undoes, drag a ring to
move it, ✕ deletes it.

![Seed panel](images/06_seed_panel.png)

**Preview.** From 3 seeds, green outlines show the average atlas barrel map scaled, rotated and shifted as one piece
onto the clicks (the same similarity transform the seeded fit starts from). It shows where the remaining barrels
should roughly be. Each seed's number is how far that rigid atlas lands from the click: the brain's shape differing
from the average atlas, which the fit corrects. The clicks are ground truth.

**Saving.** "Save seeds" (from 3 seeds) or "Save and request fit". Aim for 5–15 seeds spread over the whole field
(not all in one row); the straddlers and the arc-1 barrels constrain rotation and scale best.

![Seed tab, plain images](images/07_seed_AB126.png)

## Keys

| Key | Action |
|---|---|
| `1`–`9` / `shift`+number | fit A / fit B (Compare); image version (Seed) |
| `f`, hold `space` | flicker A/B, show B |
| `p`, `g`, `r` | use A, global is better, redo |
| `c`, `[` `]`, `h` | channel, slab, sharpen (Seed) |
| `s`, `u` | skip, undo (Seed) |

## Screenshots

The images on this page were taken with `docs/wiki/make_screenshots.py` (headless Chrome via Playwright on a built
page). Rerun it after changing the page.
