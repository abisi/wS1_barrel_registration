# Troubleshooting

## Fits

| Symptom | Likely cause | Try |
|---|---|---|
| contrast < 0.30 on every variant | barrel field not found: rotated beyond ±30°, scale off, or hidden | seeded fit |
| fits disagree on the probe barrel by whole rows / arcs; outlines regular but shifted | naming slip of the search (the grid snaps onto the wrong blobs) | flicker the fits on the slab image; seeded fit |
| outlines loose, barrels soft in the slab image | blurred L4 | sharpened variants (`sharpen` set), then seed on the sharpened images if still off |
| a hole in the field, outlines pulled into it | lesion | `lesion` set (`*_lesion`) |
| barrels clearer in one channel | autofluorescence vs other channel | compare `ch0` / `ch1` rows; the page's "ch1 band" background |
| L4 slab shows L2/3 texture (fine, not barrel-sized) | slab too shallow | check `l4_offset_um` and `depth_fits`; the `L4spec` seed image helps to see L4 alone |
| `surface_bridged_frac` > 0.1 | notched / torn surface | check the coronal figure; columns under wide damage start lower |
| seeded fit drifts off some clicks | matching pulled those barrels to other blobs | check with `click_agreement.py`; more seeds near the drifted ones |

## Review page

| Symptom | Fix |
|---|---|
| red message "can't be saved in this view" | open it with `scripts/review_server.py PAGE` (or as the claude.ai page) |
| page blank, "brain list did not load" | opened as `file://`: use the server |
| Seed tab greyed out | no seed images for that brain: rebuild with `--brains B --seed-brains B` |
| Sharpen buttons missing | rebuild with `--sharp-seed-brains B` |
| "new fits" tag stays after deciding | the tag compares fit times with the decision time: decide again after the refit |
| build runs out of memory | lower `-j` (each worker loads a `debug3d.npz` of several GB) |

## Compute (haas)

| Symptom | Fix |
|---|---|
| fits missing after a batch (`n / m fits` in `pipeline.log`) | out of memory: 15–30 GB per running brain; rerun the missing brains with fewer in parallel (`JOBS`) |
| root disk full | write runs and pages on the NAS (`/mnt/lsens-analysis/...`) |
| `pkill -f <pattern>` kills your ssh session | the pattern appears in your own command line: kill by PID |
| editing a running shell script has no effect | bash read it at start (`sed -i` makes a new file): stop it, edit, restart |
