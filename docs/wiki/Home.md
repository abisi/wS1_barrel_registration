# wS1 barrel registration wiki

Detailed documentation for the `barrelreg` package (method D2): how to run a dataset from raw brainreg output to final
per-mouse barrel maps, how to review the fits, what every metric means, and how the method works. The
[README](../../README.md) gives the short overview, installation and commands.

These pages live in the repository under `docs/wiki/` and can be copied to the GitHub wiki unchanged (images in
`docs/wiki/images/`).

| Page | Content |
|---|---|
| [Workflow](Workflow.md) | the typical process, step by step, with the commands: first pass → review → refits (sharpen, seed) → final export |
| [Review page](Review-page.md) | the web page used to compare fits, decide and click seeds; how to open it with or without claude.ai; screenshots |
| [Metrics](Metrics.md) | every quality number: page columns, `metrics.json`, `summary.csv`, click agreement, probe barrel |
| [Methods](Methods.md) | the method in one page, and the variants (channel, lesion mask, sharpening, seeded fit) |
| [Outputs and export](Outputs-and-export.md) | files per fit, the final user-picked folder, the Anatomy folders, how to use them in analysis |
| [Troubleshooting](Troubleshooting.md) | known failure modes and what to try |

Longer background notes in `docs/`:
[D2_explained.md](../D2_explained.md) (the pipeline with figures),
[methods_explained.md](../methods_explained.md) (search, CPD, Hungarian matching, leave-one-out, thin-plate splines
from first principles), [questions_and_answers.md](../questions_and_answers.md).

## Glossary

| Term | Meaning |
|---|---|
| **brain** | one cleared brain, named by its folder (e.g. `AB140`, `MH038cmdii`). A mouse can have several brains (re-imaged or re-cleared). |
| **hemisphere** | brainglobe value 1 = left, 2 = right. The AB / MH dataset is fitted on the left hemisphere only (`--hemisphere 1`). |
| **global registration** | brainreg to the Allen / Perens atlas (whole brain). Places barrels to within about 0.5–1.5 barrels. |
| **fit / local registration** | the D2 fit of the atlas barrel map onto the barrels seen in the brain's own L4. |
| **variant** | one fit of a brain with given settings (`wide_ch0`, `sharp2_ch1`, `seeded_ch0_lesion`, …). Each lives in `<variants>/<variant>/<brain>/`. |
| **pick / decision** | per brain: `use_local` (a chosen variant), `use_global` (no fit is better than the global registration) or `redo`. |
| **seeds** | barrel centres clicked by hand on the flat L4 image; they replace the automatic search. Taken as ground truth. |
| **slab** | the thin layer of tissue at L4 depth that is flattened into the 2D L4 image. Its depth is given relative to the detected L4 bottom (µm, negative = above). |
| **flat grid** | the 2D template plane (10 µm pixels, `[row, col]`) in which the barrel map, the L4 images and the seeds live. |
