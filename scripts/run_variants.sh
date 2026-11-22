#!/bin/bash
# Fit a set of D2 variants for some brains (one hemisphere), each variant in its own work folder.
#
# usage: scripts/run_variants.sh SET BASE OUT [--seeds FILE] BRAINS...
#   SET    wide      wide_ch0, wide_ch1                                         (first pass, every brain)
#          lesion    wide_ch0_lesion, wide_ch1_lesion  (brains where scripts/lesion_fraction.py flags >= 3 % lesion)
#          sharpen   sharp1_ch0, sharp1_ch1, sharp2_ch0, sharp2_ch1             (blurred L4 image)
#          seeded    seeded_ch0, seeded_ch1, seeded_ch0_lesion, seeded_ch1_lesion,
#                    seeded_sharp1_ch0, seeded_sharp2_ch0, seeded_sharp2_ch1   (needs --seeds)
#   BASE   run folder holding config.toml, template.npz, barrel_index.csv, registration_params.csv
#   OUT    variants folder: fits go to OUT/<variant>/<brain>/{results,report}
#   --seeds  seeds JSON {brain: {"points_flat_rc": {barrel: [row, col]}}} (review page db/seeds -> scripts/db_to_files.py --seeds-json)
# env: PY (python, default .venv/bin/python or python), HEMI (default 1 = left), JOBS (brains in parallel per variant,
#      default 2), THREADS (default 4). Memory: 15-30 GB per running brain; variants of a set run in parallel.
# Run from the repo folder. Progress lines go to OUT/pipeline.log.
set -u
SET=$1 BASE=$2 OUT=$3; shift 3
SEEDS=""
if [ "${1:-}" = "--seeds" ]; then SEEDS=$(realpath "$2"); shift 2; fi
BR="$*"
[ -n "$BR" ] || { echo "no brains given" >&2; exit 1; }
PY=${PY:-$( [ -x .venv/bin/python ] && echo .venv/bin/python || echo python )}
HEMI=${HEMI:-1} JOBS=${JOBS:-2} THREADS=${THREADS:-4}
mkdir -p "$OUT"
log() { echo "$(date '+%F %T') $*" >> "$OUT/pipeline.log"; }

# settings shared by every variant (the ones used for the AB / MH brains, 2026-11)
export BARRELREG_CONTRAST_REFINE=1 BARRELREG_DEBUG=1 BARRELREG_PICK=contrast BARRELREG_SEARCH=wide \
  BARRELREG_OFFSETS_RANGE=-400,150 BARRELREG_FIT_OFFSETS=-350,-300,-250,-200,-150,-100,-50,0,50,100,150

fit() {  # fit LABEL CHANNEL LESION SHARPEN_AMOUNT(0 = off)
  local lab=$1 ch=$2 les=$3 sh=$4 w="$OUT/$1"
  mkdir -p "$w"; cp "$BASE/template.npz" "$BASE/barrel_index.csv" "$w/"
  head -n 1 "$BASE/registration_params.csv" > "$w/registration_params.csv"
  for b in $BR; do grep -E "^$b," "$BASE/registration_params.csv" >> "$w/registration_params.csv"; done
  local pat='s#^$##'   # brains located by a reg_dir column: point dir_pattern at it
  head -n 1 "$BASE/registration_params.csv" | grep -q 'reg_dir' && pat='s#^dir_pattern = .*#dir_pattern = "{reg_dir}"#'
  sed -e "s#^registration_params = .*#registration_params = \"$w/registration_params.csv\"#" \
      -e "s#^work_dir = .*#work_dir = \"$w\"#" -e "s#^export_dir = .*#export_dir = \"$w/export\"#" \
      -e "$pat" "$BASE/config.toml" > "$w/config.toml"
  local env=(BARRELREG_CH=$ch BARRELREG_LESION=$les)
  [ "$sh" != 0 ] && env+=(BARRELREG_SHARPEN=$sh,60)
  [ -n "$SEEDS" ] && env+=(BARRELREG_SEEDS=$SEEDS)
  env "${env[@]}" nice -n 10 $PY -m barrelreg -c "$w/config.toml" run $BR -j $JOBS --threads $THREADS \
    --hemisphere $HEMI >> "$w/run.out" 2>&1
  log "$SET $lab: $(for b in $BR; do ls "$w/$b/results/h$HEMI/metrics.json" 2>/dev/null; done | wc -l) / $(echo $BR | wc -w) fits"
}

log "$SET start: $BR"
case $SET in
  wide)     fit wide_ch0 0 0 0 & fit wide_ch1 1 0 0 & wait ;;
  lesion)   fit wide_ch0_lesion 0 1 0 & fit wide_ch1_lesion 1 1 0 & wait ;;
  sharpen)  fit sharp1_ch0 0 0 1 & fit sharp1_ch1 1 0 1 & fit sharp2_ch0 0 0 2 & fit sharp2_ch1 1 0 2 & wait ;;
  seeded)
    [ -n "$SEEDS" ] || { echo "seeded needs --seeds FILE" >&2; exit 1; }
    fit seeded_ch0 0 0 0 & fit seeded_ch1 1 0 0 & fit seeded_ch0_lesion 0 1 0 & fit seeded_ch1_lesion 1 1 0 &
    fit seeded_sharp1_ch0 0 0 1 & wait
    fit seeded_sharp2_ch0 0 0 2 & fit seeded_sharp2_ch1 1 0 2 & wait ;;
  *) echo "unknown set $SET (wide | lesion | sharpen | seeded)" >&2; exit 1 ;;
esac
log "$SET done: $BR"
