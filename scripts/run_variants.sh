#!/bin/bash
# Fit a set of D2 variants for some brains (one hemisphere), each variant in its own work folder, on every image
# channel the brains have (channel c -> variants named *_ch<c>; a brain is fitted only on channels it has).
#
# usage: scripts/run_variants.sh SET BASE OUT [--seeds FILE] BRAINS...
#   SET    wide      wide_ch<c>                                    (first pass, every brain)
#          lesion    wide_ch<c>_lesion        (brains where scripts/lesion_fraction.py flags >= 3 % lesion)
#          sharpen   sharp1_ch<c>, sharp2_ch<c>                    (blurred L4 image)
#          seeded    seeded_ch<c>, seeded_ch<c>_lesion, seeded_sharp1_ch<c>, seeded_sharp2_ch<c>   (needs --seeds)
#   BASE   run folder holding config.toml, template.npz, barrel_index.csv, registration_params.csv
#   OUT    variants folder: fits go to OUT/<variant>/<brain>/{results,report}
#   --seeds  seeds JSON {brain: {"points_flat_rc": {barrel: [row, col]}}} (review page db/seeds -> scripts/db_to_files.py --seeds-json)
# env: CHANNELS (e.g. "0 1"; default: every channel found in the brains' brainreg folders), PY (python, default
#      .venv/bin/python or python), HEMI (default 1 = left), JOBS (brains in parallel per variant, default 2),
#      THREADS (default 4). Memory: 15-30 GB per running brain; the variants of a set run in parallel.
# Every fit is also scored on every other channel (metrics.json contrast_after_ch<c>). Run from the repo folder.
# Progress lines go to OUT/pipeline.log.
set -u
SET=$1 BASE=$2 OUT=$3; shift 3
SEEDS=""
if [ "${1:-}" = "--seeds" ]; then SEEDS=$(realpath "$2"); shift 2; fi
BR="$*"
[ -n "$BR" ] || { echo "no brains given" >&2; exit 1; }
case $SET in wide|lesion|sharpen|seeded) ;; *) echo "unknown set $SET (wide | lesion | sharpen | seeded)" >&2; exit 1 ;; esac
[ "$SET" = seeded ] && [ -z "$SEEDS" ] && { echo "seeded needs --seeds FILE" >&2; exit 1; }
PY=${PY:-$( [ -x .venv/bin/python ] && echo .venv/bin/python || echo python )}
HEMI=${HEMI:-1} JOBS=${JOBS:-2} THREADS=${THREADS:-4}
mkdir -p "$OUT"
log() { echo "$(date '+%F %T') $*" >> "$OUT/pipeline.log"; }

# settings shared by every variant (the ones used for the AB / MH brains, 2026-11)
export BARRELREG_CONTRAST_REFINE=1 BARRELREG_DEBUG=1 BARRELREG_PICK=contrast BARRELREG_SEARCH=wide \
  BARRELREG_OFFSETS_RANGE=-400,150 BARRELREG_FIT_OFFSETS=-350,-300,-250,-200,-150,-100,-50,0,50,100,150

mkcfg() {  # mkcfg DIR BRAINS...: config.toml + registration_params.csv (these brains) + template in DIR
  local w=$1; shift; mkdir -p "$w"; cp "$BASE/template.npz" "$BASE/barrel_index.csv" "$w/"
  head -n 1 "$BASE/registration_params.csv" > "$w/registration_params.csv"
  for b in "$@"; do grep -E "^$b," "$BASE/registration_params.csv" >> "$w/registration_params.csv"; done
  local pat='s#^$##'   # brains located by a reg_dir column: point dir_pattern at it
  head -n 1 "$BASE/registration_params.csv" | grep -q 'reg_dir' && pat='s#^dir_pattern = .*#dir_pattern = "{reg_dir}"#'
  sed -e "s#^registration_params = .*#registration_params = \"$w/registration_params.csv\"#" \
      -e "s#^work_dir = .*#work_dir = \"$w\"#" -e "s#^export_dir = .*#export_dir = \"$w/export\"#" \
      -e "$pat" "$BASE/config.toml" > "$w/config.toml"
}

# channels of each brain: "<brain> <c> <c> ..."
mkcfg "$OUT/.channels" $BR
CHMAP=$($PY - "$OUT/.channels/config.toml" $BR <<'EOF'
import sys
from barrelreg.config import Config
cfg = Config(sys.argv[1])
for b in sys.argv[2:]:
    print(b, *cfg.channels(cfg.reg_dir(b)))
EOF
)
CHANNELS=${CHANNELS:-$(echo "$CHMAP" | cut -s -d' ' -f2- | tr ' ' '\n' | grep -E '^[0-9]+$' | sort -nu | xargs)}
with_ch() { echo "$CHMAP" | awk -v c="$1" '{for (i = 2; i <= NF; i++) if ($i == c) {print $1; break}}' | xargs; }
log "$SET start: $BR (channels: $CHANNELS)"

fit() {  # fit LABEL CHANNEL LESION SHARPEN_AMOUNT(0 = off): the brains of BR that have CHANNEL
  local lab=$1 ch=$2 les=$3 sh=$4 w="$OUT/$1" br
  br=$(with_ch "$ch")
  [ -n "$br" ] || { log "$SET $lab: no brain has channel $ch"; return; }
  mkcfg "$w" $br
  local env=(BARRELREG_CH=$ch BARRELREG_LESION=$les)
  [ "$sh" != 0 ] && env+=(BARRELREG_SHARPEN=$sh,60)
  [ -n "$SEEDS" ] && env+=(BARRELREG_SEEDS=$SEEDS)
  env "${env[@]}" nice -n 10 $PY -m barrelreg -c "$w/config.toml" run $br -j $JOBS --threads $THREADS \
    --hemisphere $HEMI >> "$w/run.out" 2>&1
  log "$SET $lab: $(for b in $br; do ls "$w/$b/results/h$HEMI/metrics.json" 2>/dev/null; done | wc -l) / $(echo $br | wc -w) fits"
}

for c in $CHANNELS; do
  case $SET in
    wide)    fit wide_ch$c $c 0 0 & ;;
    lesion)  fit wide_ch${c}_lesion $c 1 0 & ;;
    sharpen) fit sharp1_ch$c $c 0 1 & fit sharp2_ch$c $c 0 2 & ;;
    seeded)  fit seeded_ch$c $c 0 0 & fit seeded_ch${c}_lesion $c 1 0 & ;;
  esac
done
wait
if [ "$SET" = seeded ]; then   # second wave: sharpened seeded fits
  for c in $CHANNELS; do fit seeded_sharp1_ch$c $c 0 1 & fit seeded_sharp2_ch$c $c 0 2 & done
  wait
fi
log "$SET done: $BR"
