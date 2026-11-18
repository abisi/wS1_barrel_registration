#!/bin/bash
# Move a run folder written before the results/report split to one folder per brain:
#   <run>/<ID>/{h1,h2,*.tif,*_wS1_cells_ccf.csv}         -> <run>/<ID>/results/
#   <run>/<ID>/{h*/*_flat_fit.png,*_coronal.png,*_D2_overview.png}, <run>/reports/<ID>/* -> <run>/<ID>/report/
# usage: scripts/migrate_layout.sh RUN_DIR [RUN_DIR ...]   (moves only; nothing is deleted except emptied folders)
set -e
for run in "$@"; do
  for d in "$run"/*/; do
    b=$(basename "$d")
    [ -d "$d/h1" ] || [ -d "$d/h2" ] || [ -d "$run/reports/$b" ] || continue
    mkdir -p "$d/results" "$d/report"
    for h in "$d"/h1 "$d"/h2; do
      [ -d "$h" ] || continue
      mv "$h"/*_flat_fit.png "$d/report/" 2>/dev/null || true
      mv "$h" "$d/results/"
    done
    mv "$d"/*.tif "$d"/*_wS1_cells_ccf.csv "$d/results/" 2>/dev/null || true
    mv "$d"/*_coronal.png "$d"/*_D2_overview.png "$d/report/" 2>/dev/null || true
    if [ -d "$run/reports/$b" ]; then mv "$run/reports/$b"/* "$d/report/" 2>/dev/null || true; rmdir "$run/reports/$b"; fi
    # older flat layout: reports/<ID>_*.pdf and reports/flat_views/<ID>_* (a per-brain folder, moved above, wins)
    mv -n "$run/reports/${b}"_* "$run/reports/flat_views/${b}"_* "$d/report/" 2>/dev/null || true
    rmdir "$d/results" "$d/report" 2>/dev/null || true
  done
  rmdir "$run/reports/flat_views" "$run/reports" 2>/dev/null || true
done
