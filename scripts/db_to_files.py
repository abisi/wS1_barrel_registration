"""Turn the review page's saved decisions and seeds into the files the fit and export scripts read.

usage: python scripts/db_to_files.py <db_dir> [--picks-csv user_picks_<date>.csv] [--seeds-json seeds.json]
                                     [--seed-brains B1 B2 ...]

<db_dir> holds picks/<brain>.json and seeds/<brain>.json: the db/ folder written by review_server.py, or a download
of the claude.ai page database (same one-file-per-brain layout).
--picks-csv   brain, variant, decision, note, picked_at  -> scripts/export_user_picks.py --picks
--seeds-json  {brain: {"points_flat_rc": {barrel: [row, col]}, ...}}  -> scripts/run_variants.sh seeded --seeds
              (all seeded brains, or only --seed-brains)
"""
import argparse
import csv
import glob
import json
import os


def docs(d):
    out = {}
    for f in sorted(glob.glob(os.path.join(d, "*.json"))):
        x = json.load(open(f))
        out[x.get("brain") or os.path.basename(f)[:-5]] = x
    return out


def decision(p):
    """decision of a pick doc; older docs have no `decision` field (variant "none" = global, note says redo)"""
    if p.get("decision"):
        return p["decision"]
    if p.get("variant") not in (None, "", "none"):
        return "use_local"
    return "redo" if any(w in (p.get("note") or "").lower() for w in ("redo", "redone", "sharpen")) else "use_global"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("db_dir")
    ap.add_argument("--picks-csv")
    ap.add_argument("--seeds-json")
    ap.add_argument("--seed-brains", nargs="*")
    a = ap.parse_args()
    if a.picks_csv:
        picks = docs(os.path.join(a.db_dir, "picks"))
        with open(a.picks_csv, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["brain", "variant", "decision", "note", "picked_at"])
            for b, p in sorted(picks.items()):
                dec = decision(p)
                key = p["variant"] if dec == "use_local" else dec
                # a note written for an earlier decision does not describe this one
                note = (p.get("note") or "").strip() if p.get("note_for") in (None, key) else ""
                if dec == "use_local" and "global is better" in note.lower():  # carried over from a global pick
                    note = ""
                w.writerow([b, p.get("variant", ""), dec, note, p.get("at", "")])
        print(f"{len(picks)} picks -> {a.picks_csv}")
    if a.seeds_json:
        seeds = docs(os.path.join(a.db_dir, "seeds"))
        keep = {b: {k: s[k] for k in ("points_flat_rc", "image_channel", "image_slab", "image_mode", "image_sharpen", "at")
                    if k in s}
                for b, s in seeds.items() if not a.seed_brains or b in a.seed_brains}
        json.dump(keep, open(a.seeds_json, "w"), indent=1)
        print(f"{len(keep)} seeded brains -> {a.seeds_json}")


if __name__ == "__main__":
    main()
