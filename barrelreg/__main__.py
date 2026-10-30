"""Command line:  python -m barrelreg -c config.toml <step> [brains ...]

steps (in order):
  convert-flatmap   flatmap_float.nrrd -> cropped .npy in work_dir                      (once)
  build-template    2D barrel template + inverse flatmap -> work_dir/template.npz      (once)
  run               register brains (default: every brain in registration_params)
  summarize         summary.csv + contact_sheet_D2.png over the brains
  export            overview figure, cells -> CCF, optional copy (--copy)
"""
import argparse
import os
import subprocess
import sys
import time

from .config import Config


def main(argv=None):
    ap = argparse.ArgumentParser(prog="barrelreg", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-c", "--config", default="config.toml")
    ap.add_argument("step", choices=["convert-flatmap", "build-template", "run", "summarize", "export"])
    ap.add_argument("brains", nargs="*", help="brain IDs (default: all in registration_params)")
    ap.add_argument("-j", "--jobs", type=int, default=1, help="run: brains in parallel (one process each)")
    ap.add_argument("--threads", type=int, default=5, help="run with -j: OMP threads per process")
    ap.add_argument("--hemisphere", type=int, choices=[1, 2], help="run: only this hemisphere (1 left, 2 right)")
    ap.add_argument("--no-cells", action="store_true", help="export: figures only, no cell table")
    ap.add_argument("--copy", action="store_true", help="export: also copy results to paths.export_dir")
    a = ap.parse_args(argv)
    cfg = Config(a.config)
    os.makedirs(cfg.work_dir, exist_ok=True)
    ids = a.brains or list(cfg.params.index)

    if a.step == "convert-flatmap":
        from .flatmap import convert_flatmap
        convert_flatmap(cfg.flatmap_nrrd, cfg.annotation, cfg.flatmap_crop, cfg.flatmap_crop_lo)
    elif a.step == "build-template":
        from .template import build
        build(cfg)
    elif a.step == "run" and a.jobs > 1:
        # one process per brain, a.jobs at a time; logs in work_dir/logs/<ID>.log
        os.makedirs(f"{cfg.work_dir}/logs", exist_ok=True)
        env = dict(os.environ, OMP_NUM_THREADS=str(a.threads))
        extra = ["--hemisphere", str(a.hemisphere)] if a.hemisphere else []
        procs = []
        for b in ids:
            while sum(p.poll() is None for p in procs) >= a.jobs:
                time.sleep(5)
            log = open(f"{cfg.work_dir}/logs/{b}.log", "w")
            procs.append(subprocess.Popen([sys.executable, "-m", "barrelreg", "-c", a.config, "run", b] + extra,
                                          stdout=log, stderr=subprocess.STDOUT, env=env))
        for p in procs:
            p.wait()
        print(f"done; logs in {cfg.work_dir}/logs/")
    elif a.step == "run":
        from .register import run
        for b in ids:
            try:
                run(cfg, b, (a.hemisphere,) if a.hemisphere else (1, 2))
            except Exception as e:  # keep going over brains
                import traceback
                traceback.print_exc()
                print(f"{b}: FAILED {e}", flush=True)
    elif a.step == "summarize":
        from .summarize import summarize
        summarize(cfg, ids)
    elif a.step == "export":
        from .export import combine, export
        summs = []
        for b in ids:
            try:
                summs.append(export(cfg, b, do_cells=not a.no_cells, copy_to=cfg.export_dir if a.copy else None))
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"{b}: FAILED {e}", flush=True)
        if not a.no_cells:
            combine(cfg, ids, summs)


if __name__ == "__main__":
    main()
