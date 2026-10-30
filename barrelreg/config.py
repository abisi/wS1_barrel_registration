"""Paths and per-brain registration lookup, read from a TOML file (see config.example.toml)."""
import glob
import os
import tomllib

import numpy as np
import pandas as pd
import tifffile


class Config:
    def __init__(self, path):
        with open(path, "rb") as f:
            c = tomllib.load(f)
        base = os.path.dirname(os.path.abspath(path))
        p = c["paths"]
        fmt = dict(data_root=p["data_root"])
        self.atlas_dir = p["atlas_dir"]
        self.flatmap_nrrd = p["flatmap_nrrd"]
        # annotation of the atlas the coarse registration went to (older configs: perens_annotation)
        self.registration_annotation = p.get("registration_annotation", p.get("perens_annotation"))
        self.data_root = p["data_root"]
        self.work_dir = p["work_dir"]
        self.cells_dir = p["cells_dir"].format(**fmt)
        self.export_dir = p["export_dir"].format(**fmt)
        params = p["registration_params"]
        self.registration_params = params if os.path.isabs(params) else os.path.join(base, params)
        r = c["registration"]
        self.dir_pattern = r["dir_pattern"]
        self.channel_glob = r["channel_glob"]
        self.raw_ch0_glob = r["raw_ch0_glob"]
        self.voxel_um = float(r["voxel_um"])
        self.registration_atlas_um = float(r.get("registration_atlas_um", 20.0))
        self.params = pd.read_csv(self.registration_params, dtype=str).set_index("brain")
        self._template = None

    # ------------------------------------------------------------------ files
    @property
    def flatmap_crop(self):
        return f"{self.work_dir}/flatmap_atlas_crop.npy"

    @property
    def flatmap_crop_lo(self):
        return f"{self.work_dir}/flatmap_atlas_crop_lo.npy"

    @property
    def template_file(self):
        return f"{self.work_dir}/template.npz"

    @property
    def annotation(self):
        return f"{self.atlas_dir}/annotation.tiff"

    def brain_dir(self, bid):
        return f"{self.work_dir}/{bid}"

    def reg_dir(self, bid):
        """brainreg output folder of this brain (its row in registration_params fills the pattern)."""
        return self.dir_pattern.format(data_root=self.data_root, brain=bid, **self.params.loc[bid].to_dict())

    def ds_file(self, reg, ch):
        return [f for f in glob.glob(f"{reg}/{self.channel_glob.format(ch=ch)}") if "standard" not in f][0]

    def raw_ch0(self, bid):
        return glob.glob(self.raw_ch0_glob.format(data_root=self.data_root, brain=bid))[0]

    def registration_atlas(self):
        """Annotation of the coarse-registration atlas (.npy or .tiff), memory-mapped."""
        f = self.registration_annotation
        return np.load(f, mmap_mode="r") if f.endswith(".npy") else tifffile.memmap(f)

    def template(self):
        if self._template is None:
            self._template = dict(np.load(self.template_file))
        return self._template
