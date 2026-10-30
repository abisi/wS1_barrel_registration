"""Structure tables of allen_mouse_bluebrain_barrels_10um (copy shipped in data/structures.csv).

Barrel index 1..33 = BARREL_NAMES order (A1..E8, then Alpha, Beta, Delta, Gamma); 0 = not a barrel; SEPTA = 34.
Every descendant of SSp-bfd-<barrel> (its layers) collapses onto that barrel.
"""
import os

import numpy as np
import pandas as pd

S = pd.read_csv(os.path.join(os.path.dirname(__file__), "data", "structures.csv"))

BARREL_NAMES = sorted({a.split("-")[2] for a in S.acronym if a.startswith("SSp-bfd-")},
                      key=lambda s: (len(s) > 2, s))  # A1..E8 first, then Greek straddlers
BIDX = {b: k + 1 for k, b in enumerate(BARREL_NAMES)}
SEPTA = len(BARREL_NAMES) + 1
C2 = BIDX["C2"]
BARREL_OF_ID = {}  # every atlas id that is a barrel or a descendant of one -> barrel index
L4_BARREL_IDS = set()  # SSp-bfd-<barrel>-4
for _i, _a in zip(S.id, S.acronym):
    if _a.startswith("SSp-bfd-"):
        BARREL_OF_ID[int(_i)] = BIDX[_a.split("-")[2]]
        if _a.endswith("-4"):
            L4_BARREL_IDS.add(int(_i))
BFD_IDS = np.array(sorted(S.id[S.acronym.str.startswith("SSp-bfd")]))  # SSp-bfd, its layers, barrels and theirs
L4_BFD_IDS = np.array([int(i) for i, a in zip(S.id, S.acronym) if a.startswith("SSp-bfd") and a.endswith("4")])
ISO_IDS = np.array(S.id[S.structure_id_path.str.contains("/315/")])  # isocortex
ROW_COL = {"A": "#E69F00", "B": "#56B4E9", "C": "#009E73", "D": "#F0E442", "E": "#D55E00", "S": "#CC79A7"}


def row_of(name):
    """'A'..'E' for row barrels, 'S' for the straddlers."""
    return name[0] if name[0] in "ABCDE" and name[1:].isdigit() else "S"


def collapse(lab):
    """Atlas id volume -> barrel index volume (uint8, 0 = not a barrel)."""
    u, inv = np.unique(lab, return_inverse=True)
    lut = np.array([BARREL_OF_ID.get(int(x), 0) for x in u], np.uint8)
    return lut[inv].reshape(lab.shape)


def is_l4(lab):
    """True for barrel L4 voxels (SSp-bfd-<barrel>-4)."""
    u, inv = np.unique(lab, return_inverse=True)
    lut = np.array([int(x) in L4_BARREL_IDS for x in u])
    return lut[inv].reshape(lab.shape)
