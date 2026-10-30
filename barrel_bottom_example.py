"""Cortical columns under a region, built in the sample's own voxel grid.

Everything here happens in brainreg's ``downsampled`` space: the sample's own
ch0, resampled to 20 um and reoriented to the atlas axis order (asr), on which
brainreg also writes ``registered_atlas.tiff`` and ``registered_hemispheres.tiff``
- the atlas warped onto the sample. The atlas is never resampled the other way,
so tissue geometry (surface position, cortical thickness, curvature) is the
brain's own.

For one region and one hemisphere:

1. The tissue is segmented from ch0 (``retrograde.brain_mask``) and the depth of
   every voxel is its distance to the nearest point of that segmented surface,
   which also gives each voxel its surface point.
2. The region's footprint is the set of surface points that the registered
   region's voxels map onto. Every tissue voxel whose surface point is in the
   footprint belongs to the region's columns - the full depth of cortex under
   that piece of surface, wherever the registration put the layer lines.
3. The bottom of the cortex is the surface of the registered non-cortical
   tissue below it (white matter and deeper), and d_wm is the signed distance
   to it (negative once in white matter).
4. Normalised depth = d_pia / (d_pia + d_wm): 0 at the real pia, 1 at the
   cortex bottom, >1 in white matter. Local thickness is d_pia + d_wm, so it
   follows the brain's own thickness voxel by voxel without any patching.

The registration is therefore used for two things only: which piece of surface
is the region (tangential), and where the grey/white boundary is. The second is
checked against the image in `edge_alignment`.
"""

from __future__ import annotations

import glob
import json
import sys

import numpy as np
import pandas as pd
import tifffile
from scipy import ndimage as ndi
from scipy.spatial import cKDTree

from . import config

sys.path.insert(0, str(config.REPO_ROOT))
from retrograde.brain_mask import build_brain_mask, robust_background_threshold  # noqa: E402


# --- loading -----------------------------------------------------------------

def _ch0_path(reg_dir) -> str:
    """ch0 in downsampled space. Named ``downsampled.tiff`` when brainreg ran on
    a tiff stack, ``downsampled_<input>_C0.tiff`` when it ran on the fused file."""
    plain = reg_dir / "downsampled.tiff"
    if plain.exists():
        return str(plain)
    matches = [
        path for path in glob.glob(str(reg_dir / "downsampled_*_C0.tiff"))
        if "standard" not in path
    ]
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one downsampled ch0 in {reg_dir}, got {matches}")
    return matches[0]


def load_registration(mouse: str) -> dict:
    """ch0, warped atlas and hemispheres on the 20 um working grid.

    `native_shape` is the shape of brainreg's own downsampled grid (10 um for
    the Allen registrations, which are binned 2x2x2 here; see _binned)."""
    reg_dir = config.registration_dir(mouse)
    with open(reg_dir / "brainreg.json") as handle:
        params = json.load(handle)
    if params["orientation"] != "ipr":
        raise ValueError(f"{mouse}: raw->downsampled mapping assumes ipr, got {params['orientation']}")
    out = {"reg_dir": reg_dir, "raw_voxel_um": np.array([float(v) for v in params["voxel_sizes"]])}
    if config.FACTOR == 1:
        out["ch0"] = tifffile.imread(_ch0_path(reg_dir)).astype(np.float32)
        out["atlas"] = tifffile.imread(reg_dir / "registered_atlas.tiff")
        out["hemispheres"] = tifffile.imread(reg_dir / "registered_hemispheres.tiff")
        out["native_shape"] = out["ch0"].shape
    else:
        out.update(_binned(mouse, reg_dir))
    return out


def _bin_mean(path, factor):
    """Block mean of a tiff stack, read one plane at a time."""
    with tifffile.TiffFile(path) as tif:
        pages = tif.pages
        n_z, (n_y, n_x) = len(pages), pages[0].shape
        shape = (n_z // factor, n_y // factor, n_x // factor)
        out = np.zeros(shape, np.float32)
        for z in range(shape[0] * factor):
            plane = pages[z].asarray()[:shape[1] * factor, :shape[2] * factor].astype(np.float32)
            out[z // factor] += plane.reshape(shape[1], factor, shape[2], factor).mean(axis=(1, 3))
    return out / factor, (n_z, n_y, n_x)


def _bin_labels(path, factor):
    """Labels at every `factor`-th voxel (each coarse voxel takes the label at
    its first fine voxel, 5 um off its centre for 10 -> 20 um)."""
    with tifffile.TiffFile(path) as tif:
        pages = tif.pages
        n_z, (n_y, n_x) = len(pages), pages[0].shape
        shape = (n_z // factor, n_y // factor, n_x // factor)
        out = np.zeros(shape, pages[0].dtype)
        for j in range(shape[0]):
            out[j] = pages[j * factor].asarray()[:shape[1] * factor:factor, :shape[2] * factor:factor]
    return out


def _binned(mouse: str, reg_dir) -> dict:
    """The 10 um registration binned onto the 20 um working grid, cached
    locally (the share copies are 3 x 5.5 GB)."""
    cache = config.DATA_DIR / "registration_binned" / f"{mouse}.npz"
    if cache.exists():
        stored = np.load(cache)
        return {"ch0": stored["ch0"].astype(np.float32), "atlas": stored["atlas"],
                "hemispheres": stored["hemispheres"], "native_shape": tuple(stored["native_shape"])}
    factor = config.FACTOR
    ch0, native_shape = _bin_mean(_ch0_path(reg_dir), factor)
    atlas = _bin_labels(reg_dir / "registered_atlas.tiff", factor)
    hemispheres = _bin_labels(reg_dir / "registered_hemispheres.tiff", factor)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, ch0=np.round(ch0).astype(np.uint16), atlas=atlas, hemispheres=hemispheres,
                        native_shape=np.array(native_shape))
    return {"ch0": ch0, "atlas": atlas, "hemispheres": hemispheres, "native_shape": native_shape}


def native_to_work(native: np.ndarray) -> np.ndarray:
    """Continuous voxel coordinates of the registration grid -> the 20 um working
    grid. Coarse voxel j covers fine voxels factor*j ... factor*j + factor - 1, so
    its centre sits at factor*j + (factor - 1) / 2. Identity for Perens."""
    return (native - (config.FACTOR - 1) / 2) / config.FACTOR


def work_to_native(work: np.ndarray) -> np.ndarray:
    return work * config.FACTOR + (config.FACTOR - 1) / 2


def fill_internal_holes(mask: np.ndarray) -> np.ndarray:
    """The tissue mask with enclosed cavities (ventricles) filled, so that only
    the outer brain surface counts as pia. Without this the ventricle wall is a
    surface too: deep wS1 next to an enlarged lateral ventricle (MS214 L) took
    its depth from the ventricle, and columns grew up from it, so L6 cells were
    counted as superficial. Filled in 3-D, then per coronal plane for cavities
    that open to the outside elsewhere in the stack."""
    filled = ndi.binary_fill_holes(mask)
    for plane in range(filled.shape[0]):
        filled[plane] = ndi.binary_fill_holes(filled[plane])
    return filled


def brain_mask(mouse: str, ch0: np.ndarray) -> np.ndarray:
    """Image-based tissue mask (internal holes filled, see fill_internal_holes).
    The unfilled segmentation is cached because it costs ~30 s per brain."""
    cache = config.DATA_DIR / "masks" / f"{mouse}.npz"
    if cache.exists():
        stored = np.load(cache)
        flat = np.unpackbits(stored["bits"], count=int(np.prod(stored["shape"])))
        return fill_internal_holes(flat.reshape(stored["shape"]).astype(bool))
    # robust_background_threshold rather than the default: these are raw
    # downsampled stacks, not the preprocessed standard-space ones the default
    # threshold was tuned on (see its docstring).
    mask, _ = build_brain_mask(ch0, threshold=robust_background_threshold(ch0))
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, bits=np.packbits(mask.ravel()), shape=np.array(mask.shape))
    return fill_internal_holes(mask)


def raw_to_downsampled(raw: np.ndarray, raw_voxel_um: np.ndarray, native_shape) -> np.ndarray:
    """Raw detection coordinates (ipr, raw voxels) -> continuous working-grid (asr) voxels.

    Scaling to the registration voxel size gives the ipr grid of
    ``registered_atlas_original_orientation``; ipr -> asr swaps the first two
    axes and flips both (`native_shape` = brainreg's downsampled shape). Verified
    on MS176: the flipped/transposed original-orientation atlas equals
    registered_atlas.tiff voxel for voxel, and 99.99% of detections land inside
    the image mask.
    """
    scaled = raw * (raw_voxel_um / config.NATIVE_UM)
    return native_to_work(np.column_stack([
        (native_shape[0] - 1) - scaled[:, 1],
        (native_shape[1] - 1) - scaled[:, 0],
        scaled[:, 2],
    ]))


def load_points(mouse: str, raw_voxel_um, native_shape) -> np.ndarray:
    raw = pd.read_csv(
        config.points_path(mouse),
        usecols=["coordinate_raw_axis_0", "coordinate_raw_axis_1", "coordinate_raw_axis_2"],
    ).to_numpy(float)
    return raw_to_downsampled(raw, raw_voxel_um, native_shape)


# --- atlas space ---------------------------------------------------------------

_ATLAS_CACHE: dict = {}


def load_atlas() -> dict:
    """The dataset's atlas itself, as the 'brain' for the atlas-space comparison:
    the reference image stands in for ch0 and the annotation for both the
    registered atlas and the tissue mask. Loaded once per process."""
    if not _ATLAS_CACHE:
        if config.FACTOR == 1:
            annotation = tifffile.imread(config.ATLAS_ANNOTATION)
            reference = tifffile.imread(config.ATLAS_REFERENCE).astype(np.float32)
        else:   # Allen 10 um, binned onto the 20 um working grid
            annotation = _bin_labels(config.ATLAS_ANNOTATION, config.FACTOR)
            reference, _ = _bin_mean(config.ATLAS_REFERENCE, config.FACTOR)
        # The atlas is symmetric and ships no hemisphere volume. In asr the last
        # axis runs right -> left, matching the brainreg labels (1 = left, the
        # high-index half, as checked on MS176).
        hemispheres = np.full(annotation.shape, config.HEMISPHERE_LABELS_BY_NAME["right"], np.uint8)
        hemispheres[:, :, annotation.shape[2] // 2:] = config.HEMISPHERE_LABELS_BY_NAME["left"]
        _ATLAS_CACHE.update({
            "image": reference,
            "atlas": annotation,
            "hemispheres": hemispheres,
            "mask": annotation > 0,
        })
    return _ATLAS_CACHE


def points_in_atlas(mouse: str) -> np.ndarray:
    """Detections in atlas voxels of the working grid, through the dataset's registration.

    Perens: brainreg's deformation_field_<i>.tiff live on the downsampled sample
    grid and hold the atlas coordinate along axis i in mm, sampled with
    trilinear interpolation at each point's sub-voxel position. Checked on
    MS176: 5000 random sample voxels land on the same label in the atlas as in
    registered_atlas.tiff (100%).

    Allen: the 10 um fields are 3 x 5.5 GB, so the atlas coordinates written by
    the cluster transform (all_points.csv, integer 10 um voxels) are used
    instead; their +-5 um rounding is below the 20 um working grid.
    Cached either way.
    """
    cache = config.DATA_DIR / "atlas_points" / f"{mouse}.npy"
    if cache.exists():
        return np.load(cache)
    if config.FACTOR == 1:
        reg_dir = config.registration_dir(mouse)
        with open(reg_dir / "brainreg.json") as handle:
            raw_voxel_um = np.array([float(v) for v in json.load(handle)["voxel_sizes"]])
        shape = tifffile.TiffFile(_ch0_path(reg_dir)).series[0].shape
        points = load_points(mouse, raw_voxel_um, shape)
        atlas_points = np.empty_like(points)
        for axis in range(3):
            field = tifffile.imread(reg_dir / f"deformation_field_{axis}.tiff")
            atlas_points[:, axis] = ndi.map_coordinates(field, points.T, order=1, mode="nearest") * 1000 / config.VOXEL_UM
            del field
    else:
        native = pd.read_csv(config.points_path(mouse),
                             usecols=[f"coordinate_atlas_axis_{i}" for i in range(3)]).to_numpy(float)
        atlas_points = native_to_work(native)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache, atlas_points)
    return atlas_points


# --- columns -----------------------------------------------------------------

def _crop_box(region: np.ndarray, shape) -> tuple[slice, ...]:
    coords = np.nonzero(region)
    margin = config.CROP_MARGIN_VOXELS
    return tuple(
        slice(max(int(c.min()) - margin, 0), min(int(c.max()) + margin + 1, size))
        for c, size in zip(coords, shape)
    )


def midline_sheet(hemispheres: np.ndarray) -> np.ndarray:
    """One-voxel sheet on the left/right boundary of the hemisphere labels.

    The two hemispheres touch in the image and the segmentation closes the
    fissure, so without this the medial wall has no surface and medial areas
    (PL, the medial edge of MOs/MOp) would measure depth from the dorsal or
    ventral surface instead. The registration's midline stands in for the
    fissure; removing it from the tissue mask makes it a surface.
    """
    sheet = np.zeros(hemispheres.shape, bool)
    left, right = config.HEMISPHERE_LABELS_BY_NAME["left"], config.HEMISPHERE_LABELS_BY_NAME["right"]
    a, b = hemispheres[:, :, :-1], hemispheres[:, :, 1:]
    sheet[:, :, :-1] = ((a == left) & (b == right)) | ((a == right) & (b == left))
    return sheet


def build_columns(atlas, mask, isocortex_ids, region, hemispheres=None, ch0=None):
    """Columns under `region` (a boolean volume, one hemisphere).

    Returns the crop box, the pia-depth volume of the crop, and a per-voxel dict
    for the column voxels (flat crop index, depth, normalised depth, local
    thickness, surface point). Depths are in um. With `ch0` (the full image),
    the registered cortex bottom is moved onto the grey/white edge found in the
    image (image_bottom_shift); d_wm and everything after it then use that
    bottom, and voxels gets "bottom_shift_um" and the patch table "bottom_patches". The EDT measures centre to
    centre, so half a voxel is taken off to put each surface on the face
    between the last voxel on one side and the first on the other.
    """
    box = _crop_box(region, mask.shape)
    mask_c, atlas_c, region_c = mask[box], atlas[box], region[box]
    if hemispheres is not None:
        mask_c = mask_c & ~midline_sheet(hemispheres[box])
    shape = mask_c.shape
    half = config.VOXEL_UM / 2

    # d_pia: distance to the image surface, with the surface point of each voxel.
    indices = np.empty((3,) + shape, dtype=np.int32)
    edt = ndi.distance_transform_edt(mask_c, return_indices=True, indices=indices)
    depth = (edt * config.VOXEL_UM - half).astype(np.float32)
    surface = np.ravel_multi_index(tuple(indices), shape).ravel()
    del indices, edt

    # d_wm: signed distance to the bottom of the registered cortex, positive in
    # cortex, negative below it. "Below the cortex" is registered tissue that is
    # not isocortex; tissue outside the warped atlas (the shell between the real
    # pia and the registration's pia) counts as cortex, not as white matter.
    # The smoothing is applied to the cortex, not to what lies below: the white
    # matter under S1 is only ~5 voxels thick and smoothing it erased it, which
    # put the "bottom" on the hippocampus instead (thickness ~1180 um).
    cortex = (atlas_c == 0) | np.isin(atlas_c, isocortex_ids)
    cortex = ndi.gaussian_filter(cortex.astype(np.float32), config.BOTTOM_SMOOTH_SIGMA_VOXELS) > 0.5
    below = mask_c & ~cortex
    del cortex
    to_wm = ndi.distance_transform_edt(~below) * config.VOXEL_UM - half
    into_wm = ndi.distance_transform_edt(below) * config.VOXEL_UM - half
    d_wm = np.where(below, -into_wm, to_wm).astype(np.float32)
    del below, to_wm, into_wm

    shift_patches = None
    shift_voxel = None
    if ch0 is not None:
        shift_volume, shift_patches = image_bottom_shift(ch0[box], depth, d_wm, surface, region_c, mask_c)
        d_wm = (d_wm + shift_volume).astype(np.float32)
        shift_voxel = shift_volume.ravel()
        del shift_volume

    # Footprint: surface points the registered region maps onto, closed by one
    # voxel along the surface so single-voxel holes do not cut columns out.
    footprint = np.zeros(mask_c.size, bool)
    footprint[surface[region_c.ravel()]] = True
    footprint = (ndi.binary_dilation(footprint.reshape(shape), np.ones((3, 3, 3))) & ~mask_c).ravel()

    # Keep columns in the region's own hemisphere (the midline surface is shared
    # by both) and within REGION_ENVELOPE_UM of the registered region. Without
    # the envelope, PL's columns ran sideways through orbital/motor cortex at
    # rostral levels, where no white matter separates them from PL.
    in_column = footprint[surface] & mask_c.ravel()
    envelope = ndi.distance_transform_edt(~region_c) * config.VOXEL_UM <= config.REGION_ENVELOPE_UM
    in_column &= envelope.ravel()
    if hemispheres is not None:
        own = np.bincount(hemispheres[box][region_c].ravel()).argmax()
        in_column &= (hemispheres[box] == own).ravel()
    voxel = np.flatnonzero(in_column)
    voxel_depth = depth.ravel()[voxel]
    thickness = voxel_depth + d_wm.ravel()[voxel]
    with np.errstate(divide="ignore", invalid="ignore"):
        norm = voxel_depth / thickness
    keep = (thickness >= config.MIN_THICKNESS_UM) & (norm <= config.MAX_NORM_DEPTH)

    # Shallowest registered-isocortex voxel under each surface point = where the
    # registration puts the pia; reported, not used.
    is_cortex = np.isin(atlas_c.ravel()[voxel], isocortex_ids)
    atlas_pia = pd.Series(voxel_depth[is_cortex]).groupby(surface[voxel][is_cortex]).min()

    voxels = {
        "index": voxel[keep],
        "depth_um": voxel_depth[keep],
        "norm_depth": norm[keep],
        "thickness_um": thickness[keep],
        "surface": surface[voxel][keep],
        "atlas_pia_depth_um": atlas_pia,
    }
    if shift_voxel is not None:
        voxels["bottom_shift_um"] = shift_voxel[voxel[keep]]
        voxels["bottom_patches"] = shift_patches
    return box, depth, d_wm, voxels


def _along_depth_gradient(image, depth):
    """d(image)/d(depth): the image gradient projected on the pia-depth direction."""
    depth_grad = np.gradient(ndi.gaussian_filter(depth, 1.0))
    image_grad = np.gradient(ndi.gaussian_filter(image, 1.0))
    norm = np.sqrt(sum(g ** 2 for g in depth_grad)) + 1e-6
    return sum(i * d for i, d in zip(image_grad, depth_grad)) / norm


def image_bottom_shift(ch0_crop, depth, d_wm, surface, region_c, mask_c):
    """How far (um) the grey/white edge in ch0 lies below the registered cortex
    bottom, per surface patch; returned as a per-voxel volume of the crop (each
    voxel takes its surface point's patch) plus the patch table.

    Same patch scheme as barrel_bottom: 100 um surface patches pooled over
    BARREL_POOL_RADIUS_UM. For each pooled patch, the mean d(ch0)/d(depth) is
    binned against depth relative to the registered bottom (-d_wm), and the
    edge is the strongest local maximum of IMAGE_BOTTOM_SIGN x gradient within
    +-IMAGE_BOTTOM_SEARCH_UM (sub-bin by a parabola through the peak). Only
    genuine local maxima count, as in run_columns._edge_offset, so the flank of
    the L5 -> L6 change higher up is not picked. Patches without a peak are
    filled from neighbours, then median-smoothed over the surface; if no patch
    has one the shift is 0 (registered bottom kept).
    """
    shape = depth.shape
    along = _along_depth_gradient(ch0_crop, depth).ravel() * config.IMAGE_BOTTOM_SIGN
    offset = -d_wm.ravel()
    # Candidate voxels: under the region's footprint, near the registered bottom.
    footprint = np.zeros(int(np.prod(shape)), bool)
    footprint[surface[region_c.ravel()]] = True
    reach = config.IMAGE_BOTTOM_SEARCH_UM + 3 * config.IMAGE_BOTTOM_BIN_UM
    cand = np.flatnonzero(footprint[surface] & mask_c.ravel() & (np.abs(offset) <= reach))
    width = config.IMAGE_BOTTOM_BIN_UM
    edges = np.arange(-reach, reach + width / 2, width)
    centres = (edges[:-1] + edges[1:]) / 2
    which = np.digitize(offset[cand], edges) - 1
    ok = (which >= 0) & (which < len(centres))
    cand, which = cand[ok], which[ok]

    surf_xyz = np.array(np.unravel_index(surface[cand], shape)).T
    patch_xyz, patch = np.unique(surf_xyz // config.BARREL_PATCH_VOXELS, axis=0, return_inverse=True)
    patch = patch.ravel()
    sums = np.zeros((len(patch_xyz), len(centres)))
    counts = np.zeros_like(sums)
    np.add.at(sums, (patch, which), along[cand])
    np.add.at(counts, (patch, which), 1)

    tree = cKDTree(patch_xyz.astype(float))
    radius = config.BARREL_POOL_RADIUS_UM / (config.VOXEL_UM * config.BARREL_PATCH_VOXELS)
    neighbours = tree.query_ball_point(patch_xyz.astype(float), radius)
    search = np.abs(centres) <= config.IMAGE_BOTTOM_SEARCH_UM
    raw = np.full(len(patch_xyz), np.nan)
    for i, near in enumerate(neighbours):
        n = counts[near].sum(0)
        if (n[search] > 0).mean() < 0.8:
            continue
        profile = np.where(n > 0, sums[near].sum(0) / np.maximum(n, 1), np.nan)
        profile = ndi.gaussian_filter1d(pd.Series(profile).interpolate(limit_direction="both").to_numpy(), 1)
        inner = np.arange(1, len(profile) - 1)
        peaks = inner[(profile[inner] >= profile[inner - 1]) & (profile[inner] >= profile[inner + 1])
                      & (profile[inner] > 0) & search[inner]]
        if not len(peaks):
            continue
        k = peaks[np.argmax(profile[peaks])]
        a, b, c = profile[k - 1], profile[k], profile[k + 1]
        sub = 0.5 * (a - c) / (a - 2 * b + c) if (a - 2 * b + c) != 0 else 0.0
        raw[i] = centres[k] + np.clip(sub, -0.5, 0.5) * width

    smoothed = np.full(len(patch_xyz), np.nan)
    for i, near in enumerate(neighbours):
        local = raw[near]
        local = local[np.isfinite(local)]
        if len(local):
            smoothed[i] = np.median(local)
    smoothed[~np.isfinite(smoothed)] = np.nanmedian(raw) if np.isfinite(raw).any() else 0.0

    # Every crop voxel takes the shift of the nearest patch to its surface point
    # (so the dilated column edge does not step back to 0); 0 far from the region.
    all_patch = np.array(np.unravel_index(surface, shape)).T // config.BARREL_PATCH_VOXELS
    unique, inverse = np.unique(all_patch, axis=0, return_inverse=True)
    distance, nearest = tree.query(unique.astype(float))
    per_unique = np.where(distance <= 2 * radius, smoothed[nearest], 0.0).astype(np.float32)
    shift_volume = per_unique[inverse.ravel()].reshape(shape)
    patches = pd.DataFrame({"ap": patch_xyz[:, 0], "dv": patch_xyz[:, 1], "ml": patch_xyz[:, 2],
                            "bottom_shift_raw_um": raw, "bottom_shift_um": smoothed})
    return shift_volume, patches


def edge_alignment(ch0_crop, depth, d_wm, voxels, offsets_um=np.arange(-800, 601, 20)):
    """Mean ch0 and d(ch0)/d(depth) against depth relative to the cortex bottom.

    If the registered cortex bottom sits on the real grey/white boundary, the
    image should change sharply at offset 0. Pooled over every column voxel this
    is a sensitive check on the one piece of the registration the depth relies on.
    """
    smooth = ndi.gaussian_filter(ch0_crop, 1.0)
    along = _along_depth_gradient(ch0_crop, depth).ravel()

    offset = -d_wm.ravel()[voxels["index"]]
    which = np.digitize(offset, offsets_um) - 1
    ok = (which >= 0) & (which < len(offsets_um) - 1)
    n = np.bincount(which[ok], minlength=len(offsets_um) - 1)
    with np.errstate(invalid="ignore"):
        mean_grad = np.bincount(which[ok], along[voxels["index"][ok]], len(offsets_um) - 1) / n
        mean_ch0 = np.bincount(which[ok], smooth.ravel()[voxels["index"][ok]], len(offsets_um) - 1) / n
    centres = (offsets_um[:-1] + offsets_um[1:]) / 2
    return pd.DataFrame({"offset_um": centres, "mean_ch0": mean_ch0, "mean_gradient": mean_grad, "n_voxels": n})


def assign_points(points, box, depth, d_wm, voxels):
    """Depth of every detection inside the columns.

    The voxel decides membership; both distances are interpolated at the
    point's sub-voxel position.
    """
    shape_c = depth.shape
    local = points - np.array([s.start for s in box])
    rounded = np.round(local).astype(int)
    inside = np.all((rounded >= 0) & (rounded < np.array(shape_c)), axis=1)
    flat = np.ravel_multi_index(tuple(rounded[inside].T), shape_c)

    lookup = np.full(int(np.prod(shape_c)), -1, dtype=np.int64)
    lookup[voxels["index"]] = np.arange(len(voxels["index"]))
    hit = lookup[flat]
    member = hit >= 0

    chosen = points[inside][member]
    coords = local[inside][member].T
    point_depth = np.clip(ndi.map_coordinates(depth, coords, order=1, mode="nearest"), 0, None)
    point_wm = ndi.map_coordinates(d_wm, coords, order=1, mode="nearest")
    thickness = point_depth + point_wm
    table = pd.DataFrame({
        "ap": chosen[:, 0],
        "dv": chosen[:, 1],
        "ml": chosen[:, 2],
        "depth_um": point_depth,
        "norm_depth": point_depth / thickness,
        "thickness_um": thickness,
    })
    if "barrel_bottom" in voxels:
        bottom = voxels["barrel_bottom"][hit[member]]
        table["barrel_bottom"] = bottom
        table["aligned_depth"] = aligned_depth(table["norm_depth"].to_numpy(), bottom)
        table["from_barrel_um"] = from_barrel_um(table["depth_um"].to_numpy(), bottom, thickness)
    return table


# --- barrel alignment ----------------------------------------------------------

def barrel_bottom(ch0_crop, voxels):
    """Normalised depth of the bottom of the barrel row, per column voxel.

    Barrels are visible in ch0 as a row of bright blobs in L4, and their bottom
    is by definition the L4/L5 boundary. Columns are grouped into surface
    patches; each patch's ch0 profile against normalised depth is pooled with
    its neighbours (so it spans barrels and septa), the slow pia-to-WM
    brightness trend is subtracted, and the barrel bottom is where the barrel
    peak has fallen halfway to the trough below it.

    Without detrending the half-max fell on the broad L4 -> L6 decline instead
    (~0.7 in MS176, visibly below the barrels). Barrel/septum texture (local
    variance) found the same bottoms as detrending in MS176 but threw outliers
    to 0.6-0.85 in the low-contrast MS211, so the detrended mean is used.

    Patches where no peak can be found are filled from valid neighbours, then
    the estimate is median-smoothed over the surface. Returns the per-voxel
    bottom and a per-patch table for QC.
    """
    shape = ch0_crop.shape
    smooth = ndi.gaussian_filter(ch0_crop, 0.7).ravel()[voxels["index"]]
    width = config.BARREL_PROFILE_BIN
    n_bins = int(np.ceil(config.MAX_NORM_DEPTH / width))
    centres = (np.arange(n_bins) + 0.5) * width
    which = np.minimum((voxels["norm_depth"] / width).astype(int), n_bins - 1)

    surface = np.array(np.unravel_index(voxels["surface"], shape)).T
    patch_xyz, patch = np.unique(surface // config.BARREL_PATCH_VOXELS, axis=0, return_inverse=True)
    patch = patch.ravel()
    n_patches = len(patch_xyz)
    sums = np.zeros((n_patches, n_bins))
    counts = np.zeros((n_patches, n_bins))
    np.add.at(sums, (patch, which), smooth)
    np.add.at(counts, (patch, which), 1)

    tree = cKDTree(patch_xyz.astype(float))
    radius = config.BARREL_POOL_RADIUS_UM / (config.VOXEL_UM * config.BARREL_PATCH_VOXELS)
    neighbours = tree.query_ball_point(patch_xyz.astype(float), radius)
    lo, hi = config.BARREL_PEAK_RANGE
    span = (centres > 0.15) & (centres < 0.85)

    raw = np.full(n_patches, np.nan)
    for i, near in enumerate(neighbours):
        n = counts[near].sum(0)
        if (n[span] > 0).mean() < 0.8:
            continue
        profile = np.where(n > 0, sums[near].sum(0) / np.maximum(n, 1), np.nan)
        profile = pd.Series(profile).interpolate(limit_direction="both").to_numpy()
        profile = ndi.gaussian_filter1d(profile, 1) - ndi.gaussian_filter1d(profile, 6, mode="nearest")
        window = np.flatnonzero((centres > lo) & (centres < hi))
        peak = window[np.argmax(profile[window])]
        below = np.flatnonzero((centres > centres[peak]) & (centres < centres[peak] + 0.3))
        trough = below[np.argmin(profile[below])]
        half = (profile[peak] + profile[trough]) / 2
        segment = np.arange(peak, trough + 1)
        j = segment[np.argmax(profile[segment] < half)]
        if j == peak or profile[peak] <= profile[trough]:
            continue
        raw[i] = centres[j - 1] + (profile[j - 1] - half) / (profile[j - 1] - profile[j]) * width

    # Fill, then smooth, over the surface.
    smoothed = np.full(n_patches, np.nan)
    for i, near in enumerate(neighbours):
        local = raw[near]
        local = local[np.isfinite(local)]
        if len(local):
            smoothed[i] = np.median(local)
    fallback = np.nanmedian(raw) if np.isfinite(raw).any() else config.BARREL_BOTTOM_TARGET
    smoothed[~np.isfinite(smoothed)] = fallback

    patches = pd.DataFrame({
        "ap": patch_xyz[:, 0], "dv": patch_xyz[:, 1], "ml": patch_xyz[:, 2],
        "barrel_bottom_raw": raw, "barrel_bottom": smoothed,
    })
    return smoothed[patch], patches


def from_barrel_um(depth_um, bottom, thickness_um):
    """Signed distance (um) from the column's barrel bottom: 0 at the L4/L5
    boundary, negative towards the pia, positive towards the white matter.
    Real distance, no stretching - the barrel bottom's own depth is its
    normalised depth times the local cortical thickness."""
    return depth_um - bottom * thickness_um


def aligned_depth(norm, bottom):
    """Piecewise-linear remap: pia stays 0, the local barrel bottom goes to
    BARREL_BOTTOM_TARGET, the cortex bottom stays 1, white matter is unchanged."""
    target = config.BARREL_BOTTOM_TARGET
    upper = norm * target / bottom
    lower = target + (norm - bottom) * (1 - target) / (1 - bottom)
    return np.where(norm <= bottom, upper, np.where(norm <= 1, lower, norm))


def volume_histograms(voxels) -> pd.DataFrame:
    """Voxel counts in fine depth bins, so density can be rebinned later."""
    norm_edges = np.round(np.arange(0, config.MAX_NORM_DEPTH + 1e-9, 0.01), 4)
    um_edges = np.arange(0, 3001, 10)
    rows = []
    kinds = [("norm", voxels["norm_depth"], norm_edges), ("um", voxels["depth_um"], um_edges)]
    if "barrel_bottom" in voxels:
        kinds.append(("aligned", aligned_depth(voxels["norm_depth"], voxels["barrel_bottom"]), norm_edges))
        barrel_edges = np.arange(-1500, 1501, 10)
        kinds.append(("barrel_um", from_barrel_um(voxels["depth_um"], voxels["barrel_bottom"], voxels["thickness_um"]), barrel_edges))
    for kind, values, edges in kinds:
        counts, _ = np.histogram(values, edges)
        rows.append(pd.DataFrame({"kind": kind, "bin_start": edges[:-1], "bin_end": edges[1:], "n_voxels": counts}))
    return pd.concat(rows, ignore_index=True)
