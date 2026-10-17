"""KiTS23 (kidney / tumor / cyst CT) support for Uni-Light.

Why a separate loader?  BraTS/MSD are 4-channel MRI of a fixed ~240x240x155 brain
volume, so the existing loaders z-score each modality and center-crop 128^3.
KiTS23 is single-channel CT (Hounsfield units) with variable size (512x512xN,
N = 29..1000+) and variable spacing, so a center crop would usually miss the
kidneys.  This module therefore:

  1. resamples every case once to an isotropic spacing and caches it (.npz),
  2. clips / normalises HU values,
  3. samples 128^3 patches (foreground-oversampled for training, centred on the
     kidneys for validation/test),
  4. converts labels into 3 nested sigmoid channels (the official KiTS HECs):
        ch0 = Kidney + Masses   (labels 1,2,3)
        ch1 = Kidney Masses     (labels 2,3  = tumor + cyst)
        ch2 = Tumor             (label  2)
     which keeps out_channels=3, so losses/metrics/models are unchanged.

Each sample is returned as (image[1,D,H,W], mask[3,D,H,W], sdf[3,D,H,W], case_name),
the same 4-tuple the rest of the pipeline already expects.
"""

import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.ndimage import zoom
from tqdm import tqdm

from utils.loss_metrics import BoundaryLoss

# CT intensity statistics used by nnU-Net for KiTS (foreground 0.5/99.5 percentiles, mean, std).
HU_MIN, HU_MAX = -79.0, 304.0
HU_MEAN, HU_STD = 101.0, 76.9


# ----------------------------------------------------------------------------
# Case discovery
# ----------------------------------------------------------------------------
def find_kits_files(case_dir):
    """Return (imaging_path, segmentation_path) or (None, None) if incomplete."""
    case_dir = Path(case_dir)
    img = case_dir / "imaging.nii.gz"
    seg = case_dir / "segmentation.nii.gz"
    if img.exists() and seg.exists():
        return str(img), str(seg)
    return None, None


def discover_kits_cases(data_dir):
    """All `case_XXXXX` folders under data_dir that have imaging + segmentation."""
    cases = sorted(p for p in Path(data_dir).iterdir()
                   if p.is_dir() and p.name.startswith("case_"))
    valid = [p for p in cases if all(find_kits_files(p))]
    if not valid:
        raise FileNotFoundError(
            f"No KiTS23 cases (case_*/imaging.nii.gz + segmentation.nii.gz) found under {data_dir}. "
            "Point --data-dir at the `dataset` folder created by kits23_download_data."
        )
    return valid


# ----------------------------------------------------------------------------
# One-off preprocessing (resample + HU normalisation) -> cache
# ----------------------------------------------------------------------------
def _preprocess_one(args):
    case_dir, cache_dir, spacing = args
    import nibabel as nib  # imported here so worker processes have it too

    case_dir = Path(case_dir)
    out_path = Path(cache_dir) / f"{case_dir.name}.npz"
    if out_path.exists():
        try:
            with np.load(out_path) as d:
                _ = d["seg"].shape
            return str(out_path)
        except Exception:
            out_path.unlink()  # corrupted -> redo

    img_path, seg_path = find_kits_files(case_dir)
    img_nii = nib.load(img_path)
    seg_nii = nib.load(seg_path)
    img = np.asarray(img_nii.dataobj, dtype=np.float32)       # (D, H, W) HU, KiTS array order
    seg = np.asarray(seg_nii.dataobj).astype(np.uint8)         # 0 bg, 1 kidney, 2 tumor, 3 cyst
    if img.shape != seg.shape:
        raise ValueError(f"{case_dir.name}: image {img.shape} != seg {seg.shape}")

    # Spacing along each *array* axis (nibabel zooms are already in array-axis order).
    zooms = np.asarray(img_nii.header.get_zooms()[:3], dtype=np.float64)
    factors = zooms / np.asarray(spacing, dtype=np.float64)

    img = np.clip(img, HU_MIN, HU_MAX)
    img = zoom(img, factors, order=1, mode="nearest", prefilter=False)
    seg = zoom(seg, factors, order=0, mode="nearest", prefilter=False)
    img = ((img - HU_MEAN) / HU_STD).astype(np.float16)

    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".tmp.npz")
    np.savez(tmp, image=img, seg=seg.astype(np.uint8))
    os.replace(tmp, out_path)
    return str(out_path)


def preprocess_kits_cases(case_dirs, cache_dir, spacing=(3.0, 3.0, 3.0), workers=2):
    """Resample + cache all given cases (skips ones already cached)."""
    jobs = [(str(c), str(cache_dir), tuple(spacing)) for c in case_dirs]
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    if workers <= 1:
        for j in tqdm(jobs, desc="KiTS23 preprocess"):
            _preprocess_one(j)
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for _ in tqdm(ex.map(_preprocess_one, jobs), total=len(jobs), desc="KiTS23 preprocess"):
                pass


# ----------------------------------------------------------------------------
# Dataset
# ----------------------------------------------------------------------------
class KiTS23Dataset(Dataset):
    """Patch dataset over cached KiTS23 volumes.

    mode="train": random patch, centred on a foreground voxel with prob `fg_prob`
                  (half of those on tumor/cyst voxels), plus flips/rot90/noise.
    mode="eval" : deterministic patch centred on the kidney bounding box.
                  NOTE: this is an optimistic proxy for monitoring training.
                  Use sliding-window inference over the whole volume for final numbers.
    """

    def __init__(self, case_names, cache_dir, roi_size=(128, 128, 128), mode="train",
                 compute_sdf=False, fg_prob=0.8, return_case_name=True):
        assert mode in ("train", "eval")
        self.case_names = list(case_names)
        self.cache_dir = str(cache_dir)
        self.roi = tuple(int(r) for r in roi_size)
        self.mode = mode
        self.compute_sdf = compute_sdf
        self.fg_prob = fg_prob
        self.return_case_name = return_case_name

    def __len__(self):
        return len(self.case_names)

    # -- helpers -------------------------------------------------------------
    def _choose_center(self, seg):
        shape = np.asarray(seg.shape)
        if self.mode == "train":
            if np.random.rand() < self.fg_prob:
                flat = None
                if np.random.rand() < 0.5:
                    flat = np.flatnonzero(seg >= 2)           # tumor / cyst
                if flat is None or flat.size == 0:
                    flat = np.flatnonzero(seg > 0)            # kidney or mass
                if flat.size:
                    return np.array(np.unravel_index(np.random.choice(flat), seg.shape))
            lo = np.minimum(self.roi, shape) // 2
            hi = np.maximum(shape - np.minimum(self.roi, shape) // 2, lo + 1)
            return np.array([np.random.randint(l, h) for l, h in zip(lo, hi)])
        # eval: bounding-box centre of all foreground
        idx = np.argwhere(seg > 0)
        if idx.size == 0:
            return shape // 2
        return (idx.min(0) + idx.max(0)) // 2

    def _crop(self, arr, start, fill):
        """Crop arr (D,H,W) at `start` with size roi, padding with `fill` where out of bounds."""
        out = np.full(self.roi, fill, dtype=arr.dtype)
        src_lo = np.maximum(start, 0)
        src_hi = np.minimum(np.asarray(start) + self.roi, arr.shape)
        dst_lo = src_lo - start
        dst_hi = dst_lo + (src_hi - src_lo)
        out[dst_lo[0]:dst_hi[0], dst_lo[1]:dst_hi[1], dst_lo[2]:dst_hi[2]] = \
            arr[src_lo[0]:src_hi[0], src_lo[1]:src_hi[1], src_lo[2]:src_hi[2]]
        return out

    @staticmethod
    def _augment(image, mask):
        # image: (1,D,H,W)  mask: (3,D,H,W)
        for ax in (1, 2, 3):
            if np.random.rand() < 0.5:
                image, mask = np.flip(image, axis=ax), np.flip(mask, axis=ax)
        k = np.random.randint(0, 4)
        if k:
            image, mask = np.rot90(image, k, axes=(2, 3)), np.rot90(mask, k, axes=(2, 3))
        image = image * np.random.uniform(0.9, 1.1) + np.random.uniform(-0.1, 0.1)
        image = image + np.random.normal(0, 0.05, size=image.shape).astype(np.float32)
        return np.ascontiguousarray(image, dtype=np.float32), np.ascontiguousarray(mask)

    # -- main ----------------------------------------------------------------
    def __getitem__(self, idx):
        name = self.case_names[idx]
        with np.load(os.path.join(self.cache_dir, f"{name}.npz")) as d:
            image_full = d["image"]   # (D,H,W) float16
            seg_full = d["seg"]       # (D,H,W) uint8

        center = self._choose_center(seg_full)
        start = np.clip(center - np.asarray(self.roi) // 2, 0,
                        np.maximum(np.asarray(seg_full.shape) - np.asarray(self.roi), 0))

        pad_val = float(image_full.min()) if image_full.size else 0.0
        image = self._crop(image_full, start, pad_val).astype(np.float32)[None]
        seg = self._crop(seg_full, start, 0)

        mask = np.stack([seg >= 1, seg >= 2, seg == 2], axis=0).astype(np.float32)

        if self.mode == "train":
            image, mask = self._augment(image, mask)

        # SDF is computed AFTER augmentation so it always matches the mask.
        if self.compute_sdf:
            sdf = np.stack([BoundaryLoss.compute_sdf_np(mask[c]) for c in range(3)], axis=0).astype(np.float32)
        else:
            sdf = np.zeros_like(mask)

        ret = [torch.from_numpy(np.ascontiguousarray(image)),
               torch.from_numpy(np.ascontiguousarray(mask)),
               torch.from_numpy(np.ascontiguousarray(sdf))]
        if self.return_case_name:
            ret.append(name)
        return tuple(ret)
