"""Data loading utilities for UKD-Net (BraTS 2023-GLI and MSD Task01 Brain Tumour).

Split protocol (used for every reported result):
    KFold(n_splits=5, shuffle=True, random_state=42) over the *sorted* list of cases,
    fold index 4 is the held-out evaluation fold (250 BraTS cases / 96 MSD cases).
    """

import glob
import os
import warnings

import nibabel as nib
import numpy as np
import torch
from sklearn.model_selection import KFold
from torch.utils.data import Dataset
from tqdm import tqdm

from utils.loss_metrics import BoundaryLoss

DEFAULT_ROI_SIZE = (128, 128, 128)


# ============================================================
# 1. K-fold splitting
# ============================================================
def get_5fold_split(data_dir, fold_idx, num_folds=5, seed=42):
    """Return (root, train_cases, val_cases, mode). `val_cases` is the held-out fold."""
    found_dirs = glob.glob(os.path.join(data_dir, "**/imagesTr"), recursive=True)

    if found_dirs:  # MSD layout: imagesTr / labelsTr, cases are file names
        mode = "MSD"
        images_dir = found_dirs[0]
        actual_root = os.path.dirname(images_dir)
        case_list = sorted(
            f for f in os.listdir(images_dir)
            if f.endswith((".nii.gz", ".nii")) and not f.startswith(".")
        )
    else:  # BraTS layout: one folder per case, cases are full folder paths
        mode = "BraTS"
        actual_root = data_dir
        case_list = []
        if os.path.exists(actual_root):
            case_list = sorted(
                os.path.join(actual_root, d)
                for d in os.listdir(actual_root)
                if os.path.isdir(os.path.join(actual_root, d)) and not d.startswith(".")
            )

    if not case_list:
        raise FileNotFoundError(f"No valid data found in {data_dir} (neither BraTS nor MSD layout matched)")
    if not 0 <= fold_idx < num_folds:
        raise ValueError(f"Fold index {fold_idx} out of range (0..{num_folds - 1})")

    print(f"Total Cases: {len(case_list)} | Mode: {mode}")
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=seed)
    train_idx, val_idx = list(kf.split(case_list))[fold_idx]
    return actual_root, [case_list[i] for i in train_idx], [case_list[i] for i in val_idx], mode


# ============================================================
# 2. File discovery and label mapping
# ============================================================
_PATTERNS = {
    "flair": ["-t2f", "flair"],
    "t1": ["-t1n", "t1."],
    "t1ce": ["-t1c", "t1ce", "t1gd"],
    "t2": ["-t2w", "t2."],
    "seg": ["-seg", "seg"],
}
_PRIMARY = {"flair": "-t2f", "t1": "-t1n", "t1ce": "-t1c", "t2": "-t2w", "seg": "-seg"}


def find_nii_adaptive(case_dir, mode):
    """Recursively find the NIfTI file of one BraTS-GLI modality ('flair','t1','t1ce','t2','seg')."""
    all_files = [
        os.path.join(root, f)
        for root, _, files in os.walk(case_dir)
        for f in files
        if f.endswith((".nii", ".nii.gz"))
    ]
    if not all_files:
        return None

    candidates = []
    for f_path in all_files:
        f_name = os.path.basename(f_path).lower()
        # Exclusions prevent T1/T1ce and T2/FLAIR mix-ups.
        if mode == "t1" and any(x in f_name for x in ("t1c", "t1ce", "t1gd")):
            continue
        if mode == "t2" and any(x in f_name for x in ("flair", "t2f")):
            continue
        if any(k in f_name for k in _PATTERNS[mode]):
            candidates.append(f_path)

    if not candidates:
        return None
    for c in candidates:
        if _PRIMARY[mode] in os.path.basename(c).lower():
            return c
    return candidates[0]


def seg_to_regions(seg, mode):
    """Map a label volume to the three nested regions [WT, TC, ET].

    BraTS 2023: 1=NCR, 2=ED, 3=ET -> WT={1,2,3}, TC={1,3}, ET={3}
    MSD Task01: 1=ED,  2=NCR, 3=ET -> WT={1,2,3}, TC={2,3}, ET={3}
    Returns a (3, H, W, D) array.
    """
    wt = (seg == 1) | (seg == 2) | (seg == 3)
    tc = ((seg == 2) | (seg == 3)) if mode == "MSD" else ((seg == 1) | (seg == 3))
    et = seg == 3
    return np.stack([wt, tc, et], axis=0)


# ============================================================
# 3. Shared helpers
# ============================================================
def pad_crop_to_roi(data, roi_size):
    """Zero-pad (centered) then center-crop a (C, D, H, W) array to roi_size."""
    _, d, h, w = data.shape
    td, th, tw = roi_size
    pd, ph, pw = max(0, td - d), max(0, th - h), max(0, tw - w)
    if pd or ph or pw:
        data = np.pad(data, ((0, 0), (pd // 2, pd - pd // 2), (ph // 2, ph - ph // 2), (pw // 2, pw - pw // 2)))
    _, d, h, w = data.shape
    sd, sh, sw = (d - td) // 2, (h - th) // 2, (w - tw) // 2
    return data[:, sd:sd + td, sh:sh + th, sw:sw + tw]


def zscore_nonzero(img):
    """Z-score using non-zero (brain) voxels only."""
    mask = img != 0
    if mask.any():
        v = img[mask]
        img[mask] = (v - v.mean()) / (v.std() + 1e-8)
    return img.astype(np.float32)


# ============================================================
# 4. SDF pre-computation (for the boundary loss)
# ============================================================
def precompute_sdfs_unified(data_dir, case_list, out_dir, mode, roi_size=DEFAULT_ROI_SIZE):
    """Pre-compute and cache per-case signed distance maps (float16, shape (3, D, H, W))."""
    os.makedirs(out_dir, exist_ok=True)

    for case_item in tqdm(case_list, desc="SDF Precomputing"):
        if mode == "MSD":
            case_name = case_item.replace(".nii.gz", "").replace(".nii", "")
            seg_path = os.path.join(data_dir, "labelsTr", case_item)
            if not os.path.exists(seg_path):
                seg_path = os.path.join(data_dir, "labels", case_item)
        else:
            case_name = os.path.basename(case_item.rstrip("/"))
            seg_path = find_nii_adaptive(case_item, "seg")

        out_path = os.path.join(out_dir, case_name + ".npz")

        if os.path.exists(out_path):  # reuse cache unless corrupted
            try:
                with np.load(out_path) as d:
                    _ = d["sdfs"]
                continue
            except Exception:
                os.remove(out_path)

        if not seg_path or not os.path.exists(seg_path):
            warnings.warn(f"No segmentation found for {case_name}; SDF skipped")
            continue
        try:
            seg = nib.load(seg_path).get_fdata().astype(np.int16)
        except Exception as e:
            warnings.warn(f"Could not read {seg_path}: {e}; SDF skipped")
            continue

        seg_3ch = np.transpose(seg_to_regions(seg, mode).astype(np.uint8), (0, 3, 1, 2))  # (3, D, H, W)
        seg_roi = pad_crop_to_roi(seg_3ch, roi_size)

        sdfs = np.zeros(seg_roi.shape, dtype=np.float16)
        for ci in range(3):
            sdfs[ci] = BoundaryLoss.compute_sdf_np(seg_roi[ci]).astype(np.float16)
        np.savez_compressed(out_path, sdfs=sdfs)


# ============================================================
# 5. Datasets
# ============================================================
class _BTSBase(Dataset):
    """Shared logic: SDF loading, augmentation, tensor packing."""

    mode = None

    def __init__(self, sdf_dir=None, roi_size=DEFAULT_ROI_SIZE, augment=False, return_case_name=False):
        self.sdf_dir = sdf_dir
        self.roi_size = roi_size
        self.augment = augment
        self.return_case_name = return_case_name

    def _load_sdf(self, case_name, like):
        """Load the cached SDF; a zero map (boundary loss inactive) is returned if unavailable."""
        sdf = np.zeros_like(like)
        if self.sdf_dir:
            sdf_path = os.path.join(self.sdf_dir, f"{case_name}.npz")
            if os.path.exists(sdf_path):
                try:
                    return np.load(sdf_path)["sdfs"].astype(np.float32)
                except Exception as e:
                    warnings.warn(f"Could not read SDF {sdf_path}: {e}")
            else:
                warnings.warn(f"SDF missing for {case_name}; boundary loss is inactive for this case")
        return sdf

    def _augment_3d(self, image, seg, sdf):
        """Random flips (D,H), random 90-degree rotation in the H-W plane, Gaussian noise (sigma=0.05)."""
        if np.random.rand() < 0.5:
            image, seg, sdf = image[:, ::-1], seg[:, ::-1], sdf[:, ::-1]
        if np.random.rand() < 0.5:
            image, seg, sdf = image[:, :, ::-1], seg[:, :, ::-1], sdf[:, :, ::-1]
        k = np.random.randint(0, 4)
        if k > 0:
            image = np.rot90(image, k=k, axes=(2, 3))
            seg = np.rot90(seg, k=k, axes=(2, 3))
            sdf = np.rot90(sdf, k=k, axes=(2, 3))
        image = image + np.random.normal(0, 0.05, size=image.shape).astype(np.float32)
        return np.ascontiguousarray(image), np.ascontiguousarray(seg), np.ascontiguousarray(sdf)

    def _pack(self, image, seg_3ch, case_name):
        image, seg_3ch = pad_crop_to_roi(image, self.roi_size), pad_crop_to_roi(seg_3ch, self.roi_size)
        sdf = self._load_sdf(case_name, seg_3ch)
        if self.augment:
            image, seg_3ch, sdf = self._augment_3d(image, seg_3ch, sdf)
        ret = [torch.from_numpy(np.ascontiguousarray(image)).float(),
               torch.from_numpy(np.ascontiguousarray(seg_3ch)).float(),
               torch.from_numpy(np.ascontiguousarray(sdf)).float()]
        if self.return_case_name:
            ret.append(case_name)
        return tuple(ret)


class BraTS2023Dataset(_BTSBase):
    mode = "BraTS"

    def __init__(self, case_dirs, sdf_dir=None, roi_size=DEFAULT_ROI_SIZE, augment=False, return_case_name=False):
        super().__init__(sdf_dir, roi_size, augment, return_case_name)
        self.case_dirs = case_dirs

    def __len__(self):
        return len(self.case_dirs)

    def __getitem__(self, idx):
        case_dir = self.case_dirs[idx]
        case_name = os.path.basename(case_dir.rstrip("/"))

        modalities = []
        for mod in ("flair", "t1", "t1ce", "t2"):  # channel order: FLAIR, T1, T1ce, T2
            f = find_nii_adaptive(case_dir, mod)
            if not f:
                raise FileNotFoundError(f"Missing {mod} in {case_dir}")
            modalities.append(zscore_nonzero(nib.load(f).get_fdata().astype(np.float32)))
        image = np.transpose(np.stack(modalities, axis=0), (0, 3, 1, 2))  # (4, H, W, D) -> (4, D, H, W)

        seg_file = find_nii_adaptive(case_dir, "seg")
        if not seg_file:
            raise FileNotFoundError(f"Missing seg in {case_dir}")
        seg = nib.load(seg_file).get_fdata().astype(np.int16)
        seg_3ch = np.transpose(seg_to_regions(seg, "BraTS").astype(np.float32), (0, 3, 1, 2))

        return self._pack(image, seg_3ch, case_name)


class MSDBTSDataset(_BTSBase):
    mode = "MSD"

    def __init__(self, data_dir, case_names, sdf_dir=None, roi_size=DEFAULT_ROI_SIZE, augment=False,
                 return_case_name=False):
        super().__init__(sdf_dir, roi_size, augment, return_case_name)
        self.data_dir = data_dir
        self.case_names = case_names

    def __len__(self):
        return len(self.case_names)

    def __getitem__(self, idx):
        name = self.case_names[idx]
        img_path = os.path.join(self.data_dir, "imagesTr", name)
        try:
            img_data = nib.load(img_path).get_fdata().astype(np.float32)
        except Exception as e:
            raise FileNotFoundError(f"Read error: {img_path} ({e})")

        if not (img_data.ndim == 4 and img_data.shape[-1] == 4):
            raise ValueError(f"Unexpected shape for MSD input: {img_data.shape}")
        for c in range(4):
            img_data[..., c] = zscore_nonzero(img_data[..., c])
        image = np.transpose(img_data, (3, 2, 0, 1))  # (H, W, D, C) -> (C, D, H, W)

        seg_path = os.path.join(self.data_dir, "labelsTr", name)
        if not os.path.exists(seg_path):
            seg_path = os.path.join(self.data_dir, "labels", name)
        seg = nib.load(seg_path).get_fdata().astype(np.int16)
        seg_3ch = np.transpose(seg_to_regions(seg, "MSD").astype(np.float32), (0, 3, 1, 2))

        case_name = name.replace(".nii.gz", "").replace(".nii", "")
        return self._pack(image, seg_3ch, case_name)


# ============================================================
# 6. Dataset factory
# ============================================================
def get_datasets(data_dir, fold, num_folds=5, seed=42, sdf_dir=None, roi_size=DEFAULT_ROI_SIZE):
    """Auto-detect BraTS/MSD and return (train_ds, val_ds); val_ds is the held-out fold."""
    root, train_list, val_list, mode = get_5fold_split(data_dir, fold, num_folds, seed)

    if sdf_dir:
        precompute_sdfs_unified(root, train_list + val_list, sdf_dir, mode, roi_size)

    if mode == "MSD":
        train_ds = MSDBTSDataset(root, train_list, sdf_dir=sdf_dir, roi_size=roi_size, augment=True,
                                 return_case_name=True)
        val_ds = MSDBTSDataset(root, val_list, sdf_dir=sdf_dir, roi_size=roi_size, augment=False,
                               return_case_name=True)
    elif mode == "BraTS":
        train_ds = BraTS2023Dataset(train_list, sdf_dir=sdf_dir, roi_size=roi_size, augment=True,
                                    return_case_name=True)
        val_ds = BraTS2023Dataset(val_list, sdf_dir=sdf_dir, roi_size=roi_size, augment=False,
                                  return_case_name=True)
    else:
        raise ValueError(f"Unknown mode: {mode}")
    return train_ds, val_ds
