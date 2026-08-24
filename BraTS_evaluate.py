# -*- coding: utf-8 -*-

!pip install thop

import os
import sys
import glob
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nibabel as nib
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.model_selection import KFold
from scipy.ndimage import binary_erosion, distance_transform_edt, generate_binary_structure, label as cc_label

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore")

DATA_DIR = "/kaggle/input/datasets/ahmedelzayat77/bratsglioma/Training"
STUDENT_WEIGHTS = "/kaggle/input/models/sorensalehi/student-model-brats2023/pytorch/default/1/best_distilled_student.pth"

CURRENT_FOLD = 4
NUM_FOLDS = 5
SEED = 42
ROI_SIZE = (128, 128, 128)
SPACING = (1.0, 1.0, 1.0)
NSD_TOLERANCE_MM = 2.0
BATCH_SIZE = 1
NUM_WORKERS = 2
CONNECTIVITY = 3
MIN_LESION_VOXELS = 1
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def find_nii_adaptive(case_dir, mode):
    all_files = []
    for root, _, files in os.walk(case_dir):
        for f in files:
            if f.endswith(('.nii', '.nii.gz')):
                all_files.append(os.path.join(root, f))

    patterns = {
        'flair': ['-t2f', 'flair'],
        't1':    ['-t1n', 't1.'],
        't1ce':  ['-t1c', 't1ce', 't1gd'],
        't2':    ['-t2w', 't2.'],
        'seg':   ['-seg', 'seg']
    }

    candidates = []
    for f_path in all_files:
        f_name = os.path.basename(f_path).lower()
        if mode == 't1' and any(x in f_name for x in ['t1c', 't1ce', 't1gd']):
            continue
        if mode == 't2' and any(x in f_name for x in ['flair', 't2f']):
            continue
        if any(k in f_name for k in patterns[mode]):
            candidates.append(f_path)

    if not candidates:
        return None

    primary = {
        'flair': '-t2f',
        't1': '-t1n',
        't1ce': '-t1c',
        't2': '-t2w',
        'seg': '-seg'
    }
    for c in candidates:
        if primary[mode] in os.path.basename(c).lower():
            return c
    return candidates[0]


class ParallelConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.embed = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.p1 = nn.Sequential(
            nn.Conv3d(out_channels, out_channels, 1, bias=False),
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.p2 = nn.Sequential(
            nn.Conv3d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.p3 = nn.Sequential(
            nn.Conv3d(out_channels, out_channels, 5, padding=2, bias=False),
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.pool = nn.MaxPool3d(2)

    def forward(self, x):
        feat = self.embed(x)
        o1 = self.pool(self.p1(feat))
        o2 = self.pool(self.p2(feat))
        o3 = self.pool(self.p3(feat))
        return torch.cat([o1, o2, o3], dim=1), feat


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv3d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(0.1, inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class LATUPStudent(nn.Module):
    def __init__(self, in_channels=4, out_channels=3, base_filters=16):
        super().__init__()
        self.enc1_pc = ParallelConvBlock(in_channels, base_filters)
        self.enc2 = ConvBlock(base_filters * 3, base_filters * 2)
        self.pool2 = nn.MaxPool3d(2)
        self.enc3 = ConvBlock(base_filters * 2, base_filters * 4)
        self.pool3 = nn.MaxPool3d(2)
        self.bottleneck = ConvBlock(base_filters * 4, base_filters * 8)
        self.up3 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.dec3 = ConvBlock(base_filters * 8 + base_filters * 4, base_filters * 4)
        self.up2 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.dec2 = ConvBlock(base_filters * 4 + base_filters * 2, base_filters * 2)
        self.up1 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.dec1 = ConvBlock(base_filters * 2 + base_filters, base_filters)
        self.out_conv = nn.Conv3d(base_filters, out_channels, kernel_size=1)

    def forward(self, x):
        x1, skip1 = self.enc1_pc(x)
        skip2 = self.enc2(x1)
        x2 = self.pool2(skip2)
        skip3 = self.enc3(x2)
        x3 = self.pool3(skip3)
        x_bot = self.bottleneck(x3)
        d3 = self.dec3(torch.cat([self.up3(x_bot), skip3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), skip2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), skip1], dim=1))
        return self.out_conv(d1)


class BraTSDataset(torch.utils.data.Dataset):
    def __init__(self, case_dirs, roi_size=ROI_SIZE):
        self.case_dirs = case_dirs
        self.roi_size = roi_size

    def __len__(self):
        return len(self.case_dirs)

    def _zscore_nonzero(self, img):
        mask = img != 0
        if mask.any():
            v = img[mask]
            img[mask] = (v - v.mean()) / (v.std() + 1e-8)
        return img.astype(np.float32)

    def _pad_crop(self, data):
        _, d, h, w = data.shape
        td, th, tw = self.roi_size
        pd, ph, pw = max(0, td - d), max(0, th - h), max(0, tw - w)
        if pd or ph or pw:
            data = np.pad(data, ((0, 0), (pd // 2, pd - pd // 2), (ph // 2, ph - ph // 2), (pw // 2, pw - pw // 2)))
        _, d, h, w = data.shape
        sd, sh, sw = (d - td) // 2, (h - th) // 2, (w - tw) // 2
        return data[:, sd:sd + td, sh:sh + th, sw:sw + tw]

    def __getitem__(self, idx):
        case_dir = self.case_dirs[idx]
        case_name = os.path.basename(case_dir.rstrip('/'))
        mods = []
        for mod in ['flair', 't1', 't1ce', 't2']:
            p = find_nii_adaptive(case_dir, mod)
            if p is None:
                raise FileNotFoundError(f"Missing {mod} in {case_dir}")
            mods.append(self._zscore_nonzero(nib.load(p).get_fdata().astype(np.float32)))

        image = np.transpose(np.stack(mods, axis=0), (0, 3, 1, 2))
        seg_p = find_nii_adaptive(case_dir, 'seg')
        if seg_p is None:
            raise FileNotFoundError(f"Missing seg in {case_dir}")

        seg = nib.load(seg_p).get_fdata().astype(np.int16)
        wt = ((seg == 1) | (seg == 2) | (seg == 3)).astype(np.float32)
        tc = ((seg == 1) | (seg == 3)).astype(np.float32)
        et = (seg == 3).astype(np.float32)
        label = np.transpose(np.stack([wt, tc, et], axis=0), (0, 3, 1, 2))

        image = self._pad_crop(image)
        label = self._pad_crop(label)
        return torch.FloatTensor(image.copy()), torch.FloatTensor(label.copy()), case_name


def surface(mask):
    mask = mask.astype(bool)
    if not mask.any():
        return np.zeros_like(mask, dtype=bool)
    return np.logical_xor(mask, binary_erosion(mask, iterations=1, border_value=0))


def directed_surface_distances(a, b, spacing):
    if not a.any() or not b.any():
        return np.empty(0, dtype=np.float32)
    dt = distance_transform_edt(~b, sampling=spacing)
    return dt[a].astype(np.float32)


def compute_all_metrics(pred, gt, spacing=SPACING, tolerance=NSD_TOLERANCE_MM):
    p_bin = pred.astype(bool)
    g_bin = gt.astype(bool)

    ps, gs = int(p_bin.sum()), int(g_bin.sum())
    tp = int(np.logical_and(p_bin, g_bin).sum())
    fp = int(np.logical_and(p_bin, ~g_bin).sum())
    fn = int(np.logical_and(~p_bin, g_bin).sum())
    tn = int(np.logical_and(~p_bin, ~g_bin).sum())

    dice = 1.0 if (ps == 0 and gs == 0) else (0.0 if (ps == 0 or gs == 0) else float(2.0 * tp / (ps + gs)))
    iou = 1.0 if (ps == 0 and gs == 0) else float(tp / (tp + fp + fn + 1e-8))
    precision = tp / (tp + fp) if (tp + fp) else (1.0 if (tp + fn) == 0 else 0.0)
    recall = tp / (tp + fn) if (tp + fn) else (1.0 if (tp + fp) == 0 else 0.0)
    specificity = tn / (tn + fp) if (tn + fp) else 0.0

    structure = generate_binary_structure(3, CONNECTIVITY)
    pl, npred = cc_label(p_bin, structure=structure)
    gl, ngt = cc_label(g_bin, structure=structure)
    pred_comps = [(pl == i) for i in range(1, npred + 1) if int((pl == i).sum()) >= MIN_LESION_VOXELS]
    gt_comps = [(gl == i) for i in range(1, ngt + 1) if int((gl == i).sum()) >= MIN_LESION_VOXELS]
    detected = sum(any(np.logical_and(g, p).any() for p in pred_comps) for g in gt_comps)
    lesion_recall = detected / len(gt_comps) if gt_comps else 1.0

    voxel_ml = float(np.prod(np.asarray(spacing, dtype=float)) / 1000.0)
    v_p_ml = float(ps) * voxel_ml
    v_g_ml = float(gs) * voxel_ml
    v_err = v_p_ml - v_g_ml
    abs_rve = abs(v_err / v_g_ml) if v_g_ml > 0 else (0.0 if v_p_ml == 0 else np.nan)

    if not p_bin.any() and not g_bin.any():
        hd95 = 0.0
        nsd = 1.0
    elif not p_bin.any() or not g_bin.any():
        physical = np.asarray(pred.shape, dtype=float) * np.asarray(spacing, dtype=float)
        hd95 = float(np.linalg.norm(physical))
        nsd = 0.0
    else:
        dp = directed_surface_distances(surface(p_bin), surface(g_bin), spacing)
        dg = directed_surface_distances(surface(g_bin), surface(p_bin), spacing)
        hd95 = float(np.percentile(np.concatenate([dp, dg]), 95))
        total_sd = dp.size + dg.size
        nsd = float(((dp <= tolerance).sum() + (dg <= tolerance).sum()) / total_sd) if total_sd else 1.0

    return {
        "dice": dice, "iou": iou, "precision": precision, "recall": recall,
        "specificity": specificity, "hd95": hd95, "nsd": nsd,
        "lesion_recall": lesion_recall, "pred_vol_ml": v_p_ml, "gt_vol_ml": v_g_ml,
        "abs_rve": abs_rve, "volume_mae_ml": abs(v_err)
    }


def load_student(path, device):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    model = LATUPStudent(4, 3, 16)
    ckpt = torch.load(path, map_location='cpu')
    sd = ckpt['model_state_dict'] if isinstance(ckpt, dict) and 'model_state_dict' in ckpt else (
        ckpt['state_dict'] if isinstance(ckpt, dict) and 'state_dict' in ckpt else (
            ckpt['model'] if isinstance(ckpt, dict) and 'model' in ckpt else ckpt
        )
    )
    cleaned = {}
    for k, v in sd.items():
        if not torch.is_tensor(v):
            continue
        if k.startswith('module.'):
            k = k[len('module.'):]
        if k.startswith('model.'):
            k = k[len('model.'):]
        if k.startswith('out_conv2.') or k.startswith('out_conv3.'):
            continue
        cleaned[k] = v
    model.load_state_dict(cleaned, strict=True)
    model.to(device).eval().requires_grad_(False)
    return model

def plot_evaluation_results(res_dict, save_path="brats_evaluation_plots.png"):
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    channels = ["WT", "TC", "ET"]

    for i, ch in enumerate(channels):
        ax = axes[0, i]
        data = res_dict["Student"][ch]
        df_box = pd.DataFrame({
            "Precision": [d["precision"] for d in data],
            "Recall": [d["recall"] for d in data],
            "Dice": [d["dice"] for d in data]
        })
        df_melt = df_box.melt(var_name="Metric", value_name="Value")
        palette = {"Precision": "#66c2a5", "Recall": "#fc8d62", "Dice": "#8da0cb"}
        sns.boxplot(x="Metric", y="Value", data=df_melt, ax=ax, palette=palette, width=0.6)
        ax.set_title(f"Segmentation Metrics - {ch}", fontsize=13, fontweight='bold', pad=10)
        ax.set_ylim(-0.02, 1.05)

    for i, ch in enumerate(channels):
        ax = axes[1, i]
        data = res_dict["Student"][ch]
        gt_vols = [d["gt_vol_ml"] for d in data]
        pred_vols = [d["pred_vol_ml"] for d in data]
        ax.scatter(gt_vols, pred_vols, alpha=0.55, color="#4c72b0", edgecolors="none", s=25)
        max_val = max(max(gt_vols, default=0), max(pred_vols, default=0)) * 1.05
        if max_val == 0:
            max_val = 1.0
        ax.plot([0, max_val], [0, max_val], 'r--', label="Ideal", alpha=0.85)
        ax.set_title(f"Volume Comparison - {ch}", fontsize=13, fontweight='bold', pad=10)
        ax.set_xlabel("GT Volume (mL)", fontsize=11)
        ax.set_ylabel("Pred Volume (mL)" if i == 0 else "", fontsize=11)
        ax.set_xlim(-max_val * 0.02, max_val)
        ax.set_ylim(-max_val * 0.02, max_val)
        ax.legend(loc="upper left")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def main():
    set_seed(SEED)
    cases = sorted([
        os.path.join(DATA_DIR, d)
        for d in os.listdir(DATA_DIR)
        if os.path.isdir(os.path.join(DATA_DIR, d)) and not d.startswith('.')
    ])

    kf = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=SEED)
    folds = list(kf.split(cases))
    _, val_idx = folds[CURRENT_FOLD]
    val_cases = [cases[i] for i in val_idx]

    print(f"Evaluating Fold {CURRENT_FOLD} with {len(val_cases)} cases.")

    ds = BraTSDataset(val_cases, roi_size=ROI_SIZE)
    loader = torch.utils.data.DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=(DEVICE.type == 'cuda'))

    student = load_student(STUDENT_WEIGHTS, DEVICE)
    print("Loaded student weights successfully.")

    res = {"Student": {"WT": [], "TC": [], "ET": []}}
    channels = ["WT", "TC", "ET"]

    with torch.inference_mode():
        for images, labels, _ in tqdm(loader, desc="Evaluating"):
            images = images.to(DEVICE, non_blocking=True)
            logits = student(images)
            probs = torch.sigmoid(logits)[0].cpu().numpy()
            pred = (probs >= 0.5).astype(np.uint8)
            gt = (labels[0].numpy() > 0.5).astype(np.uint8)

            for c_idx, ch in enumerate(channels):
                res["Student"][ch].append(compute_all_metrics(pred[c_idx], gt[c_idx]))

    metrics_list = ["dice", "hd95", "nsd", "precision", "recall", "specificity", "lesion_recall", "abs_rve", "volume_mae_ml"]

    print("\n" + "=" * 140)
    print(f"{'Target':<8} | " + " | ".join([f"{m:<12}" for m in metrics_list]))
    print("=" * 140)

    for ch in channels:
        vals = [np.nanmean([item[m] for item in res["Student"][ch]]) for m in metrics_list]
        val_str = " | ".join([f"{v:<12.4f}" if m not in ["hd95"] else f"{v:<12.2f}" for m, v in zip(metrics_list, vals)])
        print(f"{ch:<8} | {val_str}")

    mean_vals = [np.nanmean([[item[m] for item in res["Student"][ch]] for ch in channels]) for m in metrics_list]
    mean_str = " | ".join([f"{v:<12.4f}" if m not in ["hd95"] else f"{v:<12.2f}" for m, v in zip(metrics_list, mean_vals)])
    print(f"{'Mean':<8} | {mean_str}")
    print("-" * 140)

    plot_evaluation_results(res, save_path="brats_evaluation_plots.png")
    print("Evaluation completed and plots saved.")


if __name__ == '__main__':
    main()

import os
import time
import torch
import torch.nn as nn

try:
    from fvcore.nn import FlopCountAnalysis
except ImportError:
    FlopCountAnalysis = None


class ChannelAttention3D(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)
        self.mlp = nn.Sequential(
            nn.Conv3d(channels, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(hidden, channels, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return x * self.sigmoid(self.mlp(self.avg_pool(x)) + self.mlp(self.max_pool(x)))


class SpatialAttention3D(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv3d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = x.mean(dim=1, keepdim=True)
        maximum = x.amax(dim=1, keepdim=True)
        return x * self.sigmoid(self.conv(torch.cat((avg, maximum), dim=1)))


class SAXBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv3d(channels, channels, 3, padding=1, bias=False),
            nn.InstanceNorm3d(channels),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(channels, channels, 3, padding=1, bias=False),
            nn.InstanceNorm3d(channels),
        )
        self.channel_attention = ChannelAttention3D(channels, reduction)
        self.spatial_attention = SpatialAttention3D()
        self.activation = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        res = self.conv2(self.conv1(x))
        res = self.channel_attention(res)
        res = self.spatial_attention(res)
        return self.activation(x + res)


class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=16):
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv3d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.sax = SAXBlock(out_channels, reduction)

    def forward(self, x):
        return self.sax(self.convs(x))


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, reduction=16):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=True)
        self.convs = nn.Sequential(
            nn.Conv3d(in_channels + skip_channels, out_channels, 3, padding=1, bias=False),
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv3d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.sax = SAXBlock(out_channels, reduction)

    def forward(self, x, skip):
        return self.sax(self.convs(torch.cat((self.up(x), skip), dim=1)))


class SAXNetV1(nn.Module):
    def __init__(self, in_channels=4, out_channels=3, channels=(32, 64, 128, 256, 512)):
        super().__init__()
        c1, c2, c3, c4, c5 = channels
        self.enc1 = EncoderBlock(in_channels, c1)
        self.enc2 = EncoderBlock(c1, c2)
        self.enc3 = EncoderBlock(c2, c3)
        self.enc4 = EncoderBlock(c3, c4)
        self.bottleneck = EncoderBlock(c4, c5)
        self.pool = nn.MaxPool3d(2)
        self.dec4 = DecoderBlock(c5, c4, c4)
        self.dec3 = DecoderBlock(c4, c3, c3)
        self.dec2 = DecoderBlock(c3, c2, c2)
        self.dec1 = DecoderBlock(c2, c1, c1)
        self.output = nn.Conv3d(c1, out_channels, 1)

    def forward(self, x):
        s1 = self.enc1(x)
        s2 = self.enc2(self.pool(s1))
        s3 = self.enc3(self.pool(s2))
        s4 = self.enc4(self.pool(s3))
        x = self.bottleneck(self.pool(s4))
        x = self.dec4(x, s4)
        x = self.dec3(x, s3)
        x = self.dec2(x, s2)
        x = self.dec1(x, s1)
        return self.output(x)


def load_weights(model, path):
    if path and os.path.exists(path):
        ckpt = torch.load(path, map_location='cpu')
        state = ckpt.get('model_state_dict', ckpt.get('state_dict', ckpt.get('model', ckpt)))
        cleaned = {k.replace('module.', '').replace('model.', ''): v for k, v in state.items() if torch.is_tensor(v)}
        model.load_state_dict(cleaned, strict=False)


def compute_macs_and_tflops(model, dummy_input):
    if FlopCountAnalysis is not None:
        try:
            flops = FlopCountAnalysis(model, dummy_input)
            flops.set_op_handle('aten::upsample_trilinear3d', None)
            total_flops = flops.total()
            macs_g = (total_flops / 2) / 1e9
            tflops = total_flops / 1e12
            return macs_g, tflops
        except Exception:
            pass

    total_macs = 0
    hooks = []

    def conv3d_hook(module, inp, out):
        nonlocal total_macs
        x = inp[0]
        batch_size = x.shape[0]
        out_c, out_d, out_h, out_w = out.shape[1:]
        in_c = module.in_channels
        k_d, k_h, k_w = module.kernel_size
        groups = module.groups
        macs_per_instance = (in_c // groups) * k_d * k_h * k_w * out_c * out_d * out_h * out_w
        total_macs += batch_size * macs_per_instance

    for m in model.modules():
        if isinstance(m, nn.Conv3d):
            hooks.append(m.register_forward_hook(conv3d_hook))

    with torch.no_grad():
        _ = model(dummy_input)

    for h in hooks:
        h.remove()

    macs_g = total_macs / 1e9
    tflops = (2 * total_macs) / 1e12
    return macs_g, tflops


def benchmark_model(model, model_name, input_size=(1, 4, 128, 128, 128), device='cuda'):
    dev = torch.device(device)
    model = model.to(dev)

    total_params = sum(p.numel() for p in model.parameters())
    params_m = total_params / 1e6

    dummy_input = torch.randn(input_size).to(dev)

    model.eval()
    macs_g, tflops = compute_macs_and_tflops(model, dummy_input)

    for _ in range(10):
        with torch.no_grad():
            _ = model(dummy_input)

    if dev.type == 'cuda':
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(100):
            with torch.no_grad():
                _ = model(dummy_input)
        end.record()
        torch.cuda.synchronize()
        latency_ms = start.elapsed_time(end) / 100
    else:
        t0 = time.time()
        for _ in range(50):
            with torch.no_grad():
                _ = model(dummy_input)
        latency_ms = (time.time() - t0) * 1000 / 50

    inference_time_s = latency_ms / 1000

    im_mb = 0
    if dev.type == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            _ = model(dummy_input)
            torch.cuda.synchronize()
        im_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)

    tm_mb = 0
    if dev.type == 'cuda':
        for p in model.parameters():
            p.requires_grad = True
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        model.train()
        opt = torch.optim.Adam(model.parameters())
        opt.zero_grad()
        out = model(dummy_input)
        loss = (out[0].mean() if isinstance(out, (list, tuple)) else out.mean())
        loss.backward()
        torch.cuda.synchronize()
        tm_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)

    model.eval()
    return {
        "Model": model_name,
        "Params (M)": f"{params_m:.2f}",
        "MACs (G)": f"{macs_g:.2f}",
        "TFLOPS": f"{tflops:.4f}",
        "Infer Time (s)": f"{inference_time_s:.4f}",
        "Latency (ms)": f"{latency_ms:.2f}",
        "Infer Mem (MB)": f"{im_mb:.1f}",
        "Train Mem (MB)": f"{tm_mb:.1f}"
    }


def print_table(results):
    if not results:
        return
    headers = list(results[0].keys())
    widths = [max(len(h), max(len(str(r[h])) for r in results)) + 2 for h in headers]
    border = "+" + "+".join(["-" * w for w in widths]) + "+"
    print("\n" + border)
    print("|" + "|".join([f" {h:<{w - 2}} " for h, w in zip(headers, widths)]) + "|")
    print(border)
    for res in results:
        print("|" + "|".join([f" {str(res[h]):<{w - 2}} " for h, w in zip(headers, widths)]) + "|")
    print(border + "\n")


device = "cuda" if torch.cuda.is_available() else "cpu"
roi_size = (1, 4, 128, 128, 128)
teacher_path = "/kaggle/input/models/sorensalehi/saxnet-teacher/pytorch/default/1/best_saxnet_teacher.pth"

teacher = SAXNetV1(in_channels=4, out_channels=3)
load_weights(teacher, teacher_path)

res_teacher = benchmark_model(teacher, "Teacher (SAXNetV1)", roi_size, device)
print_table([res_teacher])

import os
import time
import torch
import torch.nn as nn

try:
    from fvcore.nn import FlopCountAnalysis
except ImportError:
    FlopCountAnalysis = None


class ParallelConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.embed = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.p1 = nn.Sequential(
            nn.Conv3d(out_channels, out_channels, 1, bias=False),
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(0.1)
        )
        self.p2 = nn.Sequential(
            nn.Conv3d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(0.1)
        )
        self.p3 = nn.Sequential(
            nn.Conv3d(out_channels, out_channels, 5, padding=2, bias=False),
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(0.1)
        )
        self.pool = nn.MaxPool3d(2)

    def forward(self, x):
        feat = self.embed(x)
        o1 = self.pool(self.p1(feat))
        o2 = self.pool(self.p2(feat))
        o3 = self.pool(self.p3(feat))
        return torch.cat([o1, o2, o3], dim=1), feat


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv3d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(0.1, inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class LATUPStudent(nn.Module):
    def __init__(self, in_channels=4, out_channels=3, base_filters=16):
        super().__init__()
        self.enc1_pc = ParallelConvBlock(in_channels, base_filters)
        self.enc2 = ConvBlock(base_filters * 3, base_filters * 2)
        self.pool2 = nn.MaxPool3d(2)
        self.enc3 = ConvBlock(base_filters * 2, base_filters * 4)
        self.pool3 = nn.MaxPool3d(2)
        self.bottleneck = ConvBlock(base_filters * 4, base_filters * 8)
        self.up3 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.dec3 = ConvBlock(base_filters * 8 + base_filters * 4, base_filters * 4)
        self.up2 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.dec2 = ConvBlock(base_filters * 4 + base_filters * 2, base_filters * 2)
        self.up1 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.dec1 = ConvBlock(base_filters * 2 + base_filters, base_filters)
        self.out_conv = nn.Conv3d(base_filters, out_channels, kernel_size=1)

    def forward(self, x):
        x1, skip1 = self.enc1_pc(x)
        skip2 = self.enc2(x1)
        x2 = self.pool2(skip2)
        skip3 = self.enc3(x2)
        x3 = self.pool3(skip3)
        x_bot = self.bottleneck(x3)
        d3 = self.dec3(torch.cat([self.up3(x_bot), skip3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), skip2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), skip1], dim=1))
        return self.out_conv(d1)


def load_weights(model, path):
    if path and os.path.exists(path):
        ckpt = torch.load(path, map_location='cpu')
        state = ckpt.get('model_state_dict', ckpt.get('state_dict', ckpt.get('model', ckpt)))
        cleaned = {}
        for k, v in state.items():
            if not torch.is_tensor(v):
                continue
            k = k.replace('module.', '').replace('model.', '')
            if k.startswith('out_conv2.') or k.startswith('out_conv3.'):
                continue
            cleaned[k] = v
        model.load_state_dict(cleaned, strict=False)


def compute_macs_and_tflops(model, dummy_input):
    if FlopCountAnalysis is not None:
        try:
            flops = FlopCountAnalysis(model, dummy_input)
            flops.set_op_handle('aten::upsample_trilinear3d', None)
            total_flops = flops.total()
            macs_g = (total_flops / 2) / 1e9
            tflops = total_flops / 1e12
            return macs_g, tflops
        except Exception:
            pass

    total_macs = 0
    hooks = []

    def conv3d_hook(module, inp, out):
        nonlocal total_macs
        x = inp[0]
        batch_size = x.shape[0]
        out_c, out_d, out_h, out_w = out.shape[1:]
        in_c = module.in_channels
        k_d, k_h, k_w = module.kernel_size
        groups = module.groups
        macs_per_instance = (in_c // groups) * k_d * k_h * k_w * out_c * out_d * out_h * out_w
        total_macs += batch_size * macs_per_instance

    for m in model.modules():
        if isinstance(m, nn.Conv3d):
            hooks.append(m.register_forward_hook(conv3d_hook))

    with torch.no_grad():
        _ = model(dummy_input)

    for h in hooks:
        h.remove()

    macs_g = total_macs / 1e9
    tflops = (2 * total_macs) / 1e12
    return macs_g, tflops


def benchmark_model(model, model_name, input_size=(1, 4, 128, 128, 128), device='cuda'):
    dev = torch.device(device)
    model = model.to(dev)

    total_params = sum(p.numel() for p in model.parameters())
    params_m = total_params / 1e6

    dummy_input = torch.randn(input_size).to(dev)

    model.eval()
    macs_g, tflops = compute_macs_and_tflops(model, dummy_input)

    for _ in range(10):
        with torch.no_grad():
            _ = model(dummy_input)

    if dev.type == 'cuda':
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(100):
            with torch.no_grad():
                _ = model(dummy_input)
        end.record()
        torch.cuda.synchronize()
        latency_ms = start.elapsed_time(end) / 100
    else:
        t0 = time.time()
        for _ in range(50):
            with torch.no_grad():
                _ = model(dummy_input)
        latency_ms = (time.time() - t0) * 1000 / 50

    inference_time_s = latency_ms / 1000

    im_mb = 0
    if dev.type == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            _ = model(dummy_input)
            torch.cuda.synchronize()
        im_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)

    tm_mb = 0
    if dev.type == 'cuda':
        for p in model.parameters():
            p.requires_grad = True
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        model.train()
        opt = torch.optim.Adam(model.parameters())
        opt.zero_grad()
        out = model(dummy_input)
        loss = (out[0].mean() if isinstance(out, (list, tuple)) else out.mean())
        loss.backward()
        torch.cuda.synchronize()
        tm_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)

    model.eval()
    return {
        "Model": model_name,
        "Params (M)": f"{params_m:.2f}",
        "MACs (G)": f"{macs_g:.2f}",
        "TFLOPS": f"{tflops:.4f}",
        "Infer Time (s)": f"{inference_time_s:.4f}",
        "Latency (ms)": f"{latency_ms:.2f}",
        "Infer Mem (MB)": f"{im_mb:.1f}",
        "Train Mem (MB)": f"{tm_mb:.1f}"
    }


def print_table(results):
    if not results:
        return
    headers = list(results[0].keys())
    widths = [max(len(h), max(len(str(r[h])) for r in results)) + 2 for h in headers]
    border = "+" + "+".join(["-" * w for w in widths]) + "+"
    print("\n" + border)
    print("|" + "|".join([f" {h:<{w - 2}} " for h, w in zip(headers, widths)]) + "|")
    print(border)
    for res in results:
        print("|" + "|".join([f" {str(res[h]):<{w - 2}} " for h, w in zip(headers, widths)]) + "|")
    print(border + "\n")


device = "cuda" if torch.cuda.is_available() else "cpu"
roi_size = (1, 4, 128, 128, 128)
student_path = "/kaggle/input/models/sorensalehi/student-model-brats2023/pytorch/default/1/best_distilled_student.pth"

student = LATUPStudent(in_channels=4, out_channels=3, base_filters=16)
load_weights(student, student_path)

res_student = benchmark_model(student, "Student (LATUP)", roi_size, device)
print_table([res_student])