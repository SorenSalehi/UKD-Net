"""Full test-set evaluation of the SAXNet teacher and the distilled LATUP student on KiTS23.

Metrics (per target and averaged over the 3 targets, same definitions as BraTS_evaluate.py):
    Dice, HD95 (mm), NSD @ 2 mm, Precision, Recall, Specificity, Lesion Recall,
    Abs RVE (absolute relative volume error), Volume MAE (mL)

Targets (KiTS23 hierarchical evaluation classes, i.e. the 3 sigmoid output channels):
    Kidney+Masses (labels 1,2,3) | Masses (2,3 = tumor + cyst) | Tumor (2)

Evaluation modes
    --eval-mode full  (default) sliding-window inference over the WHOLE resampled volume. Use this for
                      numbers you report.
    --eval-mode patch one 128^3 patch centred on the kidneys (what train.py validates on); fast and
                      optimistic, handy for smoke tests.

Metrics are computed on the resampled grid (default 3 mm isotropic, the grid the models were trained
on), NOT in the original CT space, so HD95 / NSD are limited by that resolution. --kits-spacing must
match the value used in training.

Example
    python Test.py --data-dir /content/kits23/dataset --save-dir /content/kits_smoke
"""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt, generate_binary_structure, label as cc_label

TARGETS = ["Kidney+Masses", "Masses", "Tumor"]
METRICS = ["dice", "hd95", "nsd", "precision", "recall", "specificity", "lesion_recall", "abs_rve", "volume_mae_ml"]
HEADERS = {"dice": "Dice ↑", "hd95": "HD95 (mm) ↓", "nsd": "NSD (2mm) ↑", "precision": "Precision ↑",
           "recall": "Recall ↑", "specificity": "Specificity ↑", "lesion_recall": "Lesion Recall ↑",
           "abs_rve": "Abs RVE ↓", "volume_mae_ml": "Vol MAE (mL) ↓"}
NSD_TOLERANCE_MM = 2.0
CONNECTIVITY = 3
MIN_LESION_VOXELS = 1


# ----------------------------------------------------------------------------
# Metrics (same definitions as BraTS_evaluate.compute_all_metrics)
# ----------------------------------------------------------------------------
def _surface(mask):
    if not mask.any():
        return np.zeros_like(mask, dtype=bool)
    return np.logical_xor(mask, binary_erosion(mask, iterations=1, border_value=0))


def _directed_surface_distances(a, b, spacing):
    if not a.any() or not b.any():
        return np.empty(0, dtype=np.float32)
    dt = distance_transform_edt(~b, sampling=spacing)
    return dt[a].astype(np.float32)


def compute_all_metrics(pred, gt, spacing, tolerance=NSD_TOLERANCE_MM):
    p_bin, g_bin = pred.astype(bool), gt.astype(bool)
    ps, gs = int(p_bin.sum()), int(g_bin.sum())
    tp = int(np.logical_and(p_bin, g_bin).sum())
    fp = ps - tp
    fn = gs - tp
    tn = int(p_bin.size) - tp - fp - fn

    dice = 1.0 if (ps == 0 and gs == 0) else (0.0 if (ps == 0 or gs == 0) else 2.0 * tp / (ps + gs))
    precision = tp / (tp + fp) if (tp + fp) else (1.0 if (tp + fn) == 0 else 0.0)
    recall = tp / (tp + fn) if (tp + fn) else (1.0 if (tp + fp) == 0 else 0.0)
    specificity = tn / (tn + fp) if (tn + fp) else 0.0

    # Lesion recall: fraction of GT connected components touched by at least one predicted voxel.
    # (Equivalent to the per-component loop in BraTS_evaluate.py, but fast when there are many components.)
    structure = generate_binary_structure(3, CONNECTIVITY)
    gl, ngt = cc_label(g_bin, structure=structure)
    if ngt:
        sizes = np.bincount(gl.ravel(), minlength=ngt + 1)
        valid = {i for i in range(1, ngt + 1) if sizes[i] >= MIN_LESION_VOXELS}
        hit = set(np.unique(gl[p_bin]).tolist()) - {0}
        lesion_recall = len(hit & valid) / len(valid) if valid else 1.0
    else:
        lesion_recall = 1.0

    voxel_ml = float(np.prod(spacing)) / 1000.0
    v_p, v_g = ps * voxel_ml, gs * voxel_ml
    abs_rve = abs((v_p - v_g) / v_g) if v_g > 0 else (0.0 if v_p == 0 else float("nan"))

    if ps == 0 and gs == 0:
        hd95, nsd = 0.0, 1.0
    elif ps == 0 or gs == 0:
        hd95 = float(np.linalg.norm(np.asarray(pred.shape, float) * np.asarray(spacing, float)))
        nsd = 0.0
    else:
        dp = _directed_surface_distances(_surface(p_bin), _surface(g_bin), spacing)
        dg = _directed_surface_distances(_surface(g_bin), _surface(p_bin), spacing)
        hd95 = float(np.percentile(np.concatenate([dp, dg]), 95))
        total = dp.size + dg.size
        nsd = float(((dp <= tolerance).sum() + (dg <= tolerance).sum()) / total) if total else 1.0

    return {"dice": float(dice), "hd95": hd95, "nsd": nsd, "precision": float(precision), "recall": float(recall),
            "specificity": float(specificity), "lesion_recall": float(lesion_recall),
            "abs_rve": float(abs_rve), "volume_mae_ml": float(abs(v_p - v_g))}


# ----------------------------------------------------------------------------
# Inference
# ----------------------------------------------------------------------------
def _starts(n, roi, stride):
    s = list(range(0, n - roi + 1, stride))
    if s[-1] != n - roi:
        s.append(n - roi)
    return s


def _importance_map(roi):
    """Gaussian window: centre voxels of a window count more than its borders (reduces seams)."""
    ax = np.arange(roi) - (roi - 1) / 2.0
    g = np.exp(-0.5 * (ax / (roi / 8.0)) ** 2)
    w = g[:, None, None] * g[None, :, None] * g[None, None, :]
    return np.clip(w / w.max(), 1e-3, None).astype(np.float32)


def sliding_window_probs(predict_fn, image, roi, overlap=0.5):
    """image: (1,D,H,W) float32. predict_fn((1,1,r,r,r) float32 array) -> (3,r,r,r) probabilities.
    Returns (3,D,H,W) float32 probabilities for the full volume."""
    _, D, H, W = image.shape
    pads = [max(0, roi - n) for n in (D, H, W)]
    padded = np.pad(image, ((0, 0), (0, pads[0]), (0, pads[1]), (0, pads[2])), constant_values=float(image.min()))
    _, pD, pH, pW = padded.shape
    stride = max(1, int(roi * (1.0 - overlap)))
    weight = _importance_map(roi)
    acc = np.zeros((3, pD, pH, pW), np.float32)
    cnt = np.zeros((pD, pH, pW), np.float32)
    for z in _starts(pD, roi, stride):
        for y in _starts(pH, roi, stride):
            for x in _starts(pW, roi, stride):
                patch = padded[None, :, z:z + roi, y:y + roi, x:x + roi]
                probs = predict_fn(np.ascontiguousarray(patch))
                acc[:, z:z + roi, y:y + roi, x:x + roi] += probs * weight
                cnt[z:z + roi, y:y + roi, x:x + roi] += weight
    return (acc / cnt)[:, :D, :H, :W]


def make_predict_fn(model, device, amp):
    import torch

    @torch.no_grad()
    def predict(patch):
        x = torch.from_numpy(patch).to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            logits = model(x)
        return torch.sigmoid(logits.float())[0].cpu().numpy()

    return predict


# ----------------------------------------------------------------------------
# Data helpers
# ----------------------------------------------------------------------------
def labels_to_channels(seg):
    return np.stack([seg >= 1, seg >= 2, seg == 2], axis=0).astype(np.uint8)  # same as KiTS23Dataset


def eval_patch(image, seg, roi):
    """Kidney-centred patch, identical to KiTS23Dataset(mode='eval')."""
    idx = np.argwhere(seg > 0)
    center = np.asarray(seg.shape) // 2 if idx.size == 0 else (idx.min(0) + idx.max(0)) // 2
    start = np.clip(center - roi // 2, 0, np.maximum(np.asarray(seg.shape) - roi, 0))

    def crop(a, fill):
        out = np.full((roi,) * 3, fill, dtype=a.dtype)
        lo = np.maximum(start, 0)
        hi = np.minimum(start + roi, a.shape)
        d_lo = lo - start
        d_hi = d_lo + (hi - lo)
        out[d_lo[0]:d_hi[0], d_lo[1]:d_hi[1], d_lo[2]:d_hi[2]] = a[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
        return out

    return crop(image[0], float(image.min()))[None], crop(seg, 0)


# ----------------------------------------------------------------------------
# Aggregation / reporting
# ----------------------------------------------------------------------------
def summarize(per_case):
    """per_case: list of {target: {metric: value}} -> {target: {metric: {mean,std,n}}, 'Mean': {metric: mean}}"""
    out = {}
    for t in TARGETS:
        out[t] = {}
        for m in METRICS:
            v = np.array([c[t][m] for c in per_case], dtype=np.float64)
            out[t][m] = {"mean": float(np.nanmean(v)) if np.isfinite(v).any() else None,
                         "std": float(np.nanstd(v)) if np.isfinite(v).any() else None,
                         "n": int(np.isfinite(v).sum())}
    out["Mean"] = {m: float(np.mean([out[t][m]["mean"] for t in TARGETS if out[t][m]["mean"] is not None]))
                   for m in METRICS}
    return out


def print_table(name, summary, n_cases):
    cols = [HEADERS[m] for m in METRICS]
    w = [max(len(c), 8) for c in cols]
    line = "=" * (12 + sum(x + 3 for x in w))
    print(f"\n{line}\n{name}  ({n_cases} test cases)\n{line}")
    print(f"{'Target':<14}" + " | ".join(f"{c:^{x}}" for c, x in zip(cols, w)))
    print("-" * len(line))
    for t in TARGETS + ["Mean"]:
        vals = []
        for m, x in zip(METRICS, w):
            v = summary["Mean"][m] if t == "Mean" else summary[t][m]["mean"]
            s = "n/a" if v is None or (isinstance(v, float) and math.isnan(v)) else (
                f"{v:.2f}" if m in ("hd95", "volume_mae_ml") else f"{v:.4f}")
            vals.append(f"{s:^{x}}")
        print(f"{t:<14}" + " | ".join(vals))
    print(line)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--save-dir", type=Path, required=True, help="train.py --save-dir (holds split json, cache, checkpoints)")
    p.add_argument("--data-dir", type=Path, default=None, help="KiTS23 `dataset` folder; only needed if the cache is missing")
    p.add_argument("--teacher-checkpoint", type=Path, default=None, help="default: <save-dir>/best_saxnet_teacher.pth")
    p.add_argument("--student-checkpoint", type=Path, default=None, help="default: <save-dir>/best_distilled_student.pth")
    p.add_argument("--split-file", type=Path, default=None, help="default: <save-dir>/split_70_10_20.json")
    p.add_argument("--cache-dir", type=Path, default=None, help="default: <save-dir>/kits23_cache")
    p.add_argument("--models", nargs="+", choices=("teacher", "student"), default=["teacher", "student"])
    p.add_argument("--eval-mode", choices=("full", "patch"), default="full")
    p.add_argument("--roi-size", type=int, default=128, help="must match training")
    p.add_argument("--kits-spacing", type=float, default=3.0, help="must match training (mm, isotropic)")
    p.add_argument("--overlap", type=float, default=0.5, help="sliding-window overlap (full mode)")
    p.add_argument("--limit", type=int, default=None, help="evaluate only the first N test cases")
    p.add_argument("--no-amp", action="store_true", help="run the teacher in fp32 (student is always fp32)")
    p.add_argument("--out", type=Path, default=None, help="default: <save-dir>/test_metrics_full.json")
    return p.parse_args()


def main():
    args = parse_args()
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        print("WARNING: no GPU found; 3D inference on CPU will be very slow.")
    save_dir = args.save_dir
    split_path = args.split_file or save_dir / "split_70_10_20.json"
    cache_dir = args.cache_dir or save_dir / "kits23_cache"
    out_path = args.out or save_dir / "test_metrics_full.json"
    spacing = (args.kits_spacing,) * 3
    roi = args.roi_size

    split = json.loads(split_path.read_text())
    cases = split["test"][: args.limit] if args.limit else split["test"]
    print(f"Test cases: {len(cases)} | mode={args.eval_mode} | spacing={args.kits_spacing} mm | ROI={roi}")

    # Make sure every test case is cached (only runs the preprocessing if something is missing).
    missing = [c for c in cases if not (cache_dir / f"{c}.npz").exists()]
    if missing:
        if args.data_dir is None:
            raise FileNotFoundError(f"{len(missing)} cached cases missing in {cache_dir}; pass --data-dir to rebuild.")
        from data.kits23_dataset import preprocess_kits_cases
        preprocess_kits_cases([args.data_dir / c for c in missing], cache_dir, spacing, workers=2)

    # ---- models (same construction as train.py) ----
    from models.saxnet import SAXNetV1
    from models.student import LATUPStudent

    def load(cls_name, ckpt_path):
        if cls_name == "teacher":
            model = SAXNetV1(in_channels=1, out_channels=3)
        else:
            model = LATUPStudent(in_channels=1, out_channels=3, base_filters=16, deep_supervision=True)
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        sd = state["model_state_dict"] if isinstance(state, dict) and "model_state_dict" in state else state
        model.load_state_dict(sd)
        model.to(device).eval().requires_grad_(False)
        epoch = state.get("epoch") if isinstance(state, dict) else None
        print(f"Loaded {cls_name} from {ckpt_path} (epoch {epoch})")
        return model

    ckpts = {"teacher": args.teacher_checkpoint or save_dir / "best_saxnet_teacher.pth",
             "student": args.student_checkpoint or save_dir / "best_distilled_student.pth"}
    predict = {}
    for name in args.models:
        amp = (name == "teacher") and not args.no_amp and device.type == "cuda"
        predict[name] = make_predict_fn(load(name, ckpts[name]), device, amp)

    # ---- evaluation loop (both models see exactly the same cases) ----
    per_case = {name: [] for name in args.models}
    per_case_named = {}
    t0 = time.time()
    for i, case in enumerate(cases, 1):
        with np.load(cache_dir / f"{case}.npz") as d:
            image = d["image"].astype(np.float32)[None]  # (1,D,H,W)
            seg = d["seg"]
        if args.eval_mode == "patch":
            image, seg = eval_patch(image, seg, roi)
        gt = labels_to_channels(seg)
        per_case_named[case] = {}
        for name in args.models:
            if args.eval_mode == "full":
                probs = sliding_window_probs(predict[name], image, roi, args.overlap)
            else:
                probs = predict[name](np.ascontiguousarray(image[None]))
            pred = (probs >= 0.5).astype(np.uint8)
            res = {t: compute_all_metrics(pred[c], gt[c], spacing) for c, t in enumerate(TARGETS)}
            per_case[name].append(res)
            per_case_named[case][name] = res
        msg = " | ".join(f"{n}: Dice={np.mean([per_case[n][-1][t]['dice'] for t in TARGETS]):.3f}" for n in args.models)
        print(f"[{i}/{len(cases)}] {case} {msg}  ({time.time() - t0:.0f}s)")

    # ---- report ----
    report = {
        "config": {"dataset": "KiTS23", "eval_mode": args.eval_mode, "spacing_mm": args.kits_spacing, "roi": roi,
                   "overlap": args.overlap if args.eval_mode == "full" else None, "n_cases": len(cases),
                   "nsd_tolerance_mm": NSD_TOLERANCE_MM, "targets": TARGETS,
                   "checkpoints": {n: str(ckpts[n]) for n in args.models},
                   "note": "Metrics are computed on the resampled grid, not the original CT space."},
    }
    for name in args.models:
        summary = summarize(per_case[name])
        report[name] = summary
        print_table(f"{name.upper()} - KiTS23", summary, len(cases))
    report["per_case"] = per_case_named

    def clean(o):  # JSON has no NaN
        if isinstance(o, dict):
            return {k: clean(v) for k, v in o.items()}
        if isinstance(o, float) and not math.isfinite(o):
            return None
        return o

    out_path.write_text(json.dumps(clean(report), indent=2) + "\n")
    print(f"\nSaved results to {out_path}")


if __name__ == "__main__":
    main()
