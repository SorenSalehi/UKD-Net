"""Train a SAXNet teacher, then distill it into LATUPStudent.

This is an independent training version. It does not modify or reuse the split
logic of train.py/train_teacher.py. A deterministic 70/10/20 split is persisted
to the result directory and reused on every resume.
"""

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.dataset import BraTS2023Dataset, find_nii_adaptive, precompute_sdfs_unified
from data.kits23_dataset import KiTS23Dataset, discover_kits_cases, preprocess_kits_cases
from models.saxnet import SAXNetV1
from models.student import LATUPStudent
from utils.loss_metrics import (
    BoundaryLoss,
    DWKDLoss,
    WeightedDiceBCELoss,
    calculate_dice,
    calculate_hd95_per_channel,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = ROOT / "data" / "brats2023" / "Training"
DEFAULT_SAVE_DIR = ROOT / "saxnet_distill_results_70_10_20_v1"
SPLIT_SEED = 42
ROI_SIZE = (128, 128, 128)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("brats", "kits23"), default="brats",
                        help="brats = 4-ch MRI (default, original behaviour); kits23 = 1-ch CT kidney/tumor/cyst")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--max-cases", type=int, default=None,
                        help="Use only the first N cases (smoke tests). Use a fresh --save-dir when changing it.")
    parser.add_argument("--roi-size", type=int, default=128, help="Cubic patch size (e.g. 96 to save GPU memory)")
    parser.add_argument("--kits-spacing", type=float, default=3.0, help="KiTS23 isotropic resampling spacing (mm)")
    parser.add_argument("--prep-workers", type=int, default=2, help="Processes for the one-off KiTS23 resampling")
    parser.add_argument("--save-dir", type=Path, default=DEFAULT_SAVE_DIR)
    parser.add_argument("--split-file", type=Path, default=None)
    parser.add_argument("--teacher-checkpoint", type=Path, default=None)
    parser.add_argument("--stage", choices=("all", "teacher", "student", "test"), default="all")
    parser.add_argument("--teacher-epochs", type=int, default=100)
    parser.add_argument("--student-epochs", type=int, default=100)
    parser.add_argument("--teacher-batch-size", type=int, default=1)
    parser.add_argument("--student-batch-size", type=int, default=2)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--val-interval", type=int, default=5)
    parser.add_argument("--teacher-lr", type=float, default=5e-5)
    parser.add_argument("--student-lr", type=float, default=1e-4)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument(
        "--rce-weight",
        type=float,
        default=0.1,
        help="Weight of RCE loss. Set to 0.0 to disable RCE.",
    )
    return parser.parse_args()


def set_seed(seed=SPLIT_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def discover_valid_cases(data_dir, dataset="brats"):
    if dataset == "kits23":
        return discover_kits_cases(data_dir)
    cases = sorted(p for p in Path(data_dir).iterdir() if p.is_dir() and not p.name.startswith("."))
    required = ("flair", "t1", "t1ce", "t2", "seg")
    valid = [p for p in cases if all(find_nii_adaptive(str(p), modality) for modality in required)]
    if not valid:
        raise FileNotFoundError(f"No complete BraTS training cases found under {data_dir}")
    return valid


def load_or_create_split(data_dir, split_path, dataset="brats", max_cases=None):
    valid_cases = discover_valid_cases(data_dir, dataset)
    if max_cases:
        valid_cases = valid_cases[:max_cases]
    names = [case.name for case in valid_cases]
    if split_path.exists():
        split = json.loads(split_path.read_text())
        saved = split["train"] + split["val"] + split["test"]
        if sorted(saved) != sorted(names):
            raise RuntimeError("Existing split does not match the current dataset; use a new save directory.")
    else:
        indices = np.random.default_rng(SPLIT_SEED).permutation(len(names))
        n_train = round(0.70 * len(names))
        n_val = round(0.10 * len(names))
        train_end = n_train
        val_end = n_train + n_val
        split = {
            "seed": SPLIT_SEED,
            "ratios": {"train": 0.70, "val": 0.10, "test": 0.20},
            "train": [names[i] for i in indices[:train_end]],
            "val": [names[i] for i in indices[train_end:val_end]],
            "test": [names[i] for i in indices[val_end:]],
        }
        split_path.write_text(json.dumps(split, indent=2) + "\n")
    print(
        f"Fixed split (seed={split['seed']}): train={len(split['train'])}, "
        f"val={len(split['val'])}, test={len(split['test'])}"
    )
    return split


def make_loaders(data_dir, split, teacher_batch_size, student_batch_size, workers, sdf_dir=None,
                 dataset="brats", cache_dir=None, kits_spacing=3.0, prep_workers=2):
    def paths(part):
        return [str(Path(data_dir) / name) for name in split[part]]

    common = {"num_workers": workers, "pin_memory": True, "persistent_workers": workers > 0}

    if dataset == "kits23":
        # One-off resampling to isotropic spacing; cached so later runs start instantly.
        all_dirs = paths("train") + paths("val") + paths("test")
        preprocess_kits_cases(all_dirs, cache_dir, (kits_spacing,) * 3, workers=prep_workers)
        need_sdf = sdf_dir is not None  # only the student stage uses the boundary loss
        datasets = {
            "train": KiTS23Dataset(split["train"], cache_dir, ROI_SIZE, mode="train", compute_sdf=need_sdf),
            "train_teacher": KiTS23Dataset(split["train"], cache_dir, ROI_SIZE, mode="train", compute_sdf=False),
            "val": KiTS23Dataset(split["val"], cache_dir, ROI_SIZE, mode="eval"),
            "test": KiTS23Dataset(split["test"], cache_dir, ROI_SIZE, mode="eval"),
        }
    else:
        if sdf_dir is not None:
            sdf_dir = Path(sdf_dir)
            sdf_dir.mkdir(parents=True, exist_ok=True)
            # Match train.py: precompute SDFs for the training and validation cases.
            precompute_sdfs_unified(
                str(data_dir),
                paths("train") + paths("val"),
                str(sdf_dir),
                "BraTS",
                ROI_SIZE,
            )
        datasets = {
            "train": BraTS2023Dataset(
                paths("train"), sdf_dir=str(sdf_dir) if sdf_dir else None,
                roi_size=ROI_SIZE, augment=True, return_case_name=True
            ),
            "val": BraTS2023Dataset(
                paths("val"), sdf_dir=str(sdf_dir) if sdf_dir else None,
                roi_size=ROI_SIZE, augment=False, return_case_name=True
            ),
            "test": BraTS2023Dataset(paths("test"), roi_size=ROI_SIZE, augment=False, return_case_name=True),
        }
    teacher_train = DataLoader(datasets.get("train_teacher", datasets["train"]), batch_size=teacher_batch_size, shuffle=True, **common)
    student_train = DataLoader(datasets["train"], batch_size=student_batch_size, shuffle=True, **common)
    val = DataLoader(datasets["val"], batch_size=1, shuffle=False, **common)
    test = DataLoader(datasets["test"], batch_size=1, shuffle=False, **common)
    return teacher_train, student_train, val, test


def autocast_context(device, amp):
    return torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp)


def checkpoint_state(epoch, model, optimizer, scheduler, best_dice):
    return {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_dice": best_dice,
        "split_seed": SPLIT_SEED,
        "split_ratios": (0.70, 0.10, 0.20),
    }


@torch.no_grad()
def evaluate(model, loader, device, amp, description):
    model.eval()
    all_dice = []
    for images, labels, _, _ in tqdm(loader, desc=description, leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with autocast_context(device, amp):
            logits = model(images)
        all_dice.append(calculate_dice((torch.sigmoid(logits) > 0.5).float(), labels))
    scores = np.asarray(all_dice, dtype=np.float64)
    means = scores.mean(axis=0)
    return {"mean": float(means.mean()), "wt": float(means[0]), "tc": float(means[1]), "et": float(means[2])}


def print_scores(prefix, scores):
    print(
        f"{prefix}: Dice={scores['mean']:.4f} "
        f"(WT={scores['wt']:.4f}, TC={scores['tc']:.4f}, ET={scores['et']:.4f})"
    )


@torch.no_grad()
def evaluate_student_trainpy(model, loader, device, description):
    """Match train.py validation: FP32 Dice plus HD95."""
    model.eval()
    all_dice, all_hd95 = [], []
    for images, labels, _, _ in tqdm(loader, desc=description, leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        preds = (torch.sigmoid(logits) > 0.5).float()
        all_dice.append(calculate_dice(preds, labels))
        all_hd95.append(calculate_hd95_per_channel(preds, labels, roi_size=ROI_SIZE))
    dice = np.asarray(all_dice, dtype=np.float64)
    hd95 = np.asarray(all_hd95, dtype=np.float64)
    means = dice.mean(axis=0)
    return {
        "mean": float(means.mean()),
        "wt": float(means[0]),
        "tc": float(means[1]),
        "et": float(means[2]),
        "hd95": float(np.nanmean(hd95)),
    }


def train_teacher(args, train_loader, val_loader, device, amp, in_channels=4):
    best_path = args.save_dir / "best_saxnet_teacher.pth"
    last_path = args.save_dir / "last_saxnet_teacher.pth"
    model = SAXNetV1(in_channels=in_channels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.teacher_lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.teacher_epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler(device.type, enabled=amp)
    loss_fn = WeightedDiceBCELoss(weights=[1.0, 1.0, 3.0])
    start_epoch, best_dice = 1, 0.0

    if last_path.exists():
        state = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        scheduler.load_state_dict(state["scheduler_state_dict"])
        start_epoch = state["epoch"] + 1
        best_dice = state.get("best_dice", 0.0)
        print(f"Resuming SAXNet teacher at epoch {start_epoch}; best Dice={best_dice:.4f}")

    print(f"SAXNet teacher parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M")
    for epoch in range(start_epoch, args.teacher_epochs + 1):
        model.train()
        losses, dices = [], []
        progress = tqdm(train_loader, desc=f"Teacher {epoch}/{args.teacher_epochs}")
        for images, labels, _, _ in progress:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(device, amp):
                logits = model(images)
                loss = loss_fn(logits, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            dice = calculate_dice((torch.sigmoid(logits.detach()) > 0.5).float(), labels)
            losses.append(loss.item())
            dices.append(np.mean(dice))
            progress.set_postfix(loss=f"{loss.item():.4f}", dice=f"{np.mean(dice):.3f}")
        scheduler.step()

        validate = epoch == 1 or epoch % args.val_interval == 0 or epoch == args.teacher_epochs
        if validate:
            scores = evaluate(model, val_loader, device, amp, f"Teacher val {epoch}")
            print_scores(f"Teacher epoch {epoch} | loss={np.mean(losses):.4f}", scores)
            if scores["mean"] > best_dice:
                best_dice = scores["mean"]
                torch.save(checkpoint_state(epoch, model, optimizer, scheduler, best_dice), best_path)
                print(f"Saved best SAXNet teacher: {best_path}")
        else:
            print(f"Teacher epoch {epoch}: loss={np.mean(losses):.4f}, train Dice={np.mean(dices):.4f}")
        torch.save(checkpoint_state(epoch, model, optimizer, scheduler, best_dice), last_path)
    return best_path


def load_frozen_teacher(path, device, in_channels=4):
    if not path.exists():
        raise FileNotFoundError(f"SAXNet teacher checkpoint not found: {path}")
    model = SAXNetV1(in_channels=in_channels).to(device)
    state = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    model.requires_grad_(False)
    print(f"Loaded frozen SAXNet teacher from epoch {state['epoch']} (Dice={state['best_dice']:.4f})")
    return model


def train_student(args, teacher, train_loader, val_loader, device, amp=False, in_channels=4):
    best_path = args.save_dir / "best_distilled_student.pth"
    last_path = args.save_dir / "last_distilled_student.pth"
    student = LATUPStudent(in_channels=in_channels, out_channels=3, base_filters=16, deep_supervision=True).to(device)
    optimizer = torch.optim.AdamW(student.parameters(), lr=args.student_lr, weight_decay=1e-5)
    task_loss = WeightedDiceBCELoss(weights=[1.0, 1.0, 3.0])
    kd_loss = DWKDLoss(
        temperature=3.0,
        use_rce=args.rce_weight > 0.0,
        rce_t=0.9,
    )
    boundary_loss = BoundaryLoss()
    start_epoch, best_dice = 1, 0.0

    if last_path.exists():
        state = torch.load(last_path, map_location=device, weights_only=False)
        student.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        start_epoch = state["epoch"] + 1
        best_dice = state.get("best_dice", 0.0)
        print(f"Resuming distilled student at epoch {start_epoch}; best Dice={best_dice:.4f}")

    print(f"Student parameters: {sum(p.numel() for p in student.parameters()) / 1e6:.2f} M")
    print(f"RCE weight: {args.rce_weight:.4f} | enabled={args.rce_weight > 0.0}")
    for epoch in range(start_epoch, args.student_epochs + 1):
        student.train()
        losses, dices = [], []
        progress = tqdm(train_loader, desc=f"Student {epoch}/{args.student_epochs}")
        for images, labels, sdfs, _ in progress:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            sdfs = sdfs.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.no_grad():
                teacher_logits = teacher(images)
            outputs = student(images)
            main, half, quarter = outputs
            supervised = task_loss(main, labels)
            dwkd, rce = kd_loss(main, teacher_logits)
            boundary = boundary_loss(main, labels, precomputed_sdfs=sdfs)
            aux_half = task_loss(half, F.interpolate(labels, scale_factor=0.5, mode="nearest"))
            aux_quarter = task_loss(quarter, F.interpolate(labels, scale_factor=0.25, mode="nearest"))
            # Exactly match train.py's effective weights, including Boundary Loss.
            loss = (
                0.6 * supervised + 0.2 * dwkd + args.rce_weight * rce + 0.1 * boundary
                + 0.3 * aux_half + 0.15 * aux_quarter
            )
            loss.backward()
            optimizer.step()
            dice = calculate_dice((torch.sigmoid(main.detach()) > 0.5).float(), labels)
            losses.append(loss.item())
            dices.append(np.mean(dice))
            progress.set_postfix(loss=f"{loss.item():.4f}", dice=f"{np.mean(dice):.3f}")
        validate = epoch == 1 or epoch % args.val_interval == 0 or epoch == args.student_epochs
        if validate:
            scores = evaluate_student_trainpy(student, val_loader, device, f"Student val {epoch}")
            print_scores(f"Student epoch {epoch} | loss={np.mean(losses):.4f}", scores)
            print(f"Student epoch {epoch} | HD95={scores['hd95']:.2f}")
            if scores["mean"] > best_dice:
                best_dice = scores["mean"]
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": student.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_dice": best_dice,
                    "rce_weight": args.rce_weight,
                }, best_path)
                print(f"Saved best distilled student: {best_path}")
        else:
            print(f"Student epoch {epoch}: loss={np.mean(losses):.4f}, train Dice={np.mean(dices):.4f}")
        torch.save({
            "epoch": epoch,
            "model_state_dict": student.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_dice": best_dice,
            "rce_weight": args.rce_weight,
        }, last_path)
    return best_path


def load_student(path, device, in_channels=4):
    if not path.exists():
        raise FileNotFoundError(f"Student checkpoint not found: {path}")
    model = LATUPStudent(in_channels=in_channels, out_channels=3, base_filters=16, deep_supervision=True).to(device)
    state = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state_dict"])
    return model


def main():
    global ROI_SIZE
    args = parse_args()
    ROI_SIZE = (args.roi_size,) * 3
    in_ch = 1 if args.dataset == "kits23" else 4
    args.save_dir.mkdir(parents=True, exist_ok=True)
    set_seed()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this 3D training pipeline.")
    device = torch.device("cuda")
    amp = not args.no_amp
    print(f"Device: {torch.cuda.get_device_name(0)} | AMP={amp}")

    split_path = args.split_file or (args.save_dir / "split_70_10_20.json")
    split = load_or_create_split(args.data_dir, split_path, args.dataset, args.max_cases)
    sdf_dir = args.save_dir / "sdfs" if args.stage in ("all", "student") else None
    teacher_train, student_train, val_loader, test_loader = make_loaders(
        args.data_dir, split, args.teacher_batch_size, args.student_batch_size, args.workers, sdf_dir=sdf_dir,
        dataset=args.dataset, cache_dir=args.save_dir / "kits23_cache", kits_spacing=args.kits_spacing,
        prep_workers=args.prep_workers,
    )
    teacher_path = args.teacher_checkpoint or (args.save_dir / "best_saxnet_teacher.pth")
    student_path = args.save_dir / "best_distilled_student.pth"

    if args.stage in ("all", "teacher"):
        teacher_path = train_teacher(args, teacher_train, val_loader, device, amp, in_channels=in_ch)
    if args.stage in ("all", "student"):
        teacher = load_frozen_teacher(teacher_path, device, in_channels=in_ch)
        print("Student method: train.py-compatible FP32 + SDF Boundary Loss + fixed LR")
        student_path = train_student(args, teacher, student_train, val_loader, device, amp=False, in_channels=in_ch)
        del teacher
        torch.cuda.empty_cache()
    if args.stage in ("all", "test"):
        teacher = load_frozen_teacher(teacher_path, device, in_channels=in_ch)
        student = load_student(student_path, device, in_channels=in_ch)
        teacher_scores = evaluate(teacher, test_loader, device, amp, "Teacher test")
        student_scores = evaluate(student, test_loader, device, False, "Student test")
        print_scores("FINAL SAXNet teacher test", teacher_scores)
        print_scores("FINAL distilled student test", student_scores)
        report = {"teacher": teacher_scores, "student": student_scores}
        (args.save_dir / "test_metrics.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
