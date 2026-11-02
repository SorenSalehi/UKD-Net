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


def load_student(path, device):
    if not path.exists():
        raise FileNotFoundError(f"Student checkpoint not found: {path}")
    model = LATUPStudent(in_channels=4, out_channels=3, base_filters=16, deep_supervision=True).to(device)
    state = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state_dict"])
    return model


def main():
    args = parse_args()
    args.save_dir.mkdir(parents=True, exist_ok=True)
    set_seed()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this 3D training pipeline.")
    device = torch.device("cuda")
    amp = not args.no_amp
    print(f"Device: {torch.cuda.get_device_name(0)} | AMP={amp}")

    actual_root, train_cases, val_cases, mode = load_or_create_fold(
        args.data_dir, args.save_dir, args.fold, args.num_folds
    )
    sdf_dir = args.save_dir / "sdfs" if args.stage in ("all", "student") else None
    teacher_train, student_train, val_loader = make_loaders(
        actual_root, train_cases, val_cases, mode,
        args.teacher_batch_size, args.student_batch_size, args.workers, sdf_dir=sdf_dir
    )
    teacher_path = args.teacher_checkpoint or (args.save_dir / "best_saxnet_teacher.pth")
    student_path = args.save_dir / "best_distilled_student.pth"

    if args.stage in ("all", "teacher"):
        teacher_path = train_teacher(args, teacher_train, val_loader, device, amp)
    if args.stage in ("all", "student"):
        teacher = load_frozen_teacher(teacher_path, device)
        print("Student method: train.py-compatible FP32 + SDF Boundary Loss + fixed LR")
        student_path = train_student(args, teacher, student_train, val_loader, device, amp=False)
        del teacher
        torch.cuda.empty_cache()
    if args.stage in ("all", "test"):
        teacher = load_frozen_teacher(teacher_path, device)
        student = load_student(student_path, device)
        teacher_scores = evaluate(teacher, val_loader, device, amp, "Teacher held-out fold")
        student_scores = evaluate(student, val_loader, device, False, "Student held-out fold")
        print_scores("FINAL SAXNet teacher held-out fold (NOT independent test)", teacher_scores)
        print_scores("FINAL distilled student held-out fold (NOT independent test)", student_scores)
        report = {"fold": args.fold, "validation_is_evaluation": True,
                  "teacher": teacher_scores, "student": student_scores}
        (args.save_dir / "heldout_fold_metrics.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
