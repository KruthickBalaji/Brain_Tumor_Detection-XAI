"""
train.py
Brain Tumor YOLO Classification Training & Evaluation Script
Trains YOLOv8m-cls / YOLO11m-cls on the 4-class MRI brain tumor dataset.
Evaluates Top-1 Accuracy, Precision, Recall, and Confusion Matrix, and saves weights to weights/best.pt.
"""

import os
import sys
import shutil
import json
import argparse
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from ultralytics import YOLO

# Standard classes in exact order
CLASSES = ["glioma_tumor", "meningioma_tumor", "no_tumor", "pituitary_tumor"]


def ensure_dataset_compatibility(dataset_root: Path):
    """
    Ensures that Ultralytics dataset loader can find train, val, and test splits.
    The original folder layout contains 'Training' and 'Testing'.
    Creates OS directory links/junctions from 'train' -> 'Training' and 'val'/'test' -> 'Testing' if needed.
    """
    training_dir = dataset_root / "Training"
    testing_dir = dataset_root / "Testing"

    if not training_dir.exists():
        raise FileNotFoundError(f"Expected training directory at: {training_dir}")
    if not testing_dir.exists():
        raise FileNotFoundError(f"Expected testing directory at: {testing_dir}")

    # Check for Ultralytics split conventions
    splits = {
        "train": training_dir,
        "val": testing_dir,
        "test": testing_dir,
    }

    for split_name, target_dir in splits.items():
        link_path = dataset_root / split_name
        if not link_path.exists():
            print(f"Creating link: {link_path} -> {target_dir}")
            try:
                if sys.platform == "win32":
                    # Windows directory junction
                    import subprocess
                    subprocess.run(
                        f'cmd /c mklink /J "{link_path}" "{target_dir}"',
                        shell=True,
                        check=True,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                    )
                else:
                    os.symlink(target_dir, link_path, target_is_directory=True)
            except Exception as e:
                print(f"Warning: Could not create link ({e}). Attempting fallback copy...")
                shutil.copytree(target_dir, link_path)


def plot_and_save_confusion_matrix(cm: np.ndarray, classes: list, save_path: Path):
    """Plots and exports a professional confusion matrix heatmap."""
    plt.figure(figsize=(8, 6), dpi=150)
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Brain Tumor YOLO Classification - Confusion Matrix", fontsize=13, fontweight="bold", pad=12)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    clean_labels = [c.replace("_tumor", "").title() for c in classes]
    plt.xticks(tick_marks, clean_labels, rotation=45, ha="right", fontsize=10)
    plt.yticks(tick_marks, clean_labels, fontsize=10)

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            text = f"{int(val)}" if val == int(val) else f"{val:.2f}"
            plt.text(
                j,
                i,
                text,
                horizontalalignment="center",
                verticalalignment="center",
                color="white" if val > thresh else "black",
                fontweight="bold",
            )

    plt.ylabel("True Clinical Label", fontweight="bold")
    plt.xlabel("Model Predicted Label", fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"Confusion matrix plot saved to: {save_path}")


def evaluate_model(model: YOLO, dataset_path: Path, weights_dir: Path):
    """
    Evaluates model on content/Testing and calculates:
    - Top-1 Accuracy
    - Precision, Recall, F1-Score per class and macro-average
    - Formatted Confusion Matrix
    """
    print("\n" + "=" * 60)
    print("           MODEL EVALUATION ON TESTING SET")
    print("=" * 60)

    # Run validation on test split
    metrics = model.val(data=str(dataset_path), split="test")

    # Extract metrics
    top1_acc = float(metrics.top1) * 100.0
    top5_acc = float(metrics.top5) * 100.0 if hasattr(metrics, "top5") else 100.0

    print(f"\n[+] Top-1 Accuracy: {top1_acc:.2f}%")
    if hasattr(metrics, "top5"):
        print(f"[+] Top-5 Accuracy: {top5_acc:.2f}%")

    # Extract Confusion Matrix
    cm = metrics.confusion_matrix.matrix
    num_classes = len(CLASSES)

    # Ultralytics confusion matrix might include a background column or be square
    if cm.shape[0] >= num_classes and cm.shape[1] >= num_classes:
        cm_sub = cm[:num_classes, :num_classes]
    else:
        cm_sub = cm

    print("\n[+] Confusion Matrix (Rows = Actual, Columns = Predicted):")
    print("-" * 65)
    header = f"{'Class':<18} | " + " | ".join(f"{c[:10]:<10}" for c in CLASSES)
    print(header)
    print("-" * 65)
    for i, c in enumerate(CLASSES):
        if i < cm_sub.shape[0]:
            row_vals = " | ".join(f"{int(cm_sub[i, j]):<10}" for j in range(min(num_classes, cm_sub.shape[1])))
            print(f"{c:<18} | {row_vals}")
    print("-" * 65)

    # Compute per-class Precision, Recall, and F1-Score
    precisions = []
    recalls = []
    f1_scores = []
    report_per_class = {}

    for i, c in enumerate(CLASSES):
        tp = float(cm_sub[i, i]) if i < cm_sub.shape[0] and i < cm_sub.shape[1] else 0.0
        actual_total = float(cm_sub[i, :].sum()) if i < cm_sub.shape[0] else 0.0
        pred_total = float(cm_sub[:, i].sum()) if i < cm_sub.shape[1] else 0.0

        prec = (tp / pred_total * 100.0) if pred_total > 0 else 0.0
        rec = (tp / actual_total * 100.0) if actual_total > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

        precisions.append(prec)
        recalls.append(rec)
        f1_scores.append(f1)

        report_per_class[c] = {
            "precision": round(prec, 2),
            "recall": round(rec, 2),
            "f1_score": round(f1, 2),
            "support": int(actual_total),
        }

    macro_precision = float(np.mean(precisions))
    macro_recall = float(np.mean(recalls))
    macro_f1 = float(np.mean(f1_scores))

    print("\n[+] Per-Class Performance Summary:")
    print(f"{'Class':<20} {'Precision (%)':<15} {'Recall (%)':<15} {'F1-Score (%)':<15}")
    print("-" * 65)
    for c in CLASSES:
        stat = report_per_class[c]
        print(f"{c:<20} {stat['precision']:<15.2f} {stat['recall']:<15.2f} {stat['f1_score']:<15.2f}")
    print("-" * 65)
    print(f"{'Macro Average':<20} {macro_precision:<15.2f} {macro_recall:<15.2f} {macro_f1:<15.2f}\n")

    # Save Confusion Matrix figure
    cm_plot_path = weights_dir / "confusion_matrix.png"
    plot_and_save_confusion_matrix(cm_sub, CLASSES, cm_plot_path)

    # Save metrics JSON
    report_data = {
        "top1_accuracy": round(top1_acc, 2),
        "macro_precision": round(macro_precision, 2),
        "macro_recall": round(macro_recall, 2),
        "macro_f1": round(macro_f1, 2),
        "per_class": report_per_class,
        "confusion_matrix": cm_sub.tolist(),
    }
    report_json_path = weights_dir / "evaluation_report.json"
    with open(report_json_path, "w") as f:
        json.dump(report_data, f, indent=2)
    print(f"Evaluation report JSON saved to: {report_json_path}")

    return report_data


def train(args):
    """Main training routine."""
    project_root = Path(__file__).resolve().parent
    dataset_path = (project_root / args.dataset).resolve()
    weights_dir = (project_root / "weights").resolve()
    weights_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("      BRAIN TUMOR YOLO CLASSIFICATION PIPELINE")
    print("=" * 60)
    print(f"Base Model:       {args.model}")
    print(f"Dataset Path:     {dataset_path}")
    print(f"Epochs:           {args.epochs}")
    print(f"Batch Size:       {args.batch}")
    print(f"Image Size:       {args.imgsz}")
    print(f"Early Stopping:   patience={args.patience}")
    print(f"Device:           {args.device}")
    print("=" * 60)

    # Check and prepare dataset
    ensure_dataset_compatibility(dataset_path)

    # Load base model
    print(f"\n[+] Loading base classification model: {args.model} ...")
    model = YOLO(args.model)

    if not args.eval_only:
        print("\n[+] Commencing model training...")
        train_results = model.train(
            data=str(dataset_path),
            epochs=args.epochs,
            patience=args.patience,
            batch=args.batch,
            imgsz=args.imgsz,
            device=args.device,
            project="runs/classify",
            name="brain_tumor",
            exist_ok=True,
            verbose=True,
        )

        # Locate trained best.pt
        trainer_best = getattr(getattr(model, "trainer", None), "best", None)
        if trainer_best and Path(trainer_best).exists():
            best_pt_source = Path(trainer_best)
        elif hasattr(train_results, "save_dir"):
            best_pt_source = Path(train_results.save_dir) / "weights" / "best.pt"
        else:
            best_pt_source = Path("runs/classify/brain_tumor/weights/best.pt")
        target_best_pt = weights_dir / "best.pt"

        if best_pt_source.exists():
            shutil.copy(best_pt_source, target_best_pt)
            print(f"\n[✓] Trained best weights saved to: {target_best_pt}")
            # Reload best model for validation
            model = YOLO(str(target_best_pt))
        else:
            print("[!] Warning: best.pt was not found at default run path.")
    else:
        target_best_pt = weights_dir / "best.pt"
        if target_best_pt.exists():
            model = YOLO(str(target_best_pt))
            print(f"[+] Loaded existing model from: {target_best_pt}")

    # Evaluate on content/Testing
    evaluate_model(model, dataset_path, weights_dir)
    print("\n[✓] Pipeline execution finished successfully!")


def parse_arguments():
    parser = argparse.ArgumentParser(description="Train and evaluate YOLO on Brain Tumor Classification dataset.")
    parser.add_argument("--model", type=str, default="yolov8m-cls.pt", help="Pretrained model weights file")
    parser.add_argument("--dataset", type=str, default="Brain_tumor_dataset/content", help="Dataset directory")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs (default: 30)")
    parser.add_argument("--batch", type=int, default=16, help="Batch size (default: 16)")
    parser.add_argument("--imgsz", type=int, default=224, help="Image resolution (default: 224)")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience (default: 10)")
    parser.add_argument(
        "--device",
        type=str,
        default="0" if torch.cuda.is_available() else "cpu",
        help="Device to use for training ('0', 'cpu')",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Run evaluation only using existing weights/best.pt",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    train(args)
