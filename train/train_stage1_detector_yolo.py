#!/usr/bin/env python3
"""Train stage-1 detector from data_v2 using Ultralytics YOLO.

This script reuses data_v2 class folders and builds a YOLO-format dataset where
images are treated as a single object spanning the full frame. This is suitable
for centered single-produce images and provides a practical stage-1 bootstrap.

Input (required):
  data_v2/{train,val,test}/{fresh_<produce>|stale_<produce>}/*

Outputs:
  data_v2/yolo_stage1/dataset.yaml
  data_v2/yolo_stage1/images/{train,val,test}/*
  data_v2/yolo_stage1/labels/{train,val,test}/*
  model/detector/stage1_yolo_classes.json
  model/detector/classes.names
  model/detector/stage1_yolov8_best.pt
"""

from __future__ import annotations

import argparse
import json
import math
import random
import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

DATA_V2 = Path("data_v2")
YOLO_ROOT = DATA_V2 / "yolo_stage1"
MODEL_DIR = Path("model") / "detector"
TEST_CM_PNG = MODEL_DIR / "confusion_matrix_test.png"
TEST_CM_CSV = MODEL_DIR / "confusion_matrix_test.csv"

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(frozen=True)
class Sample:
    src: Path
    split: str
    produce: str


def extract_produce(class_dir_name: str) -> str:
    lower = class_dir_name.lower()
    if "_" in lower:
        return lower.split("_", 1)[1]
    return lower


def collect_samples() -> list[Sample]:
    samples: list[Sample] = []
    for split in ("train", "val", "test"):
        split_dir = DATA_V2 / split
        if not split_dir.exists():
            continue
        for class_dir in split_dir.iterdir():
            if not class_dir.is_dir():
                continue
            produce = extract_produce(class_dir.name)
            for img in class_dir.rglob("*"):
                if img.is_file() and img.suffix.lower() in SUPPORTED_EXTS:
                    samples.append(Sample(src=img, split=split, produce=produce))
    return samples


def apply_train_fraction(samples: list[Sample], fraction: float, seed: int) -> list[Sample]:
    if fraction >= 1.0:
        return samples

    train_groups: dict[str, list[Sample]] = defaultdict(list)
    remainder: list[Sample] = []
    for item in samples:
        if item.split == "train":
            train_groups[item.produce].append(item)
        else:
            remainder.append(item)

    rng = random.Random(seed)
    reduced_train: list[Sample] = []
    before = sum(len(items) for items in train_groups.values())

    for produce, items in train_groups.items():
        rng.shuffle(items)
        keep = max(1, math.ceil(len(items) * fraction))
        reduced_train.extend(items[:keep])
        print(f"  train fraction class={produce}: keeping {keep}/{len(items)}")

    after = len(reduced_train)
    print(f"Applied stage-1 train fraction: {fraction:.3f} (seed={seed})")
    print(f"  train samples: {before} -> {after}")
    return reduced_train + remainder


def reset_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def link_or_copy(src: Path, dst: Path) -> None:
    try:
        if dst.exists():
            dst.unlink()
        dst.hardlink_to(src)
    except OSError:
        shutil.copy2(src, dst)


def build_yolo_dataset(samples: list[Sample], classes: list[str]) -> Path:
    images_root = YOLO_ROOT / "images"
    labels_root = YOLO_ROOT / "labels"

    reset_dir(images_root)
    reset_dir(labels_root)

    class_to_idx = {name: idx for idx, name in enumerate(classes)}
    counts = {"train": 0, "val": 0, "test": 0}

    for split in ("train", "val", "test"):
        (images_root / split).mkdir(parents=True, exist_ok=True)
        (labels_root / split).mkdir(parents=True, exist_ok=True)

    for i, item in enumerate(samples):
        idx = class_to_idx[item.produce]
        stem = f"{item.produce}_{i:07d}"
        dst_img = images_root / item.split / f"{stem}{item.src.suffix.lower()}"
        dst_lbl = labels_root / item.split / f"{stem}.txt"

        link_or_copy(item.src, dst_img)
        # Full-image bounding box bootstrap: x_center y_center width height
        dst_lbl.write_text(f"{idx} 0.5 0.5 1.0 1.0\n", encoding="utf-8")
        counts[item.split] += 1

    yaml_path = YOLO_ROOT / "dataset.yaml"
    yaml_lines = [
        f"path: {YOLO_ROOT.resolve()}",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        "",
        f"nc: {len(classes)}",
        f"names: {classes}",
        "",
    ]
    yaml_path.write_text("\n".join(yaml_lines), encoding="utf-8")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    (MODEL_DIR / "stage1_yolo_classes.json").write_text(
        json.dumps(classes, indent=2), encoding="utf-8"
    )
    (MODEL_DIR / "classes.names").write_text("\n".join(classes) + "\n", encoding="utf-8")

    print("Prepared YOLO dataset:")
    print(f"  classes: {len(classes)}")
    print(f"  train:   {counts['train']}")
    print(f"  val:     {counts['val']}")
    print(f"  test:    {counts['test']}")
    print(f"  yaml:    {yaml_path}")

    return yaml_path


def save_confusion_matrix(cm: list[list[int]], labels: list[str], png_path: Path, csv_path: Path, title: str) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8") as f:
        f.write("true_label," + ",".join(labels) + "\n")
        for i, row in enumerate(cm):
            f.write(labels[i] + "," + ",".join(str(v) for v in row) + "\n")

    n = len(labels)
    cell = 82
    left = 220
    top = 120
    right_pad = 40
    bottom = 120
    width = left + n * cell + right_pad
    height = top + n * cell + bottom

    max_val = max(max(row) for row in cm) if n > 0 else 1
    img = Image.new("RGB", (width, height), color="white")
    draw = ImageDraw.Draw(img)
    draw.text((16, 16), title, fill="black")
    draw.text((left, 72), "Predicted label", fill="black")
    draw.text((16, top), "True label", fill="black")

    for j, label in enumerate(labels):
        draw.text((left + j * cell + 8, top - 24), label, fill="black")
    for i, label in enumerate(labels):
        draw.text((16, top + i * cell + 30), label, fill="black")

    for i in range(n):
        for j in range(n):
            value = cm[i][j]
            shade = 255 - int((value / max_val) * 190) if max_val > 0 else 255
            color = (shade, shade, 255)
            x1 = left + j * cell
            y1 = top + i * cell
            x2 = x1 + cell
            y2 = y1 + cell
            draw.rectangle([x1, y1, x2, y2], fill=color, outline="black", width=1)
            draw.text((x1 + 24, y1 + 30), str(value), fill="black")

    png_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(png_path)


def export_stage1_confusion(metrics: Any, classes: list[str], save_dir: Path | None) -> None:
    # Prefer Ultralytics-rendered confusion matrix image when available.
    copied_png = False
    if save_dir is not None:
        for name in ("confusion_matrix.png", "confusion_matrix_normalized.png"):
            src = save_dir / name
            if src.exists():
                shutil.copy2(src, TEST_CM_PNG)
                copied_png = True
                break

    cm_obj = getattr(metrics, "confusion_matrix", None)
    matrix = getattr(cm_obj, "matrix", None)
    if matrix is None:
        if copied_png:
            print(f"Saved detector confusion matrix PNG: {TEST_CM_PNG}")
        else:
            print("Warning: detector confusion matrix was not available for export")
        return

    rows = [[int(v) for v in row] for row in matrix.tolist()]
    n = len(rows)
    if n == len(classes) + 1:
        labels = classes + ["background"]
    elif n == len(classes):
        labels = classes
    else:
        labels = [f"class_{i}" for i in range(n)]

    save_confusion_matrix(rows, labels, TEST_CM_PNG, TEST_CM_CSV, "Detector Confusion Matrix (test)")
    print(f"Saved detector confusion matrix PNG: {TEST_CM_PNG}")
    print(f"Saved detector confusion matrix CSV: {TEST_CM_CSV}")


def resolve_device(device: str) -> str:
    if device != "auto":
        return device

    try:
        import torch
    except ImportError:
        return "cpu"

    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def train_detector(
    dataset_yaml: Path,
    classes: list[str],
    weights: str,
    epochs: int,
    imgsz: int,
    batch: int,
    device: str,
    workers: int,
) -> None:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit(
            "Missing ultralytics. Install with: pip install ultralytics"
        ) from exc

    selected_device = resolve_device(device)
    print(f"Using stage-1 device: {selected_device}")

    model = YOLO(weights)
    result: Any = model.train(
        data=str(dataset_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=selected_device,
        workers=workers,
    )

    # Save best checkpoint in the detector model folder for predictable usage.
    save_dir = getattr(result, "save_dir", None)
    if save_dir is None:
        trainer = getattr(model, "trainer", None)
        save_dir = getattr(trainer, "save_dir", None)

    if save_dir is not None:
        best = Path(str(save_dir)) / "weights" / "best.pt"
        if best.exists():
            target = MODEL_DIR / "stage1_yolov8_best.pt"
            shutil.copy2(best, target)
            print(f"Saved best detector weights: {target}")
        else:
            print(f"Warning: best.pt not found under {save_dir}")
    else:
        print("Warning: could not resolve YOLO save directory to copy best.pt")

    # Evaluate on test split and export confusion matrix artifacts.
    metrics = model.val(data=str(dataset_yaml), split="test")
    val_save_dir = getattr(metrics, "save_dir", None)
    export_stage1_confusion(metrics, classes, Path(str(val_save_dir)) if val_save_dir is not None else None)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train stage-1 detector from data_v2")
    parser.add_argument(
        "--weights",
        default="model/detector/yolov8n.pt",
        help="Initial YOLO weights",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="auto", help="auto|mps|cpu|cuda[:index]")
    parser.add_argument("--workers", type=int, default=4, help="Data loader workers for YOLO train/val")
    parser.add_argument("--train-fraction", type=float, default=1.0, help="Use fraction of train split only (0,1]")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for train-fraction sampling")
    args = parser.parse_args()

    if not (0.0 < args.train_fraction <= 1.0):
        raise SystemExit("--train-fraction must be in the range (0, 1].")

    samples = collect_samples()
    if not samples:
        raise SystemExit("No samples found in data_v2. Run scripts/prepare_data_v2.py first.")

    samples = apply_train_fraction(samples, args.train_fraction, args.seed)

    classes = sorted({s.produce for s in samples})
    dataset_yaml = build_yolo_dataset(samples, classes)
    train_detector(
        dataset_yaml,
        classes,
        args.weights,
        args.epochs,
        args.imgsz,
        args.batch,
        args.device,
        args.workers,
    )


if __name__ == "__main__":
    main()
