#!/usr/bin/env python3
"""Train the stage-2 freshness classifier from data_v2 only.

This is the only supported training strategy in the repository.
Expected data layout:
  data_v2/train/fresh_<produce>/*
  data_v2/train/stale_<produce>/*
  data_v2/val/fresh_<produce>/*
  data_v2/val/stale_<produce>/*

Outputs:
  model/freshness_binary.pt
  model/freshness_binary_classes.json
"""

from __future__ import annotations

import json
import os
import pathlib
import random
import sys
from dataclasses import dataclass
from math import ceil

try:
    import torch
    import torch.nn as nn
    import torchvision.transforms as T
    from PIL import Image, ImageDraw
    from torch.utils.data import DataLoader, Dataset
    from torchvision import datasets, models
except ImportError:
    print('Missing dependencies. Install with: pip install -r requirements.txt')
    sys.exit(1)

DATA_V2 = pathlib.Path('data_v2')
TRAIN_DIR = DATA_V2 / 'train'
VAL_DIR = DATA_V2 / 'val'
TEST_DIR = DATA_V2 / 'test'
MODEL_DIR = pathlib.Path('model')
MODEL_PATH = MODEL_DIR / 'freshness_binary.pt'
CLASS_PATH = MODEL_DIR / 'freshness_binary_classes.json'
TEST_CM_PNG = MODEL_DIR / 'freshness_confusion_matrix_test.png'
TEST_CM_CSV = MODEL_DIR / 'freshness_confusion_matrix_test.csv'

IMG_SIZE = int(os.getenv('STAGE2_IMG_SIZE', '224'))
BATCH_SIZE = int(os.getenv('STAGE2_BATCH_SIZE', '32'))
EPOCHS = int(os.getenv('STAGE2_EPOCHS', '8'))
LR = float(os.getenv('STAGE2_LR', '2e-4'))
WORKERS = int(os.getenv('STAGE2_WORKERS', '2'))
TRAIN_FRACTION = float(os.getenv('STAGE2_TRAIN_FRACTION', '1.0'))
SEED = int(os.getenv('STAGE2_SEED', '42'))
REQUESTED_DEVICE = os.getenv('STAGE2_DEVICE', 'auto').strip().lower()
MOSAIC_PROB = float(os.getenv('STAGE2_MOSAIC_PROB', '0.3'))


def resolve_device(requested: str) -> str:
    if requested and requested != 'auto':
        return requested
    if torch.cuda.is_available():
        return 'cuda'
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


DEVICE = resolve_device(REQUESTED_DEVICE)
PIN_MEMORY = DEVICE == 'cuda'


def collapse_to_binary(class_name: str) -> int:
    token = class_name.lower()
    if token.startswith('fresh'):
        return 0
    if token.startswith('stale') or token.startswith('rotten'):
        return 1
    raise ValueError(f'Cannot infer freshness from class name: {class_name}')


class RightAngleRotate:
    def __call__(self, img: Image.Image) -> Image.Image:
        angle = random.choice((0, 90, 180, 270))
        return img.rotate(angle)


@dataclass
class SampleItem:
    path: str
    label: int


class BinaryFreshnessDataset(Dataset):
    def __init__(self, root: pathlib.Path, transform: T.Compose, mosaic_prob: float = 0.0) -> None:
        base = datasets.ImageFolder(root)
        self.transform = transform
        self.mosaic_prob = mosaic_prob
        self.samples: list[SampleItem] = []
        for path, class_idx in base.samples:
            class_name = base.classes[class_idx]
            self.samples.append(SampleItem(path=path, label=collapse_to_binary(class_name)))

    def __len__(self) -> int:
        return len(self.samples)

    def _load(self, idx: int) -> Image.Image:
        return Image.open(self.samples[idx].path).convert('RGB')

    def _mosaic(self, idx: int) -> Image.Image:
        picks = [idx] + [random.randrange(0, len(self.samples)) for _ in range(3)]
        imgs = [self._load(i).resize((IMG_SIZE, IMG_SIZE)) for i in picks]

        canvas = Image.new('RGB', (IMG_SIZE * 2, IMG_SIZE * 2))
        canvas.paste(imgs[0], (0, 0))
        canvas.paste(imgs[1], (IMG_SIZE, 0))
        canvas.paste(imgs[2], (0, IMG_SIZE))
        canvas.paste(imgs[3], (IMG_SIZE, IMG_SIZE))
        return canvas.resize((IMG_SIZE, IMG_SIZE))

    def __getitem__(self, idx: int):
        item = self.samples[idx]
        img = self._mosaic(idx) if random.random() < self.mosaic_prob else self._load(idx)
        return self.transform(img), torch.tensor(item.label, dtype=torch.long)


def apply_train_fraction(dataset: BinaryFreshnessDataset, fraction: float, seed: int) -> None:
    if not (0 < fraction <= 1.0):
        raise ValueError(f'STAGE2_TRAIN_FRACTION must be in (0, 1], got {fraction}')
    if fraction >= 1.0:
        return

    groups: dict[int, list[SampleItem]] = {0: [], 1: []}
    for sample in dataset.samples:
        groups[int(sample.label)].append(sample)

    rng = random.Random(seed)
    reduced: list[SampleItem] = []
    before = len(dataset.samples)
    label_names = {0: 'fresh', 1: 'stale'}

    for label in (0, 1):
        items = groups[label]
        if not items:
            continue
        rng.shuffle(items)
        keep = max(1, ceil(len(items) * fraction))
        reduced.extend(items[:keep])
        print(f'  train fraction class={label_names[label]}: keeping {keep}/{len(items)}')

    dataset.samples = reduced
    print(f'Applied stage-2 train fraction: {fraction:.3f} (seed={seed})')
    print(f'  train samples: {before} -> {len(dataset.samples)}')


def build_model() -> nn.Module:
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    classifier_layer = model.classifier[1]
    if not isinstance(classifier_layer, nn.Linear):
        raise TypeError('Unexpected MobileNet classifier structure')
    in_features = classifier_layer.in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.25),
        nn.Linear(in_features, 2),
    )
    return model.to(DEVICE)


def train_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for imgs, labels in loader:
        imgs = imgs.to(DEVICE)
        labels = labels.to(DEVICE)
        optimizer.zero_grad()
        logits = model(imgs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        total += imgs.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def eval_epoch(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for imgs, labels in loader:
        imgs = imgs.to(DEVICE)
        labels = labels.to(DEVICE)
        logits = model(imgs)
        loss = criterion(logits, labels)
        total_loss += loss.item() * imgs.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        total += imgs.size(0)
    return total_loss / total, correct / total


def save_confusion_matrix(cm: list[list[int]], labels: list[str], png_path: pathlib.Path, csv_path: pathlib.Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open('w', encoding='utf-8') as f:
        f.write('true_label,' + ','.join(labels) + '\n')
        for i, row in enumerate(cm):
            f.write(labels[i] + ',' + ','.join(str(v) for v in row) + '\n')

    n = len(labels)
    cell = 96
    left = 220
    top = 120
    right_pad = 40
    bottom = 120
    width = left + n * cell + right_pad
    height = top + n * cell + bottom

    max_val = max(max(row) for row in cm) if n > 0 else 1
    img = Image.new('RGB', (width, height), color='white')
    draw = ImageDraw.Draw(img)

    draw.text((16, 16), 'Freshness Confusion Matrix (test)', fill='black')
    draw.text((left, 72), 'Predicted label', fill='black')
    draw.text((16, top), 'True label', fill='black')

    for j, label in enumerate(labels):
        x = left + j * cell + 8
        draw.text((x, top - 24), label, fill='black')
    for i, label in enumerate(labels):
        y = top + i * cell + 34
        draw.text((16, y), label, fill='black')

    for i in range(n):
        for j in range(n):
            value = cm[i][j]
            shade = 255 - int((value / max_val) * 190) if max_val > 0 else 255
            color = (shade, shade, 255)
            x1 = left + j * cell
            y1 = top + i * cell
            x2 = x1 + cell
            y2 = y1 + cell
            draw.rectangle([x1, y1, x2, y2], fill=color, outline='black', width=1)
            draw.text((x1 + 30, y1 + 36), str(value), fill='black')

    png_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(png_path)


@torch.no_grad()
def evaluate_test_and_export(model_path: pathlib.Path, transform: T.Compose) -> None:
    if not TEST_DIR.exists():
        print('No data_v2/test found; skipping stage-2 test confusion matrix export.')
        return

    test_ds = BinaryFreshnessDataset(TEST_DIR, transform, mosaic_prob=0.0)
    if len(test_ds) == 0:
        print('Empty data_v2/test; skipping stage-2 test confusion matrix export.')
        return

    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=WORKERS,
        pin_memory=PIN_MEMORY,
    )
    model = torch.load(model_path, map_location=DEVICE, weights_only=False)
    model.eval()

    cm = [[0, 0], [0, 0]]
    correct = 0
    total = 0

    for imgs, labels in test_loader:
        imgs = imgs.to(DEVICE)
        labels = labels.to(DEVICE)
        logits = model(imgs)
        preds = logits.argmax(1)

        for truth, pred in zip(labels.tolist(), preds.tolist()):
            cm[int(truth)][int(pred)] += 1

        correct += (preds == labels).sum().item()
        total += imgs.size(0)

    test_acc = correct / total if total > 0 else 0.0
    save_confusion_matrix(cm, ['fresh', 'stale'], TEST_CM_PNG, TEST_CM_CSV)
    print(f'Test samples: {total}')
    print(f'Test acc: {test_acc:.3f}')
    print(f'Test confusion matrix PNG: {TEST_CM_PNG}')
    print(f'Test confusion matrix CSV: {TEST_CM_CSV}')


def main() -> None:
    if not TRAIN_DIR.exists() or not VAL_DIR.exists():
        print('Missing data_v2/train or data_v2/val. Run scripts/prepare_data_v2.py first.')
        sys.exit(1)

    train_tf = T.Compose(
        [
            RightAngleRotate(),
            T.RandomResizedCrop(IMG_SIZE, scale=(0.65, 1.0)),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(p=0.15),
            T.ColorJitter(brightness=0.35, contrast=0.25, saturation=0.25, hue=0.03),
            T.RandomAffine(degrees=0, translate=(0.08, 0.08), scale=(0.9, 1.1), shear=8),
            T.RandomPerspective(distortion_scale=0.25, p=0.25),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    val_tf = T.Compose(
        [
            T.Resize(256),
            T.CenterCrop(IMG_SIZE),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    random.seed(SEED)
    torch.manual_seed(SEED)

    train_ds = BinaryFreshnessDataset(TRAIN_DIR, train_tf, mosaic_prob=MOSAIC_PROB)
    apply_train_fraction(train_ds, TRAIN_FRACTION, SEED)
    val_ds = BinaryFreshnessDataset(VAL_DIR, val_tf, mosaic_prob=0.0)

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=WORKERS,
        pin_memory=PIN_MEMORY,
        persistent_workers=WORKERS > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=WORKERS,
        pin_memory=PIN_MEMORY,
        persistent_workers=WORKERS > 0,
    )

    model = build_model()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    best_val = -1.0

    print(f'Using device: {DEVICE}')
    print(f'Workers: {WORKERS}')
    print(f'Train fraction: {TRAIN_FRACTION:.3f} (seed={SEED})')
    print(f'Train samples: {len(train_ds)}')
    print(f'Val samples:   {len(val_ds)}')

    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc = train_epoch(model, train_loader, criterion, optimizer)
        vl_loss, vl_acc = eval_epoch(model, val_loader, criterion)
        scheduler.step()
        print(f'Epoch {epoch:02d}  loss={tr_loss:.4f}/{vl_loss:.4f}  acc={tr_acc:.3f}/{vl_acc:.3f}')

        if vl_acc > best_val:
            best_val = vl_acc
            torch.save(model, MODEL_PATH)
            print(f'  saved best model: val_acc={best_val:.3f}')

    CLASS_PATH.write_text(json.dumps(['fresh', 'stale'], indent=2), encoding='utf-8')
    evaluate_test_and_export(MODEL_PATH, val_tf)
    print(f'Best val acc: {best_val:.3f}')
    print(f'Model saved: {MODEL_PATH}')
    print(f'Classes saved: {CLASS_PATH}')


if __name__ == '__main__':
    main()
