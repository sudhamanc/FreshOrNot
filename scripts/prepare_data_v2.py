#!/usr/bin/env python3
"""Build merged and normalized dataset at data_v2/{train,val,test}.

This script collects image files from multiple source datasets and maps class names
into a canonical schema:
  fresh_<produce>
  stale_<produce>

It performs content-based deduplication and writes a manifest for traceability.
"""

from __future__ import annotations

import csv
import hashlib
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

ROOT = Path(__file__).resolve().parents[1]
DATA_V2 = ROOT / "data_v2"
MANIFEST_PATH = DATA_V2 / "manifest.csv"
SUMMARY_PATH = DATA_V2 / "summary.txt"

SOURCE_PATHS = [
    ROOT / "data" / "Train",
    ROOT / "data" / "Test",
    ROOT / "data" / "external" / "raghavrpotdar_fresh_and_stale",
    ROOT / "data" / "external" / "muhriddinmuxiddinov_fruits_and_vegetables" / "Fruits_Vegetables_Dataset(12000)",
    ROOT / "data" / "external" / "ulnnproject_food_freshness" / "Dataset",
    ROOT / "data" / "external" / "filipemonteir_fresh_and_rotten" / "Dataset" / "Visual_Dataset",
]

PRODUCE_ALIASES = {
    "apple": "apple",
    "apples": "apple",
    "banana": "banana",
    "bananas": "banana",
    "orange": "orange",
    "oranges": "orange",
    "tomato": "tomato",
    "tomatoes": "tomato",
    "potato": "potato",
    "potatoes": "potato",
    "cucumber": "cucumber",
    "cucumbers": "cucumber",
    "okra": "okra",
    "okara": "okra",
    "bittergroud": "bittergourd",
    "bittergourds": "bittergourd",
    "bittergourd": "bittergourd",
    "capsicum": "capsicum",
    "capciscum": "capsicum",
    "bellpepper": "bellpepper",
    "bellpeppers": "bellpepper",
    "pepper": "pepper",
    "peppers": "pepper",
    "carrot": "carrot",
    "carrots": "carrot",
    "mango": "mango",
    "mangoes": "mango",
    "strawberry": "strawberry",
    "strawberries": "strawberry",
}


@dataclass
class Record:
    source_file: Path
    canonical_class: str
    preferred_split: str | None
    sha1: str


def is_image(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTS


def file_sha1(path: Path) -> str:
    hasher = hashlib.sha1()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def parse_freshness_and_produce(raw_name: str) -> tuple[str, str] | None:
    normalized = re.sub(r"[^a-zA-Z0-9]+", "", raw_name).lower()

    freshness = None
    produce_token = normalized
    for token, mapped in (("fresh", "fresh"), ("rotten", "stale"), ("stale", "stale")):
        if normalized.startswith(token):
            freshness = mapped
            produce_token = normalized[len(token):]
            break

    if freshness is None:
        return None

    produce = PRODUCE_ALIASES.get(produce_token)
    if not produce:
        return None

    return freshness, produce


def preferred_split_for_path(path: Path) -> str | None:
    parts = {p.lower() for p in path.parts}
    if "test" in parts:
        return "test"
    if "validation" in parts or "val" in parts:
        return "val"
    if "train" in parts:
        return "train"
    return None


def iter_source_images(base: Path) -> list[Record]:
    if not base.exists():
        return []

    records: list[Record] = []
    for class_dir in base.rglob("*"):
        if not class_dir.is_dir():
            continue

        parsed = parse_freshness_and_produce(class_dir.name)
        if not parsed:
            continue

        freshness, produce = parsed
        canonical_class = f"{freshness}_{produce}"

        for img in class_dir.iterdir():
            if not img.is_file() or not is_image(img):
                continue

            records.append(
                Record(
                    source_file=img,
                    canonical_class=canonical_class,
                    preferred_split=preferred_split_for_path(img),
                    sha1=file_sha1(img),
                )
            )

    return records


def split_from_hash(sha1: str) -> str:
    bucket = int(sha1[:2], 16) % 10
    if bucket <= 7:
        return "train"
    if bucket == 8:
        return "val"
    return "test"


def reset_output() -> None:
    if DATA_V2.exists():
        shutil.rmtree(DATA_V2)
    (DATA_V2 / "train").mkdir(parents=True, exist_ok=True)
    (DATA_V2 / "val").mkdir(parents=True, exist_ok=True)
    (DATA_V2 / "test").mkdir(parents=True, exist_ok=True)


def main() -> None:
    print("Collecting source images...")
    all_records: list[Record] = []
    for source in SOURCE_PATHS:
        source_records = iter_source_images(source)
        print(f"  {source}: {len(source_records)} images")
        all_records.extend(source_records)

    if not all_records:
        raise SystemExit("No source images found. Check source paths.")

    reset_output()

    seen_hashes: set[str] = set()
    written = 0
    deduped = 0
    split_counts = {"train": 0, "val": 0, "test": 0}
    class_counts: dict[str, int] = {}

    with MANIFEST_PATH.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sha1", "split", "class", "source_file", "target_file"])

        for rec in all_records:
            if rec.sha1 in seen_hashes:
                deduped += 1
                continue

            seen_hashes.add(rec.sha1)
            split = rec.preferred_split or split_from_hash(rec.sha1)
            target_dir = DATA_V2 / split / rec.canonical_class
            target_dir.mkdir(parents=True, exist_ok=True)

            target_name = f"{rec.sha1[:12]}{rec.source_file.suffix.lower()}"
            target_file = target_dir / target_name
            shutil.copy2(rec.source_file, target_file)

            writer.writerow(
                [
                    rec.sha1,
                    split,
                    rec.canonical_class,
                    str(rec.source_file.relative_to(ROOT)),
                    str(target_file.relative_to(ROOT)),
                ]
            )

            written += 1
            split_counts[split] += 1
            class_counts[rec.canonical_class] = class_counts.get(rec.canonical_class, 0) + 1

    with SUMMARY_PATH.open("w") as f:
        f.write(f"total_source_images={len(all_records)}\n")
        f.write(f"written_unique_images={written}\n")
        f.write(f"deduped_images={deduped}\n")
        f.write(f"train={split_counts['train']}\n")
        f.write(f"val={split_counts['val']}\n")
        f.write(f"test={split_counts['test']}\n")
        f.write("\n[class_counts]\n")
        for cls in sorted(class_counts):
            f.write(f"{cls}={class_counts[cls]}\n")

    print("Done.")
    print(f"  Unique images: {written}")
    print(f"  Deduped images: {deduped}")
    print(f"  Split counts: {split_counts}")
    print(f"  Manifest: {MANIFEST_PATH}")
    print(f"  Summary: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
