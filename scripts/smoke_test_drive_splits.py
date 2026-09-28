#!/usr/bin/env python3
"""Validate uploaded train/val/test lists and paired images on Google Drive."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from PIL import Image

from create_grouped_splits import TASKS, classify


def read_names(path: Path) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing list: {path}")
    names = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(names) != len(set(names)):
        raise ValueError(f"Duplicate entries in {path}")
    return names


def sha256_names(names: list[str]) -> str:
    normalized = "".join(f"{name}\n" for name in names).encode("utf-8")
    return hashlib.sha256(normalized).hexdigest()


def decode_pair(task_dir: Path, name: str) -> None:
    shapes = []
    for folder in ("imgs", "GT"):
        path = task_dir / folder / name
        with Image.open(path) as image:
            image.load()
            if image.mode not in {"RGB", "RGBA", "L"}:
                image = image.convert("RGB")
            shapes.append(image.size)
    if shapes[0] != shapes[1]:
        raise ValueError(f"Input/GT shape mismatch for {task_dir.name}/{name}: {shapes}")


def validate_task(
    task: str, task_dir: Path, expected: dict[str, object], decode_per_split: int
) -> None:
    split_names = {
        split: read_names(task_dir / f"{split}.txt")
        for split in ("train", "val", "test")
    }
    for split, names in split_names.items():
        expected_count = int(expected["counts"][split])
        expected_hash = str(expected["sha256"][split])
        actual_hash = sha256_names(names)
        if len(names) != expected_count or actual_hash != expected_hash:
            raise ValueError(
                f"{task}/{split}: expected n={expected_count}, sha256={expected_hash}; "
                f"got n={len(names)}, sha256={actual_hash}"
            )

    group_sets = {
        split: {classify(task, name)[0] for name in names}
        for split, names in split_names.items()
    }
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = group_sets[left] & group_sets[right]
        if overlap:
            raise ValueError(f"{task}: scene leakage {left}/{right}: {sorted(overlap)[:5]}")

    imgs = set(path.name for path in (task_dir / "imgs").iterdir() if path.is_file())
    targets = set(path.name for path in (task_dir / "GT").iterdir() if path.is_file())
    required = set().union(*(set(names) for names in split_names.values()))
    missing_imgs = sorted(required - imgs)
    missing_targets = sorted(required - targets)
    if missing_imgs or missing_targets:
        raise FileNotFoundError(
            f"{task}: missing imgs={missing_imgs[:5]}, missing GT={missing_targets[:5]}"
        )

    decoded = 0
    for split in ("train", "val", "test"):
        names = split_names[split]
        if not names:
            raise ValueError(f"{task}/{split} is empty")
        sample_count = min(decode_per_split, len(names))
        if sample_count == 1:
            sample_indices = [0]
        else:
            sample_indices = [round(i * (len(names) - 1) / (sample_count - 1)) for i in range(sample_count)]
        for index in sample_indices:
            decode_pair(task_dir, names[index])
            decoded += 1

    counts = {split: len(names) for split, names in split_names.items()}
    print(f"{task}: OK | counts={counts} | decoded_pairs={decoded}")


def main() -> None:
    default_dataset_root = Path("/content/gdrive/MyDrive/Facultad/tesis/Datasets/Classifier")
    if not default_dataset_root.exists():
        default_dataset_root = Path("/content/drive/MyDrive/Facultad/tesis/Datasets/Classifier")
    default_manifest = Path("/content/maxim/datasets/splits_v2/split_manifest.json")
    if Path("/content/split_manifest.json").is_file():
        default_manifest = Path("/content/split_manifest.json")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=default_dataset_root,
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=default_manifest,
    )
    parser.add_argument("--decode-per-split", type=int, default=2)
    args, unknown = parser.parse_known_args()
    # IPython kernels inject ``-f <connection-file>`` into sys.argv. Accept
    # only that known pair so genuine CLI typos still fail loudly.
    if unknown:
        expected_kernel_args = len(unknown) == 2 and unknown[0] == "-f"
        expected_kernel_args &= unknown[1].endswith(".json")
        if not expected_kernel_args or "ipykernel" not in sys.modules:
            parser.error(f"unrecognized arguments: {' '.join(unknown)}")
    if args.decode_per_split < 1:
        raise ValueError("decode-per-split must be positive")

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if int(manifest["seed"]) != 42 or float(manifest["validation_fraction"]) != 0.10:
        raise ValueError("Unexpected split protocol in manifest")
    for task in TASKS:
        validate_task(
            task,
            args.dataset_root / task,
            manifest["tasks"][task],
            args.decode_per_split,
        )
    print("DATASET SPLIT SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
