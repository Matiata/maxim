#!/usr/bin/env python3
"""Create deterministic train/val/test lists without scene leakage.

The existing test assignment is preserved at group level: if any variant of a
scene is already in test, every variant of that scene is assigned to test before
validation groups are selected from the remaining training data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import defaultdict
from pathlib import Path


TASKS = ("deblur", "dehaze", "denoise", "derain", "enhance")


def _match(pattern: str, name: str):
    return re.match(pattern, Path(name).stem, flags=re.IGNORECASE)


def classify(task: str, name: str) -> tuple[str, str]:
    """Return ``(scene_group, source_stratum)`` for a dataset filename."""
    if task == "deblur":
        match = _match(r"(GOPR\d+_\d+_\d+)-\d+$", name)
        if match:
            return f"gopro-sequence:{match.group(1).upper()}", "gopro-sequence"
        match = _match(r"\d+from(GOPR\d+)(?:\.MP4)?$", name)
        if match:
            return f"gopro-video:{match.group(1).upper()}", "gopro-video"
        match = _match(r"(RB_scene\d+)_blur_\d+$", name)
        if match:
            return f"realblur:{match.group(1).lower()}", "realblur"

    elif task == "dehaze":
        match = _match(r"(SOTS_(indoor|outdoor)_\d+)_", name)
        if match:
            return match.group(1).lower(), f"sots-{match.group(2).lower()}"
        match = _match(r"(HSTS_\d+)$", name)
        if match:
            return match.group(1).lower(), "hsts"

    elif task == "denoise":
        match = _match(r"SIDD_(\d{4})_", name)
        if match:
            stratum = "sidd-legacy" if Path(name).stem.endswith("_NOISY") else "sidd-full"
            return f"sidd:{match.group(1)}", stratum

    elif task == "derain":
        match = _match(r"rain_(?:light|heavy)_(train|test)-(\d+)x\d+$", name)
        if match:
            # The official train/test sources use independent clean images even
            # when their numeric identifiers coincide.
            return f"rain:{match.group(1).lower()}:{match.group(2)}", "rain"

    elif task == "enhance":
        match = _match(r"LOL_(?:train|test)_(\d+)$", name)
        if match:
            return f"lol:{match.group(1)}", "lol"
        match = _match(r"a(\d+)-", name)
        if match:
            return f"fivek:{match.group(1)}", "fivek"

    raise ValueError(f"Unrecognized {task} filename: {name!r}")


def _read_list(path: Path) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing input list: {path}")
    names = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(names) != len(set(names)):
        raise ValueError(f"Duplicate filenames in {path}")
    return names


def _sha256_names(names: list[str]) -> str:
    content = "".join(f"{name}\n" for name in names).encode("utf-8")
    return hashlib.sha256(content).hexdigest()


def _stable_shuffle(values: list[str], seed_material: str) -> list[str]:
    digest = hashlib.sha256(seed_material.encode("utf-8")).digest()
    rng = random.Random(int.from_bytes(digest[:8], "big"))
    values = list(sorted(values))
    rng.shuffle(values)
    return values


def _select_groups_near_target(
    group_sizes: dict[str, int], target_samples: int, seed_material: str
) -> set[str]:
    """Select whole groups with a deterministic sample count near the target."""
    groups = _stable_shuffle(list(group_sizes), seed_material)
    if not groups or target_samples <= 0:
        return set()
    if len(set(group_sizes.values())) == 1:
        group_size = next(iter(group_sizes.values()))
        count = max(1, round(target_samples / group_size))
        count = min(count, max(1, len(groups) - 1))
        return set(groups[:count])

    # Exact subset-sum states are small here (bounded by the number of samples).
    # Stable shuffled order provides deterministic tie-breaking.
    backpointer: dict[int, tuple[int, str] | None] = {0: None}
    for group in groups:
        size = group_sizes[group]
        for subtotal in sorted(tuple(backpointer), reverse=True):
            candidate = subtotal + size
            if candidate not in backpointer:
                backpointer[candidate] = (subtotal, group)

    allowed = [total for total in backpointer if 0 < total < sum(group_sizes.values())]
    best_total = min(allowed, key=lambda total: (abs(total - target_samples), total))
    selected: set[str] = set()
    subtotal = best_total
    while subtotal:
        previous, group = backpointer[subtotal]  # type: ignore[misc]
        selected.add(group)
        subtotal = previous
    return selected


def _group_stratum(task: str, records: list[tuple[str, str]]) -> str:
    strata = {stratum for _, stratum in records}
    if task == "denoise" and "sidd-full" in strata:
        return "sidd-full"
    if len(strata) != 1:
        raise ValueError(f"Scene group mixes incompatible sources: {strata}")
    return next(iter(strata))


def create_task_split(
    task: str, train_names: list[str], test_names: list[str], val_fraction: float, seed: int
) -> tuple[dict[str, list[str]], dict[str, object]]:
    if set(train_names) & set(test_names):
        raise ValueError(f"Exact filenames overlap between {task} train and test.")

    records: dict[str, list[tuple[str, str]]] = defaultdict(list)
    original_split: dict[str, str] = {}
    for split_name, names in (("train", train_names), ("test", test_names)):
        for name in names:
            group, stratum = classify(task, name)
            records[group].append((name, stratum))
            original_split[name] = split_name

    original_test_groups = {
        classify(task, name)[0] for name in test_names
    }
    remaining_groups = set(records) - original_test_groups
    groups_by_stratum: dict[str, list[str]] = defaultdict(list)
    for group in remaining_groups:
        groups_by_stratum[_group_stratum(task, records[group])].append(group)

    val_groups: set[str] = set()
    stratum_summary: dict[str, object] = {}
    for stratum, groups in sorted(groups_by_stratum.items()):
        sizes = {group: len(records[group]) for group in groups}
        available_samples = sum(sizes.values())
        target = max(1, round(available_samples * val_fraction))
        selected = _select_groups_near_target(
            sizes, target, f"{seed}:{task}:{stratum}"
        )
        val_groups.update(selected)
        stratum_summary[stratum] = {
            "available_groups": len(groups),
            "available_samples": available_samples,
            "validation_groups": len(selected),
            "validation_samples": sum(sizes[group] for group in selected),
            "target_samples": target,
        }

    assignment: dict[str, str] = {}
    for group, group_records in records.items():
        split_name = "test" if group in original_test_groups else (
            "val" if group in val_groups else "train"
        )
        for name, _ in group_records:
            assignment[name] = split_name

    # Preserve the source-list order. Test starts with its historical entries,
    # followed by train entries moved there to close leaked scene groups.
    ordered_names = train_names + test_names
    result = {
        "train": [name for name in train_names if assignment[name] == "train"],
        "val": [name for name in train_names if assignment[name] == "val"],
        "test": test_names + [
            name for name in train_names if assignment[name] == "test"
        ],
    }

    split_groups = {
        split_name: {classify(task, name)[0] for name in names}
        for split_name, names in result.items()
    }
    if split_groups["train"] & split_groups["val"]:
        raise AssertionError(f"{task}: train/val group leakage")
    if split_groups["train"] & split_groups["test"]:
        raise AssertionError(f"{task}: train/test group leakage")
    if split_groups["val"] & split_groups["test"]:
        raise AssertionError(f"{task}: val/test group leakage")
    if set().union(*(set(names) for names in result.values())) != set(ordered_names):
        raise AssertionError(f"{task}: split lost or added filenames")

    moved_to_test = [
        name for name in train_names if original_split[name] == "train" and assignment[name] == "test"
    ]
    moved_test_groups = sorted({classify(task, name)[0] for name in moved_to_test})
    metadata = {
        "task": task,
        "seed": seed,
        "validation_fraction": val_fraction,
        "counts": {split: len(names) for split, names in result.items()},
        "group_counts": {split: len(groups) for split, groups in split_groups.items()},
        "sha256": {split: _sha256_names(names) for split, names in result.items()},
        "historical_train_samples_moved_to_test_for_group_closure": len(moved_to_test),
        "historical_train_groups_moved_to_test": len(moved_test_groups),
        "historical_train_group_ids_moved_to_test": moved_test_groups,
        "strata": stratum_summary,
    }
    return result, metadata


def write_splits(
    input_root: Path, output_root: Path, val_fraction: float, seed: int, overwrite: bool
) -> dict[str, object]:
    if not 0.0 < val_fraction < 0.5:
        raise ValueError("val_fraction must be between 0 and 0.5.")
    manifest: dict[str, object] = {
        "seed": seed,
        "validation_fraction": val_fraction,
        "tasks": {},
    }
    pending: dict[str, tuple[dict[str, list[str]], dict[str, object]]] = {}
    for task in TASKS:
        train_names = _read_list(input_root / task / "train.txt")
        test_names = _read_list(input_root / task / "test.txt")
        pending[task] = create_task_split(
            task, train_names, test_names, val_fraction, seed
        )

    destinations = [
        output_root / task / f"{split}.txt"
        for task in TASKS for split in ("train", "val", "test")
    ] + [output_root / "split_manifest.json"]
    existing = [path for path in destinations if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            f"Refusing to overwrite {len(existing)} existing output files; use --overwrite."
        )

    for task, (splits, metadata) in pending.items():
        task_dir = output_root / task
        task_dir.mkdir(parents=True, exist_ok=True)
        for split_name, names in splits.items():
            (task_dir / f"{split_name}.txt").write_text(
                "".join(f"{name}\n" for name in names), encoding="utf-8"
            )
        manifest["tasks"][task] = metadata  # type: ignore[index]
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "split_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--val-fraction", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    manifest = write_splits(
        args.input_root, args.output_root, args.val_fraction, args.seed, args.overwrite
    )
    for task, metadata in manifest["tasks"].items():
        print(task, metadata["counts"], metadata["sha256"])


if __name__ == "__main__":
    main()
