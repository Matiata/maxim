"""Shared metric aggregation and checkpoint metadata for MAXIM experiments."""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np


SELECTION_METRIC = "psnr_macro"
SUPPORTED_SELECTION_METRICS = ("psnr_macro", "psnr_weighted")


def aggregate_task_psnr(
    task_psnr_sum: Sequence[float], task_count: Sequence[int]
) -> dict[str, object]:
    """Aggregate per-image PSNR sums into explicit task/macro/weighted metrics.

    Tasks with no samples are represented by ``NaN`` and excluded from the macro
    mean. The weighted value is the per-image mean over every present task.
    """
    psnr_sum = np.asarray(task_psnr_sum, dtype=np.float64)
    counts = np.asarray(task_count, dtype=np.int64)
    if psnr_sum.ndim != 1 or counts.ndim != 1 or psnr_sum.shape != counts.shape:
        raise ValueError(
            "task_psnr_sum and task_count must be one-dimensional arrays "
            "with the same shape."
        )
    if np.any(counts < 0):
        raise ValueError("task_count cannot contain negative values.")
    present = counts > 0
    if not np.any(present):
        raise ValueError("At least one task must contain evaluation samples.")
    if not np.all(np.isfinite(psnr_sum[present])):
        raise ValueError("PSNR sums for present tasks must be finite.")

    task_psnr = np.divide(
        psnr_sum,
        counts,
        out=np.full(psnr_sum.shape, np.nan, dtype=np.float64),
        where=present,
    )
    total_count = int(np.sum(counts))
    psnr_weighted = float(np.sum(task_psnr[present] * counts[present]) / total_count)
    psnr_macro = float(np.mean(task_psnr[present]))
    return {
        "psnr_weighted": psnr_weighted,
        "psnr_macro": psnr_macro,
        "task_psnr": task_psnr,
        "task_count": counts,
        "sample_count": total_count,
    }


def build_best_metric_payload(
    epoch: int,
    metrics: Mapping[str, object],
    task_names: Sequence[str],
    selection_metric: str = SELECTION_METRIC,
) -> dict[str, object]:
    """Build the complete, unambiguous metadata stored with a best checkpoint."""
    if selection_metric not in SUPPORTED_SELECTION_METRICS:
        raise ValueError(f"Unsupported selection metric: {selection_metric!r}")
    required = ("psnr_weighted", "psnr_macro", "task_psnr", "task_count")
    missing = [key for key in required if key not in metrics]
    if missing:
        raise KeyError(f"Missing checkpoint metrics: {missing}")

    task_psnr = np.asarray(metrics["task_psnr"], dtype=np.float64)
    task_count = np.asarray(metrics["task_count"], dtype=np.int64)
    if len(task_names) != len(task_psnr) or task_psnr.shape != task_count.shape:
        raise ValueError("task_names, task_psnr and task_count must have equal lengths.")

    return {
        "epoch": int(epoch),
        "selection_metric": selection_metric,
        "selection_value": float(metrics[selection_metric]),
        "psnr_macro": float(metrics["psnr_macro"]),
        "psnr_weighted": float(metrics["psnr_weighted"]),
        "task_names": list(task_names),
        "task_psnr": task_psnr.tolist(),
        "task_count": task_count.tolist(),
    }


def read_selection_value(
    payload: Mapping[str, object], selection_metric: str = SELECTION_METRIC
) -> tuple[float, int]:
    """Validate new-format best-checkpoint metadata and return value and epoch."""
    stored_metric = payload.get("selection_metric")
    if stored_metric != selection_metric:
        raise ValueError(
            f"Checkpoint was selected by {stored_metric!r}, not {selection_metric!r}."
        )
    if selection_metric not in payload:
        raise KeyError(f"Missing {selection_metric!r} in best checkpoint metadata.")
    value = float(payload[selection_metric])
    epoch = int(payload.get("epoch", -1))
    if not np.isfinite(value) or epoch < 1:
        raise ValueError(f"Invalid best checkpoint metadata: {payload}")
    return value, epoch
