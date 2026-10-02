"""Shared metric aggregation and checkpoint metadata for MAXIM experiments."""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np


SELECTION_METRIC = "psnr_macro"
SUPPORTED_SELECTION_METRICS = ("psnr_macro", "psnr_weighted")
def _aggregate_task_mean(
    task_sum: Sequence[float],
    task_count: Sequence[int],
    *,
    metric_name: str,
) -> dict[str, object]:
    """Aggregate per-image metric sums by task, weighted mean and macro mean."""
    values_sum = np.asarray(task_sum, dtype=np.float64)
    counts = np.asarray(task_count, dtype=np.int64)
    if values_sum.ndim != 1 or counts.ndim != 1 or values_sum.shape != counts.shape:
        raise ValueError(
            f"task_{metric_name}_sum and task_count must be one-dimensional "
            "arrays with the same shape."
        )
    if np.any(counts < 0):
        raise ValueError("task_count cannot contain negative values.")
    present = counts > 0
    if not np.any(present):
        raise ValueError("At least one task must contain evaluation samples.")
    if not np.all(np.isfinite(values_sum[present])):
        raise ValueError(
            f"{metric_name.upper()} sums for present tasks must be finite."
        )

    task_metric = np.divide(
        values_sum,
        counts,
        out=np.full(values_sum.shape, np.nan, dtype=np.float64),
        where=present,
    )
    total_count = int(np.sum(counts))
    weighted = float(
        np.sum(task_metric[present] * counts[present]) / total_count
    )
    macro = float(np.mean(task_metric[present]))
    return {
        f"{metric_name}_weighted": weighted,
        f"{metric_name}_macro": macro,
        f"task_{metric_name}": task_metric,
        "task_count": counts,
        "sample_count": total_count,
    }


def aggregate_task_psnr(
    task_psnr_sum: Sequence[float], task_count: Sequence[int]
) -> dict[str, object]:
    """Aggregate per-image PSNR sums into explicit task/macro/weighted metrics.

    Tasks with no samples are represented by ``NaN`` and excluded from the macro
    mean. The weighted value is the per-image mean over every present task.
    """
    return _aggregate_task_mean(
        task_psnr_sum, task_count, metric_name="psnr"
    )


def aggregate_task_ssim(
    task_ssim_sum: Sequence[float], task_count: Sequence[int]
) -> dict[str, object]:
    """Aggregate per-image SSIM sums using the same task counts as PSNR."""
    return _aggregate_task_mean(
        task_ssim_sum, task_count, metric_name="ssim"
    )


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

    payload = {
        "epoch": int(epoch),
        "selection_metric": selection_metric,
        "selection_value": float(metrics[selection_metric]),
        "psnr_macro": float(metrics["psnr_macro"]),
        "psnr_weighted": float(metrics["psnr_weighted"]),
        "task_names": list(task_names),
        "task_psnr": task_psnr.tolist(),
        "task_count": task_count.tolist(),
    }
    ssim_keys = ("ssim_macro", "ssim_weighted", "task_ssim")
    available_ssim_keys = [key for key in ssim_keys if key in metrics]
    if available_ssim_keys and len(available_ssim_keys) != len(ssim_keys):
        missing_ssim = [key for key in ssim_keys if key not in metrics]
        raise KeyError(f"Incomplete SSIM checkpoint metrics: {missing_ssim}")
    if available_ssim_keys:
        task_ssim = np.asarray(metrics["task_ssim"], dtype=np.float64)
        if task_ssim.shape != task_count.shape:
            raise ValueError("task_ssim and task_count must have equal lengths.")
        if not np.all(np.isfinite(task_ssim[task_count > 0])):
            raise ValueError("SSIM values for present tasks must be finite.")
        payload.update(
            {
                "ssim_macro": float(metrics["ssim_macro"]),
                "ssim_weighted": float(metrics["ssim_weighted"]),
                "task_ssim": task_ssim.tolist(),
                "ssim_config": {
                    "data_range": 1.0,
                    "filter_size": 11,
                    "filter_sigma": 1.5,
                    "k1": 0.01,
                    "k2": 0.03,
                    "color_space": "RGB",
                    "channel_reduction": "mean",
                    "padding": "VALID",
                    "clipping": "none",
                },
            }
        )
    return payload


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
