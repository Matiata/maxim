#!/usr/bin/env python3
"""Genera curvas de entrenamiento para las corridas actuales de MAXIM+MoE.

Lee los cuatro archivos training.log de runs_logs y produce una figura PDF por
métrica:

* fig_loss.pdf: pérdida media de entrenamiento.
* fig_psnr_macro.pdf: PSNR macro de validación.
* fig_psnr_weighted.pdf: PSNR ponderado de validación.

Las continuaciones baseline +10 y shared residual +10 parten del baseline
entrenado durante 30 épocas. Por eso sus épocas 1..10 se representan como
épocas acumuladas 31..40. En las curvas de validación se agrega el punto fuente
de la época 30 para hacer explícito el warm start. No se agrega ese punto a la
pérdida de entrenamiento porque no existe una medición de época 0 comparable.

Uso:
    python tesis/make_figs.py
    python tesis/make_figs.py --check-only
    python tesis/make_figs.py --output-dir /ruta/de/salida
"""

from __future__ import annotations

import argparse
import ast
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

# Evita warnings y recompilación de fuentes en entornos con HOME de solo lectura.
os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(os.environ.get("TMPDIR", "/tmp")) / "maxim-matplotlib-cache"),
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "tesis" / "figs"
SOURCE_EPOCH = 30

NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"


@dataclass(frozen=True)
class RunSpec:
    label: str
    directory: Path
    color: str
    marker: str
    linestyle: str = "-"
    epoch_offset: int = 0
    uses_source_validation: bool = False

    @property
    def log_path(self) -> Path:
        return self.directory / "training.log"


RUNS = (
    RunSpec(
        label="Baseline desde cero",
        directory=ROOT / "runs_logs" / "baseline_S-2_newsplit_scratch",
        color="#0072B2",
        marker="o",
    ),
    RunSpec(
        label="Baseline +10",
        directory=ROOT / "runs_logs" / "baseline_S-2_from_newsplit_ckpt_extra10",
        color="#56B4E9",
        marker="s",
        linestyle="--",
        epoch_offset=SOURCE_EPOCH,
        uses_source_validation=True,
    ),
    RunSpec(
        label="Shared residual +10",
        directory=ROOT
        / "runs_logs"
        / "moe_shared_residual_from_newsplit_baseline_extra10",
        color="#D55E00",
        marker="^",
        epoch_offset=SOURCE_EPOCH,
        uses_source_validation=True,
    ),
    RunSpec(
        label="Token-choice K=8",
        directory=ROOT
        / "runs_logs"
        / "moe_token_choice_k8_top2_S-2_scratch",
        color="#009E73",
        marker="D",
    ),
)


plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman", "cmr10"],
        "mathtext.fontset": "dejavuserif",
        "font.size": 9,
        "axes.titlesize": 9,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.linewidth": 0.7,
        "lines.linewidth": 1.4,
        "grid.linewidth": 0.5,
        "grid.alpha": 0.45,
        "figure.dpi": 150,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    }
)


def _parse_python_dict(body: str) -> dict[str, float]:
    """Extrae únicamente escalares numéricos de un diccionario impreso."""
    normalized = re.sub(r"np\.float(?:32|64)\(([^)]*)\)", r"\1", body)
    try:
        raw = ast.literal_eval("{" + normalized + "}")
    except (SyntaxError, ValueError):
        return {}

    parsed: dict[str, float] = {}
    for key, value in raw.items():
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            parsed[str(key)] = float(value)
    return parsed


def _extract_named(body: str, name: str, *, unit: str = "") -> float | None:
    pattern = rf"(?<!SSIM\s)\b{re.escape(name)}\s*=\s*({NUMBER})"
    if unit:
        pattern += rf"\s*{re.escape(unit)}"
    match = re.search(pattern, body, flags=re.IGNORECASE)
    return float(match.group(1)) if match else None


def parse_training_log(
    path: os.PathLike[str] | str,
) -> tuple[dict[int, dict[str, float]], dict[int, dict[str, float]]]:
    """Parsea los formatos actuales baseline y MoE.

    Si una corrida fue reanudada y una época aparece más de una vez, conserva
    la última línea resumen de esa época. Las líneas por paso se ignoran.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"No se encontró el log: {path}")

    train: dict[int, dict[str, float]] = {}
    validation: dict[int, dict[str, float]] = {}

    baseline_train = re.compile(
        r"^Epoch\s+(\d+)\s+train metrics:\s*\{(.*)\}\s*$"
    )
    compact_train = re.compile(r"^Epoch\s+(\d+)\s+train:\s*(.*)$")
    current_validation = re.compile(
        r"^Epoch\s+(\d+)\s+validation:\s*(.*)$"
    )

    with path.open(encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()

            match = baseline_train.match(line)
            if match:
                metrics = _parse_python_dict(match.group(2))
                if metrics:
                    train[int(match.group(1))] = metrics
                continue

            match = compact_train.match(line)
            if match:
                body = match.group(2)
                metrics = {
                    key: value
                    for key, value in {
                        "loss": _extract_named(body, "loss"),
                        "psnr_weighted": _extract_named(
                            body, "weighted", unit="dB"
                        ),
                        "psnr_macro": _extract_named(body, "macro", unit="dB"),
                    }.items()
                    if value is not None
                }
                if metrics:
                    train[int(match.group(1))] = metrics
                continue

            match = current_validation.match(line)
            if match:
                body = match.group(2)
                metrics = {
                    key: value
                    for key, value in {
                        "loss": _extract_named(body, "loss"),
                        "psnr_weighted": _extract_named(
                            body, "weighted", unit="dB"
                        ),
                        "psnr_macro": _extract_named(body, "macro", unit="dB"),
                    }.items()
                    if value is not None
                }
                if metrics:
                    validation[int(match.group(1))] = metrics

    if not train:
        raise ValueError(f"No se encontraron resúmenes de entrenamiento en {path}")
    if not validation:
        raise ValueError(f"No se encontraron resúmenes de validación en {path}")
    return train, validation


def _validate_epochs(
    metrics: Mapping[int, Mapping[str, float]],
    *,
    required_key: str,
    label: str,
    split: str,
) -> None:
    missing = [epoch for epoch, values in metrics.items() if required_key not in values]
    if missing:
        raise ValueError(
            f"{label}: falta {required_key!r} en {split}, épocas {missing}"
        )


def _series(
    metrics: Mapping[int, Mapping[str, float]],
    key: str,
    epoch_offset: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    epochs = sorted(metrics)
    x = np.asarray([epoch + epoch_offset for epoch in epochs], dtype=np.int64)
    y = np.asarray([metrics[epoch][key] for epoch in epochs], dtype=np.float64)
    if not np.all(np.isfinite(y)):
        raise ValueError(f"La serie {key!r} contiene valores no finitos.")
    return x, y


def load_runs() -> dict[str, dict[str, object]]:
    parsed: dict[str, dict[str, object]] = {}
    for spec in RUNS:
        train, validation = parse_training_log(spec.log_path)
        _validate_epochs(
            train, required_key="loss", label=spec.label, split="train"
        )
        _validate_epochs(
            validation,
            required_key="psnr_macro",
            label=spec.label,
            split="validation",
        )
        _validate_epochs(
            validation,
            required_key="psnr_weighted",
            label=spec.label,
            split="validation",
        )
        parsed[spec.label] = {
            "spec": spec,
            "train": train,
            "validation": validation,
        }
    return parsed


def _source_validation(
    parsed: Mapping[str, Mapping[str, object]],
) -> Mapping[str, float]:
    source = parsed["Baseline desde cero"]["validation"]
    assert isinstance(source, dict)
    if SOURCE_EPOCH not in source:
        raise ValueError(
            f"El baseline fuente no contiene validación de época {SOURCE_EPOCH}."
        )
    return source[SOURCE_EPOCH]


def print_summary(parsed: Mapping[str, Mapping[str, object]]) -> None:
    print("=== Logs reconocidos ===")
    for spec in RUNS:
        run = parsed[spec.label]
        train = run["train"]
        validation = run["validation"]
        assert isinstance(train, dict)
        assert isinstance(validation, dict)
        train_epochs = sorted(train)
        val_epochs = sorted(validation)
        # Ante empates causados por el redondeo del log, mostrar la última
        # época. La selección real del checkpoint sigue viniendo de su metadata.
        macro_epoch = max(
            val_epochs,
            key=lambda epoch: (validation[epoch]["psnr_macro"], epoch),
        )
        weighted_epoch = max(
            val_epochs,
            key=lambda epoch: (validation[epoch]["psnr_weighted"], epoch),
        )
        print(
            f"{spec.label:24s} "
            f"train={train_epochs[0]}..{train_epochs[-1]} ({len(train_epochs)}) | "
            f"val={val_epochs[0]}..{val_epochs[-1]} ({len(val_epochs)}) | "
            f"mejor macro={validation[macro_epoch]['psnr_macro']:.4f} "
            f"(época {macro_epoch + spec.epoch_offset}) | "
            f"mejor ponderado={validation[weighted_epoch]['psnr_weighted']:.4f} "
            f"(época {weighted_epoch + spec.epoch_offset})"
        )


def plot_metric(
    parsed: Mapping[str, Mapping[str, object]],
    *,
    split: str,
    key: str,
    ylabel: str,
    output_path: Path,
    log_scale: bool = False,
    include_source: bool = False,
) -> None:
    fig, ax = plt.subplots(figsize=(5.6, 3.2))
    source = _source_validation(parsed) if include_source else None

    for spec in RUNS:
        metrics = parsed[spec.label][split]
        assert isinstance(metrics, dict)
        x, y = _series(metrics, key, spec.epoch_offset)

        if include_source and spec.uses_source_validation:
            assert source is not None
            x = np.insert(x, 0, SOURCE_EPOCH)
            y = np.insert(y, 0, float(source[key]))

        ax.plot(
            x,
            y,
            color=spec.color,
            marker=spec.marker,
            linestyle=spec.linestyle,
            markersize=3.0,
            markevery=max(1, len(x) // 10),
            label=spec.label,
        )

    if any(spec.epoch_offset for spec in RUNS):
        ax.axvline(
            SOURCE_EPOCH,
            color="#777777",
            linestyle=":",
            linewidth=1.0,
            label="Inicio de continuaciones",
        )

    ax.set_xlabel("Época acumulada")
    ax.set_ylabel(ylabel)
    if log_scale:
        ax.set_yscale("log")
    ax.grid(True, linestyle="--")
    ax.legend(frameon=True, framealpha=0.9, edgecolor="none", ncol=2)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    print(f"Figura escrita: {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directorio de salida (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Valida y resume los logs sin generar ni sobrescribir figuras.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    parsed = load_runs()
    print_summary(parsed)

    if args.check_only:
        print("\nValidación terminada; no se escribieron figuras.")
        return

    output_dir = args.output_dir.resolve()
    plot_metric(
        parsed,
        split="train",
        key="loss",
        ylabel="Pérdida de entrenamiento",
        output_path=output_dir / "fig_loss.pdf",
        log_scale=True,
    )
    plot_metric(
        parsed,
        split="validation",
        key="psnr_macro",
        ylabel="PSNR macro de validación (dB)",
        output_path=output_dir / "fig_psnr_macro.pdf",
        include_source=True,
    )
    plot_metric(
        parsed,
        split="validation",
        key="psnr_weighted",
        ylabel="PSNR ponderado de validación (dB)",
        output_path=output_dir / "fig_psnr_weighted.pdf",
        include_source=True,
    )


if __name__ == "__main__":
    main()
