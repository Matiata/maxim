#!/usr/bin/env python3
"""Gráficos de calidad frente a compresión de las versiones con expertos RCAB.

Lee los resultados ya descargados en runs_logs y no necesita JAX:

* compresion_test_niveles: diferencias de PSNR en test, las que usa la tesis.
* compresion_validacion_niveles: las mismas en validación, con --split val.
* benchmark_maxim_niveles: GFLOP de todas las versiones, medidos en una misma
  corrida.

Produce, en tesis/figs:

* fig_calidad_vs_flops.pdf: Δ PSNR macro frente a S0_sel contra la reducción
  de GFLOP frente a B0.
* fig_calidad_por_tarea.pdf: heatmap de Δ PSNR por tarea frente a S0_sel.

Con --split val, los mismos gráficos con sufijo _val. Las versiones con dos
semillas usan la diferencia promediada entre semillas y su intervalo; las de una
semilla usan la réplica v1. Todas las diferencias son pareadas imagen a imagen,
con bootstrap estratificado por tarea.

Uso:
    python3 tesis/fig_calidad_vs_compresion.py
    python3 tesis/fig_calidad_vs_compresion.py --split val --output-dir /ruta --png
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(os.environ.get("TMPDIR", "/tmp")) / "maxim-matplotlib-cache"),
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter

REPO = Path(__file__).resolve().parents[1]
RUNS = REPO / "runs_logs"
VAL_PATH = RUNS / "compresion_validacion_niveles" / "validacion_compresion_niveles_results.json"
TEST_PATH = RUNS / "compresion_test_niveles" / "test_final_compresion_niveles_results.json"
BENCH_PATH = RUNS / "benchmark_maxim_niveles" / "benchmark_niveles_results.json"

TASKS = ["deblur", "dehaze", "denoise", "derain", "enhance"]
LEVELS = [("", "nivel 2"), ("_l12", "niveles 1–2"), ("_l012", "niveles 0–2")]
FAMILIES = {
    "S1": {"label": "S1: módulo común", "color": "#eb6834", "marker": "s"},
    "S2": {"label": "S2: cinco expertos", "color": "#2a78d6", "marker": "o"},
}
REFERENCE_COLOR = "#52514e"
CRITERION_DB = -0.05
# Rango del eje Y: centrado en la zona del criterio. Los puntos por debajo se
# marcan en el borde con una flecha y su valor.
Y_LIMITS = (-0.75, 0.35)

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
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#8a8984",
        "axes.linewidth": 0.6,
        "xtick.color": "#52514e",
        "ytick.color": "#52514e",
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "savefig.dpi": 200,
    }
)


def comma(value: float, decimals: int) -> str:
    return f"{value:.{decimals}f}".replace(".", ",").replace("-", "−")


def comma_formatter(decimals: int) -> FuncFormatter:
    return FuncFormatter(lambda value, _pos: comma(value, decimals))


def load(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Falta {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def variant_difference(result: dict, variant: str) -> dict:
    """Δ PSNR frente a S0_sel: macro, tareas, intervalos y cantidad de semillas."""
    averaged = result["seed_averaged_differences"].get(f"{variant} - S0_sel")
    if averaged is not None:
        psnr, seeds = averaged["psnr"], 2
    else:
        psnr, seeds = result["paired_differences"][f"{variant}_v1 - S0_sel"]["psnr"], 1
    return {
        "macro": psnr["macro"],
        "macro_ci": tuple(psnr["macro_ci95"]),
        "task": dict(zip(TASKS, psnr["task"])),
        "seeds": seeds,
    }


def baseline_difference(result: dict) -> dict:
    """B0 − S0_sel a partir de S0_sel − B0."""
    psnr = result["paired_differences"]["S0_sel - B0"]["psnr"]
    low, high = psnr["macro_ci95"]
    return {"macro": -psnr["macro"], "macro_ci": (-high, -low)}


def costs(bench: dict) -> dict:
    """Reducción de parámetros y de GFLOP a 256×256 frente a B0, en %."""
    params = {name: spec["parameters"]["total_parameters"]
              for name, spec in bench["variants"].items()}
    flops = {}
    for key, entry in bench["measurements"].items():
        group, _, name = key.rpartition("/")
        if group == "inference/256x256" and entry["lowered_cost_analysis"]["available"]:
            flops[name] = entry["lowered_cost_analysis"]["values"]["flops"]
    out = {}
    for name in params:
        out[name] = {
            "params": 100 * (1 - params[name] / params["B0"]),
            "flops": 100 * (1 - flops[name] / flops["B0"]),
        }
    return out


def scatter_panel(ax, result: dict, cost: dict, axis: str, title: str) -> None:
    ax.axhspan(CRITERION_DB, 0.0, color="#f0efec", zorder=0, lw=0)
    ax.axhline(0.0, color="#8a8984", lw=0.6, zorder=1)
    ax.axhline(CRITERION_DB, color="#8a8984", lw=0.6, ls=(0, (3, 2)), zorder=1)

    for name, label in (("S0_sel", "shared"), ("B0", "baseline")):
        y, ci = ((0.0, None) if name == "S0_sel" else
                 (baseline_difference(result)["macro"], baseline_difference(result)["macro_ci"]))
        x = cost[name][axis]
        if ci:
            ax.plot([x, x], ci, color=REFERENCE_COLOR, lw=1.0, zorder=2)
        ax.plot(x, y, marker="D", ms=5, color="white", mec=REFERENCE_COLOR, mew=1.2, zorder=3)
        above = name == "S0_sel"
        ax.annotate(label, (x, y), xytext=(-5, 4 if above else -4), textcoords="offset points",
                    ha="right", va="bottom" if above else "top", fontsize=7, color="#52514e")

    for family, style in FAMILIES.items():
        xs, ys = [], []
        for suffix, level_label in LEVELS:
            variant = f"{family}_d32{suffix}"
            diff = variant_difference(result, variant)
            x, y = cost[variant][axis], diff["macro"]
            ax.plot([x, x], diff["macro_ci"], color=style["color"], lw=1.2, zorder=4)
            filled = diff["seeds"] == 2
            if y < Y_LIMITS[0]:
                # Fuera de rango: flecha en el borde inferior con el valor real.
                edge = Y_LIMITS[0] + 0.03
                ax.annotate("", (x, Y_LIMITS[0]), xytext=(x, edge + 0.07),
                            arrowprops={"arrowstyle": "-|>", "color": style["color"],
                                        "lw": 1.2, "mutation_scale": 8}, zorder=5)
                ax.annotate(comma(y, 2), (x, edge + 0.07), xytext=(-5, 0),
                            textcoords="offset points", ha="right", va="center",
                            fontsize=7, color="#52514e")
                y = Y_LIMITS[0]
            else:
                ax.plot(x, y, marker=style["marker"], ms=6, zorder=5,
                        color=style["color"] if filled else "white",
                        mec=style["color"], mew=1.4)
            xs.append(x)
            ys.append(y)
        ax.plot(xs, ys, color=style["color"], lw=1.0, alpha=0.6, zorder=3,
                clip_on=True)

    ax.set_title(title)
    ax.set_ylim(*Y_LIMITS)
    ax.set_yticks([-0.6, -0.4, -0.2, 0.0, 0.2])
    ax.xaxis.set_major_formatter(comma_formatter(0))
    ax.yaxis.set_major_formatter(comma_formatter(1))
    ax.grid(axis="y", color="#e4e3df", lw=0.5, zorder=0)


def annotate_levels(ax, result: dict, cost: dict, axis: str) -> None:
    """Rótulo de nivel junto al punto de S2, sólo en un panel."""
    placement = {"": ((-4, 6), "right", "bottom"), "_l12": ((4, 6), "left", "bottom"),
                 "_l012": ((-6, -4), "right", "top")}
    for suffix, level_label in LEVELS:
        variant = f"S2_d32{suffix}"
        x, y = cost[variant][axis], variant_difference(result, variant)["macro"]
        offset, ha, va = placement[suffix]
        ax.annotate(level_label, (x, y), xytext=offset, textcoords="offset points",
                    ha=ha, va=va, fontsize=7, color="#52514e")


def scatter_figure(result: dict, cost: dict, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(4.8, 3.3))
    scatter_panel(ax, result, cost, "flops", "")
    annotate_levels(ax, result, cost, "flops")
    ax.set_ylabel("Δ PSNR macro frente a shared (dB)")
    ax.set_xlabel("Reducción de GFLOP frente a B0 (%)")
    points = [cost[name]["flops"] for name in cost]
    ax.set_xlim(min(points) - 2.5, max(points) + 3)
    handles = [
        Line2D([], [], marker=s["marker"], color=s["color"], mec=s["color"], ms=6, lw=1.0,
               label=s["label"]) for s in FAMILIES.values()
    ] + [
        Line2D([], [], marker="o", color="#8a8984", mfc="white", mec="#8a8984", ms=6, lw=0,
               label="una semilla"),
        Line2D([], [], color="#8a8984", lw=0.6, ls=(0, (3, 2)),
               label=f"criterio, {comma(CRITERION_DB, 2)} dB"),
    ]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.2),
              ncol=2, frameon=False, handlelength=1.8, columnspacing=1.5)
    fig.savefig(path)
    plt.close(fig)


DIVERGING = LinearSegmentedColormap.from_list(
    "rojo_gris_azul", ["#b3261e", "#e34948", "#f0efec", "#3987e5", "#1c5cab"])


def heatmap_panel(ax, result: dict, title: str, norm) -> object:
    rows, labels = [], []
    for suffix, level_label in LEVELS:
        for family in FAMILIES:
            variant = f"{family}_d32{suffix}"
            diff = variant_difference(result, variant)
            rows.append([diff["task"][t] for t in TASKS] + [diff["macro"]])
            seeds = "" if diff["seeds"] == 2 else " ¹"
            labels.append(f"{variant}{seeds}")
    image = ax.imshow(rows, cmap=DIVERGING, norm=norm, aspect="auto")
    for i, row in enumerate(rows):
        for j, value in enumerate(row):
            color = "white" if abs(value) > 0.3 else "#0b0b0b"
            weight = "bold" if j == len(TASKS) else "normal"
            ax.text(j, i, comma(value, 2), ha="center", va="center",
                    fontsize=7, color=color, fontweight=weight)
    ax.set_xticks(range(len(TASKS) + 1), TASKS + ["macro"], rotation=0)
    ax.set_yticks(range(len(labels)), labels)
    ax.tick_params(length=0)
    ax.axvline(len(TASKS) - 0.5, color="white", lw=2)
    for boundary in (1.5, 3.5):
        ax.axhline(boundary, color="white", lw=2)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(title)
    return image


def heatmap_figure(result: dict, path: Path) -> None:
    limit = 0.5
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)
    fig, ax = plt.subplots(figsize=(5.8, 2.7))
    fig.subplots_adjust(left=0.2, right=0.84, top=0.97, bottom=0.16)
    image = heatmap_panel(ax, result, "", norm)
    cax = fig.add_axes([0.87, 0.25, 0.025, 0.65])
    bar = fig.colorbar(image, cax=cax, extend="both")
    bar.set_label("Δ PSNR frente a shared (dB)")
    bar.ax.yaxis.set_major_formatter(comma_formatter(2))
    bar.outline.set_visible(False)
    fig.text(0.2, 0.02, "¹ Una semilla. Sin marca: promedio de dos semillas. "
             f"Colores saturados más allá de ±{comma(limit, 1)} dB.",
             fontsize=7, color="#52514e")
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--output-dir", type=Path, default=REPO / "tesis" / "figs")
    parser.add_argument("--png", action="store_true", help="también guarda PNG")
    parser.add_argument("--split", choices=["test", "val"], default="test",
                        help="split de las diferencias; la tesis usa test")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    result = load(TEST_PATH if args.split == "test" else VAL_PATH)
    cost = costs(load(BENCH_PATH))
    suffix_name = "" if args.split == "test" else "_val"
    outputs = {
        f"fig_calidad_vs_flops{suffix_name}": lambda p: scatter_figure(result, cost, p),
        f"fig_calidad_por_tarea{suffix_name}": lambda p: heatmap_figure(result, p),
    }
    for name, build in outputs.items():
        for suffix in ([".pdf", ".png"] if args.png else [".pdf"]):
            path = args.output_dir / f"{name}{suffix}"
            build(path)
            print(path)


if __name__ == "__main__":
    main()
