#!/usr/bin/env python3
"""Grouped bar chart: PSNR por tarea para los cuatro modelos evaluados.

Parsea la salida ya capturada de `eval_metrics.py` (tesis/eval_metrics_output.txt)
--igual que make_figs.py parsea los logs de entrenamiento-- así que no requiere
JAX/Flax ni los checkpoints/datasets. Como los cuatro modelos se evalúan con
el mismo protocolo (mismo recorte central, misma submuestra de hasta 200
imágenes/tarea, misma métrica), es la única figura del trabajo donde el
mono-tarea puede compararse contra el resto en las cinco tareas de forma
homogénea; reemplaza en ese rol a la antigua fig_qualitative_all.pdf (que
comparaba una única imagen por tarea, sin agregación estadística).

Uso:
    python3 tesis/fig_task_bars.py
"""

import os
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "tesis", "figs")
LOG_PATH = os.path.join(ROOT, "tesis", "eval_metrics_output.txt")
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman", "cmr10"],
    "mathtext.fontset": "dejavuserif",
    "font.size": 9,
    "axes.titlesize": 9,
    "axes.labelsize": 9,
    "legend.fontsize": 7.5,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.linewidth": 0.7,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})

C_MULTI = "#0072B2"    # blue          -> MAXIM multi-task (S-3, warm start)
C_MULTI_S = "#CC79A7"  # reddish purple -> MAXIM multi-task (S-2, from scratch; fair)
C_SINGLE = "#009E73"   # green         -> MAXIM single-task baseline
C_MOE = "#D55E00"      # orange        -> MAXIM+MoE

TASKS = ["deblur", "dehaze", "denoise", "derain", "enhance"]

# (etiqueta en el log, nombre para la leyenda, color, tarea de entrenamiento
# o None si el modelo se entrenó sobre la mezcla completa).
MODELS = [
    ("MAXIM single-task enhance", "MAXIM mono-tarea",           C_SINGLE,   "enhance"),
    ("MAXIM multi-task warm",     "MAXIM multi-tarea (preentr.)", C_MULTI,   None),
    ("MAXIM multi-task scratch",  "MAXIM multi-tarea",          C_MULTI_S,  None),
    ("MAXIM+MoE",                 "MAXIM+MoE",                  C_MOE,      None),
]

HEADER_RE = re.compile(r"^\[\d+/\d+\]\s+(.+?)\s+\(\w+, variante \S+\)\s*$")
TASK_RE = re.compile(
    r"^\s*(\w+)\s+PSNR=\s*([\d.]+)\s+SSIM=([\d.]+)\s+\(n=(\d+)\)\s*$")


def parse_eval_log(path):
    """label -> {task: (psnr, ssim)}."""
    results = {}
    label = None
    with open(path) as f:
        for line in f:
            m = HEADER_RE.match(line)
            if m:
                label = m.group(1)
                results.setdefault(label, {})
                continue
            m = TASK_RE.match(line)
            if m and label is not None:
                task, psnr, ssim, _n = m.groups()
                if task in TASKS:
                    results[label][task] = (float(psnr), float(ssim))
    return results


def fig_task_bars(results):
    fig, ax = plt.subplots(figsize=(6.6, 3.0))

    n_models = len(MODELS)
    group_w = 0.8
    bar_w = group_w / n_models
    x = np.arange(len(TASKS))

    legend_handles = []
    for i, (label, legend, color, own_task) in enumerate(MODELS):
        if label not in results:
            print(f"[aviso] falta '{label}' en el log; se omite del gráfico.")
            continue
        vals = [results[label].get(t, (0.0, 0.0))[0] for t in TASKS]
        offset = (i - (n_models - 1) / 2) * bar_w
        hatches = ["//" if t == own_task else None for t in TASKS]
        for xi, v, h in zip(x + offset, vals, hatches):
            ax.bar(xi, v, width=bar_w * 0.92, color=color, hatch=h,
                   edgecolor="white", linewidth=0.4)
        legend_handles.append(Patch(facecolor=color, label=legend))

    ax.set_xticks(x)
    ax.set_xticklabels(TASKS)
    ax.set_ylabel("PSNR (dB)")
    ax.set_ylim(0, 50)
    ax.grid(True, axis="y", linestyle="--", alpha=0.45)
    ax.legend(handles=legend_handles, loc="upper left", ncols=2, frameon=False,
              handlelength=1.4, columnspacing=1.0)
    # Sin título ni notas dentro de la figura: esa información va en el
    # \caption de la tesis, como en el resto de las figuras.

    out_path = os.path.join(OUT, "fig_task_comparison.pdf")
    fig.savefig(out_path)
    plt.close(fig)
    print("Figura escrita en", out_path)


def main():
    if not os.path.exists(LOG_PATH):
        raise SystemExit(f"No se encontró {LOG_PATH}; correr eval_metrics.py "
                          f"primero y guardar su salida ahí.")
    results = parse_eval_log(LOG_PATH)
    print(f"Modelos encontrados en el log: {list(results.keys())}")
    for label, _, _, _ in MODELS:
        if label not in results:
            print(f"  [aviso] '{label}' no está en el log actual.")
    fig_task_bars(results)


if __name__ == "__main__":
    main()
