#!/usr/bin/env python3
"""Parse training logs and generate clean vector figures for the thesis.

Reads the raw training logs in moe_outputs/ and no_moe_outputs/ and produces
the PDF figures used in the Results section. Re-run with:

    /tmp/plotenv/bin/python tesis/make_figs.py

Outputs are written to tesis/figs/.
"""

import os
import re
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "tesis", "figs")
os.makedirs(OUT, exist_ok=True)

# ---------------------------------------------------------------------------
# Global style: clean, academic, serif to match the LaTeX body font.
# ---------------------------------------------------------------------------
plt.rcParams.update({
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
})

# Colour-blind friendly palette (Okabe-Ito subset).
C_MULTI = "#0072B2"    # blue          -> MAXIM multi-task (S-3, warm start)
C_MULTI_S = "#CC79A7"  # reddish purple -> MAXIM multi-task (S-2, from scratch; fair)
C_SINGLE = "#009E73"   # green         -> MAXIM single-task baseline
C_MOE = "#D55E00"      # orange        -> MAXIM+MoE
C_AUX = "#5A5A5A"      # grey          -> references / second axis


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------
def parse_moe(path):
    """MoE epoch summaries: 'Epoch N Training/Validation Metrics: {...}'."""
    tr, va = {}, {}
    re_tr = re.compile(r"Epoch (\d+) Training Metrics: \{(.+)\}")
    re_va = re.compile(r"Epoch (\d+) Validation Metrics: \{(.+)\}")

    def parse_dict(body):
        body = re.sub(r"np\.float32\(([^)]*)\)", r"\1", body)
        d = {}
        for k, v in re.findall(r"'(\w+)':\s*([-\d.eE]+)", body):
            d[k] = float(v)
        return d

    with open(path) as f:
        for line in f:
            m = re_tr.search(line)
            if m:
                tr[int(m.group(1))] = parse_dict(m.group(2))
                continue
            m = re_va.search(line)
            if m:
                va[int(m.group(1))] = parse_dict(m.group(2))
    return tr, va


def parse_no_moe(path):
    """no-MoE epoch summaries: 'Epoch N train/val metrics: {...}'."""
    tr, va = {}, {}
    re_tr = re.compile(r"Epoch (\d+) train metrics: \{(.+)\}")
    re_va = re.compile(r"Epoch (\d+) val metrics: \{(.+)\}")

    def parse_dict(body):
        body = re.sub(r"np\.float32\(([^)]*)\)", r"\1", body)
        d = {}
        for k, v in re.findall(r"'(\w+)':\s*([-\d.eE]+)", body):
            d[k] = float(v)
        return d

    with open(path) as f:
        for line in f:
            m = re_tr.search(line)
            if m:
                tr[int(m.group(1))] = parse_dict(m.group(2))
                continue
            m = re_va.search(line)
            if m:
                va[int(m.group(1))] = parse_dict(m.group(2))
    return tr, va


def series(d, key):
    ep = sorted(d.keys())
    return np.array(ep), np.array([d[e][key] for e in ep])


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
# MoE: corrida 2026-07 desde cero (S-2, random init) con el random_crop corregido.
# La corrida vieja (moe_outputs/training_outs_2.txt) entrenaba contra parches
# desalineados y quedo obsoleta.
moe_tr, moe_va = parse_moe(os.path.join(ROOT, "moe_outputs", "multi_task_scratch.txt"))
# single_task.txt concatena varios intentos (1-21, 1-15 y la corrida final 1-30);
# el parser guarda la ULTIMA ocurrencia de cada epoca, que es la corrida completa.
st_tr, st_va = parse_no_moe(os.path.join(ROOT, "no_moe_outputs", "single_task.txt"))
mt_tr, mt_va = parse_no_moe(os.path.join(ROOT, "no_moe_outputs", "multi_task.txt"))
# Fair comparison: multi-task, S-2 backbone, random init (same scale + init as the MoE).
mts_tr, mts_va = parse_no_moe(os.path.join(ROOT, "no_moe_outputs", "multi_task_scratch.txt"))

NUM_EXPERTS = 5
LN_K = math.log(NUM_EXPERTS)

# Quick textual summary (also used to sanity-check the LaTeX numbers).
def best(d, key="psnr"):
    ep, v = series(d, key)
    i = int(np.argmax(v))
    return ep[i], v[i]

print("=== Best validation PSNR (dB) ===")
for name, d in [("MAXIM multi (S-3 warm)", mt_va),
                ("MAXIM multi (S-2 scratch)", mts_va),
                ("MAXIM single-task", st_va),
                ("MAXIM+MoE", moe_va)]:
    e, p = best(d)
    print(f"  {name:26s}: {p:6.2f} dB  (epoch {e})")
print("=== MoE router (last epoch) ===")
le = max(moe_tr)
print(f"  entropy={moe_tr[le]['entropy']:.4f}  ln(K)={LN_K:.4f}  "
      f"balance={moe_tr[le]['balance']:.4f}")


# ---------------------------------------------------------------------------
# Figure 1 — Validation PSNR vs epoch (headline comparison, matched budget)
# ---------------------------------------------------------------------------
def fig_val_psnr():
    fig, ax = plt.subplots(figsize=(3.4, 2.7))
    EMAX = 30  # matched training budget

    for d, c, lab, mk in [
        (mt_va, C_MULTI, "MAXIM multi-tarea (S-3, pre-entr.)", "o"),
        (mts_va, C_MULTI_S, "MAXIM multi-tarea (S-2, aleat.)", "D"),
        (st_va, C_SINGLE, "MAXIM mono-tarea (S-2, aleat.)", "s"),
        (moe_va, C_MOE, "MAXIM+MoE (S-2, aleat.)", "^"),
    ]:
        ep, v = series(d, "psnr")
        mask = ep <= EMAX
        ax.plot(ep[mask], v[mask], color=c, marker=mk, markersize=2.8,
                markevery=2, label=lab)

    ax.set_xlabel("Época")
    ax.set_ylabel("PSNR de validación (dB)")
    ax.set_xlim(1, EMAX)
    ax.set_ylim(12, 27)
    ax.grid(True, linestyle="--")
    # All curves climb above ~19 dB after the first epochs, so the lower-right
    # region stays empty for the legend.
    ax.legend(loc="lower right", frameon=True, framealpha=0.88, edgecolor="none",
              handlelength=1.6, fontsize=7.0)
    fig.savefig(os.path.join(OUT, "fig_val_psnr.pdf"))
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2 — MoE dynamics: (a) train/val PSNR convergence, (b) uniform routing
# ---------------------------------------------------------------------------
def fig_moe_dynamics():
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(7.0, 2.7))

    # (a) PSNR convergence over the 30-epoch run (fixed random_crop)
    ep, tr = series(moe_tr, "psnr")
    epv, va = series(moe_va, "psnr")
    axL.plot(ep, tr, color=C_MOE, label="Entrenamiento")
    axL.plot(epv, va, color=C_MULTI, label="Validación")
    axL.set_xlabel("Época")
    axL.set_ylabel("PSNR (dB)")
    axL.set_ylim(2, 25)
    axL.grid(True, linestyle="--")
    axL.legend(loc="lower right", frameon=False, handlelength=1.6)
    axL.set_title("(a) PSNR del modelo MoE", fontsize=9)

    # (b) router entropy and balance stay pinned at the uniform regime.
    # Zoomed axes: entropy moves within [1.600, 1.609] (max ln 5 = 1.6094) and
    # balance within [1.001, 1.010] (min 1.0), so a full-range axis would show
    # two flat lines.
    ep, ent = series(moe_tr, "entropy")
    _, bal = series(moe_tr, "balance")
    axR.plot(ep, ent, color=C_MOE, label="Entropía del ruteador")
    axR.axhline(LN_K, color=C_AUX, linestyle=":", linewidth=1.1,
                label=r"$\ln K$ (máx., uniforme)")
    axR.set_xlabel("Época")
    axR.set_ylabel("Entropía (nats)")
    axR.set_ylim(1.55, 1.62)
    axR.grid(True, linestyle="--")
    axR.set_title("(b) Diagnóstico del ruteador", fontsize=9)

    axR2 = axR.twinx()
    axR2.plot(ep, bal, color=C_SINGLE, linestyle="--", label="Balance")
    axR2.axhline(1.0, color=C_SINGLE, linestyle=":", linewidth=0.8, alpha=0.6)
    axR2.set_ylabel("Balance de carga", color=C_SINGLE)
    axR2.tick_params(axis="y", labelcolor=C_SINGLE)
    axR2.set_ylim(0.99, 1.06)

    # merged legend for panel (b)
    h1, l1 = axR.get_legend_handles_labels()
    h2, l2 = axR2.get_legend_handles_labels()
    axR.legend(h1 + h2, l1 + l2, loc="center right", frameon=False,
               handlelength=1.6, fontsize=7.5)

    fig.tight_layout(w_pad=2.0)
    fig.savefig(os.path.join(OUT, "fig_moe_dynamics.pdf"))
    plt.close(fig)


# The qualitative figures (fig_qualitative.pdf, fig_qualitative_enhance.pdf)
# are generated by tesis/eval_metrics.py in the Colab/Drive environment with
# the trained checkpoints + datasets; this script does not touch them.

fig_val_psnr()
fig_moe_dynamics()
print("\nFigures written to", OUT)
for f in sorted(os.listdir(OUT)):
    print("  ", f)
