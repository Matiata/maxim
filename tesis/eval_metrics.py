#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluación de PSNR + SSIM y figura cualitativa para la tesis MAXIM+MoE.

Este script NO se puede ejecutar en cualquier máquina: necesita
  (1) los checkpoints entrenados (mono-tarea, multi-tarea y MoE),
  (2) los datasets de las cinco tareas (carpetas imgs/ GT/ test.txt), y
  (3) un entorno con JAX/Flax + el paquete `maxim` instalado.
Está pensado para correr en el mismo entorno (Colab/Drive) donde se entrenó,
reutilizando exactamente la construcción de modelos de los notebooks
`maxim/moeTrainer.ipynb` y `maxim/maximTrainer_no_moe.ipynb`.

Produce:
  * Las filas de SSIM (y PSNR de control) para la Tabla I de la tesis, listas
    para pegar en `tesis.tex` reemplazando los "---".
  * La figura cualitativa `tesis/figs/fig_qualitative.pdf` que usa la tesis,
    sobrescribiendo la plantilla generada por `make_figs.py`.

Uso:
    python tesis/eval_metrics.py
Ajustar antes el bloque CONFIG con las rutas reales.
"""

import os
import io
import sys
import collections
import numpy as np


def log(msg=""):
    """print con flush inmediato: evita que el output se vea 'colgado' en Colab.

    Al correr con `!python script.py`, la salida estándar queda en buffer y los
    mensajes no aparecen hasta que el buffer se llena o el proceso termina (por
    eso un Ctrl-C mostraba de golpe lo que parecía estar congelado). Forzando
    flush el progreso se ve en vivo (equivale a correr `python -u`).
    """
    print(msg, flush=True)

# --------------------------------------------------------------------------- #
# CONFIG
# --------------------------------------------------------------------------- #
PATCH_SIZE = 256
SEED = 42
NUM_EXPERTS = 5

# Progreso / submuestreo.
# EVAL_LIMIT: máximo de imágenes evaluadas POR TAREA (None = todas; con batch=4
# equivale a EVAL_LIMIT/4 batches). Las imágenes se eligen como una submuestra
# ALEATORIA UNIFORME con semilla fija (ver iter_val_pairs), así que el PSNR/SSIM
# promedio sigue siendo un estimador INSESGADO de la métrica sobre el test
# completo: solo baja la precisión (error estándar ~ 1/sqrt(EVAL_LIMIT)), no
# introduce sesgo. 200 da una estimación estable; subir para más precisión,
# bajar para un smoke test. Las tareas con menos de EVAL_LIMIT imágenes (denoise,
# enhance, dehaze) se evalúan completas igual.
EVAL_LIMIT = 200
# Cada cuántas imágenes se imprime una línea de progreso dentro de cada tarea.
PROGRESS_EVERY = 50

# Carpeta raíz con una subcarpeta por tarea (cada una con imgs/, GT/, test.txt).
DATA_ROOT = "/content/gdrive/MyDrive/Facultad/tesis/Datasets/Classifier"
TASKS = ["deblur", "dehaze", "denoise", "derain", "enhance"]

# Checkpoints entrenados. Cada entrada: (etiqueta, tipo, variante, ruta).
#   tipo: "base"  -> MAXIM sin MoE (mono o multi)
#         "moe"   -> variante MAXIM+MoE
CHECKPOINTS = [
    ("MAXIM mono-tarea",  "base", "S-2", "/content/gdrive/MyDrive/Facultad/tesis/ckpts/maxim_no_moe/single_task_enhance/best_checkpoint/checkpoint_29"),
    ("MAXIM multi-tarea", "base", "S-3", "/content/gdrive/MyDrive/Facultad/tesis/ckpts/maxim_no_moe/multi_task_all/best_checkpoint/checkpoint_29"),
    ("MAXIM+MoE",         "moe",  "S-2", "/content/gdrive/MyDrive/Facultad/tesis/ckpts/moe_training/best_checkpoint/checkpoint_41"),
]

# La figura cualitativa contrasta estos modelos (por etiqueta) contra la GT.
# El orden aquí es el orden de las columnas en la figura.
QUALITATIVE_MONO = "MAXIM mono-tarea"
QUALITATIVE_BASELINE = "MAXIM multi-tarea"
QUALITATIVE_MOE = "MAXIM+MoE"

# Figura principal: una fila por tarea, multi-tarea vs MoE (sin el modelo
# mono-tarea, que solo es especialista en enhance y bajaría en las demás filas).
OUT_FIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figs",
                       "fig_qualitative.pdf")

# Figura secundaria: una sola fila (tarea enhance) que sí incluye el modelo
# mono-tarea, comparándolo de forma justa con multi-tarea y MoE en su tarea.
QUALITATIVE_ENHANCE_TASK = "enhance"
OUT_FIG_ENHANCE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figs",
                               "fig_qualitative_enhance.pdf")

_MODEL_CONFIGS = {
    "variant": "", "dropout_rate": 0.1, "num_outputs": 3,
    "use_bias": True, "num_supervision_scales": 3,
}


# --------------------------------------------------------------------------- #
# SSIM  --  implementación estándar con ventana gaussiana 11x11, sigma 1.5.
# Usa skimage si está disponible (recomendado); si no, una versión propia.
# Imágenes en rango [0, 1]; data_range = 1.0.
# --------------------------------------------------------------------------- #
def compute_ssim(pred, target, data_range=1.0):
    """SSIM media sobre un batch NHWC (promedio por canal)."""
    pred = np.asarray(pred, np.float64)
    target = np.asarray(target, np.float64)
    try:
        from skimage.metrics import structural_similarity as sk_ssim
        vals = []
        for i in range(pred.shape[0]):
            vals.append(sk_ssim(target[i], pred[i], data_range=data_range,
                                channel_axis=-1, gaussian_weights=True,
                                sigma=1.5, use_sample_covariance=False))
        return float(np.mean(vals))
    except Exception:
        pass

    # Fallback sin skimage: ventana gaussiana via scipy.
    from scipy.ndimage import gaussian_filter
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2

    def _ssim_one(a, b):  # a, b: HxWxC
        out = []
        for c in range(a.shape[-1]):
            x, y = a[..., c], b[..., c]
            mux = gaussian_filter(x, 1.5)
            muy = gaussian_filter(y, 1.5)
            mux2, muy2, muxy = mux * mux, muy * muy, mux * muy
            sx = gaussian_filter(x * x, 1.5) - mux2
            sy = gaussian_filter(y * y, 1.5) - muy2
            sxy = gaussian_filter(x * y, 1.5) - muxy
            ssim_map = ((2 * muxy + C1) * (2 * sxy + C2)) / (
                (mux2 + muy2 + C1) * (sx + sy + C2))
            out.append(ssim_map.mean())
        return float(np.mean(out))

    return float(np.mean([_ssim_one(target[i], pred[i])
                          for i in range(pred.shape[0])]))


def compute_psnr(pred, target):
    pred = np.asarray(pred, np.float64) * 255.0
    target = np.asarray(target, np.float64) * 255.0
    mse = np.maximum(np.mean((pred - target) ** 2), 1e-6)
    return float(20.0 * np.log10(255.0 / np.sqrt(mse)))


# --------------------------------------------------------------------------- #
# Datos  --  loader de validación determinista (center-crop a PATCH_SIZE).
# Sigue el layout de los notebooks: <task>/imgs, <task>/GT, <task>/test.txt
# --------------------------------------------------------------------------- #
from PIL import Image


def _load_img(path):
    return np.asarray(Image.open(path).convert("RGB"), np.float32) / 255.0


def _center_crop(img, size):
    h, w = img.shape[:2]
    if h < size or w < size:
        ph, pw = max(0, size - h), max(0, size - w)
        img = np.pad(img, ((0, ph), (0, pw), (0, 0)), mode="reflect")
        h, w = img.shape[:2]
    top, left = (h - size) // 2, (w - size) // 2
    return img[top:top + size, left:left + size]


def iter_val_pairs(task, limit=None):
    """Genera (input, target) en [0,1], recortados a PATCH_SIZE, para una tarea.

    Si `limit` se especifica y es menor que el conjunto, se evalúa sobre una
    submuestra ALEATORIA UNIFORME (semilla fija = SEED), no sobre las primeras
    `limit` imágenes. Como las métricas se reportan como promedio sobre el test,
    una submuestra uniforme es un estimador insesgado de esa media; tomar las
    primeras N sesgaría el resultado si test.txt está ordenado por escena/condición.
    """
    base = os.path.join(DATA_ROOT, task)
    list_file = os.path.join(base, "test.txt")
    with open(list_file) as f:
        names = [ln.strip() for ln in f if ln.strip()]
    if limit and limit < len(names):
        rng = np.random.default_rng(SEED)
        idx = np.sort(rng.permutation(len(names))[:limit])
        names = [names[i] for i in idx]
    for name in names:
        ip = os.path.join(base, "imgs", name)
        tp = os.path.join(base, "GT", name)
        if not (os.path.exists(ip) and os.path.exists(tp)):
            continue
        x = _center_crop(_load_img(ip), PATCH_SIZE)
        y = _center_crop(_load_img(tp), PATCH_SIZE)
        if x.shape == y.shape:
            yield x, y


def count_val_pairs(task, limit=None):
    """Cantidad de nombres listados en test.txt (cota superior de imágenes)."""
    list_file = os.path.join(DATA_ROOT, task, "test.txt")
    try:
        with open(list_file) as f:
            n = sum(1 for ln in f if ln.strip())
    except OSError:
        return 0
    return min(n, limit) if limit else n


# --------------------------------------------------------------------------- #
# Modelos  --  construcción y restauración (espejo de los notebooks).
# --------------------------------------------------------------------------- #
def _build_state(kind, variant):
    import importlib
    import jax, jax.numpy as jnp
    from jax import random
    import ml_collections
    import optax
    from flax.training import train_state

    class TrainState(train_state.TrainState):
        batch_stats: any = None

    cfg = ml_collections.ConfigDict(_MODEL_CONFIGS)
    cfg.variant = variant
    maxim_mod = importlib.import_module("maxim.models.maxim")
    maxim_model = maxim_mod.Model(**cfg)

    rng = random.PRNGKey(SEED)
    dummy = jnp.ones([1, PATCH_SIZE, PATCH_SIZE, 3])

    if kind == "base":
        model = maxim_model
        variables = model.init(rng, dummy, train=True)
    else:  # moe
        router_mod = importlib.import_module("maxim.models.router")
        moe_mod = importlib.import_module("maxim.models.moe")
        router = router_mod.RouterModel()
        experts = [moe_mod.ExpertHead() for _ in range(NUM_EXPERTS)]
        model = moe_mod.MaximMoE(maxim_model, router, experts)
        variables = model.init(rng, dummy, train=True)

    tx = optax.adamw(1e-4)  # irrelevante en evaluación, solo para armar el state
    state = TrainState.create(apply_fn=model.apply, params=variables["params"],
                              tx=tx, batch_stats=variables.get("batch_stats"))
    return model, state


def _restore(state, ckpt_dir):
    from flax.training import checkpoints
    return checkpoints.restore_checkpoint(ckpt_dir=ckpt_dir, target=state)


def _predict(kind, model, state, x_nhwc):
    """Devuelve la imagen reconstruida en [0,1] (numpy NHWC)."""
    import jax.numpy as jnp
    variables = {"params": state.params}
    if state.batch_stats is not None:
        variables["batch_stats"] = state.batch_stats
    if kind == "base":
        preds = model.apply(variables, jnp.asarray(x_nhwc), train=False)
        out = preds[-1][-1] if isinstance(preds, list) else preds
    else:  # moe -> (pred, gates)
        out, _ = model.apply(variables, jnp.asarray(x_nhwc), train=False)
    return np.clip(np.asarray(out), 0.0, 1.0)


# --------------------------------------------------------------------------- #
# Evaluación
# --------------------------------------------------------------------------- #
def evaluate_checkpoint(label, kind, variant, ckpt_dir, batch=4):
    log("  -> construyendo modelo...")
    model, state = _build_state(kind, variant)
    log(f"  -> restaurando checkpoint: {ckpt_dir}")
    state = _restore(state, ckpt_dir)
    log("  -> modelo listo. El primer batch compila (JIT) y puede tardar ~1 min.")

    per_task = {}
    all_psnr, all_ssim = [], []
    for task in TASKS:
        total = count_val_pairs(task, EVAL_LIMIT)
        log(f"  [{task:8s}] evaluando hasta {total} imágenes...")
        xs, ys = [], []
        ps, ss = [], []
        done, last_report = 0, 0
        for x, y in iter_val_pairs(task, limit=EVAL_LIMIT):
            xs.append(x); ys.append(y)
            if len(xs) == batch:
                pred = _predict(kind, model, state, np.stack(xs))
                tgt = np.stack(ys)
                ps.append(compute_psnr(pred, tgt))
                ss.append(compute_ssim(pred, tgt))
                done += len(xs)
                xs, ys = [], []
                if done - last_report >= PROGRESS_EVERY:
                    last_report = done
                    log(f"    {task:8s} {done}/{total}  "
                        f"PSNR~{np.mean(ps):5.2f}  SSIM~{np.mean(ss):.3f}")
        if xs:
            pred = _predict(kind, model, state, np.stack(xs))
            tgt = np.stack(ys)
            ps.append(compute_psnr(pred, tgt))
            ss.append(compute_ssim(pred, tgt))
            done += len(xs)
        if ps:
            per_task[task] = (np.mean(ps), np.mean(ss))
            all_psnr.append(np.mean(ps)); all_ssim.append(np.mean(ss))
            log(f"  {task:8s}  PSNR={np.mean(ps):6.2f}  "
                f"SSIM={np.mean(ss):.4f}  (n={done})")
        else:
            log(f"  {task:8s}  [sin imágenes válidas]")

    psnr, ssim = float(np.mean(all_psnr)), float(np.mean(all_ssim))
    log(f"  {'GLOBAL':8s}  PSNR={psnr:6.2f}  SSIM={ssim:.4f}")
    return psnr, ssim, (model, state, kind)


def _render_qualitative(models, model_labels, tasks, out_path):
    """Renderiza una grilla cualitativa y la guarda en out_path.

    Una fila por tarea en `tasks`; columnas = Entrada + un modelo por cada
    etiqueta de `model_labels` presente en `models` + Referencia. Devuelve True
    si escribió la figura.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    name = os.path.basename(out_path)
    model_cols = [lbl for lbl in model_labels if lbl in models]
    if not model_cols:
        log(f"[aviso] sin modelos disponibles para {name}; se omite.")
        return False
    col_titles = ["Entrada (degradada)"] + model_cols + ["Referencia"]
    ncols = len(col_titles)

    rows = []
    log(f"\nGenerando {name} ({len(tasks)} fila(s); columnas: "
        f"{', '.join(model_cols)})...")
    for task in tasks:
        for x, y in iter_val_pairs(task, limit=1):
            preds = [_predict(models[lbl][2], models[lbl][0], models[lbl][1], x[None])[0]
                     for lbl in model_cols]
            rows.append((task, x, preds, y))
            log(f"  fila {task} lista.")
            break
    if not rows:
        log(f"[aviso] sin imágenes para {name}; se omite.")
        return False

    fig, axes = plt.subplots(len(rows), ncols,
                             figsize=(1.75 * ncols, 1.8 * len(rows)))
    if len(rows) == 1:
        axes = axes[None, :]
    for r, (task, x, preds, y) in enumerate(rows):
        for c, img in enumerate([x] + preds + [y]):
            ax = axes[r, c]
            ax.imshow(np.clip(img, 0, 1))
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(col_titles[c], fontsize=8)
            if c == 0:
                ax.set_ylabel(task, fontsize=8)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    # quitar la marca de plantilla (si existe) para que make_figs.py no
    # sobrescriba la figura real en la próxima corrida.
    stem = os.path.splitext(os.path.basename(out_path))[0]
    flag = os.path.join(os.path.dirname(out_path), f".{stem}_is_placeholder")
    if os.path.exists(flag):
        os.remove(flag)
    log(f"Figura escrita en {out_path}")
    return True


def make_qualitative(models):
    """Figura principal de la tesis (fig_qualitative.pdf): una fila por tarea,
    contrasta multi-tarea vs MoE contra la GT. No incluye el modelo mono-tarea
    porque solo es especialista en enhance y bajaría en las otras filas."""
    _render_qualitative(models,
                        [QUALITATIVE_BASELINE, QUALITATIVE_MOE],
                        TASKS, OUT_FIG)


def make_qualitative_enhance(models):
    """Figura secundaria (fig_qualitative_enhance.pdf): una sola fila para la
    tarea enhance, comparando mono-tarea vs multi-tarea vs MoE contra la GT.
    Aquí el modelo mono-tarea (single_task_enhance) sí es comparable, porque es
    su tarea de entrenamiento."""
    _render_qualitative(models,
                        [QUALITATIVE_MONO, QUALITATIVE_BASELINE, QUALITATIVE_MOE],
                        [QUALITATIVE_ENHANCE_TASK], OUT_FIG_ENHANCE)


def main():
    log(f"Evaluando {len(CHECKPOINTS)} checkpoints sobre {len(TASKS)} tareas.")
    log(f"DATA_ROOT = {DATA_ROOT}")
    if EVAL_LIMIT:
        log(f"** Submuestra de {EVAL_LIMIT} imágenes/tarea (aleatoria uniforme, "
            f"seed={SEED}); el promedio es estimación insesgada del test completo **")

    results, models = {}, {}
    for i, (label, kind, variant, ckpt) in enumerate(CHECKPOINTS, 1):
        log("\n" + "=" * 60)
        log(f"[{i}/{len(CHECKPOINTS)}] {label}  ({kind}, variante {variant})")
        log("=" * 60)
        try:
            psnr, ssim, handle = evaluate_checkpoint(label, kind, variant, ckpt)
            results[label] = (psnr, ssim)
            models[label] = handle
        except Exception as e:
            log(f"  [ERROR] {label}: {e}")

    log("\n--- Filas LaTeX (PSNR / SSIM) para la Tabla I ---")
    for label, (psnr, ssim) in results.items():
        p = f"{psnr:.1f}".replace(".", "{,}")
        s = f"{ssim:.3f}".replace(".", "{,}")
        log(f"  {label}: PSNR ${p}$   SSIM ${s}$")

    if any(lbl in models for lbl in
           (QUALITATIVE_MONO, QUALITATIVE_BASELINE, QUALITATIVE_MOE)):
        make_qualitative(models)
        make_qualitative_enhance(models)
    else:
        log("\n[aviso] faltan checkpoints para la figura cualitativa.")


if __name__ == "__main__":
    main()
