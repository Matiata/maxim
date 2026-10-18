"""
Flax/JAX Router for MoE (lightweight)

- Backbone: small depthwise-separable convolutional network inspired by MobileNet
  (keeps it simple and dependency-free)
- Head: small MLP classifier
- Training loop: Optax Adam, cross-entropy
- Checkpointing: flax.training.checkpoints
- Inference: route_image(...) returns (task_name, confidence)

Edit dataset loading and augmentation to match your pipelines.
"""

from typing import Any, Callable, Sequence, Tuple, Dict
import time
import jax
import jax.numpy as jnp
from jax import random
import optax
import os
from flax import linen as nn
from flax.training import train_state, checkpoints
import numpy as np
from PIL import Image
import tensorflow as tf

# -----------------------------
# Configuration / Hyperparams
# -----------------------------
NUM_CLASSES = 4
TASK_NAMES = ["denoise", "deblur", "derain", "dehaze", "enhance"]
TASK_DICT = {
    "denoise": 0,
    "deblur": 1,
    "derain": 2,
    "dehaze": 3,
    "enhance": 4,
}
PATCH_SIZE = 225
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
NUM_EPOCHS = 20
SEED = 42
CKPT_DIR = "./router_checkpoints"

# -----------------------------
# Utilities
# -----------------------------


def preprocess_pil(img: Image.Image, image_size: int = PATCH_SIZE) -> np.ndarray:
    """Convert PIL image to float32 numpy array shaped (H,W,3) normalized to [0,1]."""
    img = img.convert("RGB")
    img = img.resize((image_size, image_size), resample=Image.BILINEAR)
    arr = np.array(img).astype(np.float32) / 255.0
    # HWC format is standard for Flax/JAX
    return arr


def load_image(filepath):
    """Load and preprocess image."""
    img = Image.open(filepath).convert("RGB")
    img = np.asarray(img, np.float32) / 255.0
    return img


def make_shape_even(image):
    """Pad the image to have even shapes."""
    height, width = image.shape[0], image.shape[1]
    padh = 1 if height % 2 != 0 else 0
    padw = 1 if width % 2 != 0 else 0
    image = jnp.pad(image, [(0, padh), (0, padw), (0, 0)], mode="reflect")
    return image


def mod_padding_symmetric(image, factor=64):
    """Padding the image to be divided by factor."""
    height, width = image.shape[0], image.shape[1]
    height_pad, width_pad = (
        ((height + factor) // factor) * factor,
        ((width + factor) // factor) * factor,
    )
    padh = height_pad - height if height % factor != 0 else 0
    padw = width_pad - width if width % factor != 0 else 0
    image = jnp.pad(
        image, [(padh // 2, padh // 2), (padw // 2, padw // 2), (0, 0)], mode="reflect"
    )
    return image


def random_crop(image, crop_size):
    """Random crop for data augmentation."""
    h, w = image.shape[0], image.shape[1]

    if h > crop_size and w > crop_size:
        top = np.random.randint(0, h - crop_size)
        left = np.random.randint(0, w - crop_size)

        image = image[top : top + crop_size, left : left + crop_size, :]

    return image


def random_flip(image):
    """Random horizontal and vertical flip."""
    if np.random.rand() > 0.5:
        image = np.fliplr(image)

    if np.random.rand() > 0.5:
        image = np.flipud(image)

    return image


def random_rotation(image):
    """Random 90-degree rotation."""
    k = np.random.randint(0, 4)
    image = np.rot90(image, k=k)
    return image


# -----------------------------
# Model: small DW-Conv backbone
# -----------------------------


class DepthwiseConv(nn.Module):
    kernel_size: Tuple[int, int]
    strides: Tuple[int, int] = (1, 1)
    padding: str = "SAME"

    @nn.compact
    def __call__(self, x):
        in_ch = x.shape[-1]
        # Flax conv expects (N, H, W, C) by default
        # We'll use `feature_group_count=in_ch` for depthwise
        x = nn.Conv(
            features=in_ch,
            kernel_size=self.kernel_size,
            strides=self.strides,
            padding=self.padding,
            feature_group_count=in_ch,
            use_bias=False,
        )(x)
        return x


class DwSepBlock(nn.Module):
    out_ch: int
    stride: int = 1

    @nn.compact
    def __call__(self, x):
        # Depthwise
        x = DepthwiseConv(kernel_size=(3, 3), strides=(self.stride, self.stride))(x)
        x = nn.BatchNorm(use_running_average=False)(x)
        x = nn.relu(x)
        # Pointwise
        x = nn.Conv(features=self.out_ch, kernel_size=(1, 1), use_bias=False)(x)
        x = nn.BatchNorm(use_running_average=False)(x)
        x = nn.relu(x)
        return x


class SmallBackbone(nn.Module):
    """Small lightweight backbone producing a global-pooled embedding.

    Input shape: (N, H, W, C) with C=3
    Output: (N, embedding_dim)
    """

    embedding_dim: int = 576

    @nn.compact
    def __call__(self, x, train: bool = True):
        # x: NCHW
        assert x.ndim == 4
        # initial conv
        x = nn.Conv(
            features=32,
            kernel_size=(3, 3),
            strides=(2, 2),
            padding="SAME",
            use_bias=False,
        )(x)
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = nn.relu(x)

        # a few depthwise separable blocks
        x = DwSepBlock(out_ch=64, stride=1)(x)
        x = DwSepBlock(out_ch=96, stride=2)(x)
        x = DwSepBlock(out_ch=160, stride=2)(x)
        x = DwSepBlock(out_ch=self.embedding_dim, stride=2)(x)

        # global avg pool
        # x shape: (N, H, W, C)
        x = x.mean(axis=(1, 2))  # (N, C)
        return x


class RouterHead(nn.Module):
    num_classes: int = NUM_CLASSES

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        x = nn.Dense(128)(x)
        x = nn.relu(x)
        x = nn.Dense(self.num_classes)(x)
        return x


class RouterModel(nn.Module):
    embedding_dim: int = 576
    num_classes: int = NUM_CLASSES

    def setup(self):
        self.backbone = SmallBackbone(embedding_dim=self.embedding_dim)
        self.head = RouterHead(num_classes=self.num_classes)

    def __call__(self, x, train: bool = True):
        feats = self.backbone(x, train=train)
        logits = self.head(feats)
        return logits


# -----------------------------
# Train state
# -----------------------------


class TrainState(train_state.TrainState):
    batch_stats: Any = None  # for BatchNorm


# -----------------------------
# Loss / metrics
# -----------------------------


def cross_entropy_loss(logits, labels):
    onehot = jax.nn.one_hot(labels, logits.shape[-1])
    loss = optax.softmax_cross_entropy(logits=logits, labels=onehot)
    return loss.mean()


@jax.jit
def compute_metrics(logits, labels):
    loss = cross_entropy_loss(logits, labels)
    acc = jnp.mean(jnp.argmax(logits, -1) == labels)
    return {"loss": loss, "accuracy": acc}


# -----------------------------
# Training / Eval step
# -----------------------------


@jax.jit
def train_step(state: TrainState, batch: Tuple[jnp.ndarray, jnp.ndarray]):
    imgs, labels = batch  # imgs: NCHW float32, labels: (N,)

    def loss_fn(params):
        logits, new_model_state = state.apply_fn(
            {"params": params, "batch_stats": state.batch_stats},
            imgs,
            train=True,
            mutable=["batch_stats"],
        )
        loss = cross_entropy_loss(logits, labels)
        return loss, (logits, new_model_state)

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, (logits, new_model_state)), grads = grad_fn(state.params)
    state = state.apply_gradients(
        grads=grads, batch_stats=new_model_state["batch_stats"]
    )
    metrics = compute_metrics(logits, labels)
    return state, metrics


@jax.jit
def eval_step(state: TrainState, batch: Tuple[jnp.ndarray, jnp.ndarray]):
    imgs, labels = batch
    variables = {"params": state.params, "batch_stats": state.batch_stats}
    logits = state.apply_fn(variables, imgs, train=False, mutable=False)
    metrics = compute_metrics(logits, labels)
    return metrics


# -----------------------------
# Init / Trainer
# -----------------------------


def create_train_state(rng, learning_rate=LEARNING_RATE):
    model = RouterModel()
    input_shape = (1, PATCH_SIZE, PATCH_SIZE, 3)
    variables = model.init(rng, jnp.ones(input_shape, jnp.float32), train=True)
    params = variables["params"]
    batch_stats = variables.get("batch_stats")

    tx = optax.adamw(learning_rate=learning_rate, weight_decay=WEIGHT_DECAY)
    state = TrainState.create(
        apply_fn=model.apply, params=params, tx=tx, batch_stats=batch_stats
    )
    return state


# -----------------------------
# Simple dataset helpers (placeholders)
# -----------------------------


def make_fake_dataset(num_per_class=1000, rng_seed=0):
    rng = np.random.RandomState(rng_seed)
    images = []
    labels = []
    for label in range(NUM_CLASSES):
        for i in range(num_per_class):
            # generate random image noise as placeholder
            img = rng.rand(PATCH_SIZE, PATCH_SIZE, 3).astype(np.float32)
            images.append(img)
            labels.append(label)
    images = np.stack(images)
    labels = np.array(labels, dtype=np.int32)
    return images, labels


def read_lines_from_file(basepath, task):
    filepath = os.path.join(basepath, task)
    with open(filepath, "r", encoding="utf-8") as f:
        paths = [line.strip() for line in f if line.strip()]

    imgs_dir = os.path.join(basepath, "imgs")
    for path in paths:
        p = os.path.join(imgs_dir, path)
        if os.path.exists(p):
            yield p


def create_dataset(data_dir, batch_size, is_training=True):
    """Create TensorFlow dataset for training/validation."""
    print(
        f"Creating {'training' if is_training else 'validation'} dataset from {data_dir}"
    )
    images = []
    labels = []

    for task in TASK_NAMES:
        for img_path in read_lines_from_file(data_dir, task):
            images.append(img_path)
            labels.append(task)

    def load_and_preprocess(input_path, label):
        """Load and preprocess a single pair of images."""
        input_img = preprocess_pil(Image.open(input_path.numpy().decode()))
        orig_h, orig_w = input_img.shape[:2]

        # Padding images to have even shapes
        input_img = make_shape_even(input_img)
        even_h, even_w = input_img.shape[:2]

        # Padding images to be multiples of 64
        input_img = mod_padding_symmetric(input_img, factor=64)
        pad_h, pad_w = input_img.shape[:2]

        if is_training:
            # Data augmentation
            input_img = random_crop(input_img, PATCH_SIZE)
            input_img = random_flip(input_img)
            input_img = random_rotation(input_img)

        return (
            input_img.astype(np.float32),
            tf.cast(label, tf.int32),
            np.array([orig_h, orig_w, even_h, even_w, pad_h, pad_w], np.int32),
        )

    dataset = tf.data.Dataset.from_tensor_slices((images, labels))

    if is_training:
        dataset = dataset.shuffle(buffer_size=1000)

    dataset = dataset.map(
        lambda x, y: tf.py_function(
            func=load_and_preprocess,
            inp=[x, y],
            Tout=[tf.float32, tf.int32, tf.int32],
        ),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    dataset = dataset.batch(batch_size, drop_remainder=is_training)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset, len(images)


def data_loader(
    images: np.ndarray, labels: np.ndarray, batch_size=BATCH_SIZE, shuffle=True
):
    n = len(images)
    idxs = np.arange(n)
    if shuffle:
        np.random.shuffle(idxs)
    for i in range(0, n, batch_size):
        batch_idx = idxs[i : i + batch_size]
        yield images[batch_idx], labels[batch_idx]


# -----------------------------
# Saving / Loading
# -----------------------------


def save_checkpoint(state: TrainState, step: int):
    save_dict = {
        "params": state.params,
        "batch_stats": state.batch_stats,
        "opt_state": state.opt_state,
    }
    checkpoints.save_checkpoint(
        ckpt_dir=CKPT_DIR, target=save_dict, step=step, overwrite=True
    )


def load_checkpoint(state: TrainState):
    ckpt = checkpoints.restore_checkpoint(ckpt_dir=CKPT_DIR, target=None)
    if ckpt:
        state = state.replace(
            params=ckpt["params"], batch_stats=ckpt.get("batch_stats")
        )
    return state


# -----------------------------
# Inference helper
# -----------------------------


def route_image_pil(
    state: TrainState, pil_img: Image.Image, rng=None
) -> Tuple[str, float]:
    arr = preprocess_pil(pil_img)  # HWC float32
    arr = jnp.array(arr)
    arr = arr[None, ...]  # 1HWC
    variables = {"params": state.params, "batch_stats": state.batch_stats}
    logits = state.apply_fn(variables, arr, train=False, mutable=False)
    probs = jax.nn.softmax(logits, axis=-1)
    probs = np.array(probs[0])
    idx = int(np.argmax(probs))
    return TASK_NAMES[idx], float(probs[idx])


# -----------------------------
# Full training loop (example)
# -----------------------------


def train_example():
    rng = random.PRNGKey(SEED)
    state = create_train_state(rng)

    # For demo: fake dataset. Replace with your actual loader that returns NCHW float32 arrays.
    images, labels = make_fake_dataset(num_per_class=500)

    num_steps = 0
    for epoch in range(NUM_EPOCHS):
        t0 = time.time()
        # training
        for batch_imgs, batch_labels in data_loader(
            images, labels, batch_size=BATCH_SIZE, shuffle=True
        ):
            # to jax arrays
            batch_imgs = jnp.array(batch_imgs)
            batch_labels = jnp.array(batch_labels)
            state, metrics = train_step(state, (batch_imgs, batch_labels))
            num_steps += 1

        # evaluation - single pass
        eval_metrics = {"loss": 0.0, "accuracy": 0.0}
        n_batches = 0
        for batch_imgs, batch_labels in data_loader(
            images, labels, batch_size=BATCH_SIZE, shuffle=False
        ):
            batch_imgs = jnp.array(batch_imgs)
            batch_labels = jnp.array(batch_labels)
            m = eval_step(state, (batch_imgs, batch_labels))
            eval_metrics["loss"] += float(m["loss"])
            eval_metrics["accuracy"] += float(m["accuracy"])
            n_batches += 1
        eval_metrics = {k: v / n_batches for k, v in eval_metrics.items()}

        t1 = time.time()
        print(
            f"Epoch {epoch + 1}/{NUM_EPOCHS} - eval_loss: {eval_metrics['loss']:.4f} - eval_acc: {eval_metrics['accuracy']:.4f} - time: {t1 - t0:.1f}s"
        )

        # save checkpoint per epoch
        save_checkpoint(state, step=epoch + 1)

    return state


# -----------------------------
# If run as script, demo training
# -----------------------------
if __name__ == "__main__":
    # Quick demo run (with fake data)
    final_state = train_example()
    # Example inference
    from PIL import Image

    dummy = Image.new("RGB", (PATCH_SIZE, PATCH_SIZE), color=(128, 128, 128))
    task, conf = route_image_pil(final_state, dummy)
    print("Routed to", task, "conf", conf)

# ==============================
# MoE Pipeline (JAX/Flax)
# ==============================
# Este pipeline asume que:
#   - Tenés varios expertos MAXIM ya entrenados.
#   - Usás el router definido arriba.
#   - Querés hacer hot-swap CPU⇄GPU para evitar sobrecargar VRAM.
#   - Cada experto está guardado como un checkpoint Flax.


# -------------------------------------------------------------
# 1) Loader genérico para expertos MAXIM
# -------------------------------------------------------------
def load_expert(model_def, ckpt_path: str, device: str = "cpu"):
    """
    Carga un experto MAXIM desde un checkpoint.
    delivery:
        model_def: instancia del MAXIM (ej: MAXIM(config))
        ckpt_path: ruta al checkpoint
        device: "cpu" o "cuda"/"gpu"
    """
    params = checkpoints.restore_checkpoint(ckpt_path, target=None)
    if device == "cpu":
        params = jax.device_put(params, jax.devices("cpu")[0])
    else:
        params = jax.device_put(params, jax.devices("gpu")[0])
    return params


# -------------------------------------------------------------
# 2) Diccionario de expertos (lazy loading)
# -------------------------------------------------------------
class ExpertPool:
    def __init__(self, expert_defs: Dict[str, object], ckpts: Dict[str, str]):
        self.expert_defs = expert_defs
        self.ckpts = ckpts
        self.cache = {k: None for k in expert_defs.keys()}  # hot-swap

    def load(self, name: str, device="gpu"):
        if self.cache[name] is None:
            model_def = self.expert_defs[name]
            ckpt_path = self.ckpts[name]
            self.cache[name] = load_expert(model_def, ckpt_path, device=device)
        return self.cache[name]

    def unload(self, name: str):
        self.cache[name] = None  # libera VRAM
        if hasattr(jax, "clear_caches"):
            jax.clear_caches()
        elif hasattr(jax, "clear_backends"):
            jax.clear_backends()


# -------------------------------------------------------------
# 3) Pipeline completo MoE
# -------------------------------------------------------------
@jax.jit
def run_expert(model_def, params, image_tensor):
    return model_def.apply(params, image_tensor, train=False)


def moe_pipeline(router, router_params, expert_pool: ExpertPool, image_np):
    """
    Pipeline completo MoE:
      - Preprocesa imagen
      - El router predice la tarea
      - Se carga el experto correspondiente en GPU
      - Se ejecuta y se devuelve la imagen procesada
    """
    # --- Preprocesar imagen para el router
    img = (image_np / 255.0).astype(jnp.float32)
    img = jnp.expand_dims(img, axis=0)  # [1, H, W, 3]

    # --- Predicción de tarea
    # Necesitamos pasar batch_stats si el modelo usa BatchNorm
    variables = {"params": router_params}
    # Si tienes batch_stats guardados en el router, deberías pasarlos aquí también.
    # Asumimos que router_params contiene solo los parámetros entrenables.
    # Si router_params es un TrainState, úsalo directamente.

    # Nota: router.apply devuelve logits. Necesitamos argmax.
    logits = router.apply(variables, img, train=False, mutable=False)
    task_idx = int(jnp.argmax(logits, axis=-1)[0])

    task_names = list(expert_pool.expert_defs.keys())
    chosen_task = task_names[task_idx]

    # --- Cargar experto a GPU
    expert_params = expert_pool.load(chosen_task, device="gpu")
    expert_def = expert_pool.expert_defs[chosen_task]

    # --- Ejecutar experto
    output = run_expert(expert_def, expert_params, img)

    # --- Convertir a uint8
    out_np = (jnp.clip(output[0], 0, 1) * 255).astype(jnp.uint8)
    return out_np, chosen_task


# -------------------------------------------------------------
# 4) API de inferencia tipo "end-to-end"
# -------------------------------------------------------------


def process_image_moe(router, router_params, expert_pool, pil_img: Image.Image):
    arr = np.array(pil_img)
    out_arr, chosen = moe_pipeline(router, router_params, expert_pool, arr)
    return Image.fromarray(np.array(out_arr)), chosen
