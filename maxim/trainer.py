# Copyright 2022 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Training script for MAXIM model."""

import functools
import importlib
import os
from typing import Any, Dict

from absl import app
from absl import flags
from absl import logging
import flax
from flax.training import train_state, checkpoints
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
import optax
from PIL import Image
import tensorflow as tf

FLAGS = flags.FLAGS

flags.DEFINE_enum(
    "task",
    "Denoising",
    ["Denoising", "Deblurring", "Deraining", "Dehazing", "Enhancement"],
    "Task to train.",
)
flags.DEFINE_string("train_dir", "", "Training data directory.")
flags.DEFINE_string("val_dir", "", "Validation data directory.")
flags.DEFINE_string("output_dir", "./checkpoints", "Output directory for checkpoints.")
flags.DEFINE_integer("batch_size", 8, "Batch size for training.")
flags.DEFINE_integer("num_epochs", 300, "Number of training epochs.")
flags.DEFINE_float("learning_rate", 2e-4, "Initial learning rate.")
flags.DEFINE_float("warmup_epochs", 10, "Number of warmup epochs.")
flags.DEFINE_integer("log_every", 100, "Log every n steps.")
flags.DEFINE_integer("save_every", 1000, "Save checkpoint every n steps.")
flags.DEFINE_integer("patch_size", 256, "Training patch size.")
flags.DEFINE_string("resume_from", "", "Resume training from checkpoint.")
flags.DEFINE_float("weight_decay", 1e-4, "Weight decay for optimizer.")

_MODEL_FILENAME = "maxim"

_MODEL_VARIANT_DICT = {
    "Denoising": "S-3",
    "Deblurring": "S-3",
    "Deraining": "S-2",
    "Dehazing": "S-2",
    "Enhancement": "S-2",
}

_MODEL_CONFIGS = {
    "variant": "",
    "dropout_rate": 0.1,
    "num_outputs": 3,
    "use_bias": True,
    "num_supervision_scales": 3,
}


class TrainState(train_state.TrainState):
    """Extended train state with batch statistics."""

    batch_stats: Any = None


def create_learning_rate_schedule(base_lr, warmup_epochs, total_steps, steps_per_epoch):
    """Create learning rate schedule with warmup and cosine decay."""
    warmup_steps = warmup_epochs * steps_per_epoch

    warmup_fn = optax.linear_schedule(
        init_value=0.0, end_value=base_lr, transition_steps=warmup_steps
    )

    cosine_fn = optax.cosine_decay_schedule(
        init_value=base_lr, decay_steps=total_steps - warmup_steps, alpha=1e-6
    )

    schedule_fn = optax.join_schedules(
        schedules=[warmup_fn, cosine_fn], boundaries=[warmup_steps]
    )

    return schedule_fn


def create_train_state(rng, model, learning_rate_fn, weight_decay):
    """Create initial training state."""
    # Initialize with dummy input
    dummy_input = jnp.ones([1, 256, 256, 3])
    variables = model.init(rng, dummy_input, train=True)

    params = variables["params"]
    batch_stats = variables.get("batch_stats", None)

    # Create optimizer
    tx = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate=learning_rate_fn, weight_decay=weight_decay),
    )

    return TrainState.create(
        apply_fn=model.apply, params=params, tx=tx, batch_stats=batch_stats
    )


def load_image(filepath):
    """Load and preprocess image."""
    img = Image.open(filepath).convert("RGB")
    img = np.asarray(img, np.float32) / 255.0
    return img


def random_crop(image, target, crop_size):
    """Random crop for data augmentation."""
    h, w = image.shape[0], image.shape[1]

    if h > crop_size and w > crop_size:
        top = np.random.randint(0, h - crop_size)
        left = np.random.randint(0, w - crop_size)

        image = image[top : top + crop_size, left : left + crop_size]
        target = target[top : top + crop_size, left : left + crop_size]

    return image, target


def random_flip(image, target):
    """Random horizontal and vertical flip."""
    if np.random.rand() > 0.5:
        image = np.fliplr(image)
        target = np.fliplr(target)

    if np.random.rand() > 0.5:
        image = np.flipud(image)
        target = np.flipud(target)

    return image, target


def random_rotation(image, target):
    """Random 90-degree rotation."""
    k = np.random.randint(0, 4)
    image = np.rot90(image, k=k)
    target = np.rot90(target, k=k)
    return image, target


def create_dataset(data_dir, batch_size, patch_size, is_training=True):
    """Create TensorFlow dataset for training/validation."""

    input_dir = os.path.join(data_dir, "input")
    target_dir = os.path.join(data_dir, "target")

    input_files = sorted(tf.io.gfile.glob(os.path.join(input_dir, "*")))
    target_files = sorted(tf.io.gfile.glob(os.path.join(target_dir, "*")))

    def load_and_preprocess(input_path, target_path):
        """Load and preprocess a single pair of images."""
        input_img = load_image(input_path.numpy().decode())
        target_img = load_image(target_path.numpy().decode())

        if is_training:
            # Data augmentation
            input_img, target_img = random_crop(input_img, target_img, patch_size)
            input_img, target_img = random_flip(input_img, target_img)
            input_img, target_img = random_rotation(input_img, target_img)

        return input_img.astype(np.float32), target_img.astype(np.float32)

    dataset = tf.data.Dataset.from_tensor_slices((input_files, target_files))

    if is_training:
        dataset = dataset.shuffle(buffer_size=1000)

    dataset = dataset.map(
        lambda x, y: tf.py_function(
            load_and_preprocess, [x, y], [tf.float32, tf.float32]
        ),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    dataset = dataset.batch(batch_size, drop_remainder=is_training)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


def compute_loss(preds, targets, num_scales=3):
    """Compute multi-scale L1 loss."""
    total_loss = 0.0

    if isinstance(preds, list):
        # Multi-stage outputs
        for stage_preds in preds:
            if isinstance(stage_preds, list):
                # Multi-scale outputs within a stage
                for scale_idx, pred in enumerate(stage_preds):
                    weight = 0.5 ** (num_scales - scale_idx - 1)
                    loss = jnp.mean(jnp.abs(pred - targets))
                    total_loss += weight * loss
            else:
                loss = jnp.mean(jnp.abs(stage_preds - targets))
                total_loss += loss
    else:
        total_loss = jnp.mean(jnp.abs(preds - targets))

    return total_loss


def compute_psnr(pred, target):
    """Compute PSNR metric."""
    mse = jnp.mean((pred - target) ** 2)
    psnr = 20.0 * jnp.log10(1.0 / jnp.sqrt(mse + 1e-10))
    return psnr


@functools.partial(jax.jit, static_argnums=(3,))
def train_step(state, batch_input, batch_target, num_scales):
    """Single training step."""

    def loss_fn(params):
        if state.batch_stats is not None:
            preds, updates = state.apply_fn(
                {"params": params, "batch_stats": state.batch_stats},
                batch_input,
                train=True,
                mutable=["batch_stats"],
            )
            new_batch_stats = updates["batch_stats"]
        else:
            preds = state.apply_fn({"params": params}, batch_input, train=True)
            new_batch_stats = None

        loss = compute_loss(preds, batch_target, num_scales)

        # Get final prediction for metrics
        final_pred = preds[-1][-1] if isinstance(preds, list) else preds
        psnr = compute_psnr(final_pred, batch_target)

        return loss, (psnr, new_batch_stats)

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, (psnr, new_batch_stats)), grads = grad_fn(state.params)

    state = state.apply_gradients(grads=grads)

    if new_batch_stats is not None:
        state = state.replace(batch_stats=new_batch_stats)

    metrics = {
        "loss": loss,
        "psnr": psnr,
    }

    return state, metrics


@functools.partial(jax.jit, static_argnums=(2,))
def eval_step(state, batch_input, batch_target):
    """Single evaluation step."""
    if state.batch_stats is not None:
        preds = state.apply_fn(
            {"params": state.params, "batch_stats": state.batch_stats},
            batch_input,
            train=False,
        )
    else:
        preds = state.apply_fn({"params": state.params}, batch_input, train=False)

    # Get final prediction
    final_pred = preds[-1][-1] if isinstance(preds, list) else preds

    loss = jnp.mean(jnp.abs(final_pred - batch_target))
    psnr = compute_psnr(final_pred, batch_target)

    return {"loss": loss, "psnr": psnr}


def train_epoch(state, train_dataset, num_scales, epoch):
    """Train for one epoch."""
    batch_metrics = []

    for step, (batch_input, batch_target) in enumerate(train_dataset):
        batch_input = jnp.array(batch_input)
        batch_target = jnp.array(batch_target)

        state, metrics = train_step(state, batch_input, batch_target, num_scales)
        batch_metrics.append(metrics)

        if (step + 1) % FLAGS.log_every == 0:
            metrics_np = jax.device_get(metrics)
            logging.info(
                f"Epoch {epoch}, Step {step + 1}: "
                f'loss = {metrics_np["loss"]:.4f}, '
                f'psnr = {metrics_np["psnr"]:.2f} dB'
            )

    # Compute epoch metrics
    epoch_metrics = {
        k: np.mean([m[k] for m in batch_metrics]) for k in batch_metrics[0].keys()
    }

    return state, epoch_metrics


def evaluate(state, val_dataset):
    """Evaluate on validation set."""
    batch_metrics = []

    for batch_input, batch_target in val_dataset:
        batch_input = jnp.array(batch_input)
        batch_target = jnp.array(batch_target)

        metrics = eval_step(state, batch_input, batch_target)
        batch_metrics.append(metrics)

    # Compute average metrics
    avg_metrics = {
        k: np.mean([jax.device_get(m[k]) for m in batch_metrics])
        for k in batch_metrics[0].keys()
    }

    return avg_metrics


def main(_):
    # Set random seed
    rng = jax.random.PRNGKey(42)

    # Create output directory
    os.makedirs(FLAGS.output_dir, exist_ok=True)

    # Load model
    model_mod = importlib.import_module(f"maxim.models.{_MODEL_FILENAME}")
    model_configs = ml_collections.ConfigDict(_MODEL_CONFIGS)
    model_configs.variant = _MODEL_VARIANT_DICT[FLAGS.task]
    model = model_mod.Model(**model_configs)

    # Create datasets
    train_dataset = create_dataset(
        FLAGS.train_dir, FLAGS.batch_size, FLAGS.patch_size, is_training=True
    )
    val_dataset = create_dataset(
        FLAGS.val_dir, FLAGS.batch_size, FLAGS.patch_size, is_training=False
    )

    # Calculate steps
    train_size = len(list(train_dataset))
    steps_per_epoch = train_size
    total_steps = steps_per_epoch * FLAGS.num_epochs

    # Create learning rate schedule
    learning_rate_fn = create_learning_rate_schedule(
        FLAGS.learning_rate, FLAGS.warmup_epochs, total_steps, steps_per_epoch
    )

    # Create train state
    rng, init_rng = jax.random.split(rng)
    state = create_train_state(init_rng, model, learning_rate_fn, FLAGS.weight_decay)

    # Resume from checkpoint if specified
    if FLAGS.resume_from:
        state = checkpoints.restore_checkpoint(FLAGS.resume_from, state)
        logging.info(f"Resumed from checkpoint: {FLAGS.resume_from}")

    logging.info(f"Starting training for {FLAGS.num_epochs} epochs...")
    logging.info(f"Total steps: {total_steps}, Steps per epoch: {steps_per_epoch}")

    best_psnr = 0.0

    for epoch in range(FLAGS.num_epochs):
        # Training
        state, train_metrics = train_epoch(
            state, train_dataset, model_configs.num_supervision_scales, epoch
        )

        logging.info(
            f"Epoch {epoch} training: "
            f'loss = {train_metrics["loss"]:.4f}, '
            f'psnr = {train_metrics["psnr"]:.2f} dB'
        )

        # Validation
        val_metrics = evaluate(state, val_dataset)
        logging.info(
            f"Epoch {epoch} validation: "
            f'loss = {val_metrics["loss"]:.4f}, '
            f'psnr = {val_metrics["psnr"]:.2f} dB'
        )

        # Save checkpoint
        if (epoch + 1) % 10 == 0 or val_metrics["psnr"] > best_psnr:
            ckpt_path = os.path.join(FLAGS.output_dir, f"checkpoint_{epoch}")
            checkpoints.save_checkpoint(
                ckpt_dir=ckpt_path,
                target={"opt": {"target": state.params}},
                step=epoch,
                overwrite=True,
            )
            logging.info(f"Saved checkpoint to {ckpt_path}")

            if val_metrics["psnr"] > best_psnr:
                best_psnr = val_metrics["psnr"]
                best_ckpt_path = os.path.join(FLAGS.output_dir, "best_checkpoint")
                checkpoints.save_checkpoint(
                    ckpt_dir=best_ckpt_path,
                    target={"opt": {"target": state.params}},
                    step=epoch,
                    overwrite=True,
                )
                logging.info(f"New best model! PSNR: {best_psnr:.2f} dB")

    logging.info("Training completed!")


if __name__ == "__main__":
    app.run(main)
