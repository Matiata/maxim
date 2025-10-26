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
flags.DEFINE_string(
    "train_dir",
    "",
    "Training data directory.",
)
flags.DEFINE_string(
    "val_dir",
    "",
    "Validation data directory.",
)
flags.DEFINE_string("dataset_dir", "", "Path to the dataset.")
flags.DEFINE_string("output_dir", "./checkpoints", "Output directory for checkpoints.")
flags.DEFINE_integer("batch_size", 10, "Batch size for training.")
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


def load_image(filepath):
    """Load and preprocess image."""
    try:
        img = Image.open(filepath).convert("RGB")
        img = np.asarray(img, np.float32) / 255.0
        return img
    except Exception as e:
        print(f"Error loading image {filepath}: {e}")
        return np.zeros((1, 1, 3), np.float32)

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


def read_lines_from_file(basepath, filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    existing = []
    missing = []

    for line in lines:
        p = os.path.join(basepath, line)
        if os.path.exists(p):
            existing.append(p)
        else:
            missing.append(p)

    return existing, missing


def create_dataset(data_dir, batch_size, patch_size, is_training=True):
    """Create TensorFlow dataset for training/validation."""

    input_dir = (
        os.path.join(data_dir, "train")
        if is_training
        else os.path.join(data_dir, "test")
    )
    target_dir = os.path.join(data_dir, "GT")
    files_list = (
        os.path.join(data_dir, "train.txt")
        if is_training
        else os.path.join(data_dir, "test.txt")
    )

    # # Alternatively, you can load all files in the directory
    # input_files = sorted(tf.io.gfile.glob(os.path.join(input_dir, "*")))
    # target_files = sorted(tf.io.gfile.glob(os.path.join(target_dir, "*")))

    input_files, _ = read_lines_from_file(input_dir, files_list)
    target_files, _ = read_lines_from_file(target_dir, files_list)

    print(
        f"Found {len(input_files)} input files and {len(target_files)} target files for {'training' if is_training else 'testing'}\n"
    )

    def load_and_preprocess(input_path, target_path):
        """Load and preprocess a single pair of images."""
        input_img = load_image(input_path.numpy().decode())
        target_img = load_image(target_path.numpy().decode())

        if np.count_nonzero(input_img) == 0 or np.count_nonzero(target_img) == 0:
            return np.zeros((1, 1, 3), np.float32), np.zeros((1, 1, 3), np.float32)

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

    # dataset = dataset.filter(
    #     lambda x, y: tf.logical_and(
    #         tf.reduce_any(tf.not_equal(x, 0)),
    #         tf.reduce_any(tf.not_equal(y, 0))
    #     )
    # )

    dataset = dataset.batch(batch_size, drop_remainder=is_training)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset

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
    print(f"model_configs:\n{model_configs}")

    # Create datasets, already batched: (2, FLAGS.batch_size, 256, 256, 3)
    # first dimension is train/test, then batch dimension, then image dimensions: height, width, channels
    train_dataset = create_dataset(
        FLAGS.dataset_dir, FLAGS.batch_size, FLAGS.patch_size, is_training=True
    )
    test_dataset = create_dataset(
        FLAGS.dataset_dir, FLAGS.batch_size, FLAGS.patch_size, is_training=False
    )
    print(f"datasets lens: train: {len(train_dataset)}, test: {len(test_dataset)}")
    print(f"Example batch shape: {next(iter(train_dataset))[0].shape}")
    
    steps_per_epoch = len(train_dataset) # At least one step per batch
    total_steps = steps_per_epoch * FLAGS.num_epochs
    
    # Create learning rate schedule
    learning_rate_fn = create_learning_rate_schedule(
        FLAGS.learning_rate, FLAGS.warmup_epochs, total_steps, steps_per_epoch
    )



if __name__ == "__main__":
    app.run(main)
