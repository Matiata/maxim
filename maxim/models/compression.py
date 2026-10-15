"""Modules for compressing MAXIM with per-task experts.

Contents:

* ``TaskBottleneckBank``: replacement for the RCAB conv1/conv2 pair,
  ``1x1 C->d -> GELU -> depthwise 3x3 -> 1x1 d->C``. With one expert it is the
  common small module (S1). With several experts and oracle ``task_id`` it
  gathers the selected expert's weights and executes only that expert (S2).
* ``SelectiveSharedResidualMaximMoE``: same function and same parameter tree
  as ``moe.SharedResidualMaximMoE`` in oracle mode, but it executes only the
  selected residual head instead of all five.
* ``transplant_params``: strict initialization of a compressed variant from a
  shared checkpoint, with an explicit list of new and removed tensors.
* ``parameter_family``: groups parameter paths into block families.

Selection by weight gathering works for batch 1 and for mixed-task batches,
because each example receives its own expert weights. The
``all_experts_reference`` selection computes every expert and masks the
result; it exists only to verify the semantics of the gathered path.
"""

import re
from typing import Any

from flax import linen as nn
from flax import traverse_util
from flax.core import unfreeze
import jax
from jax import lax
import jax.numpy as jnp
import numpy as np


def _fan_in_normal(fan_in):
  stddev = float(np.sqrt(1.0 / fan_in))

  def init(key, shape, dtype=jnp.float32):
    return stddev * jax.random.normal(key, shape, dtype)

  return init


def _bottleneck_batch(x, in_kernel, in_bias, dw_kernel, dw_bias,
                      out_kernel, out_bias):
  """Applies one bottleneck expert to a whole batch ``[B, H, W, C]``."""
  width = in_kernel.shape[-1]
  h = jnp.einsum("bhwc,cd->bhwd", x, in_kernel) + in_bias
  h = nn.gelu(h)
  h = lax.conv_general_dilated(
      h,
      dw_kernel[:, :, None, :],
      window_strides=(1, 1),
      padding="SAME",
      dimension_numbers=("NHWC", "HWIO", "NHWC"),
      feature_group_count=width,
  ) + dw_bias
  return jnp.einsum("bhwd,dc->bhwc", h, out_kernel) + out_bias


def _bottleneck_single(x, *params):
  return _bottleneck_batch(x[None], *params)[0]


class TaskBottleneckBank(nn.Module):
  """Bank of ``num_experts`` bottleneck mixers that keeps shape ``[B,H,W,C]``."""

  features: int
  width: int
  num_experts: int = 1
  use_bias: bool = True
  selection: str = "gather"

  @nn.compact
  def __call__(self, x, task_id=None):
    if not self.use_bias:
      raise ValueError("TaskBottleneckBank always uses biases.")
    if x.shape[-1] != self.features:
      raise ValueError(f"Expected {self.features} channels, got {x.shape}.")
    e, c, d = self.num_experts, self.features, self.width
    zeros = nn.initializers.zeros
    params = (
        self.param("in_kernel", _fan_in_normal(c), (e, c, d)),
        self.param("in_bias", zeros, (e, d)),
        self.param("dw_kernel", _fan_in_normal(9), (e, 3, 3, d)),
        self.param("dw_bias", zeros, (e, d)),
        self.param("out_kernel", _fan_in_normal(d), (e, d, c)),
        self.param("out_bias", zeros, (e, c)),
    )
    params = tuple(p.astype(x.dtype) for p in params)
    if e == 1:
      return _bottleneck_batch(x, *(p[0] for p in params))

    if task_id is None:
      raise ValueError("task_id is required when num_experts > 1.")
    task_id = jnp.asarray(task_id, dtype=jnp.int32).reshape(-1)
    if task_id.shape[0] != x.shape[0]:
      raise ValueError("task_id must contain one id per example.")
    if self.selection == "gather":
      selected = tuple(jnp.take(p, task_id, axis=0) for p in params)
      return jax.vmap(_bottleneck_single)(x, *selected)
    if self.selection == "all_experts_reference":
      outputs = jnp.stack(
          [_bottleneck_batch(x, *(p[i] for p in params)) for i in range(e)],
          axis=1)
      gates = jax.nn.one_hot(task_id, e, dtype=x.dtype)
      return jnp.einsum("be,behwc->bhwc", gates, outputs)
    raise ValueError(f"Unknown selection={self.selection!r}")


class _ConvParams(nn.Module):
  """Declares ``kernel``/``bias`` exactly like ``nn.Conv`` 3x3 without applying it."""

  features: int
  zero_init: bool = False

  @nn.compact
  def __call__(self, in_features):
    kernel_init = (nn.initializers.zeros if self.zero_init
                   else nn.initializers.lecun_normal())
    kernel = self.param("kernel", kernel_init,
                        (3, 3, in_features, self.features))
    bias = self.param("bias", nn.initializers.zeros, (self.features,))
    return kernel, bias


class _ResidualHeadParams(nn.Module):
  """Parameter-only twin of ``moe.ResidualExpertHead``."""

  out_channels: int = 3
  num_hidden_layers: int = 2

  @nn.compact
  def __call__(self, in_features):
    hidden = [
        _ConvParams(in_features, name=f"hidden_conv_{index}")(in_features)
        for index in range(self.num_hidden_layers)
    ]
    output = _ConvParams(self.out_channels, zero_init=True,
                         name="output_conv")(in_features)
    return {"hidden": hidden, "output": output}


def _conv3x3(x, kernel, bias):
  y = lax.conv_general_dilated(
      x, kernel, window_strides=(1, 1), padding="SAME",
      dimension_numbers=("NHWC", "HWIO", "NHWC"))
  return y + bias


def residual_head_apply(x, head):
  """Same computation as ``moe.ResidualExpertHead`` for a batch ``[B,H,W,C]``."""
  shortcut = x
  h = x
  for kernel, bias in head["hidden"]:
    h = nn.gelu(_conv3x3(h, kernel, bias))
  h = nn.gelu(h + shortcut)
  kernel, bias = head["output"]
  return _conv3x3(h, kernel, bias)


def _residual_head_single(x, head):
  return residual_head_apply(x[None], head)[0]


class SelectiveSharedResidualMaximMoE(nn.Module):
  """Shared residual MAXIM with oracle experts and real selective execution.

  Parameter tree: ``maxim/...`` and ``experts_{i}/{hidden_conv_j,output_conv}``,
  identical to ``SharedResidualMaximMoE``, so its checkpoints restore directly.
  ``task_id`` is also forwarded to MAXIM for task-expert RCAB mixers.
  """

  maxim: nn.Module
  num_experts: int = 5
  num_hidden_layers: int = 2
  out_channels: int = 3
  correction_scale: float = 1.0
  selection: str = "gather"

  @nn.compact
  def __call__(self, x, train=True, task_id=None):
    if task_id is None:
      raise ValueError("task_id is required for oracle routing.")
    task_id = jnp.asarray(task_id, dtype=jnp.int32).reshape(-1)
    if task_id.shape[0] != x.shape[0]:
      raise ValueError("task_id must contain exactly one id per input image.")

    outputs_all, feats = self.maxim(
        x, train=train, return_features=True, task_id=task_id)
    channels = feats.shape[-1]
    heads = [
        _ResidualHeadParams(
            out_channels=self.out_channels,
            num_hidden_layers=self.num_hidden_layers,
            name=f"experts_{index}")(channels)
        for index in range(self.num_experts)
    ]
    heads = jax.tree_util.tree_map(
        lambda leaf: leaf.astype(feats.dtype), heads)
    gates = jax.nn.one_hot(task_id, self.num_experts, dtype=x.dtype)
    with jax.named_scope("selected_expert_head"):
      if self.selection == "gather":
        stacked = jax.tree_util.tree_map(lambda *leaves: jnp.stack(leaves),
                                         *heads)
        selected = jax.tree_util.tree_map(
            lambda leaf: jnp.take(leaf, task_id, axis=0), stacked)
        correction = jax.vmap(_residual_head_single)(feats, selected)
      elif self.selection == "all_experts_reference":
        corrections = jnp.stack(
            [residual_head_apply(feats, head) for head in heads], axis=1)
        correction = jnp.einsum("be,behwc->bhwc", gates, corrections)
      else:
        raise ValueError(f"Unknown selection={self.selection!r}")

    predictions = [list(stage_outputs) for stage_outputs in outputs_all]
    predictions[-1][-1] = (
        predictions[-1][-1] + self.correction_scale * correction)
    return predictions, gates


# --------------------------------------------------------------------------
# Parameter bookkeeping


def flatten_params(tree):
  return traverse_util.flatten_dict(unfreeze(tree))


_LEVEL_BLOCK = re.compile(r"stage_\d+_(?:encoder|decoder)_block_(\d+)$")


def rcab_conv_level(path):
  """Returns the U-Net level of an RCAB conv1/conv2 tensor, else None."""
  if len(path) < 2 or path[-2] not in ("conv1", "conv2"):
    return None
  if not any(str(part).startswith("channel_attention_block_1") and
             not str(part).startswith("channel_attention_block_1_")
             for part in path):
    return None
  for part in path:
    match = _LEVEL_BLOCK.match(str(part))
    if match:
      return int(match.group(1))
  return None


def parameter_family(path):
  """Coarse block family of a parameter path (tuple of keys)."""
  text = "/".join(str(part) for part in path)
  if text.startswith("experts_"):
    return "expert_heads"
  if "local_mixer" in text:
    return "rcab_local_mixer"
  if "GridGmlpLayer" in text or "GridGating" in text:
    return "gmlp_grid_global"
  if "BlockGmlpLayer" in text or "BlockGating" in text:
    return "gmlp_block_local"
  if "cross_gating" in text or "SplitHeadMultiAxisGating" in text or (
      "input_fuse_sam" in text):
    return "cross_gating"
  level = rcab_conv_level(path)
  if level is not None:
    return f"rcab_conv_level_{level}"
  if "channel_attention_block" in text:
    return "rcab_other"
  if "SplitHeadMultiAxisGmlpLayer" in text:
    return "mab_projections"
  if "UpSampleRatio" in text:
    return "resize"
  if "supervised_attention" in text:
    return "sam"
  return "other"


def parameter_families(params):
  totals = {}
  for path, value in flatten_params(params).items():
    family = parameter_family(path)
    totals[family] = totals.get(family, 0) + int(np.prod(value.shape))
  return dict(sorted(totals.items()))


def transplant_params(source_params, target_params, *,
                      allowed_new=lambda path: False,
                      allowed_removed=lambda path: False):
  """Builds target params from source params with a strict audit.

  Every target tensor is copied from the source with identical path, shape and
  dtype, or it must satisfy ``allowed_new`` and keeps its fresh initialization.
  Every source tensor not copied must satisfy ``allowed_removed``.
  """
  source = flatten_params(source_params)
  target = flatten_params(target_params)
  merged, copied, new, problems = {}, [], [], []
  for path, value in target.items():
    if path in source:
      origin = source[path]
      if tuple(origin.shape) != tuple(value.shape):
        problems.append(f"shape {path}: {origin.shape} != {value.shape}")
      elif np.dtype(origin.dtype) != np.dtype(value.dtype):
        problems.append(f"dtype {path}: {origin.dtype} != {value.dtype}")
      merged[path] = origin
      copied.append(path)
    elif allowed_new(path):
      merged[path] = value
      new.append(path)
    else:
      problems.append(f"unexpected new tensor {path}")
  removed = [path for path in source if path not in target]
  problems.extend(f"unexpected removed tensor {path}" for path in removed
                  if not allowed_removed(path))
  if problems:
    raise ValueError("transplant_params: " + "; ".join(problems[:10]))
  count = lambda paths, tree: int(sum(np.prod(tree[p].shape) for p in paths))
  report = {
      "copied_tensors": len(copied),
      "copied_parameters": count(copied, source),
      "new_tensors": len(new),
      "new_parameters": count(new, target),
      "removed_tensors": len(removed),
      "removed_parameters": count(removed, source),
      "new_paths": ["/".join(map(str, p)) for p in sorted(new)],
      "removed_paths": ["/".join(map(str, p)) for p in sorted(removed)],
  }
  return traverse_util.unflatten_dict(merged), report


def is_new_local_mixer(path):
  return "local_mixer" in path


def removed_rcab_conv_at(levels):
  levels = tuple(levels)
  return lambda path: rcab_conv_level(path) in levels


# --------------------------------------------------------------------------
# Variant catalogue shared by benchmark and future training notebooks.


VARIANTS: dict[str, dict[str, Any]] = {
    "B0": {"family": "baseline", "init": "baseline_checkpoint"},
    "B0_remat": {"family": "baseline", "init": "baseline_checkpoint",
                 "maxim": {"remat_blocks": True}},
    "S0": {"family": "shared_dense", "init": "shared_checkpoint"},
    "S0_sel": {"family": "shared_selective", "init": "shared_checkpoint"},
    "S1_d32": {"family": "shared_selective", "init": "shared_plus_new",
               "maxim": {"rcab_mixer": "bottleneck", "rcab_mixer_levels": (2,),
                         "rcab_mixer_width": 32, "rcab_mixer_num_experts": 1}},
    "S2_d32": {"family": "shared_selective", "init": "shared_plus_new",
               "maxim": {"rcab_mixer": "bottleneck", "rcab_mixer_levels": (2,),
                         "rcab_mixer_width": 32, "rcab_mixer_num_experts": 5}},
    "S2_d32_todos": {"family": "shared_selective", "init": "shared_plus_new",
                     "maxim": {"rcab_mixer": "bottleneck",
                               "rcab_mixer_levels": (2,),
                               "rcab_mixer_width": 32,
                               "rcab_mixer_num_experts": 5,
                               "rcab_mixer_selection": "all_experts_reference"}},
    "S1_d64": {"family": "shared_selective", "init": "shared_plus_new",
               "maxim": {"rcab_mixer": "bottleneck", "rcab_mixer_levels": (2,),
                         "rcab_mixer_width": 64, "rcab_mixer_num_experts": 1}},
    "S2_d64": {"family": "shared_selective", "init": "shared_plus_new",
               "maxim": {"rcab_mixer": "bottleneck", "rcab_mixer_levels": (2,),
                         "rcab_mixer_width": 64, "rcab_mixer_num_experts": 5}},
    "S0_sel_sin_rcab2": {"family": "shared_selective",
                         "init": "shared_plus_new",
                         "maxim": {"rcab_mixer": "none",
                                   "rcab_mixer_levels": (2,)}},
    "ref_sin_cross_gating": {"family": "baseline", "init": "random",
                             "maxim": {"use_cross_gating": False}},
    "ref_sin_gmlp": {"family": "baseline", "init": "random",
                     "maxim": {"use_global_mlp": False}},
}


def variant_replaced_levels(name):
  spec = VARIANTS[name].get("maxim", {})
  if spec.get("rcab_mixer", "dense") == "dense":
    return ()
  return tuple(spec.get("rcab_mixer_levels", ()))


def build_variant(name, base_config, num_tasks=5):
  """Returns the Flax model of a catalogue variant."""
  from maxim.models.maxim import Model
  from maxim.models.moe import ResidualExpertHead, SharedResidualMaximMoE
  from maxim.models.router import RouterModel

  spec = VARIANTS[name]
  config = dict(base_config)
  config.update(spec.get("maxim", {}))
  backbone = Model(**config)
  family = spec["family"]
  if family == "baseline":
    return backbone
  if family == "shared_dense":
    return SharedResidualMaximMoE(
        backbone, RouterModel(),
        [ResidualExpertHead(num_hidden_layers=2, zero_init_output=True)
         for _ in range(num_tasks)],
        routing_mode="oracle", correction_scale=1.0)
  if family == "shared_selective":
    return SelectiveSharedResidualMaximMoE(
        backbone, num_experts=num_tasks, num_hidden_layers=2,
        correction_scale=1.0)
  raise ValueError(family)
