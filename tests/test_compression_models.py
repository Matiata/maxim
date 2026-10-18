"""CPU checks for the compression modules (skipped when JAX is unavailable).

They use a reduced S-2 configuration so they run in a couple of minutes.
"""

import importlib.util
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_COMMIT = "98984438176afb21694dd1ec29b02fd9b40426c1"

try:
  import jax
  import jax.numpy as jnp
  import numpy as np
  HAVE_JAX = True
except ImportError:  # pragma: no cover
  HAVE_JAX = False

if HAVE_JAX:
  sys.path.insert(0, str(REPO_ROOT))
  from maxim.models import compression
  from maxim.models.compression import (
      VARIANTS, build_variant, flatten_params, transplant_params,
      is_new_local_mixer, removed_rcab_conv_at, variant_replaced_levels)
  from maxim.models.maxim import Model

SMALL = {
    "variant": "S-2", "features": 8, "dropout_rate": 0.1, "num_outputs": 3,
    "use_bias": True, "num_supervision_scales": 3,
    "block_size_hr": (4, 4), "grid_size_hr": (4, 4),
    "block_size_lr": (2, 2), "grid_size_lr": (2, 2),
}
SHAPE = (2, 32, 32, 3)


def _old_maxim_module():
  source = subprocess.check_output(
      ["git", "-C", str(REPO_ROOT), "show",
       f"{CHECKPOINT_COMMIT}:maxim/models/maxim.py"], text=True)
  path = Path(tempfile.mkdtemp()) / "maxim_checkpoint_commit.py"
  path.write_text(source, encoding="utf-8")
  spec = importlib.util.spec_from_file_location("maxim_checkpoint_commit", path)
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


def _max_diff(a, b):
  leaves_a = jax.tree_util.tree_leaves(a)
  leaves_b = jax.tree_util.tree_leaves(b)
  assert len(leaves_a) == len(leaves_b)
  return max(float(jnp.max(jnp.abs(x - y))) for x, y in zip(leaves_a, leaves_b))


def _conv_count(compiled_text):
  return compiled_text.count(" convolution(")


@unittest.skipUnless(HAVE_JAX, "JAX not installed")
class CompressionModelTest(unittest.TestCase):

  @classmethod
  def setUpClass(cls):
    rng = np.random.default_rng(0)
    cls.x = jnp.asarray(rng.random(SHAPE, dtype=np.float32))
    cls.tasks = jnp.asarray([1, 3], dtype=jnp.int32)
    cls.key = jax.random.PRNGKey(0)
    cls.s0 = build_variant("S0", SMALL)
    cls.s0_vars = cls.s0.init(cls.key, cls.x, train=False, task_id=cls.tasks)
    # Non-zero expert outputs so that selection errors are visible.
    params = flatten_params(cls.s0_vars["params"])
    for path in list(params):
      if path[0].startswith("experts_") and path[-2] == "output_conv":
        params[path] = 0.05 * jax.random.normal(
            jax.random.PRNGKey(hash(path) % 1000), params[path].shape)
    from flax import traverse_util
    cls.s0_vars = {"params": traverse_util.unflatten_dict(params)}

  def test_default_maxim_matches_checkpoint_commit(self):
    old = _old_maxim_module()
    old_model = old.Model(**SMALL)
    new_model = Model(**SMALL)
    old_vars = old_model.init(self.key, self.x, train=False)
    new_vars = new_model.init(self.key, self.x, train=False)
    self.assertEqual(
        jax.tree_util.tree_structure(old_vars),
        jax.tree_util.tree_structure(new_vars))
    self.assertEqual(_max_diff(old_vars, new_vars), 0.0)
    for train in (False, True):
      rngs = {"dropout": jax.random.PRNGKey(3)} if train else None
      out_old = old_model.apply(old_vars, self.x, train=train, rngs=rngs)
      out_new = new_model.apply(old_vars, self.x, train=train, rngs=rngs)
      self.assertEqual(_max_diff(out_old, out_new), 0.0)

  def test_remat_keeps_tree_outputs_and_gradients(self):
    plain = Model(**SMALL)
    remat = Model(**SMALL, remat_blocks=True)
    variables = plain.init(self.key, self.x, train=False)
    remat_vars = remat.init(self.key, self.x, train=False)
    self.assertEqual(jax.tree_util.tree_structure(variables),
                     jax.tree_util.tree_structure(remat_vars))
    rngs = {"dropout": jax.random.PRNGKey(5)}

    def loss(model, params):
      out = model.apply({"params": params}, self.x, train=True, rngs=rngs)
      return jnp.mean(jnp.abs(out[-1][-1] - self.x))

    g_plain = jax.jit(jax.grad(lambda p: loss(plain, p)))(variables["params"])
    g_remat = jax.jit(jax.grad(lambda p: loss(remat, p)))(variables["params"])
    self.assertLess(_max_diff(g_plain, g_remat), 1e-6)

  def test_selective_head_matches_dense_shared(self):
    sel = build_variant("S0_sel", SMALL)
    sel_vars = sel.init(self.key, self.x, train=False, task_id=self.tasks)
    self.assertEqual(jax.tree_util.tree_structure(sel_vars),
                     jax.tree_util.tree_structure(self.s0_vars))
    for task_ids in ([0, 0], [1, 3], [4, 2]):
      task_ids = jnp.asarray(task_ids, dtype=jnp.int32)
      dense = self.s0.apply(self.s0_vars, self.x, train=False, task_id=task_ids)
      selective = sel.apply(self.s0_vars, self.x, train=False, task_id=task_ids)
      self.assertLess(_max_diff(dense[0], selective[0]), 1e-5)

  def _transplanted(self, name):
    model = build_variant(name, SMALL)
    fresh = model.init(self.key, self.x, train=False, task_id=self.tasks)
    params, report = transplant_params(
        self.s0_vars["params"], fresh["params"],
        allowed_new=is_new_local_mixer,
        allowed_removed=removed_rcab_conv_at(variant_replaced_levels(name)))
    return model, {"params": params}, report

  def test_transplant_removes_only_deep_rcab_convs(self):
    _, _, report = self._transplanted("S2_d32")
    # 2 stages x (encoder + decoder) x 2 RCAB x (conv1, conv2) x (kernel, bias)
    self.assertEqual(report["removed_tensors"], 32)
    self.assertTrue(all("block_2" in p for p in report["removed_paths"]))
    self.assertEqual(report["new_tensors"], 8 * 6)
    with self.assertRaises(ValueError):
      transplant_params(self.s0_vars["params"],
                        self._transplanted("S2_d32")[1]["params"])

  def test_parameter_counts_follow_analytic_formula(self):
    c, d = 4 * SMALL["features"], 32
    per_module = c * d + d + 9 * d + d + d * c + c
    pair = 2 * (9 * c * c + c)
    counts = {}
    for name in ("S0", "S1_d32", "S2_d32", "S0_sel_sin_rcab2"):
      _, variables, _ = (self._transplanted(name) if name != "S0"
                         else (None, self.s0_vars, None))
      counts[name] = sum(int(np.prod(v.shape)) for v in
                         jax.tree_util.tree_leaves(variables["params"]))
    self.assertEqual(counts["S0"] - counts["S0_sel_sin_rcab2"], 8 * pair)
    self.assertEqual(counts["S1_d32"] - counts["S0_sel_sin_rcab2"],
                     8 * per_module)
    self.assertEqual(counts["S2_d32"] - counts["S0_sel_sin_rcab2"],
                     8 * 5 * per_module)

  def test_bank_gather_matches_all_experts_reference(self):
    bank = compression.TaskBottleneckBank(features=16, width=8, num_experts=5)
    ref = compression.TaskBottleneckBank(features=16, width=8, num_experts=5,
                                         selection="all_experts_reference")
    x = jax.random.normal(self.key, (3, 8, 8, 16))
    tasks = jnp.asarray([4, 0, 4], dtype=jnp.int32)
    variables = bank.init(self.key, x, tasks)
    self.assertLess(_max_diff(bank.apply(variables, x, tasks),
                              ref.apply(variables, x, tasks)), 1e-5)

  def test_selective_paths_do_not_compile_unselected_experts(self):
    x1 = self.x[:1]
    t1 = self.tasks[:1]

    def conv_count(model, variables):
      fn = jax.jit(lambda v, x, t: model.apply(v, x, train=False, task_id=t))
      return _conv_count(fn.lower(variables, x1, t1).compile().as_text())

    sel = build_variant("S0_sel", SMALL)
    n_dense = conv_count(self.s0, self.s0_vars)
    n_sel = conv_count(sel, self.s0_vars)
    self.assertEqual(n_dense - n_sel, 4 * 3)
    s1, v1, _ = self._transplanted("S1_d32")
    s2, v2, _ = self._transplanted("S2_d32")
    self.assertEqual(conv_count(s1, v1), conv_count(s2, v2))

  def test_mixed_task_training_step_reaches_only_selected_experts(self):
    s2, variables, _ = self._transplanted("S2_d32")
    rngs = {"dropout": jax.random.PRNGKey(7)}

    def loss(params):
      out, _ = s2.apply({"params": params}, self.x, train=True, rngs=rngs,
                        task_id=self.tasks)
      return jnp.mean(jnp.abs(out[-1][-1] - self.x))

    grads = flatten_params(jax.jit(jax.grad(loss))(variables["params"]))
    bank_grads = [g for p, g in grads.items() if "local_mixer" in p
                  and p[-1] == "in_kernel"]
    self.assertEqual(len(bank_grads), 8)
    for grad in bank_grads:
      per_expert = np.asarray(jnp.sum(jnp.abs(grad), axis=(1, 2)))
      self.assertTrue(np.all(per_expert[[1, 3]] > 0))
      self.assertTrue(np.all(per_expert[[0, 2, 4]] == 0))
    for name, value in grads.items():
      self.assertTrue(bool(jnp.all(jnp.isfinite(value))), name)

  def test_catalogue_is_complete(self):
    for name in VARIANTS:
      build_variant(name, SMALL)


if __name__ == "__main__":
  unittest.main()
