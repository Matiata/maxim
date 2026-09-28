#!/usr/bin/env python3
"""Small JAX/Flax smoke test for TokenChoiceMaximMoE without datasets."""

import importlib.util
import sys
import types

import jax
import jax.numpy as jnp
from flax import linen as nn


def load_uploaded_moe():
    """Load /content/token_moe.py while stubbing its legacy type imports."""
    maxim_package = types.ModuleType("maxim")
    maxim_package.__path__ = []
    models_package = types.ModuleType("maxim.models")
    models_package.__path__ = []
    maxim_module = types.ModuleType("maxim.models.maxim")
    router_module = types.ModuleType("maxim.models.router")

    class StubMaxim(nn.Module):
        pass

    class StubRouter(nn.Module):
        pass

    maxim_module.MAXIM = StubMaxim
    router_module.RouterModel = StubRouter
    sys.modules.update(
        {
            "maxim": maxim_package,
            "maxim.models": models_package,
            "maxim.models.maxim": maxim_module,
            "maxim.models.router": router_module,
        }
    )
    spec = importlib.util.spec_from_file_location("token_moe", "/content/token_moe.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


moe = load_uploaded_moe()


class FakeMaxim(nn.Module):
    @nn.compact
    def __call__(self, x, train=True, return_features=False):
        features = nn.Conv(4, (1, 1), name="features")(x)
        base = nn.Conv(3, (1, 1), name="output")(features) + x
        outputs = [[base]]
        return (outputs, features) if return_features else outputs


model = moe.TokenChoiceMaximMoE(
    maxim=FakeMaxim(),
    router=moe.TokenRouter(num_experts=8, hidden_features=16),
    experts=[moe.TokenExpert(hidden_multiplier=2.0) for _ in range(8)],
    shared_output_head=moe.ExpertHead(),
    top_k=2,
    router_noise_std=0.05,
)
key = jax.random.PRNGKey(42)
first = jax.random.normal(key, (1, 4, 6, 3))
variables = model.init(
    {"params": key, "routing": jax.random.fold_in(key, 1)},
    first,
    train=True,
    dense_routing=False,
)

for shape in ((1, 4, 6, 3), (2, 6, 4, 3)):
    inputs = jax.random.normal(jax.random.fold_in(key, shape[0]), shape)
    predictions, dense, sparse = model.apply(variables, inputs, train=False)
    assert predictions[-1][-1].shape == shape
    assert dense.shape == (shape[0], shape[1] * shape[2], 8)
    assert sparse.shape == dense.shape
    assert jnp.all(dense > 0)
    assert jnp.allclose(dense.sum(axis=-1), 1.0, atol=1e-6)
    assert jnp.allclose(sparse.sum(axis=-1), 1.0, atol=1e-6)
    assert jnp.all((sparse > 0).sum(axis=-1) == 2)


def loss_fn(params):
    predictions, dense, _ = model.apply(
        {"params": params}, first, train=False, dense_routing=True
    )
    usage = dense.mean(axis=(0, 1))
    balance = dense.shape[-1] * jnp.sum(usage**2)
    return jnp.mean(predictions[-1][-1] ** 2) + 0.01 * balance


grads = jax.grad(loss_fn)(variables["params"])


def tree_norm(tree):
    leaves = jax.tree_util.tree_leaves(tree)
    return float(jnp.sqrt(sum(jnp.sum(leaf**2) for leaf in leaves)))


router_norm = tree_norm(grads["router"])
expert_norms = [tree_norm(grads[f"experts_{index}"]) for index in range(8)]
assert jnp.isfinite(router_norm) and router_norm > 0
assert all(jnp.isfinite(value) and value > 0 for value in expert_norms)

controlled_logits = jnp.asarray(
    [[[9., 8., 0., -1., -2., -3., -4., -5.],
      [0., -1., 9., 8., -2., -3., -4., -5.]]]
)
controlled_dense = jax.nn.softmax(controlled_logits, axis=-1)
controlled_sparse = moe.normalized_top_k_gates(
    controlled_logits, controlled_dense, 2
)
pairs = jnp.argsort(controlled_sparse, axis=-1)[..., -2:]
assert not jnp.array_equal(pairs[:, 0], pairs[:, 1])

print("TOKEN-CHOICE SMOKE TEST PASSED")
print(f"router_grad_norm={router_norm:.6e}")
print("expert_grad_norms=" + ",".join(f"{value:.6e}" for value in expert_norms))
