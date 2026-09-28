import importlib.util
from pathlib import Path
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]
MOE_SOURCE = (REPO_ROOT / "maxim" / "models" / "moe.py").read_text(
    encoding="utf-8"
)
K8_NOTEBOOK = (REPO_ROOT / "maxim" / "moeTrainer_latent_k8.ipynb").read_text(
    encoding="utf-8"
)


class TokenChoiceSourceTest(unittest.TestCase):
    def test_router_operates_on_tokens_without_global_pooling(self):
        token_router = MOE_SOURCE.split("class TokenRouter", 1)[1].split(
            "class TokenExpert", 1
        )[0]
        self.assertIn("tokens.ndim != 3", token_router)
        self.assertNotIn("mean(axis=(1, 2))", token_router)

    def test_model_routes_decoder_features_and_returns_both_gate_types(self):
        model = MOE_SOURCE.split("class TokenChoiceMaximMoE", 1)[1].split(
            "class MaximMoE", 1
        )[0]
        self.assertIn("return_features=True", model)
        self.assertIn("features.reshape(batch, height * width, channels)", model)
        self.assertIn("dense_router_probs", model)
        self.assertIn("sparse_gates", model)
        self.assertIn("shared_output_head", model)
        self.assertNotIn("task_id", model)

    def test_k8_notebook_uses_token_choice_model(self):
        self.assertIn("TokenChoiceMaximMoE", K8_NOTEBOOK)
        self.assertIn("dense_router_probs", K8_NOTEBOOK)
        self.assertIn("sparse_gates", K8_NOTEBOOK)
        self.assertIn("DENSE_ROUTING_WARMUP_EPOCHS = 2", K8_NOTEBOOK)
        self.assertIn("ROUTER_NOISE_STD = 0.05", K8_NOTEBOOK)
        self.assertNotIn("router_mod.RouterModel", K8_NOTEBOOK)


@unittest.skipUnless(
    importlib.util.find_spec("jax") and importlib.util.find_spec("flax"),
    "JAX/Flax are exercised by the Colab smoke test",
)
class TokenChoiceFunctionalTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import jax
        import jax.numpy as jnp
        from flax import linen as nn

        from maxim.models.moe import (
            ExpertHead,
            TokenChoiceMaximMoE,
            TokenExpert,
            TokenRouter,
            normalized_top_k_gates,
        )

        cls.jax = jax
        cls.jnp = jnp
        cls.nn = nn
        cls.ExpertHead = ExpertHead
        cls.TokenChoiceMaximMoE = TokenChoiceMaximMoE
        cls.TokenExpert = TokenExpert
        cls.TokenRouter = TokenRouter
        cls.normalized_top_k_gates = staticmethod(normalized_top_k_gates)

        class FakeMaxim(nn.Module):
            @nn.compact
            def __call__(self, x, train=True, return_features=False):
                features = nn.Conv(4, (1, 1), name="features")(x)
                base = nn.Conv(3, (1, 1), name="output")(features) + x
                outputs = [[base]]
                return (outputs, features) if return_features else outputs

        cls.FakeMaxim = FakeMaxim

    def test_controlled_tokens_can_choose_different_expert_pairs(self):
        logits = self.jnp.asarray(
            [[[9.0, 8.0, 0.0, -1.0], [0.0, -1.0, 9.0, 8.0]]]
        )
        dense = self.jax.nn.softmax(logits, axis=-1)
        sparse = self.normalized_top_k_gates(logits, dense, 2)
        selected = self.jnp.argsort(sparse, axis=-1)[..., -2:]

        self.assertEqual(sparse.shape, (1, 2, 4))
        self.assertTrue(self.jnp.allclose(sparse.sum(axis=-1), 1.0))
        self.assertTrue(self.jnp.all((sparse > 0).sum(axis=-1) == 2))
        self.assertFalse(self.jnp.array_equal(selected[:, 0], selected[:, 1]))

    def test_dense_balance_reaches_every_logit(self):
        logits = self.jnp.arange(16, dtype=self.jnp.float32).reshape(2, 8) / 7.0

        def balance(value):
            probs = self.jax.nn.softmax(value, axis=-1)
            usage = probs.mean(axis=0)
            return 8 * self.jnp.sum(usage**2)

        grads = self.jax.grad(balance)(logits)
        self.assertTrue(self.jnp.all(self.jnp.isfinite(grads)))
        self.assertTrue(self.jnp.all(self.jnp.abs(grads) > 0))

    def test_complete_model_preserves_rgb_shape_for_two_input_shapes(self):
        model = self.TokenChoiceMaximMoE(
            maxim=self.FakeMaxim(),
            router=self.TokenRouter(num_experts=4, hidden_features=8),
            experts=[self.TokenExpert(hidden_multiplier=2.0) for _ in range(4)],
            shared_output_head=self.ExpertHead(),
            top_k=2,
        )
        first = self.jnp.zeros((1, 4, 6, 3))
        variables = model.init(self.jax.random.PRNGKey(0), first, train=False)

        for shape in ((1, 4, 6, 3), (2, 6, 4, 3)):
            inputs = self.jnp.zeros(shape)
            predictions, dense, sparse = model.apply(
                variables, inputs, train=False
            )
            self.assertEqual(predictions[-1][-1].shape, shape)
            self.assertEqual(dense.shape, (shape[0], shape[1] * shape[2], 4))
            self.assertEqual(sparse.shape, dense.shape)
            self.assertTrue(self.jnp.all(dense > 0))
            self.assertTrue(self.jnp.allclose(dense.sum(axis=-1), 1.0))
            self.assertTrue(self.jnp.allclose(sparse.sum(axis=-1), 1.0))
            self.assertTrue(self.jnp.all((sparse > 0).sum(axis=-1) == 2))


if __name__ == "__main__":
    unittest.main()
