import functools

from flax import linen as nn
import jax
from jax import nn as jax_nn
import jax.numpy as jnp

from maxim.models.maxim import MAXIM
from maxim.models.router import RouterModel


Conv3x3 = functools.partial(nn.Conv, kernel_size=(3, 3))
Conv1x1 = functools.partial(nn.Conv, kernel_size=(1, 1))


class ExpertHead(nn.Module):
    out_channels: int = 3
    use_bias: bool = True

    @nn.compact
    def __call__(self, x):
        x = Conv3x3(
            self.out_channels,
            padding="SAME",
            use_bias=self.use_bias,
            name="output_conv",
        )(x)
        return x


class ResidualExpertHead(nn.Module):
    """Higher-capacity residual reconstruction head for one task.

    The hidden convolutions preserve the MAXIM feature width, so the final
    ``output_conv`` has the same input/output contract as ``ExpertHead``. This
    keeps the optional warm-start path compatible with MAXIM's final output
    convolution while adding nonlinear task-specific processing.
    """

    out_channels: int = 3
    use_bias: bool = True
    num_hidden_layers: int = 2

    @nn.compact
    def __call__(self, x):
        if self.num_hidden_layers < 1:
            raise ValueError("num_hidden_layers must be at least 1")

        shortcut = x
        hidden_channels = x.shape[-1]
        h = x
        for index in range(self.num_hidden_layers):
            h = Conv3x3(
                hidden_channels,
                padding="SAME",
                use_bias=self.use_bias,
                name=f"hidden_conv_{index}",
            )(h)
            h = nn.gelu(h)

        # Internal residual connection keeps the extra capacity close to the
        # identity at initialization and preserves stable gradient flow.
        h = nn.gelu(h + shortcut)
        return Conv3x3(
            self.out_channels,
            padding="SAME",
            use_bias=self.use_bias,
            name="output_conv",
        )(h)


class TokenRouter(nn.Module):
    """Shared per-token router with no spatial or cross-token pooling."""

    num_experts: int
    hidden_features: int = 128

    @nn.compact
    def __call__(self, tokens):
        if tokens.ndim != 3:
            raise ValueError(
                f"TokenRouter expects [B, N, C], got shape {tokens.shape}."
            )
        x = nn.LayerNorm(name="input_norm")(tokens)
        x = nn.Dense(self.hidden_features, name="hidden")(x)
        x = nn.gelu(x)
        return nn.Dense(self.num_experts, name="logits")(x)


class TokenExpert(nn.Module):
    """Residual MLP expert mapping each feature token from C to C."""

    hidden_multiplier: float = 2.0

    @nn.compact
    def __call__(self, tokens):
        if tokens.ndim != 3:
            raise ValueError(
                f"TokenExpert expects [B, N, C], got shape {tokens.shape}."
            )
        channels = tokens.shape[-1]
        hidden_features = max(1, int(channels * self.hidden_multiplier))
        x = nn.LayerNorm(name="input_norm")(tokens)
        x = nn.Dense(hidden_features, name="hidden")(x)
        x = nn.gelu(x)
        return nn.Dense(channels, name="output")(x)


def normalized_top_k_gates(logits, dense_probs, top_k):
    """Mask and renormalize probabilities independently for every token."""
    num_experts = logits.shape[-1]
    if not 1 <= top_k <= num_experts:
        raise ValueError(
            f"top_k must be in [1, {num_experts}], got {top_k}."
        )
    if top_k == num_experts:
        return dense_probs
    _, top_indices = jax.lax.top_k(logits, top_k)
    mask = jnp.sum(
        jax_nn.one_hot(top_indices, num_experts, dtype=dense_probs.dtype),
        axis=-2,
    )
    sparse_gates = dense_probs * mask
    return sparse_gates / jnp.sum(sparse_gates, axis=-1, keepdims=True)


class TokenChoiceMaximMoE(nn.Module):
    """MAXIM with independent top-k expert selection for every feature token.

    The router sees only the final decoder features. Experts transform tokens
    in feature space (C -> C); their weighted residual is reshaped to a feature
    map and reconstructed by one shared RGB head.
    """

    maxim: MAXIM
    router: TokenRouter
    experts: list[TokenExpert]
    shared_output_head: ExpertHead
    top_k: int = 2
    temperature: float = 1.0
    router_noise_std: float = 0.0

    def __call__(self, x, train=True, dense_routing=False):
        num_experts = len(self.experts)
        if num_experts < 1:
            raise ValueError("TokenChoiceMaximMoE requires at least one expert.")
        if self.router.num_experts != num_experts:
            raise ValueError(
                "Router/expert count mismatch: "
                f"{self.router.num_experts} router outputs for {num_experts} experts."
            )
        if self.temperature <= 0:
            raise ValueError("temperature must be positive.")

        outputs_all, features = self.maxim(x, train=train, return_features=True)
        batch, height, width, channels = features.shape
        tokens = features.reshape(batch, height * width, channels)

        router_logits = self.router(tokens)
        routing_logits = router_logits
        if train and self.router_noise_std > 0:
            routing_logits = routing_logits + self.router_noise_std * jax.random.normal(
                self.make_rng("routing"), routing_logits.shape
            )
        dense_router_probs = nn.softmax(
            routing_logits / self.temperature, axis=-1
        )
        sparse_gates = normalized_top_k_gates(
            routing_logits, dense_router_probs, self.top_k
        )
        mixture_gates = jnp.where(
            jnp.asarray(dense_routing), dense_router_probs, sparse_gates
        )

        mixed_residual = jnp.zeros_like(tokens)
        for expert_index, expert in enumerate(self.experts):
            expert_residual = expert(tokens)
            mixed_residual = mixed_residual + (
                mixture_gates[..., expert_index, None] * expert_residual
            )
        routed_features = (tokens + mixed_residual).reshape(
            batch, height, width, channels
        )
        final_prediction = self.shared_output_head(routed_features) + x

        predictions = [list(stage_outputs) for stage_outputs in outputs_all]
        predictions[-1][-1] = final_prediction
        return predictions, dense_router_probs, sparse_gates


class MaximMoE(nn.Module):
    """MAXIM with deep supervision and an MoE final reconstruction head."""

    maxim: MAXIM
    router: RouterModel
    experts: list[ExpertHead]
    routing_mode: str = "learned"
    top_k: int = 0

    def _route(self, x, train, task_id):
        num_experts = len(self.experts)
        if self.routing_mode == "oracle":
            if task_id is None:
                raise ValueError("task_id is required for oracle routing.")
            task_id = jnp.asarray(task_id, dtype=jnp.int32).reshape(-1)
            if task_id.shape[0] != x.shape[0]:
                raise ValueError(
                    "task_id must contain exactly one id per input image."
                )
            gates = jax_nn.one_hot(task_id, num_experts, dtype=x.dtype)
        elif self.routing_mode == "learned":
            router_logits = self.router(x, train=train)
            temperature = 1.5 if train else 1.0
            probabilities = nn.softmax(router_logits / temperature, axis=-1)
            if self.top_k:
                if not 1 <= self.top_k <= num_experts:
                    raise ValueError(
                        f"top_k must be in [1, {num_experts}], got {self.top_k}."
                    )
                if self.top_k < num_experts:
                    _, top_indices = jax.lax.top_k(router_logits, self.top_k)
                    mask = jnp.sum(
                        jax_nn.one_hot(top_indices, num_experts, dtype=x.dtype),
                        axis=-2,
                    )
                    probabilities = probabilities * mask
                    probabilities = probabilities / jnp.sum(
                        probabilities, axis=-1, keepdims=True
                    )
            gates = probabilities
        else:
            raise ValueError(
                f"Unknown routing_mode={self.routing_mode!r}; "
                "expected 'oracle' or 'learned'."
            )
        return gates

    def __call__(self, x, train=True, task_id=None):
        gates = self._route(x, train, task_id)
        num_experts = len(self.experts)

        # Preserve MAXIM's deep-supervision outputs and use only the last
        # decoder features as input to the expert reconstruction heads.
        outputs_all, feats = self.maxim(
            x, train=train, return_features=True
        )

        # Expert projections
        expert_outputs = []
        for expert in self.experts:
            expert_outputs.append(expert(feats))

        expert_outputs = jnp.stack(expert_outputs, axis=1)
        # [B, E, H, W, 3]

        # Mixture
        gates = gates[:, :, None, None, None]
        mixed = jnp.sum(gates * expert_outputs, axis=1)

        # Residual
        mixed = mixed + x

        # Replace only MAXIM's final full-resolution prediction. The remaining
        # stage/scale outputs keep their original computation graph and receive
        # auxiliary supervision during training.
        predictions = [list(stage_outputs) for stage_outputs in outputs_all]
        predictions[-1][-1] = mixed

        return predictions, gates.squeeze((2, 3, 4))
