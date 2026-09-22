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
