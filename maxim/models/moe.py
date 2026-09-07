import functools

from flax import linen as nn
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


class MaximMoE(nn.Module):
    """MAXIM with deep supervision and an MoE final reconstruction head."""

    maxim: MAXIM
    router: RouterModel
    experts: list[ExpertHead]
    routing_mode: str = "learned"

    def __call__(self, x, train=True, task_id=None):
        num_experts = len(self.experts)
        if self.routing_mode == "oracle":
            if task_id is None:
                raise ValueError("task_id is required for oracle routing.")
            task_id = jnp.asarray(task_id, dtype=jnp.int32).reshape(-1)
            if task_id.shape[0] != x.shape[0]:
                raise ValueError(
                    "task_id must contain exactly one id per input image."
                )
            # The dataset task index is the expert index. This deterministic
            # routing removes the learned router from the oracle experiment.
            gates = jax_nn.one_hot(task_id, num_experts, dtype=x.dtype)  # [B, E]
        elif self.routing_mode == "learned":
            router_logits = self.router(x, train=train)
            temperature = 1.5 if train else 1.0
            gates = nn.softmax(router_logits / temperature, axis=-1)  # [B, E]
        else:
            raise ValueError(
                f"Unknown routing_mode={self.routing_mode!r}; "
                "expected 'oracle' or 'learned'."
            )

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
