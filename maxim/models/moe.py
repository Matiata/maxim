from maxim import MAXIM
from router import RouterModel
import functools
from flax import linen as nn
import jax.numpy as jnp

Conv3x3 = functools.partial(nn.Conv, kernel_size=(3, 3))
Conv1x1 = functools.partial(nn.Conv, kernel_size=(1, 1))

class ExpertHead(nn.Module):
    out_channels: int = 3
    use_bias: bool = True

    @nn.compact
    def __call__(self, x):
        x = Conv3x3(self.out_channels, use_bias=self.use_bias)(x)
        return x

class MaximMoE(nn.Module):
    maxim: MAXIM
    router: RouterModel
    experts: list[ExpertHead]

    def __call__(self, x, train=True):
        # Router on the image
        router_logits = self.router(x, train=train)
        gates = nn.softmax(router_logits, axis=-1)  # [B, E]

        # MAXIM features
        outputs_all = self.maxim(x, train=train)
        feats = outputs_all[-1][-1]  # [B, H, W, C]

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

        return mixed, gates
