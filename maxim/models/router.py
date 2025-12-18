import functools
from typing import Tuple
from flax import linen as nn

Conv3x3 = functools.partial(nn.Conv, kernel_size=(3, 3))
Conv1x1 = functools.partial(nn.Conv, kernel_size=(1, 1))

# -----------------------------
# Model: small DW-Conv backbone
# -----------------------------
class DepthwiseConv(nn.Module):
    kernel_size: Tuple[int, int]
    strides: Tuple[int, int] = (1, 1)
    padding: str = "SAME"

    @nn.compact
    def __call__(self, x):
        in_ch = x.shape[-1]
        # Flax conv expects (N, H, W, C) by default
        # We'll use `feature_group_count=in_ch` for depthwise
        x = nn.Conv(
            features=in_ch,
            kernel_size=self.kernel_size,
            strides=self.strides,
            padding=self.padding,
            feature_group_count=in_ch,
            use_bias=False,
        )(x)
        return x


class DwSepBlock(nn.Module):
    out_ch: int
    stride: int = 1

    @nn.compact
    def __call__(self, x, train: bool = True):
        # Depthwise
        x = DepthwiseConv(kernel_size=(3, 3), strides=(self.stride, self.stride))(x)
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = nn.relu(x)
        # Pointwise
        x = Conv1x1(features=self.out_ch, use_bias=False)(x)
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = nn.relu(x)
        return x


class SmallBackbone(nn.Module):
    """Small lightweight backbone producing a global-pooled embedding.

    Input shape: (N, H, W, C) with C=3
    Output: (N, embedding_dim)
    """

    embedding_dim: int = 576

    @nn.compact
    def __call__(self, x, train: bool = True):
        # x: NHWC
        assert x.ndim == 4
        # initial conv
        x = Conv3x3(
            features=32,
            strides=(2, 2),
            padding="SAME",
            use_bias=False,
        )(x)
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = nn.relu(x)

        # a few depthwise separable blocks
        x = DwSepBlock(out_ch=64, stride=1)(x, train=train)
        x = DwSepBlock(out_ch=96, stride=2)(x, train=train)
        x = DwSepBlock(out_ch=160, stride=2)(x, train=train)
        x = DwSepBlock(out_ch=self.embedding_dim, stride=2)(x, train=train)

        # global avg pool
        # x shape: (N, H, W, C)
        x = x.mean(axis=(1, 2))  # (N, C)
        return x


class RouterHead(nn.Module):
    num_classes: int = 5

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        x = nn.Dense(128)(x)
        x = nn.relu(x)
        x = nn.Dense(self.num_classes)(x)
        return x

class RouterModel(nn.Module):
    embedding_dim: int = 576
    num_classes: int = 5

    def setup(self):
        self.backbone = SmallBackbone(embedding_dim=self.embedding_dim)
        self.head = RouterHead(num_classes=self.num_classes)

    def __call__(self, x, train: bool = True):
        feats = self.backbone(x, train=train)
        logits = self.head(feats)
        return logits