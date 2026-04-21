"""
Three architectures are provided:

* ``LeNet5`` - a configurable LeNet-5 where the four OED factors (number of
  convolutional layers, filter size, dropout percentage, filter counts) can be
  varied.  Level 1 of every factor reproduces the original LeNet-5.
* ``AlexNet`` - the architecture described in Section II.B.2 of the paper
  (5 conv + 3 max-pool + 3 FC, softmax output of size 3, ReLU activations,
  120x120x4 input).
* ``resnet50`` - a thin wrapper around ``torchvision.models.resnet50`` adapted
  to a 3-class output, matching the paper's ResNet baseline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import torch
from torch import nn
from torchvision import models as tv_models


# ---------------------------------------------------------------------------
# LeNet-5 (configurable to match the OED factor levels)
# ---------------------------------------------------------------------------

@dataclass
class LeNetConfig:
    """Factors of the OED expressed as a LeNet-5 configuration.

    The paper varies four factors (A-D) with three levels each:

    * **A** number of convolutional layers: ``2`` (default LeNet-5), ``3`` or ``4``
    * **B** filter (kernel) size: ``5``, ``7`` or ``11``
    * **C** dropout probability inserted after each conv block: ``0``, ``0.2``
      or ``0.5``
    * **D** number of filters per conv layer.  The paper lists three level
      families ``(6, 16, [16])``, ``(24, 64, [64])``, ``(96, 256, [256])``; the
      bracketed number is reused for any additional layers beyond the first
      two when factor A is ``3`` or ``4``.
    """

    num_conv_layers: int = 2
    filter_size: int = 5
    dropout: float = 0.0
    filters: Sequence[int] = field(default_factory=lambda: (6, 16, 16))
    num_classes: int = 3
    input_channels: int = 3
    input_size: int = 32  # classic LeNet-5 input size
    fc_sizes: Sequence[int] = (120, 84)

    def filters_expanded(self) -> list[int]:
        """Return a full list of filter counts of length ``num_conv_layers``."""
        if self.num_conv_layers < 1:
            raise ValueError("num_conv_layers must be >= 1")
        base = list(self.filters)
        if not base:
            raise ValueError("filters must not be empty")
        # Ensure at least two reference values; third is used for any extras
        if len(base) == 1:
            base = base + [base[0], base[0]]
        elif len(base) == 2:
            base = base + [base[-1]]
        out = base[: self.num_conv_layers]
        while len(out) < self.num_conv_layers:
            out.append(base[2])
        return out


class LeNet5(nn.Module):
    """Configurable LeNet-5 style network.

    The architecture is identical to the classic LeNet-5 when the configuration
    is left at its default values.  Additional convolutional blocks are
    appended (each followed by 2x2 max-pooling until the spatial dimension
    drops below 2x2) when ``num_conv_layers`` is greater than 2.
    """

    def __init__(self, config: LeNetConfig | None = None) -> None:
        super().__init__()
        self.config = config or LeNetConfig()
        cfg = self.config

        filters = cfg.filters_expanded()
        padding = cfg.filter_size // 2

        conv_blocks: list[nn.Module] = []
        in_channels = cfg.input_channels
        spatial = cfg.input_size
        for idx, out_channels in enumerate(filters):
            conv_blocks.append(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=cfg.filter_size,
                    padding=padding,
                )
            )
            conv_blocks.append(nn.ReLU(inplace=True))
            # alternate avg-pool (classic LeNet) / max-pool (paper says 2x2
            # max-pool for OED tested combinations); keep max-pool for OED.
            if spatial >= 2:
                conv_blocks.append(nn.MaxPool2d(kernel_size=2, stride=2))
                spatial //= 2
            if cfg.dropout > 0:
                conv_blocks.append(nn.Dropout2d(p=cfg.dropout))
            in_channels = out_channels

        self.features = nn.Sequential(*conv_blocks)

        # Determine flattened feature size with a dry forward pass.
        with torch.no_grad():
            dummy = torch.zeros(
                1, cfg.input_channels, cfg.input_size, cfg.input_size
            )
            flat_dim = self.features(dummy).flatten(1).shape[1]

        classifier: list[nn.Module] = [nn.Flatten()]
        prev = flat_dim
        for hidden in cfg.fc_sizes:
            classifier.append(nn.Linear(prev, hidden))
            classifier.append(nn.ReLU(inplace=True))
            if cfg.dropout > 0:
                classifier.append(nn.Dropout(p=cfg.dropout))
            prev = hidden
        classifier.append(nn.Linear(prev, cfg.num_classes))
        self.classifier = nn.Sequential(*classifier)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        x = self.features(x)
        x = self.classifier(x)
        return x


# ---------------------------------------------------------------------------
# AlexNet (paper describes a compact AlexNet variant, input 120x120x4)
# ---------------------------------------------------------------------------

class AlexNet(nn.Module):
    """AlexNet-style CNN.

    * 5 convolutional layers, 3 max-pool layers, 3 fully-connected layers.
    * Input image is ``input_channels x input_size x input_size`` (120 by
      default; the paper uses 4 input channels, RGBA).
    * Dropout is applied after the first two FC layers to address overfitting.
    * Output is a log-softmax over ``num_classes`` classes (default 3:
      fire, water, grass).
    """

    def __init__(
        self,
        num_classes: int = 3,
        input_channels: int = 4,
        input_size: int = 120,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()

        self.features = nn.Sequential(
            # conv1 + pool1
            nn.Conv2d(input_channels, 96, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            # conv2 + pool2
            nn.Conv2d(96, 256, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            # conv3
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # conv4
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # conv5 + pool3
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, input_channels, input_size, input_size)
            flat_dim = self.features(dummy).flatten(1).shape[1]

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_dim, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(1024, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x


# ---------------------------------------------------------------------------
# ResNet50 wrapper (matches the paper's ResNet baseline)
# ---------------------------------------------------------------------------

def resnet50(
    num_classes: int = 3,
    input_channels: int = 3,
    pretrained: bool = False,
) -> nn.Module:
    """Return a ResNet-50 adapted to ``num_classes`` outputs.

    Parameters
    ----------
    num_classes:
        Number of output classes (3 in the paper).
    input_channels:
        Number of input channels.  If different from 3, the first conv layer
        is replaced with one that accepts the new channel count.
    pretrained:
        Whether to initialise from ImageNet weights.  Defaults to ``False``
        to reflect the low-data training scenario described in the paper.
    """

    weights = tv_models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
    model = tv_models.resnet50(weights=weights)
    if input_channels != 3:
        model.conv1 = nn.Conv2d(
            input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def build_model(name: str, **kwargs) -> nn.Module:
    """Factory used by the experiment runners.

    ``name`` is one of ``"lenet5"``, ``"alexnet"`` or ``"resnet50"``.
    """

    name = name.lower()
    if name in {"lenet", "lenet5", "lenet-5"}:
        config = kwargs.pop("config", None)
        if config is None:
            config = LeNetConfig(**kwargs)
        return LeNet5(config)
    if name == "alexnet":
        return AlexNet(**kwargs)
    if name in {"resnet", "resnet50"}:
        return resnet50(**kwargs)
    raise ValueError(f"Unknown model name: {name!r}")


__all__ = [
    "LeNetConfig",
    "LeNet5",
    "AlexNet",
    "resnet50",
    "build_model",
]
