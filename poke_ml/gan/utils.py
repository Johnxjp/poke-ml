import torch
import torch.nn as nn

from ..datatypes import Tensor


def deconv(
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    stride: int = 2,
    padding: int = 1,
    batch_norm: bool = True,
):
    """
    New transpose convolutional layer with embedded batch norm
    """

    layers = []
    layers.append(
        nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
    )

    if batch_norm:
        layers.append(nn.BatchNorm2d(out_channels))

    return nn.Sequential(*layers)


def conv(
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    stride: int = 2,
    padding: int = 1,
    batch_norm: bool = True,
):
    """
    New transpose convolutional layer with embedded batch norm
    """

    layers = []
    layers.append(
        nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
    )

    if batch_norm:
        layers.append(nn.BatchNorm2d(out_channels))

    return nn.Sequential(*layers)


class GaussianNoise(nn.Module):
    def __init__(self, mean: float = 0, std: float = 1) -> None:
        super().__init__()
        self.std = std
        self.mean = mean

    def forward(self, x: Tensor) -> Tensor:
        if self.training:
            noise = torch.zeros_like(x).normal_(mean=0, std=self.std)
            return x + noise
        return x
