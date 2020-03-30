import torch
import torch.nn as nn


def deconv(
    in_channels,
    out_channels,
    kernel_size,
    stride=2,
    padding=1,
    batch_norm=True,
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
    in_channels,
    out_channels,
    kernel_size,
    stride=2,
    padding=1,
    batch_norm=True,
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
    def __init__(self, mean=0, std=1):
        super().__init__()
        self.std = std
        self.mean = mean

    def forward(self, x):
        if self.training:
            noise = torch.zeros_like(x).normal_(mean=0, std=self.std)
            return x + noise
        return x
