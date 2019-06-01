import torch
import torch.nn as nn


def deconv(
    in_channels,
    out_channels,
    kernel_size,
    stride=2,
    padding=1,
    batch_norm=True
):
    """
    New transpose convolutional layer with embedded batch norm
    """

    layers = []
    layers.append(
        nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size, stride=stride,
            padding=padding, bias=False
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
    batch_norm=True
):
    """
    New transpose convolutional layer with embedded batch norm
    """

    layers = []
    layers.append(
        nn.Conv2d(
            in_channels, out_channels, kernel_size, stride=stride,
            padding=padding, bias=False
        )
    )

    if batch_norm:
        layers.append(nn.BatchNorm2d(out_channels))

    return nn.Sequential(*layers)


class GaussianNoise(nn.Module):
    def __init__(self, stdev=0.1):
        super().__init__()
        self.stdev = stdev
        self.noise = torch.tensor(0, requires_grad=False)

    def foward(self, x):
        if self.training:
            noise = self.noise.repeat(*x.size()).normal_(std=self.stdev)
            return x + noise
        return x
