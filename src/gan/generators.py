import torch
import torch.nn as nn
import torch.nn.functional as F
from .helpers import deconv as BatchNormDeconv


class BasicFCGenerator(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout):
        super().__init__()

        self.layers = nn.ModuleList([])
        for d_out in hidden_dim:
            self.layers.append(nn.Linear(input_dim, d_out))
            input_dim = d_out

        self.out = nn.Linear(input_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        for l in self.layers:
            x = F.leaky_relu(l(x))
            x = self.dropout(x)

        return torch.tanh(self.out(x))


class DCGenerator(nn.Module):
    def __init__(
        self,
        z_size,
        output_w,
        output_h,
        n_layers=3,
        kernel_size=4,
        conv_dim=32,
        padding=1,
        is_rgb=True,
    ):
        super().__init__()
        """
        The image dimensions should be multiples of 2
        """
        self.input_w = output_w // (2 ** n_layers)
        self.input_h = output_h // (2 ** n_layers)
        self.conv_depth = conv_dim * (2 ** (n_layers - 1))
        fc_output_size = self.conv_depth * self.input_h * self.input_w

        self.deconv_layers = nn.ModuleList([])
        self.fc = nn.Linear(z_size, fc_output_size)
        input_channels = self.conv_depth
        output_channels = input_channels // 2
        for i in range(n_layers - 1):
            layer = BatchNormDeconv(
                input_channels, output_channels, kernel_size, padding=padding
            )
            self.deconv_layers.append(layer)
            input_channels = output_channels
            output_channels = input_channels // 2

        output_channels = 3 if is_rgb else 1
        self.final_conv = BatchNormDeconv(
            input_channels,
            output_channels,
            kernel_size,
            padding=padding,
            batch_norm=False,
        )

    def forward(self, x):
        x = self.fc(x)
        x = x.view(x.size(0), self.conv_depth, self.input_h, self.input_w)
        for deconv in self.deconv_layers:
            x = F.relu(deconv(x))
        x = torch.tanh(self.final_conv(x))
        return x
