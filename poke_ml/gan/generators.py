import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import deconv as BatchNormDeconv
from ..datatypes import Tensor


class BasicFCGenerator(nn.Module):
    def __init__(
        self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float
    ) -> None:
        super().__init__()

        self.layers = nn.ModuleList([])
        for d_out in hidden_dim:
            self.layers.append(nn.Linear(input_dim, d_out))
            input_dim = d_out

        self.out = nn.Linear(input_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        for l in self.layers:
            x = F.leaky_relu(l(x))
            x = self.dropout(x)

        return torch.tanh(self.out(x))


class DCGenerator(nn.Module):
    def __init__(
        self,
        z_size: int,
        output_w: int,
        output_h: int,
        n_layers: int = 3,
        kernel_size: int = 4,
        conv_dim: int = 32,
        padding: int = 1,
        is_rgb: bool = True,
    ) -> None:
        super().__init__()
        """
        The image dimensions should be multiples of 2
        """
        self.input_w = output_w // (2 ** n_layers)
        self.input_h = output_h // (2 ** n_layers)
        self.conv_depth = conv_dim * (2 ** (n_layers - 1))
        fc_output_size = self.conv_depth * self.input_h * self.input_w

        self.fc = nn.Linear(z_size, fc_output_size)
        self.lrelu_slope = 0.1
        self.lrelu = nn.LeakyReLU(self.lrelu_slope)
        self.deconv_layers = nn.ModuleList([])
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

    def init_weights(self) -> None:
        def kaiming_init(weight):
            nn.init.kaiming_normal_(weight, a=self.lrelu_slope)

        kaiming_init(self.deconv_layers[0][0].weight)
        kaiming_init(self.deconv_layers[1][0].weight)

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc(x)
        x = x.view(x.size(0), self.conv_depth, self.input_h, self.input_w)
        for deconv in self.deconv_layers:
            x = self.lrelu(deconv(x))
        x = torch.tanh(self.final_conv(x))
        return x


class CGAN(nn.Module):
    def __init__(
        self,
        n_labels: int,
        z_size: int,
        output_w: int,
        output_h: int,
        embedding_dim: int = 100,
        n_layers: int = 3,
        kernel_size: int = 4,
        conv_dim: int = 32,
        padding: int = 1,
        is_rgb: bool = True,
    ) -> None:

        super().__init__()
        self.input_w = output_w // (2 ** n_layers)
        self.input_h = output_h // (2 ** n_layers)
        self.conv_depth = conv_dim * (2 ** (n_layers - 1))

        # Sizes have been determined to ensure the correct output shape of
        # the image
        fc_output_size = self.conv_depth * self.input_h * self.input_w

        self.embedding = nn.Embedding(n_labels, embedding_dim)
        self.proj1 = nn.Linear(embedding_dim, fc_output_size // 2)
        self.proj2 = nn.Linear(z_size, 200)
        self.proj3 = nn.Linear(200, fc_output_size // 2)
        self.proj4 = nn.Linear(fc_output_size, fc_output_size)
        self.lrelu_slope = 0.1
        self.lrelu = nn.LeakyReLU(self.lrelu_slope)

        self.deconv_layers = nn.ModuleList([])
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

    def init_weights(self) -> None:
        def kaiming_init(weight):
            nn.init.kaiming_normal_(weight, a=self.lrelu_slope)

        kaiming_init(self.proj1.weight)
        kaiming_init(self.proj2.weight)
        kaiming_init(self.proj3.weight)
        kaiming_init(self.proj4.weight)
        kaiming_init(self.deconv_layers[0][0].weight)
        kaiming_init(self.deconv_layers[1][0].weight)

    def forward(self, z: Tensor, y: torch.LongTensor) -> Tensor:
        label_embedding = self.embedding(y.squeeze())
        x1 = self.lrelu(self.proj1(label_embedding))
        x2 = self.proj2(z)
        x2 = self.proj3(x2)
        x = torch.cat((x1, x2), dim=-1)
        x = self.proj4(x)
        x = x.view(x.size(0), self.conv_depth, self.input_h, self.input_w)
        for deconv in self.deconv_layers:
            x = self.lrelu(deconv(x))
        x = torch.tanh(self.final_conv(x))
        return x
