import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import conv as BatchNormConv
from .utils import GaussianNoise


class FCDiscriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout):
        super().__init__()

        self.layers = nn.ModuleList([])
        for d_out in hidden_dim:
            self.layers.append(nn.Linear(input_dim, d_out))
            input_dim = d_out

        self.out = nn.Linear(input_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        for l in self.layers:
            x = F.leaky_relu(l(x))
            x = self.dropout(x)

        return self.out(x)


class DCDiscriminator(nn.Module):
    def __init__(
        self,
        input_h,
        input_w,
        n_layers=3,
        kernel_size=4,
        conv_dim=32,
        padding=1,
        is_rgb=True,
    ):
        """
        Specify the input dimension of images. This is to compute the final
        output size.

        Also, specify n_layers which is the number of convolutional operations
        to perform.

        conv_dim is the size multiplier of the output channels. The output
        channels are doubled each layer starting at conv_dim
        """
        super().__init__()
        self.lrelu_slope = 0.1
        self.lrelu = nn.LeakyReLU(self.lrelu_slope)
        self.noise = GaussianNoise()

        input_channels = 3 if is_rgb else 1
        output_channels = conv_dim

        self.conv_layers = nn.ModuleList([])
        for i in range(n_layers):
            batch_norm = True if i > 0 else False
            layer = BatchNormConv(
                input_channels,
                output_channels,
                kernel_size,
                padding=padding,
                batch_norm=batch_norm,
            )
            self.conv_layers.append(layer)
            input_channels = output_channels
            output_channels = output_channels * 2

        # Height and width are halved each conv layer
        output_w = input_w // (2 ** n_layers)
        output_h = input_h // (2 ** n_layers)
        fc_output_size = input_channels * output_h * output_w
        self.fc = nn.Linear(fc_output_size, 1)

    def init_weights(self):
        def kaiming_init(weight):
            nn.init.kaiming_normal_(weight, a=self.lrelu_slope)

        kaiming_init(self.conv_layers[0][0].weight)
        kaiming_init(self.conv_layers[1][0].weight)
        kaiming_init(self.conv_layers[2][0].weight)

    def forward(self, x):
        x = self.noise(x)
        for conv in self.conv_layers:
            x = self.lrelu(conv(x))
        x = x.view(x.size(0), -1)
        return self.fc(x)


class CDim(nn.Module):
    def __init__(
        self,
        n_labels: int,
        embedding_dim: int,
        input_h: int,
        input_w: int,
        n_layers=3,
        kernel_size=4,
        conv_dim=32,
        padding=1,
        is_rgb=True,
    ):
        """
        Specify the input dimension of images. This is to compute the final
        output size.

        Also, specify n_layers which is the number of convolutional operations
        to perform.

        conv_dim is the size multiplier of the output channels. The output
        channels are doubled each layer starting at conv_dim
        """
        super().__init__()
        self.lrelu_slope = 0.1
        self.lrelu = nn.LeakyReLU(self.lrelu_slope)
        self.conv_layers = nn.ModuleList([])
        self.noise = GaussianNoise()

        self.image_h = input_h
        self.image_w = input_w
        self.embedding = nn.Embedding(n_labels, embedding_dim)
        self.proj1 = nn.Linear(embedding_dim, input_h * input_w)

        input_channels = (3 if is_rgb else 1) + 1  # +1 is for concat
        output_channels = conv_dim

        # WHAT A MESS! FIX
        for i in range(n_layers):
            batch_norm = True if i > 0 else False
            layer = BatchNormConv(
                input_channels,
                output_channels,
                kernel_size,
                padding=padding,
                batch_norm=batch_norm,
            )
            self.conv_layers.append(layer)
            input_channels = output_channels
            output_channels = output_channels * 2

        # Height and width are halved each conv layer
        output_w = input_w // (2 ** n_layers)
        output_h = input_h // (2 ** n_layers)
        fc_output_size = input_channels * output_h * output_w
        self.fc = nn.Linear(fc_output_size, 1)

    def init_weights(self):
        def kaiming_init(weight):
            nn.init.kaiming_normal_(weight, a=self.lrelu_slope)

        kaiming_init(self.proj1.weight)
        kaiming_init(self.conv_layers[0][0].weight)
        kaiming_init(self.conv_layers[1][0].weight)
        kaiming_init(self.conv_layers[2][0].weight)

    def forward(self, x, y):
        y = self.embedding(y)
        y = self.lrelu(self.proj1(y))
        y = y.reshape(y.size(0), 1, self.image_h, self.image_w)
        x = torch.cat((x, y), dim=1)
        x = self.noise(x)
        for conv in self.conv_layers:
            x = self.lrelu(conv(x))
        x = x.view(x.size(0), -1)
        return self.fc(x)
