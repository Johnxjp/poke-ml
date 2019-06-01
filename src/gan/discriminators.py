import torch.nn as nn
import torch.nn.functional as F
from .helpers import conv, GaussianNoise


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

    def __init__(self, conv_dim=32, is_greyscale=False):
        super().__init__()
        kernel_size = 4
        self.noise = GaussianNoise(0.2)
        self.conv_dim = conv_dim
        self.input_dim = 1 if is_greyscale else 3
        self.conv1 = conv(
            self.input_dim, conv_dim, kernel_size, batch_norm=False)
        self.conv2 = conv(conv_dim, conv_dim * 2, kernel_size)
        self.conv3 = conv(conv_dim * 2, self.conv_dim * 4, kernel_size)

        self.fc = nn.Linear(conv_dim * 5 * 5 * 4, 1)

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x), 0.2)
        x = F.leaky_relu(self.conv2(x), 0.2)
        x = F.leaky_relu(self.conv3(x), 0.2)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x
