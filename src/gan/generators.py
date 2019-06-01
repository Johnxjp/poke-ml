import torch
import torch.nn as nn
import torch.nn.functional as F
from .helpers import deconv


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

    def __init__(self, z_size, conv_dim=32, is_greyscale=False):
        super().__init__()
        kernel_size = 4
        self.conv_dim = conv_dim
        self.output_dim = 1 if is_greyscale else 3
        self.fc = nn.Linear(z_size, conv_dim * 5 * 5 * 4)
        self.tconv1 = deconv(conv_dim * 4, conv_dim * 2, kernel_size)
        self.tconv2 = deconv(conv_dim * 2, conv_dim, kernel_size)
        self.tconv3 = deconv(
            conv_dim, self.output_dim, kernel_size, batch_norm=False)

    def forward(self, x):
        x = self.fc(x)
        x = x.view(x.size(0), self.conv_dim * 4, 5, 5)
        x = F.relu(self.tconv1(x))
        x = F.relu(self.tconv2(x))
        x = torch.tanh(self.tconv3(x))
        return x
