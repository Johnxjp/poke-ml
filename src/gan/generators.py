import torch
import torch.nn as nn
import torch.nn.functional as F


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


class PokeTypeFCGenerator(nn.Module):

    def __init__(self, noise_dim, n_types):
        super().__init__()

    def forward(self, x):
        pass
