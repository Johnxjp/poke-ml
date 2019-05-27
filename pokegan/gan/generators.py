import torch.nn as nn


class SimpleFCGenerator(nn.Module):

    def __init__(self, noise_dim):
        super().__init__()

    def forward(self, x):
        pass


class PokeTypeFCGenerator(nn.Module):

    def __init__(self, noise_dim, n_types):
        super().__init__()

    def forward(self, x):
        pass
