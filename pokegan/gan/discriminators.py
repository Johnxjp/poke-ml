import torch.nn as nn
import torch.nn.functional as F


class FCDiscriminator(nn.Module):

    def __init__(self, input_dim, hdim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hdim)
        self.fc2 = nn.Linear(hdim, 1)

    def forward(self, x):
        # TODO: Flatten inside or outside?
        self.x = F.leaky_relu(self.fc1(x))
        return self.fc2(x)
