import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
import torch.utils.data
import torch.utils.data.distributed
import numpy as np

num_hidden_units = 512


class Generator(nn.Module):
    def __init__(self,
                 z_dim,
                 nlabels,
                 size,
                 embed_size=256,
                 nfilter=64,
                 nfilter_max=512,
                 **kwargs):
        super().__init__()

        self.z_dim = z_dim
        self.activation = nn.ReLU()
        # self.activation = actvn
        self.layer1 = nn.Linear(z_dim, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, 2)

        init_factor = 1
        torch.nn.init.kaiming_uniform_(self.layer1.weight, init_factor)
        torch.nn.init.kaiming_uniform_(self.layer2.weight, init_factor)
        torch.nn.init.kaiming_uniform_(self.layer3.weight, init_factor)
        torch.nn.init.zeros_(self.layer1.bias)
        torch.nn.init.zeros_(self.layer2.bias)
        torch.nn.init.zeros_(self.layer3.bias)

    def forward(self, z, y):
        feature = self.activation(self.layer1(z))
        feature = self.activation(self.layer2(feature))
        feature = self.layer3(feature)
        return feature


class Discriminator(nn.Module):
    def __init__(self,
                 z_dim,
                 nlabels,
                 size,
                 embed_size=256,
                 nfilter=64,
                 nfilter_max=1024):
        super().__init__()

        self.activation = nn.ReLU()
        self.layer1 = nn.Linear(2, num_hidden_units)
        self.layer2 = nn.Linear(num_hidden_units, num_hidden_units)
        self.layer3 = nn.Linear(num_hidden_units, 1)

    def forward(self, z, y):
        feature = self.activation(self.layer1(z))
        feature = self.activation(self.layer2(feature))
        feature = self.layer3(feature)
        return feature


def actvn(x):
    out = F.leaky_relu(x, 0.)
    return out
