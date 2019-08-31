import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
import torch.utils.data
import torch.utils.data.distributed
import numpy as np


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
        self.activation = nn.Tanh()
        self.layer1 = nn.Linear(z_dim, 512)
        self.layer2 = nn.Linear(512, 512)
        self.layer3 = nn.Linear(512, 2)

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

        self.activation = nn.Tanh()
        self.layer1 = nn.Linear(2, 512)
        self.layer2 = nn.Linear(512, 512)
        self.layer3 = nn.Linear(512, 1)

    def forward(self, z, y):
        feature = self.activation(self.layer1(z))
        feature = self.activation(self.layer2(feature))
        feature = self.layer3(feature)
        return feature


def actvn(x):
    out = F.leaky_relu(x, 2e-1)
    return out
