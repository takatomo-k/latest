import torch
import numbers
from torch.nn.parameter import Parameter
from torch.nn.modules import Module
from torch.nn.functional import batch_norm
from functools import reduce
from operator import mul

class LayerNorm(Module):
    ''' Layer normalization module '''

    def __init__(self, shape, eps=1e-3):
        super(LayerNorm, self).__init__()

        self.eps = eps
        self.param_gain = Parameter(torch.ones(shape), requires_grad=True)
        self.param_bias = Parameter(torch.zeros(shape), requires_grad=True)

    def forward(self, z):
        if z.size(1) == 1:
            return z
        mu = torch.mean(z, dim=1, keepdim=True)
        sigma = torch.std(z, dim=1, keepdim=True)
        ln_out = (z - mu.expand_as(z)) / (sigma.expand_as(z) + self.eps)
        ln_out = self.param_gain * ln_out + self.param_bias
        return ln_out
