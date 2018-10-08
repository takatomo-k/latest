import math
import torch

from torch.nn import Module
from torch.nn import Parameter
from torch.autograd import Variable
from torch import nn
from torch.nn import functional as F

class MOELinear(Module) :
    def __init__(self, in_features, out_features, n_experts, bias=True) :
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.n_experts = n_experts

        self.linears = nn.ModuleList([])
        for ii in range(n_experts) :

            pass
        raise NotImplementedError()
