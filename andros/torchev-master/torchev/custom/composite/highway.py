import torch
from torch.autograd import Variable
from torch.nn.modules import Module
from torch.nn.parameter import Parameter
from torch.nn import functional as F
from torch import nn
from torch.nn import init
from ...utils.helper import torchauto, tensorauto

from utilbox.config_util import ConfigParser

class HighwayFNN(Module) :
    def __init__(self, out_features, num_layers, fn_act=F.leaky_relu):
        super(HighwayFNN, self).__init__()
        self.num_layers = num_layers
        self.nonlinear = nn.ModuleList([nn.Linear(out_features, out_features) for _ in range(num_layers)])
        self.gate = nn.ModuleList([nn.Linear(out_features, out_features) for _ in range(num_layers)])
        self.fn_act = fn_act

    def forward(self, x):
        """
        :param x: tensor with shape of [batch_size, size]
        :return: tensor with shape of [batch_size, size]
        applies sigm(x) * f(G(x)) + (1 - sigm(x)) * Q(x) transformation | G and Q is affine transformation,
        f is non-linear transformation, sigm(x) is affine transformation with sigmoid non-linearity
        and * is element-wise multiplication
        """
        
        for layer in range(self.num_layers):
            gate = F.sigmoid(self.gate[layer](x))
            nonlinear = self.fn_act(self.nonlinear[layer](x))
            x = gate * nonlinear + (1 - gate) * x 

        return x

