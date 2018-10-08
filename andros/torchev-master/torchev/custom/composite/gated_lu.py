import torch
from torch.autograd import Variable
from torch.nn.modules import Module
from torch.nn.parameter import Parameter
from torch.nn import functional as F
from torch import nn
from torch.nn import init
from ...utils.helper import torchauto, tensorauto
from ...nn.modules.conv import Conv2dEv

from utilbox.config_util import ConfigParser

class GatedLinearUnit(Module) :
    def __init__(self, in_features,  out_features):
        super(GatedLinearUnit, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.gate = nn.Linear(in_features, out_features)

    def forward(self, x):
        """
        :param x: tensor with shape of [batch_size, in_features]
        :return: tensor with shape of [batch_size, out_features]
        applies linear(x) * sigmoid(gate(x))
        """
        x = self.linear(x) * F.sigmoid(self.gate(x))        
        return x

class GatedConv2dLinearUnit(Module) :
    def __init__(self, *args, **kwargs) :
        super(GatedConv2dLinearUnit, self).__init__()
        self.linear = Conv2dEv(*args, **kwargs)
        self.gate = Conv2dEv(*args, **kwargs)

    def forward(self, x) :
        x = self.linear(x) * F.sigmoid(self.gate(x))
        return x

    pass
