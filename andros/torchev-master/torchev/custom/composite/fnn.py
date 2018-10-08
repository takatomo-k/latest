import torch
from torch.autograd import Variable
from torch.nn.modules import Module
from torch.nn.parameter import Parameter
from torch.nn import functional as F
from torch import nn
from torch.nn import init
from ...utils.helper import torchauto, tensorauto

from utilbox.config_util import ConfigParser

class FNN(Module) :
    def __init__(self, in_features, hid_sizes=[512, 512], do=0.2, fn_act=F.leaky_relu):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(in_features, hid_sizes[0] if len(hid_sizes) > 0 else out_features))
        for ii in range(1, len(hid_sizes)) :
            self.layers.append(nn.Linear(hid_sizes[ii-1], hid_sizes[ii]))
            pass
        self.do = do
        self.fn_act = fn_act
        pass

    def forward(self, x):
        for ii in range(len(self.layers)) :
            x = self.layers[ii](x)
            x = self.fn_act(x)
            x = F.dropout(x, self.do, training=self.training)
        return x
    pass
