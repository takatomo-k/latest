import torch
from torch.autograd import Variable
from torch.nn.modules import Module
from torch.nn.parameter import Parameter
from torch.nn import functional as F
from torch import nn
from torch.nn import init
from ...utils.helper import torchauto, tensorauto
from ...nn.modules.conv import Conv1dEv, Conv2dEv

from utilbox.config_util import ConfigParser

class MultiscaleConvNd(Module) :
    """
    Ref : 
    
    Tips : better set padding 'same' and no stride
    """
    def __init__(self, conv_nd, in_channels, out_channels, kernel_sizes,
            stride=1, padding='same', dilation=1, groups=1, bias=True) :
        super(MultiscaleConvNd, self).__init__()
        assert isinstance(kernel_sizes, (list, tuple))
        out_channels = ConfigParser.list_parser(out_channels, len(kernel_sizes))
        self.out_channels = sum(out_channels)
        
        self.multi_convnd = nn.ModuleList()
        for ii in range(len(kernel_sizes)) :
            if conv_nd == 1 :
                _conv_nd_lyr = Conv1dEv
            elif conv_nd == 2 :
                _conv_nd_lyr = Conv2dEv
            self.multi_convnd.append(_conv_nd_lyr(in_channels, out_channels[ii], 
                kernel_sizes[ii], stride, padding, dilation, groups, bias))
        pass

    def forward(self, input) :
        result = []
        for ii in range(len(self.multi_convnd)) :
            result.append(self.multi_convnd[ii](input))
        result = torch.cat(result, 1) # combine in filter axis  #
        return result
        pass

class MultiscaleConv1d(MultiscaleConvNd) :
    def __init__(self, in_channels, out_channels, kernel_sizes,
            stride=1, padding='same', dilation=1, groups=1, bias=True) :
        super(MultiscaleConv1d, self).__init__(1, in_channels, out_channels, 
                kernel_sizes, stride, padding, dilation, groups, bias)

class MultiscaleConv2d(MultiscaleConvNd) :
    def __init__(self, in_channels, out_channels, kernel_sizes,
            stride=1, padding='same', dilation=1, groups=1, bias=True) :
        super(MultiscaleConv2d, self).__init__(2, in_channels, out_channels, 
                kernel_sizes, stride, padding, dilation, groups, bias)
