import math
import torch

from torch.nn import Module
from torch.nn import Parameter
from torch.autograd import Variable
from torch import nn
from torch.nn import functional as F
from ...utils.helper import torchauto

class Conv1dEv(Module) :
    def __init__(self, in_channels, out_channels, kernel_size, 
            stride=1, padding=0, dilation=1, groups=1, bias=True) :
        r"""
        A guide to convolution arithmetic for deep learning
        https://arxiv.org/pdf/1603.07285v1.pdf

        padding : ['same', 'valid', 'full']
        """
        super(Conv1dEv, self).__init__()
        self.cutoff = False
        if padding == 'valid' :
            padding = 0 
        elif padding == 'full' :
            padding = kernel_size-1
        elif padding == 'same' :
            padding = kernel_size // 2
            if kernel_size % 2 == 0 :
                self.cutoff = True
            pass
        self.conv_lyr = nn.Conv1d(in_channels, out_channels, kernel_size, 
                stride, padding, dilation, groups, bias)
        pass

    def forward(self, input) :
        output = self.conv_lyr(input)
        if self.cutoff :
            h = output.size(2) - 1
            output = output[:, :, 0:h]
        return output
    pass

class Conv2dEv(Module) :
    def __init__(self, in_channels, out_channels, kernel_size, 
            stride=1, padding=0, dilation=1, groups=1, bias=True) :
        r"""
        A guide to convolution arithmetic for deep learning
        https://arxiv.org/pdf/1603.07285v1.pdf

        padding : ['same', 'valid', 'full']
        """
        super(Conv2dEv, self).__init__()
        self.cutoff = [False, False]
        if isinstance(kernel_size, int) :
            kernel_size = (kernel_size, kernel_size)
        if padding == 'valid' :
            padding = (0, 0)
        elif padding == 'full' :
            padding = (kernel_size[0]-1, kernel_size[1]-1)
        elif padding == 'same' :
            padding = (kernel_size[0] // 2, kernel_size[1] // 2)
            for ii in range(len(kernel_size)) :
                if kernel_size[ii] % 2 == 0 :
                    self.cutoff[ii] = True
            pass
        self.conv_lyr = nn.Conv2d(in_channels, out_channels, kernel_size, 
                stride, padding, dilation, groups, bias)
        pass

    def forward(self, input) :
        output = self.conv_lyr(input)
        if any(self.cutoff) :
            h, w = output.size()[2:]
            if self.cutoff[0] :
                h -= 1
            if self.cutoff[1] :
                w -= 1
            output = output[:, :, 0:h, 0:w]
        return output
    pass
