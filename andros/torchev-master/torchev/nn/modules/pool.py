import math
import torch

from torch.nn import Module
from torch.nn import Parameter
from torch.autograd import Variable
from torch import nn
from torch.nn import functional as F
from ...utils.helper import torchauto


class MaxPool1dEv(Module) :
    def __init__(self, kernel_size, stride=None, padding=0, 
            dilation=1, return_indices=False, ceil_mode=False) :
        r"""
        A guide to convolution arithmetic for deep learning
        https://arxiv.org/pdf/1603.07285v1.pdf

        padding : ['same', 'valid', 'full']
        """
        super(MaxPool1dEv, self).__init__()
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
        self.pool_lyr = nn.MaxPool1d(kernel_size, stride=stride, 
                padding=padding, dilation=dilation, 
                return_indices=return_indices, ceil_mode=ceil_mode)
        pass

    def forward(self, input) :
        output = self.pool_lyr(input)
        if self.cutoff :
            h = output.size(2)
            if self.cutoff :
                h -= 1
            output = output[:, :, 0:h]
        return output

class MaxPool2dEv(Module) :
    def __init__(self, kernel_size, stride=None, padding=0, 
            dilation=1, return_indices=False, ceil_mode=False) :
        r"""
        A guide to convolution arithmetic for deep learning
        https://arxiv.org/pdf/1603.07285v1.pdf

        padding : ['same', 'valid', 'full']
        """
        super(MaxPool2dEv, self).__init__()
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
        self.pool_lyr = nn.MaxPool2d(kernel_size, stride=stride, 
                padding=padding, dilation=dilation, 
                return_indices=return_indices, ceil_mode=ceil_mode)
        pass

    def forward(self, input) :
        output = self.pool_lyr(input)
        if any(self.cutoff) :
            h, w = output.size()[2:]
            if self.cutoff[0] :
                h -= 1
            if self.cutoff[1] :
                w -= 1
            output = output[:, :, 0:h, 0:w]
        return output
    pass
