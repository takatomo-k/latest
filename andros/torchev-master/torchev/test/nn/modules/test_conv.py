import pytest
import numpy as np
import torch
from torch.autograd import Variable
from torchev.nn.modules.conv import Conv1dEv, Conv2dEv

# test 2d #
BATCH = 2
IN_CH = 3
OUT_CH = 6
H = 4
W = 4
def gen_conv1d_input(batch, in_ch, h) :
    return Variable(torch.randn(batch, in_ch, h))
def gen_conv2d_input(batch, in_ch, h, w) :
    return Variable(torch.randn(batch, in_ch, h, w))

class TestConv1dEv :
    def test_conv1ev_same(self) :
        x = gen_conv1d_input(BATCH, IN_CH, H)
        conv = Conv1dEv(IN_CH, OUT_CH, kernel_size=2, padding='same')
        y = conv(x)
        assert list(y.size()) == [BATCH, OUT_CH, H]

    def test_conv1ev_same_odd(self) :
        x = gen_conv1d_input(BATCH, IN_CH, H)    
        conv = Conv1dEv(IN_CH, OUT_CH, kernel_size=3, padding='same')
        y = conv(x)
        assert list(y.size()) == [BATCH, OUT_CH, H]
        pass

class TestConv2dEv :
    def test_conv2ev_same(self) :
        x = gen_conv2d_input(BATCH, IN_CH, H, W)    
        conv = Conv2dEv(IN_CH, OUT_CH, kernel_size=[2, 2], padding='same')
        y = conv(x)
        assert list(y.size()) == [BATCH, OUT_CH, H, W]
        pass

    def test_conv2ev_same_odd(self) :
        x = gen_conv2d_input(BATCH, IN_CH, H, W)    
        conv = Conv2dEv(IN_CH, OUT_CH, kernel_size=[3, 2], padding='same')
        y = conv(x)
        assert list(y.size()) == [BATCH, OUT_CH, H, W]
        pass
