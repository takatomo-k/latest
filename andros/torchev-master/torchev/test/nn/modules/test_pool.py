import pytest
import numpy as np
import torch
from torch.autograd import Variable
from torchev.nn.modules.pool import MaxPool1dEv, MaxPool2dEv

# test 2d #
BATCH = 2
IN_CH = 3
OUT_CH = 6
H = 4
W = 4
def gen_pool1d_input(batch, in_ch, h) :
    return Variable(torch.randn(batch, in_ch, h))
def gen_pool2d_input(batch, in_ch, h, w) :
    return Variable(torch.randn(batch, in_ch, h, w))

class TestMaxPool1dEv :
    def test_pool1ev_same(self) :
        x = gen_pool1d_input(BATCH, IN_CH, H)
        pool = MaxPool1dEv(kernel_size=2, stride=1, padding='same')
        y = pool(x)
        assert list(y.size()) == [BATCH, IN_CH, H]

    def test_pool1ev_same_odd(self) :
        x = gen_pool1d_input(BATCH, IN_CH, H)    
        pool = MaxPool1dEv(kernel_size=3, stride=1, padding='same')
        y = pool(x)
        assert list(y.size()) == [BATCH, IN_CH, H]
        pass

class TestMaxPool2dEv :
    def test_pool2ev_same(self) :
        x = gen_pool2d_input(BATCH, IN_CH, H, W)    
        pool = MaxPool2dEv(kernel_size=[2, 2], stride=1, padding='same')
        y = pool(x)
        assert list(y.size()) == [BATCH, IN_CH, H, W]
        pass

    def test_pool2ev_same_odd(self) :
        x = gen_pool2d_input(BATCH, IN_CH, H, W)    
        pool = MaxPool2dEv(kernel_size=[3, 3], stride=1, padding='same')
        y = pool(x)
        assert list(y.size()) == [BATCH, IN_CH, H, W]
        pass
