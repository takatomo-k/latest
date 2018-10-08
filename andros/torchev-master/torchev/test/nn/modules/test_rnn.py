import pytest
import numpy as np
import torch
from torch.autograd import Variable
from torchev.nn.modules import HMLSTMCell

NLEN = 10
NDIM = 2
BATCH = 3
NHID = 1

class TestHMLSTMBackend :
    def test_hmsltm(self) :
        x = Variable(torch.FloatTensor(NLEN, BATCH, NDIM).normal_())
        h_l_tm1 = Variable(torch.FloatTensor(BATCH, NHID).fill_(1))
        h_lp1_tm1 = Variable(torch.FloatTensor(BATCH, NHID).fill_(1))
        c_tm1 = Variable(torch.FloatTensor(BATCH, NHID).zero_())
        z_l_tm1 = Variable(torch.FloatTensor(BATCH).fill_(1))
        z_lm1_t = Variable(torch.FloatTensor(BATCH).fill_(1))
        hmlstm = HMLSTMCell(NDIM, NHID)
        h_1, c_1, z_1 = hmlstm(x[0], (h_l_tm1, h_lp1_tm1, c_tm1, z_l_tm1, z_lm1_t))
        res = ((1-h_1)**2).sum()
        res.backward()
        pass
