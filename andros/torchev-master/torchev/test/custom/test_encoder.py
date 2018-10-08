import pytest
import numpy as np

import torch
from torch.nn import functional as F
from torch.autograd import Variable
from torchev.custom import StandardRNNEncoder

BATCH, MAX_ENC_LEN, ENC_DIM = 5, 10, 2
DEC_DIM = 3
ENC_LEN = [6, 4, 2, 10, 7]
DEC_IN_SIZE = 3

def create_ctx() :
    return Variable(torch.randn(BATCH, MAX_ENC_LEN, ENC_DIM))

def create_state() :
    return Variable(torch.randn(BATCH, DEC_DIM))

def create_mask() :
    mask = torch.ByteTensor(BATCH, MAX_ENC_LEN).fill_(False)
    for ii in range(BATCH) :
        mask[ii, 0:ENC_LEN[ii]] = True
    mask = Variable(mask)
    return mask

def create_dec_in() :
    return Variable(torch.randn(BATCH, DEC_IN_SIZE))
def create_dec_mask() :
    return Variable(torch.FloatTensor(BATCH).bernoulli_().float())

class TestStandardRNNEncoder :
    def test_encoder_standard(self) :
        x = create_ctx()
        enc_mdl = StandardRNNEncoder(ENC_DIM, [5, 5])
        res = enc_mdl(x)
        pass
    pass
