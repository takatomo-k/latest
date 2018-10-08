import pytest
import numpy as np

import torch
from torch.nn import functional as F
from torch.autograd import Variable
from torchev.custom.composite.cbhg import CBHG1d

ENC_LEN = [300, 100, 80, 70, 3]
BATCH, MAX_ENC_LEN, ENC_DIM = 5, max(ENC_LEN), 128 

def test_cbhg() :
    input = Variable(torch.randn(BATCH, MAX_ENC_LEN, ENC_DIM)*100)
    l_cbhg = CBHG1d(ENC_DIM)
    res = l_cbhg(input, ENC_LEN)
    pass

def test_cbhg_mask() :
    input = Variable(torch.randn(BATCH, MAX_ENC_LEN, ENC_DIM))
    l_cbhg = CBHG1d(ENC_DIM, use_bn=False)
    res = l_cbhg(input, ENC_LEN)

    for ii in range(BATCH) :
        res_ii = l_cbhg(input[ii:ii+1, 0:ENC_LEN[ii]], ENC_LEN[ii:ii+1])
        assert np.allclose(res_ii.data.numpy().sum(), res[ii].data.numpy().sum())
    pass
