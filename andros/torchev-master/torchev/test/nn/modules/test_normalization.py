import pytest
import numpy as np
import torch
from torch.autograd import Variable
from torchev.nn.modules.normalization import LayerNorm

def test_layernorm() :
    input = Variable(torch.rand(5, 10)*3+2)
    lyr = LayerNorm(10)
    assert np.allclose(lyr(input).mean().data[0], 0, atol=1e-3)
    assert np.allclose(lyr(input).std().data[0], 1, atol=0.05)
