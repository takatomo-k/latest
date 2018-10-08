import pytest
import torch
from torch.autograd import Variable
from torchev.math.function import where

import numpy as np

def test_where() :
    c = Variable(torch.FloatTensor(100, 500).bernoulli_())
    x = Variable(torch.FloatTensor(100, 500).normal_(), requires_grad=True)
    y = Variable(torch.FloatTensor(100, 500).normal_(), requires_grad=True)
    res = where(c, x, y) 
    cost = res.sum()
    cost.backward()
    # import ipdb; ipdb.set_trace()
    assert np.all((x.grad.data.numpy() != 0) == (c.data.numpy()))
    assert np.all((y.grad.data.numpy() != 0) == (c.data.numpy() == 0))
    pass
