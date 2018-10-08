import torch

from torch.autograd import Variable

from torchev.utils.seq_util import downsampling

x = Variable(torch.randn(5, 10, 3))

def test_downsampling_last() :
    assert torch.equal(downsampling({'type':'last', 'step':3}, x)[0].data, 
            x.data[:, 2::3])
    pass
