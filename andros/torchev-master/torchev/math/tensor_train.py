import torch
import numpy as np

def tt_dot(in_modes, out_modes, ranks, input, weight, bias=None) :
    assert len(in_modes) == len(out_modes) == len(ranks)-1
    assert input.shape[1] == np.prod(in_modes)
    res = input
    res = res.view(-1, int(np.prod(in_modes)))
    res = res.transpose(1, 0)
    res = res.contiguous()
    dim = len(in_modes)
    for ii in range(dim) :
        res = res.view(ranks[ii] * in_modes[ii], -1)
        res = torch.matmul(weight[ii], res)
        res = res.view(out_modes[ii], -1)
        res = res.transpose(1, 0)
        res = res.contiguous()
    res = res.view(-1, int(np.prod(out_modes)))

    if bias is not None :
        res += bias
    return res
