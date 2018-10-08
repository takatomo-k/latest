import torch
from torch.autograd import Variable
from .helper import torchauto, tensorauto

def downsampling(cfg, mat, mat_len=None, axis=1) :

    assert axis<mat.dim() >= axis

    if cfg['type'] == 'none' :
        return mat, mat_len
    elif cfg['type'] == 'last' :
        """
        take only last variable every n-step
        e.g. (3 step) : 1 2 [3] 4 5 [6] 7 [8] <EOS>
        """
        assert cfg['step'] >= 1, "step must be positive number"
        step = cfg['step']
        selected_idx = tensorauto(mat, torch.arange(step-1, mat.size(axis), step).long())

        if isinstance(mat, Variable) :
            selected_idx = Variable(selected_idx)

        res = mat.index_select(axis, selected_idx)
        res_len = None
        if mat_len is not None :
            res_len = [x//step for x in mat_len]
        return res, res_len
    else :
        raise NotImplementedError
    pass
