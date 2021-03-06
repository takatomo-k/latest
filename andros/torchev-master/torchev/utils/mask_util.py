import torch
from .helper import torchauto

def generate_seq_mask(seq_len, device=-1, max_len=None) :
    """
    seq_len : list of each batch length
    """
    batch = len(seq_len)
    max_len = max(seq_len) if max_len is None else max_len
    mask = torchauto(device).FloatTensor(batch, max_len).zero_()
    for ii in range(batch) :
        mask[ii, 0:seq_len[ii]] = 1.0
    return mask
