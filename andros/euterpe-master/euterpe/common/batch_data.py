import torch
import numpy as np
from torchev.utils.helper import tensorauto, torchauto
from torch.autograd import Variable
from torchev.utils.mask_util import generate_seq_mask
from ..data.data_generator import group_feat_timestep
from ..config import constant

def batch_speech(device, feat_list, feat_sil=None, group=None, start_sil=0, end_sil=0) :
    if group is not None :
        # grouping feat per x frame into 1 frame #
        feat_list = [group_feat_timestep(feat_ii, group) for feat_ii in feat_list]
    if feat_sil is not None :
        feat_sil = np.tile(feat_sil, group)
    feat_len = [len(x)+start_sil+end_sil for x in feat_list]
    batch = len(feat_list)
    max_feat_len = max(feat_len)
    ndim = feat_list[0].shape[-1]

    feat_mat = np.zeros((batch, max_feat_len, ndim), dtype='float32') + \
            (feat_sil if feat_sil is not None else 0)
    for ii in range(batch) :
        feat_mat[ii, start_sil:start_sil+len(feat_list[ii])] = feat_list[ii]

    feat_mat = torch.from_numpy(feat_mat).float()
    feat_mat = tensorauto(device, feat_mat)
    return feat_mat, feat_len

def batch_text(device, text_list, add_bos=True, add_eos=True) :
    """
    return text_mat, text_len
    """
    assert all(isinstance(x, list) for x in text_list)
    text_idx_list = text_list
    batch = len(text_list)
    if add_bos :
        text_idx_list = [[constant.BOS]+x for x in text_idx_list]
    if add_eos :
        text_idx_list = [x+[constant.EOS] for x in text_idx_list]
    text_len = [len(x) for x in text_idx_list] # -1 because we shift mask by 1 for input output #
    text_mat = np.full((batch, max(text_len)), constant.PAD, dtype='int64')
    for ii in range(batch) :
        text_mat[ii, 0:text_len[ii]] = text_idx_list[ii]
    text_mat = tensorauto(device, torch.from_numpy(text_mat))
    return text_mat, text_len

def batch_speech_text(device, feat_list, text_list, feat_sil=None, group=None, start_sil=0, end_sil=0) :
    feat_mat, feat_len = batch_speech(device, feat_list, feat_sil, group, start_sil=start_sil, end_sil=end_sil)
    text_mat, text_len = batch_text(device, text_list)
    return feat_mat, feat_len, text_mat, text_len

def batch_select(obj, idx) :
    if isinstance(obj, torch.autograd.Variable) :
        res = obj.index_select(0, Variable(torchauto(obj.data).LongTensor(idx)))
    elif isinstance(obj, torch.tensor._TensorBase) :
        res = obj.index_select(0, torchauto(obj).LongTensor(idx))
    elif isinstance(obj, list) :
        res = [obj[ii] for ii in idx]
    else :
        raise ValueError("obj type is not supported")
    return res 
    pass
def batch_sorter(obj, key, reverse=True) :
    """
    function to sort object first axis based on key
        usual case : sort decreasing for bidirectional RNN input
    """
    sorted_idx = np.argsort(key).tolist()
    if reverse :
        sorted_idx = sorted_idx[::-1]
    return batch_select(obj, sorted_idx)


def list_1d_to_tensor_2d(obj, max_len=None) :
    if max_len is None :
        max_len = max([len(x) for x in obj])
    batch = len(obj)
    result = obj[0].new(batch, max_len).fill_(constant.PAD)
    for ii in range(batch) :
        result[ii, 0:len(obj[ii])] = obj[ii]
    return result
