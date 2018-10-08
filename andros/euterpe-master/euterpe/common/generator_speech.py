import argparse
import os
import json
import yaml
import sys
import tempfile

import pickle
import numpy as np

from utilbox import math_util, signal_util
from utilbox.regex_util import regex_key_val
from utilbox.data_util import iter_minibucket

from ..data.data_generator import generate_feat_opts, feat_sil_from_stat
from ..common.batch_data import batch_text, batch_speech, batch_speech_text
from ..common.helper import TacotronHelper
from ..data.data_iterator import TextIterator, DataIteratorNP
from ..model_tts.seq2seq.tacotron_core import TacotronType
from .. import model_tts
from ..util.plot_attention import crop_attention_matrix

import torch
from torch.autograd import Variable
from torchev.utils.serializer import ModelSerializer
from torchev.utils.helper import tensorauto, torchauto
from torchev.utils.mask_util import generate_seq_mask

def convert_tensor_to_seq_list(batch_mat, seq_len) :
    result = []
    for ii in range(len(seq_len)) :
        curr_result = batch_mat[ii].data.cpu().numpy()
        if seq_len[ii] != -1 :
            curr_result = curr_result[0:seq_len[ii]]
        result.append(curr_result)
    return result

AVAILABLE_MODEL = (model_tts.seq2seq.tacotron_core.TacotronV1Core, 
        model_tts.seq2seq.tacotron_core.TacotronV1CoreMSpkContinuous,
        model_tts.seq2seq.tacotron_core.TacotronV2Core)

def decode_greedy_pred_torch(model, text_mat, text_len, group, feat_sil, 
        max_target=1000, aux_info=None) :
    """
    decode free-path with its own predicted feature as the input
    """
    assert isinstance(model, AVAILABLE_MODEL), "model is not supported"
    if not isinstance(text_mat, Variable) :
        text_mat = Variable(text_mat)
    batch = text_mat.size(0)
    model.reset()
    model.eval()
    model.encode(text_mat, text_len)
    if aux_info is not None :
        if isinstance(aux_info['speaker_vector'], list) :
            aux_info['speaker_vector'] = Variable(tensorauto(model, torch.from_numpy(np.stack(aux_info['speaker_vector']).astype('float32'))))
        model.set_aux_info(aux_info)
        
    feats_core = []
    feats_att = []
    feat_sil = np.tile(feat_sil, group).astype('float32')
    feat_sil = tensorauto(model, torch.from_numpy(feat_sil).unsqueeze(0).expand(batch, feat_sil.shape[0]))
    feat_sil_var = Variable(feat_sil)
    prev_feat = feat_sil_var # 1 dim #
    idx = 0
    feat_len = [-1 for _ in range(batch)]
    while True :
        curr_feat, curr_decatt_res, curr_bern_end = model.decode(prev_feat)
        feats_core.append(curr_feat)
        feats_att.append(curr_decatt_res['att_output']['p_ctx'])
        idx += 1 # increase index #
        prev_feat = curr_feat
        
        # check if batch bb already finished or not #
        curr_bern_end = curr_bern_end[:, 0].data
        dist_to_sil = (torch.abs(curr_feat - feat_sil_var)).sum(1).data
        for bb in range(batch) :
            # output frame end is logit (not sigmoid)
            if feat_len[bb] == -1 and curr_bern_end[bb] > 0.0 :
                feat_len[bb] = idx
        
        if idx >= max_target or all([x != -1 for x in feat_len]) :
            # too long or all samples already STOP
            break
        pass

    feats_core = torch.stack(feats_core, dim=1)

    # TODO : masking #

    # reshape
    feats_core = feats_core.view(batch, feats_core.shape[1] * group, -1)
    feat_len = [x*group for x in feat_len]
    feats_att = torch.stack(feats_att, dim=1)
    return feats_core, feat_len, feats_att

def decode_greedy_pred(model, text_mat, text_len, group, feat_sil, 
        max_target=1000, aux_info=None) :

    feats_core, feat_len, feats_att = decode_greedy_pred_torch(model, text_mat, text_len, 
            group, feat_sil, max_target=max_target, aux_info=aux_info)
    return convert_tensor_to_seq_list(feats_core, feat_len), feat_len, \
            crop_attention_matrix(feats_att.data.cpu().numpy(), feat_len, model.dec_att_lyr.ctx_len)


def decode_greedy_tf_torch(model, text_mat, text_len, feat_mat, feat_len, group, feat_sil, 
        max_target=1000, aux_info=None) :
    """
    decode with teacher forcing method by using ground truth feature as the input
    """
    assert isinstance(model, AVAILABLE_MODEL), "model is not supported"
    if not isinstance(text_mat, Variable) :
        text_mat = Variable(text_mat)
    batch = text_mat.size(0)
    model.reset()
    model.eval()
    model.encode(text_mat, text_len)
    if aux_info is not None :
        if isinstance(aux_info['speaker_vector'], list) :
            aux_info['speaker_vector'] = Variable(tensorauto(model, torch.from_numpy(np.stack(aux_info['speaker_vector']).astype('float32'))))
        model.set_aux_info(aux_info)

    feats_core = []
    feats_att = []
    feat_sil = np.tile(feat_sil, group).astype('float32')
    feat_sil = tensorauto(model, torch.from_numpy(feat_sil).unsqueeze(0).expand(batch, feat_sil.shape[0]))
    feat_sil_var = Variable(feat_sil)
    feat_mat_input = feat_mat[:, 0:-1]
    feat_mask = Variable(generate_seq_mask([x-1 for x in feat_len], model))

    dec_len = feat_mat_input.size(1)
    for ii in range(dec_len) :
        curr_feat, curr_decatt_res, curr_bern_end = model.decode(feat_mat[:, ii], feat_mask[:, ii])

        feats_core.append(curr_feat)
        feats_att.append(curr_decatt_res['att_output']['p_ctx'])
        pass

    feats_core = torch.stack(feats_core, dim=1)
    feats_core = feats_core * feat_mask.unsqueeze(-1)
    feats_core = feats_core.view(batch, feats_core.shape[1] * group, -1)
    feats_att = torch.stack(feats_att, dim=1)
    return feats_core, feat_len, feats_att

def decode_greedy_tf(model, text_mat, text_len, feat_mat, feat_len, group, feat_sil, 
        max_target=1000, aux_info=None) :
    feats_core, feat_len, feats_att = decode_greedy_tf(model, text_mat, text_len, 
            feat_mat, feat_len, group, feat_sil, 
        max_target=max_target, aux_info=aux_info)
    return convert_tensor_to_seq_list(feats_core, feat_len), feat_len, convert_tensor_to_seq_list(feats_att, feat_len)

def eval_gen_speech_quality(feats, feat_len, att_mat) :
    # TODO improve criterion
    batch_size = len(feat_len)
    quality = [None for _ in range(batch_size)]
    for bb in range(batch_size) :
        if feat_len[bb] < 0 :
            quality[bb] = 0
        else :
            quality[bb] = 1
    return quality

