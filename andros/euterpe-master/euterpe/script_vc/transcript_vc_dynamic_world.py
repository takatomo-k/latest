import argparse
import os
import json
import sys
import tempfile

from __init__ import * 

import pickle
from scipy.io import wavfile
from utilbox import beamsearch, math_util
import numpy as np
from utilbox.metric_util import edit_distance
from data.data_generator import generate_feat_opts, feat_sil_from_stat
from common.batch_data import batch_speech
import model_tts
from utilbox import world_vocoder_util
import warnings
import librosa

import torch
from torch.autograd import Variable
from torchev.utils.serializer import ModelSerializer
from torchev.utils.helper import tensorauto, torchauto

def draw_2d(mat, path) :
    import matplotlib; matplotlib.use('Agg')
    from pylab import plt
    plt.imshow(mat)
    plt.savefig(path)

def draw_1d(mat, path) :
    import matplotlib; matplotlib.use('Agg')
    from pylab import plt
    plt.plot(mat)
    plt.savefig(path)

def parse() :
    parser = argparse.ArgumentParser(description='convert voice using static model')
    parser.add_argument('--model', type=str, help='model weight')
    parser.add_argument('--config', type=str, help='model config', default=None)
    parser.add_argument('--feat_cfg', type=str, help='feature config file path')
    parser.add_argument('--data_cfg', type=str, help='data config (contains scaler)')
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--wav', type=str, help='path to wav file')
    return parser.parse_args()

def greedy_decoder_torch(model, feat_source_mat, feat_source_len, feat_sil) :
    MAX_DURATION = 400 # hardcoded
    EOS_THRESHOLD = 0.1
    batch = feat_source_mat.size(0)
    model.reset()
    model.eval()
    model.encode(feat_source_mat, feat_source_len)
    feat_sil = np.tile(feat_sil, group).astype('float32')
    feat_sil = tensorauto(model, torch.from_numpy(feat_sil).unsqueeze(0).repeat(batch, 1))
    feat_sil_var = Variable(feat_sil)
    prev_feat = feat_sil_var # 1 dim #
    idx = 0
    feat_pred_len = [-1 for _ in range(batch)]
    feats_pred_core = []
    alignments = []
    while True :
        curr_feat, curr_decatt_res = model.decode(prev_feat)
        feats_pred_core.append(curr_feat)
        alignments.append(curr_decatt_res['att_output']['p_ctx'])
        prev_feat = curr_feat
        
        # check if batch bb already finished or not #
        dist_to_sil = (torch.abs(curr_feat - feat_sil_var)).sum(1).data
        print(idx, dist_to_sil.tolist())
        for bb in range(batch) :
            if feat_pred_len[bb] == -1 and dist_to_sil[bb] < EOS_THRESHOLD :
                feat_pred_len[bb] = idx
        idx += 1
        
        if idx > MAX_DURATION or all([x != -1 for x in feat_pred_len]) :
            break
        pass

    feats_pred_core = torch.stack(feats_pred_core, dim=1)

    alignments = torch.stack(alignments, dim=1)
    return feats_pred_core, alignments, feat_pred_len
    pass

# TODO : implement beam search when reduce blank (?)
if __name__ == '__main__' :
    opts = vars(parse())
        
    feat_cfg = json.load(open(opts['feat_cfg']))
    # load model structure and weight #
    if opts['config'] is None :
        opts['config'] = os.path.splitext(opts['model'])[0]+'.cfg'

    model = ModelSerializer.load_config(opts['config'])
    model.train(False)
    model.reset()
    model.load_state_dict(torch.load(opts['model']))
    
    if opts['gpu'] >= 0 :
        torch.cuda.set_device(opts['gpu'])
        model.cuda() 

    data_cfg = json.load(open(opts['data_cfg']))

    feat_stat = pickle.load(open(data_cfg['feat_stat'], 'rb'))
    feat_sil = feat_sil_from_stat(feat_stat)

    group = model.enc_in_size // feat_sil.size

    # encoding latin1 for python 3 #
    scaler = pickle.load(open(data_cfg['scaler'], 'rb'), encoding='latin1') if 'scaler' in data_cfg else None
    
    # convert source wav -> feature #
    feat_list = [generate_feat_opts(path=opts['wav'], cfg=feat_cfg)] # convert to list #
    feat_list = [np.hstack(x) if isinstance(x, (list, tuple)) else x for x in feat_list]
    if scaler is not None :
        feat_list = [scaler.transform(x) for x in feat_list]
    feat_mat, feat_len = batch_speech(opts['gpu'], feat_list, feat_sil=feat_sil, group=group, start_sil=1) 
    feat_mat = Variable(feat_mat)
    # convert source -> target feature 
    model.train(False)
    best_feat, _, _ = greedy_decoder_torch(model, feat_mat, feat_len, feat_sil)

    # post processing #
    batch = len(feat_list)
    best_feat = best_feat.data.cpu().numpy()
    best_feat = best_feat.reshape(batch, best_feat.shape[1] * group, -1)

    print("DECODING DONE...")
    # inverse feat #
    if scaler is not None :
        best_feat = scaler.inverse_transform(best_feat)
    
    # save result #
    # import ipdb; ipdb.set_trace()
    transcription_res_list = []
    tmpdir = tempfile.mkdtemp()
    MCEP_DIM = 59 # HARCODED 
    for ii in range(batch) :
        transcription_res = {}
        world_feat = world_vocoder_util.feat2world(best_feat[ii, :, 0:1], best_feat[ii, :, 1:2], best_feat[ii, :, 2:3], best_feat[ii, :, 3:])
        rate, signal_post = world_vocoder_util.world_synthesis(*world_feat)
        assert rate == 16000

        transcription_res = {}
        tmpwavpost = os.path.join(tmpdir, 'sample_post_{}.wav'.format(ii))
        librosa.output.write_wav(tmpwavpost, signal_post, rate)
        transcription_res['wav_post'] = tmpwavpost

        transcription_res_list.append(transcription_res)
        pass
    print(json.dumps(transcription_res_list, indent=2))
    pass

