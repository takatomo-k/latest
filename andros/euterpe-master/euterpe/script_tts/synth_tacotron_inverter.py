import argparse
import os
import json
import yaml
import sys
import tempfile

import warnings
import librosa
import pickle
import numpy as np
from tqdm import tqdm
from utilbox import beamsearch, math_util
from utilbox.metric_util import edit_distance
from utilbox.regex_util import regex_key_val
from utilbox.data_util import iter_minibucket

from euterpe.data.data_generator import generate_feat_opts, feat_sil_from_stat
from euterpe.common.batch_data import batch_text, batch_speech, batch_speech_text
from euterpe.common.helper import TacotronHelper
from euterpe.common.loader import DataLoader
from euterpe.data.data_iterator import TextIterator, DataIteratorNP
from euterpe.model_tts.seq2seq import tacotron_core
from euterpe.util import plot_attention

import torch
from torch.autograd import Variable
from torchev.utils.serializer import ModelSerializer
from torchev.utils.helper import tensorauto, torchauto
from torchev.utils.mask_util import generate_seq_mask

def parse() :
    parser = argparse.ArgumentParser(description='params')
    parser.add_argument('--model', type=str, help='model weight')
    parser.add_argument('--config', type=str, help='model config', default=None)
    parser.add_argument('--feat_cfg', type=str, help='feature config file path', default=None)
    parser.add_argument('--data_cfg_src', type=str, help='data config for mel spectrogram')
    parser.add_argument('--data_cfg_tgt', type=str, help='data config for linear spectrogram')
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--path', type=str, help='path to save generated wav', default=None)
    parser.add_argument('--chunk', type=int, default=10, help='number of sequences proce,sed per mini-batch')
    parser.add_argument('--dump', action='store_true', default=False)
    return parser.parse_args()

def invert_mel_to_linear(model, feat_mat, feat_len, group=1) :
    # model.reset()
    model.eval()
    pred_out = model(feat_mat, feat_len)
    return pred_out

if __name__ == '__main__' :
    opts = vars(parse())
        
    feat_cfg = json.load(open(opts['feat_cfg']))
    # load model structure and weight #
    if opts['config'] is None :
        opts['config'] = os.path.splitext(opts['model'])[0]+'.cfg'

    model = ModelSerializer.load_config(opts['config'])
    model.train(False)
    model.load_state_dict(torch.load(opts['model']))

    # tacotron util helper #
    helper = TacotronHelper(feat_cfg)

    if opts['gpu'] >= 0 :
        torch.cuda.set_device(opts['gpu'])
        model.cuda() 

    data_cfg_target = yaml.load(open(opts['data_cfg_tgt']))
    data_iterator_target = DataLoader.load_feat_single(feat_path=data_cfg_target['feat']['all'])
    
    data_cfg_source = yaml.load(open(opts['data_cfg_src']))
    data_iterator_source = DataLoader.load_feat_single(feat_path=data_cfg_source['feat']['all'])
    feat_stat = pickle.load(open(data_cfg_source['feat']['stat'], 'rb'))
    feat_sil = feat_sil_from_stat(feat_stat)

    # encoding latin1 for python 3 #
    scaler = pickle.load(open(data_cfg_source['scaler'], 'rb'), encoding='latin1') if data_cfg_source.get('scaler',None) is not None else None

    ### generation process ###
    # create folder for generated result #
    if opts['path'] is None :
        tmpdir = tempfile.mkdtemp()
        os.makedirs(os.path.join(tmpdir, 'meta'), exist_ok=False)
        print('\nCreate temporary dir: {}'.format(tmpdir), file=sys.stderr)
    else :
        tmpdir = opts['path']
        # if not exist, create new dir #
        if not os.path.isdir(tmpdir) :
            os.makedirs(tmpdir, exist_ok=False)
            os.makedirs(os.path.join(tmpdir, 'meta'), exist_ok=False)
        else :
            assert os.listdir(tmpdir) == [], "target directory path must be empty"
            os.makedirs(os.path.join(tmpdir, 'meta'), exist_ok=False)
    
    """
    2 modes
    [+] stdin from other scripts (input must be yaml / json formatted)
    [+] read feature from cfg.yaml file
    """
    if os.isatty(0) :
        print('not supported for now', file=sys.stderr)
    else :
        # read from stdin
        print('Feat input redirected from previous script ...', file=sys.stderr)
        list_info = yaml.load(sys.stdin.read())
        list_kv = [(_info['key'], _info['feat']) for _info in list_info]
        list_klen = [(_info['key'], _info['feat_len']) for _info in list_info]
        data_iterator_infer = DataIteratorNP(feat_kv=list_kv, feat_len_kv=list_klen)

    group = model.in_size // data_iterator_source.get_feat_dim()

    sorted_data_idx = np.argsort(data_iterator_infer.get_feat_length())[::-1].tolist()
    data_rr = iter_minibucket(sorted_data_idx, opts['chunk'], shuffle=False, excludes=[])
    list_saved = []
    for rr in tqdm(list(data_rr), ascii=True, ncols=50) :
        curr_feat_list = data_iterator_infer.get_feat_by_index(rr)
        feat_mat, feat_len = batch_speech(opts['gpu'], curr_feat_list, 
                feat_sil=feat_sil, group=group, start_sil=0, end_sil=0)
        pred_out = invert_mel_to_linear(model, Variable(feat_mat), feat_len, group=group)
        # convert to cpu & numpy #
        pred_out = pred_out.data.cpu().numpy()
        pred_out = [pred_out[ii, 0:feat_len[ii]] for ii in range(len(rr))]
        # save results
        for ii in range(len(rr)) :
            _info = {}
            key_ii = data_iterator_infer.get_key_by_index(rr[ii])
            path_feat_ii = os.path.join(tmpdir, '{}.npz'.format(key_ii))
            feat_ii = pred_out[ii][0:feat_len[ii]]
            np.savez(path_feat_ii, key=key_ii, feat=feat_ii)
            # invert to wav & save the file #
            feat_ii_group_1 = feat_ii.reshape(-1, data_iterator_target.get_feat_dim())
            signal = helper.inv_spectrogram(feat_ii_group_1.T)
            path_wav_ii = os.path.join(tmpdir, '{}.wav'.format(key_ii))
            helper.save_wav(signal, path_wav_ii)

            _info['key'] = key_ii
            _info['feat'] = path_feat_ii
            _info['feat_len'] = feat_len[ii]
            _info['wav'] = path_wav_ii
            list_saved.append(_info)
            pass 

    # print scp #
    list_saved = [list_saved[x] for x in np.argsort(sorted_data_idx)]

    with open(os.path.join(tmpdir, 'meta', 'feat.scp'), 'w') as f :
        f.write('\n'.join(['{} {}'.format(_info['key'], _info['feat']) for _info in list_saved]))

    with open(os.path.join(tmpdir, 'meta', 'feat_len.scp'), 'w') as f :
        f.write('\n'.join(['{} {}'.format(_info['key'], _info['feat_len']) for _info in list_saved]))

    with open(os.path.join(tmpdir, 'meta', 'wav.scp'), 'w') as f :
        f.write('\n'.join(['{} {}'.format(_info['key'], _info['wav']) for _info in list_saved]))

    # log #
    with open(os.path.join(tmpdir, 'meta', 'info.log'), 'w') as f :
        f.write(yaml.dump({'model':opts['model']}, default_flow_style=False))

    if opts['dump'] :
        print(yaml.dump(list_saved, default_flow_style=False))
    print('Path: {}'.format(tmpdir))
