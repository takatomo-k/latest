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

from euterpe.data.data_generator import generate_feat_opts, feat_sil_from_stat, list_feat_sil_strip
from euterpe.common.batch_data import batch_text, batch_speech, batch_speech_text
from euterpe.common.helper import TacotronHelper
from euterpe.common import generator_speech
from euterpe.common.loader import DataLoader
from euterpe.data.data_iterator import TextIterator, DataIteratorNP
from euterpe.model_tts.seq2seq.tacotron_core import TacotronType
from euterpe.util import plot_attention

import warnings
import librosa

import torch
from torch.autograd import Variable
from torchev.utils.serializer import ModelSerializer
from torchev.utils.helper import tensorauto, torchauto
from torchev.utils.mask_util import generate_seq_mask


from tqdm import tqdm

def parse() :
    parser = argparse.ArgumentParser(description='params')
    parser.add_argument('--model', type=str, help='model weight')
    parser.add_argument('--config', type=str, help='model config', default=None)
    parser.add_argument('--feat_cfg', type=str, help='feature config file path')
    parser.add_argument('--data_cfg', type=str, help='data config (contain vocab and scaler)')
    parser.add_argument('--text', type=str, help='input text (multiline delimited by ||)')
    parser.add_argument('--spkvec', type=str, default=None, help='path to speaker vector\n TODO : mass generate with scp')
    parser.add_argument('--search', type=str, choices=['greedy'], default='greedy')
    parser.add_argument('--mode', type=str, choices=['pred', 'tf'], default='pred',
            help='pred: decoder generates prediction based on its own result\ntf (teacher forcing): decoder generate prediction based on ground truth label from data_cfg')
    parser.add_argument('--set', type=str, default=None, help='subset of utterance (in case we don\'t need to transcript whole wav_scp')
    parser.add_argument('--plot_att', action='store_true', default=False)
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--path', type=str, help='path to save generated feat + att mat', default=None)
    parser.add_argument('--chunk', type=int, default=10, help='number of text sentences processed per mini-batch')
    # additional for multispeaker or other information
    parser.add_argument('--dump', action='store_true', default=False)
    parser.add_argument('--max_target', type=int, default=400, help='stop generation after len(x) > max_target')

    return parser.parse_args()


if __name__ == '__main__' :
    opts = vars(parse())
        
    feat_cfg = json.load(open(opts['feat_cfg']))
    # load model structure and weight #
    if opts['config'] is None :
        opts['config'] = os.path.splitext(opts['model'])[0]+'.cfg'

    model = ModelSerializer.load_config(opts['config'])
    model.train(False)
    model.load_state_dict(torch.load(opts['model']))

    aux_info = None
    
    if opts['gpu'] >= 0 :
        torch.cuda.set_device(opts['gpu'])
        model.cuda() 

    data_cfg = yaml.load(open(opts['data_cfg']))
    # data_second_cfg = yaml.load(open(opts['data_second_cfg']))

    feat_stat = pickle.load(open(data_cfg['feat']['stat'], 'rb'))
    feat_sil = feat_sil_from_stat(feat_stat)

    # feat_second_stat = pickle.load(open(data_second_cfg['feat']['stat'], 'rb'))
    # feat_second_sil = feat_sil_from_stat(feat_second_stat)
    group = model.dec_in_size // feat_sil.size

    # encoding latin1 for python 3 #
    scaler = pickle.load(open(data_cfg['scaler'], 'rb'), encoding='latin1') if data_cfg.get('scaler',None) is not None else None

    # scaler_second = pickle.load(open(data_second_cfg['scaler'], 'rb'), encoding='latin1') if data_second_cfg.get('scaler',None) is not None else None

    # generate label text #
    # if not a file, then read as normal text input #
    list_kv = None
    if opts['text'] is not None :
        if not os.path.exists(opts['text']) :
            text_list = [x.strip() for x in opts['text'].split('||')]
            key_list = ['text_{}'.format(ii) for ii in range(len(text_list))]
            list_kv = list(zip(key_list, text_list))
            assert opts['mode'] == 'pred', "free text input can't be used with teacher forcing mode"
        else :
            list_kv = regex_key_val.findall(open(opts['text']).read())
            key_list, text_list = list(map(list, zip(*list_kv)))

    map_text2idx = json.load(open(data_cfg['text']['vocab']))
    if model.TYPE == TacotronType.MULTI_SPEAKER :
        feat_spkvec_iterator = DataIteratorNP(data_cfg['misc']['spkvec'])

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
    

    if list_kv is None :
        if opts['set'] is not None :
            data_iterator = DataLoader.load_feat_single(feat_path=data_cfg['feat']['all'], key=opts['set'])
            text_iterator = DataLoader.load_text_single(text_path=data_cfg['text']['all'], 
                    key=opts['set'], vocab=map_text2idx)
        else :
            data_iterator = DataLoader.load_feat_single(feat_path=data_cfg['feat']['all'])
            text_iterator = DataLoader.load_text_single(text_path=data_cfg['text']['all'], 
                    vocab=map_text2idx)
    else :
        assert opts['mode'] == 'pred'
        text_iterator = TextIterator(text_kv=list_kv, map_text2idx=map_text2idx)

    # sort text from longest to shortest #
    sorted_text_idx = np.argsort(text_iterator.get_text_length())[::-1].tolist()
    data_rr = iter_minibucket(sorted_text_idx, opts['chunk'], shuffle=False, excludes=[])
    list_saved = []
    for rr in tqdm(list(data_rr), ascii=True, ncols=50) :
        # optional #
        aux_info = None
        # TODO REMOVE THIS 
        # case prediction mode
        if opts['mode'] == 'pred' :
            curr_key_list = text_iterator.get_key_by_index(rr)
            if model.TYPE == TacotronType.MULTI_SPEAKER :
                if opts['spkvec'] is None :
                    _spk_vec = np.stack(feat_spkvec_iterator.get_feat_by_key(curr_key_list)).astype('float32')
                    _spk_vec = Variable(tensorauto(opts['gpu'], torch.from_numpy(_spk_vec)))
                elif os.path.exists(opts['spkvec']) :
                    _spk_vec = np.load(opts['spkvec'])['feat'][None, :].astype('float32')
                    _spk_vec = np.repeat(_spk_vec, len(rr), axis=0)
                    _spk_vec = Variable(tensorauto(opts['gpu'], torch.from_numpy(_spk_vec)))
                else :
                    _spk_vec = feat_spkvec_iterator.get_feat_by_key(opts['spkvec'])
                    _spk_vec = _spk_vec[None, :].astype('float32')
                    _spk_vec = np.repeat(_spk_vec, len(rr), axis=0)
                    _spk_vec = Variable(tensorauto(opts['gpu'], torch.from_numpy(_spk_vec)))
                aux_info = {'speaker_vector':_spk_vec}
            else :
                aux_info = None
            curr_text_list = text_iterator.get_text_by_key(curr_key_list)
            text_mat, text_len = batch_text(opts['gpu'], curr_text_list)
            pred_feat, pred_len, pred_att = generator_speech.decode_greedy_pred_torch(model, 
                    Variable(text_mat), text_len, group, feat_sil, aux_info=aux_info, max_target=opts['max_target'])
        elif opts['mode'] == 'tf' :
            curr_key_list = text_iterator.get_key_by_index(rr)
            curr_text_list = text_iterator.get_text_by_key(curr_key_list)
            curr_feat_list = data_iterator.get_feat_by_key(curr_key_list)
            if model.TYPE == TacotronType.MULTI_SPEAKER :
                _spk_vec = np.stack(feat_spkvec_iterator.get_feat_by_key(curr_key_list)).astype('float32')
                _spk_vec = Variable(tensorauto(opts['gpu'], torch.from_numpy(_spk_vec)))
                aux_info = {'speaker_vector' : _spk_vec}
            feat_mat, feat_len, text_mat, text_len = batch_speech_text(opts['gpu'], curr_feat_list, curr_text_list, feat_sil=feat_sil, group=group, start_sil=1, end_sil=0)
            pred_feat, pred_len, pred_att = generator_speech.decode_greedy_tf_torch(model, 
                    Variable(text_mat), text_len, Variable(feat_mat), feat_len, 
                    group=group, feat_sil=feat_sil, aux_info=aux_info)
            pred_len = data_iterator.get_feat_length_by_key(text_iterator.get_key_by_index(rr))

        pred_feat = pred_feat.data.cpu().numpy()
        pred_att = pred_att.data.cpu().numpy()
        # save results
        for ii in range(len(rr)) :
            _info = {}
            key_ii = text_iterator.get_key_by_index(rr[ii])
            path_feat_ii = os.path.join(tmpdir, '{}.npz'.format(key_ii))
            feat_ii = pred_feat[ii][0:pred_len[ii]] # cutoff remaining len #
            if opts['mode'] == 'pred' :
                # feat_ii = list_feat_sil_strip([feat_ii])[0]
                pass
            # OPT : strip silence by statistic #
            att_ii = pred_att[ii] # decoder steps are reduced
            np.savez(path_feat_ii, key=key_ii, feat=feat_ii)
            if opts['plot_att'] :
                path_plotatt_ii = os.path.join(tmpdir, '{}_att.png'.format(key_ii))
                path_attmat_ii = os.path.join(tmpdir, '{}_att.npz'.format(key_ii))
                att_ii = plot_attention.crop_attention_matrix(att_ii[None, :, :], 
                        [len(feat_ii) // group], 
                        [len(text_iterator.get_text_by_key(key_ii))])[0]
                plot_attention.plot_softmax_attention(att_ii, None, path_plotatt_ii)
                np.savez(path_attmat_ii, key=key_ii, feat=att_ii)
                _info['att_plot'] = path_plotatt_ii
                _info['att_mat'] = path_attmat_ii
            _info['key'] = key_ii
            _info['feat'] = path_feat_ii
            _info['feat_len'] = len(feat_ii)
            _info['text'] = ' '.join(text_iterator.get_text_by_key(key_ii, convert_to_idx=False))
            list_saved.append(_info)
            pass 

    # print scp #
    list_saved = [list_saved[x] for x in np.argsort(sorted_text_idx)]

    with open(os.path.join(tmpdir, 'meta', 'feat.scp'), 'w') as f :
        f.write('\n'.join(['{} {}'.format(_info['key'], _info['feat']) for _info in list_saved]))

    with open(os.path.join(tmpdir, 'meta', 'feat_len.scp'), 'w') as f :
        f.write('\n'.join(['{} {}'.format(_info['key'], _info['feat_len']) for _info in list_saved]))

    with open(os.path.join(tmpdir, 'meta', 'text'), 'w') as f :
        f.write('\n'.join(['{} {}'.format(_info['key'], _info['text']) for _info in list_saved]))

    if opts['plot_att'] :
        with open(os.path.join(tmpdir, 'meta', 'att_mat.scp'), 'w') as f :
            f.write('\n'.join(['{} {}'.format(_info['key'], _info['att_mat']) for _info in list_saved]))
        
    # log #
    with open(os.path.join(tmpdir, 'meta', 'info.log'), 'w') as f :
        f.write(yaml.dump({'model':opts['model']}, default_flow_style=False))

    if opts['dump'] :
        print(yaml.dump(list_saved, default_flow_style=False))
