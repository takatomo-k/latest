import argparse
import json, yaml
import re
import time

import multiprocessing
from concurrent.futures import ProcessPoolExecutor
import subprocess
import os
import logging
import itertools
import warnings
import pickle
import tempfile
import numpy as np
import operator
from functools import partial

from utilbox.metric_util import edit_distance
from utilbox.regex_util import regex_key_val
from utilbox.data_util import iter_minibucket, iter_minibatches, iter_minibucket_block

from euterpe.data.data_generator import generate_feat_opts
from euterpe.data.data_iterator import TextIterator
from euterpe.common.batch_data import batch_speech, batch_text
from euterpe.common.generator_text import greedy_search, beam_search, teacher_forcing
from euterpe.common.loader import DataLoader
from euterpe.config import constant
from euterpe.util import plot_attention

import torch
from torch.autograd import Variable
from torch.nn import functional as F
from torchev.utils.serializer import ModelSerializer
from torchev.utils.helper import tensorauto, torchauto, vars_index_select
from tqdm import tqdm

def parse() :
    parser = argparse.ArgumentParser(description='params')
    parser.add_argument('--model', type=str, help='model weight', required=True)
    parser.add_argument('--config', type=str, help='model config', default=None)
    parser.add_argument('--wav_scp', type=str, help='wav file path', required=False, default=None)
    parser.add_argument('--feat_scp', type=str, help='feat file path', required=False, default=None)
    parser.add_argument('--set', type=str, default=None, help='subset of utterance (in case we don\'t need to transcript whole wav_scp')
    parser.add_argument('--feat_cfg', type=str, help='feature config file path')

    parser.add_argument('--data_cfg', type=str, help='config file with feat_stat and vocab', required=True)
    parser.add_argument('--search', type=str, choices=['greedy', 'beam'], default='greedy')
    parser.add_argument('--kbeam', type=int, default=5, help='number of beam for each sequence (beam search)')
    parser.add_argument('--coeff_lp', type=float, default=1.0, help='coefficient for length penalty (beam search)')
    parser.add_argument('--max_target', type=int, help='max target for producing hypothesis', required=True)
    
    parser.add_argument('--chunk', type=int, default=None, help='chunk size for each iteration')
    parser.add_argument('--chunkblk', type=int, default=None, help='maximum number of block per iteration')
    parser.add_argument('--gpu', type=int, default=-1, help='gpu id for decoding process (-1 for cpu)')
    # parser.add_argument('--cutoff', type=int, default=-1, help='skip evaluation from too long utts (out of memory)')
    # dump all information in yaml format
    parser.add_argument('--plot_att', action='store_true', default=False)
    parser.add_argument('--path', type=str, default=None, help='path to save transcript result (+ att_mat if plot_att is True)')
    parser.add_argument('--dump', action='store_true', default=False)
    parser.add_argument('--mode', type=str, choices=['pred', 'tf'], default='pred',
            help='''pred: decoder generates prediction based on its own result\n
            tf (teacher forcing): decoder generate prediction based on ground truth label from data_cfg''')
    return parser.parse_args()

if __name__ == '__main__' :
    args = parse()
    assert (args.chunk is None) != (args.chunkblk is None)
    
    # init logger #
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # load model structure and weight #
    if args.config is None :
        args.config = os.path.splitext(args.model)[0]+'.cfg'
    model = ModelSerializer.load_config(args.config)
    model.train(False)
    model.load_state_dict(torch.load(args.model))

    if args.gpu >= 0 :
        torch.cuda.set_device(args.gpu)
        model.cuda()
    
    # laod dict and feature scaler #
    data_cfg = yaml.load(open(args.data_cfg))

    map_text2idx = json.load(open(data_cfg['text']['vocab']))
    map_idx2text = dict([(y,x) for (x,y) in map_text2idx.items()])

    scaler = pickle.load(open(data_cfg['feat']['scaler'], 'rb')) if data_cfg['feat'].get('scaler', None) is not None else None
    feat_cfg = json.load(open(args.feat_cfg))
    # args check
    if args.mode == 'tf' :
        pass
    elif args.mode == 'pred' :
        pass

    if args.wav_scp is None and args.feat_scp is None :
        logger.info('fallback ... read feat_scp from data_cfg')
        args.feat_scp = data_cfg['feat']['all']

    if args.mode == 'tf' :
        assert args.data_cfg is not None
        text_iterator = TextIterator(path=data_cfg['text']['all'], map_text2idx=map_text2idx)

    if args.wav_scp is not None :
        # list all wav files #
        list_key_wav = regex_key_val.findall(open(args.wav_scp).read())
        if args.set is not None :
            if os.path.exists(args.set) :
                list_key_wav = DataLoader._subset_data(list_key_wav, DataLoader._read_key(args.set))
            else :
                args.set = args.set.split(' ')
                list_key_wav = DataLoader._subset_data(list_key_wav, args.set)
        list_key_wav = sorted(list_key_wav, key=lambda x : x[0])
        # lazy load -- saving memory #
        def lazy_generate_feat(path, cfg) :
            _feat = generate_feat_opts(path=path, cfg=cfg)
            if scaler is not None :
                _feat = scaler.transform(_feat)
            return _feat
        list_feat = []
        list_feat_len = []
        list_key = [x[0] for x in list_key_wav]
        for k, v in list_key_wav :
            list_feat.append(partial(lazy_generate_feat, path=v, cfg=feat_cfg))
            list_feat_len.append(os.path.getsize(v))
    elif args.feat_scp is not None :
        list_key_feat = regex_key_val.findall(open(args.feat_scp).read())
        list_feat_len = regex_key_val.findall(open(args.feat_scp.replace('.scp', '_len.scp')).read())
        if args.set is not None :
            list_key_feat = DataLoader._subset_data(list_key_feat, DataLoader._read_key(args.set))
            list_feat_len = DataLoader._subset_data(list_feat_len, DataLoader._read_key(args.set))
        list_key_feat = sorted(list_key_feat, key=lambda x : x[0])
        list_feat_len = sorted(list_feat_len, key=lambda x : x[0])
        list_feat_len = [int(x[1]) for x in list_feat_len]
        list_key = [x[0] for x in list_key_feat]

        def lazy_get_feat(path) :
            _feat = np.load(path)['feat']
            return _feat

        list_feat = []
        for k, v in list_key_feat :
            list_feat.append(partial(lazy_get_feat, v))

    logger.info("PID : %d"%os.getpid())

    if args.path is None :
        tmpdir = tempfile.mkdtemp()
        os.makedirs(os.path.join(tmpdir, 'meta'), exist_ok=False)
        logger.info('Create temporary dir: {}'.format(tmpdir))
    else :
        tmpdir = args.path
        # if not exist, create new dir #
        if not os.path.isdir(tmpdir) :
            os.makedirs(tmpdir, exist_ok=False)
            os.makedirs(os.path.join(tmpdir, 'meta'), exist_ok=False)
        else :
            assert os.listdir(tmpdir) == [], "target directory path must be empty"
            os.makedirs(os.path.join(tmpdir, 'meta'), exist_ok=False)

    def save_att(info) :
        _path_att_img = os.path.join(tmpdir, info['key']+'_att.png')
        plot_attention.plot_softmax_attention(info['att'], None, _path_att_img)
        _path_att_npz = os.path.join(tmpdir, info['key']+'_att.npz')
        np.savez(_path_att_npz, key=info['key'], feat=info['att'])
        return _path_att_img, _path_att_npz

    start = time.time()
    # cache list #
    list_saved = []

    ori_key_pos = dict((x, y) for y, x in enumerate(list_key))
    sorted_feat_idx = np.argsort(list_feat_len)[::-1].tolist()
    sorted_feat_len = operator.itemgetter(*sorted_feat_idx)(list_feat_len)
    
    if args.chunk is not None :
        data_rr = iter_minibucket(sorted_feat_idx, args.chunk, shuffle=False, excludes=[])
    elif args.chunkblk is not None :
        data_rr = iter_minibucket_block(sorted_feat_idx, args.chunkblk, sorted_feat_len, pad_block=True, shuffle=False, excludes=[])
        data_rr = [list(reversed(x)) for x in data_rr]

    for rr in tqdm(list(data_rr), ascii=True, ncols=50) :
        curr_key_list = [list_key[rrii] for rrii in rr]
        curr_feat_list = [list_feat[rrii]() for rrii in rr] # lazy load call
        if args.mode == 'pred' :
            pass
        elif args.mode == 'tf' :
            curr_text_list = text_iterator.get_text_by_key(curr_key_list)
            text_mat, text_len = batch_text(args.gpu, curr_text_list) 
        else :
            raise ValueError
        curr_feat_list = [np.hstack(x) if isinstance(x, (list, tuple)) else x for x in curr_feat_list]
        feat_mat, feat_len = batch_speech(args.gpu, curr_feat_list, feat_sil=None)

        if args.mode == 'pred' :
            if args.search == 'greedy' :
                curr_best_hypothesis, _, curr_best_att = greedy_search(model, feat_mat, feat_len, map_text2idx, args.max_target)
            elif args.search == 'beam' :
                assert args.kbeam is not None, "kbeam must be specified"
                curr_best_hypothesis, _, curr_best_att = beam_search(model, feat_mat, feat_len, map_text2idx, args.max_target, args.kbeam, args.coeff_lp)
            else :
                raise ValueError('search method is not defined')
        elif args.mode == 'tf' :
            _, _, curr_best_att = teacher_forcing(model, feat_mat, feat_len, text_mat, text_len, map_text2idx, args.max_target, exclude_eos=False)
            curr_best_hypothesis = curr_text_list

        for ii, rrii in enumerate(rr) :
            key_ii = list_key[rrii]
            text_ii =  curr_best_hypothesis[ii]
            att_ii = curr_best_att[ii]
            _info = {'key':key_ii, 'text':' '.join([map_idx2text[jj] for jj in text_ii]), 'att':att_ii}
            if args.plot_att :
                _, _att_mat_ii_path = save_att(_info)
                _info['att_mat_path'] = _att_mat_ii_path
            list_saved.append(_info)
        pass

    # return to original index #
    # list_saved = [list_saved[ii] for ii in np.argsort(sorted_feat_idx)]
    list_saved = sorted(list_saved, key=lambda x: ori_key_pos[x['key']])

    # save key & text
    with open(os.path.join(tmpdir, 'meta', 'text_hyp'), 'w') as f :
        for ii, infoii in enumerate(list_saved) :
            f.write('{} {}\n'.format(infoii['key'], infoii['text']))

    # optional : save attention matrix #
    if args.plot_att :

        # multi processing #
        with open(os.path.join(tmpdir, 'meta', 'att_mat.scp'), 'w') as f_att_mat_scp, \
                open(os.path.join(tmpdir, 'meta', 'att_mat_len.scp'), 'w') as f_att_mat_len :
            for ii, infoii in enumerate(list_saved) :
                f_att_mat_scp.write('{} {}\n'.format(list_saved[ii]['key'], infoii['att_mat_path']))
                f_att_mat_len.write('{} {}\n'.format(list_saved[ii]['key'], len(infoii['att'])))

    logger.info(('Result path: {}'.format(tmpdir)))
    logger.info(("Runtime: %.1f s"%(time.time()-start)))
    pass
