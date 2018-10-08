from __init__ import *
import sys
import os
import argparse
import yaml
import pickle
import tempfile
from tqdm import tqdm

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

import torch
from torch.autograd import Variable
from torch import nn

from utilbox.data_util import iter_minibucket

from torchev.utils.serializer import ModelSerializer
from torchev.utils.helper import tensorauto, torchauto

from euterpe.common.batch_data import batch_speech
from euterpe.common.loader import DataLoader
from euterpe.data.data_generator import group_feat_timestep, feat_sil_from_stat
from euterpe.data.data_iterator import DataIteratorNP

def parse() :
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_cfg', type=str, required=True)
    parser.add_argument('--key', type=str, required=True)
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--batchsize', type=int, default=20)
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--vis', type=str, default=None)
    parser.add_argument('--path', type=str, default=None, help='path to save speaker vector')
    parser.add_argument('--dump', action='store_true', default=False)
    return parser.parse_args()

if __name__ == '__main__':
    args = parse()

    args.config = os.path.splitext(args.model)[0]+'.cfg'
    data_cfg = yaml.load(open(args.data_cfg))

    # construct all subset feat & feat_len 
    if os.path.exists(args.key) :
        _key = DataLoader._read_key(args.key)
    else :
        _key = args.key.split()
    _feat_all = DataLoader._read_key_val(data_cfg['feat']['all'])
    _feat_len_all_path = '{}_len{}'.format(*os.path.splitext(data_cfg['feat']['all']))
    _feat_len_all = DataLoader._read_key_val(_feat_len_all_path)
    _feat_kv = DataLoader._subset_data(_feat_all, _key)
    _feat_len_kv = DataLoader._subset_data(_feat_len_all, _key)

    model = ModelSerializer.load_config(args.config)
    model.load_state_dict(torch.load(args.model))
    model.eval() # set as eval mode
    
    if args.gpu >= 0 :
        torch.cuda.set_device(args.gpu)
        torch.cuda.manual_seed(123)
        model.cuda(args.gpu)
    
    # get feature and information
    data_info = pd.read_csv(data_cfg['misc']['info'], sep=',')
    map_key2spk = dict((str(k), str(v)) for k, v in yaml.load(open(data_cfg['misc']['key2spk'])).items())
    map_spk2id = dict((y, x) for x, y in enumerate(data_info['SPK'].astype(str).unique()))
    map_id2spk = dict((y, x) for x, y in map_spk2id.items())
    map_spk2key = dict((y, x) for x, y in map_key2spk.items())

    feat_iterator = DataIteratorNP(feat_kv=_feat_kv, feat_len_kv=_feat_len_kv)
    feat_len = feat_iterator.get_feat_length()
    sorted_feat_idx = np.argsort(feat_len).tolist()
    sorted_feat_len = [feat_len[x] for x in sorted_feat_idx] 

    feat_stat = pickle.load(open(data_cfg['feat']['stat'], 'rb'))
    feat_sil = feat_sil_from_stat(feat_stat)

    all_embed = []
    all_speaker_id = []
    all_key = []

    for rr in tqdm(list(iter_minibucket(sorted_feat_idx, args.batchsize, shuffle=False)), ncols=60) :
        curr_feat_list = feat_iterator.get_feat_by_index(rr)
        curr_key_list = feat_iterator.get_key_by_index(rr)
        curr_speaker_list = [map_key2spk[x] for x in curr_key_list] 
        curr_speaker_list_id = [map_spk2id[x] for x in curr_speaker_list]
        all_speaker_id.extend(curr_speaker_list_id)
        all_key.extend(curr_key_list)
        feat_mat, feat_len = batch_speech(args.gpu, curr_feat_list, 
                feat_sil=feat_sil, group=1, start_sil=1, end_sil=1)

        res_embed = model(Variable(feat_mat), feat_len)
        res_embed = res_embed.cpu().data.numpy()
        all_embed.append(res_embed)
        pass
    
    all_embed = np.concatenate(all_embed).astype(np.float64)
    all_speaker_id = np.array(all_speaker_id, dtype=np.int32)

    # save speaker vector 
    if args.path is None :
        args.path = tempfile.mkdtemp()
        print('Create temporary dir: {}'.format(args.path), file=sys.stderr)

    if os.path.exists(args.path) :
        assert os.listdir(args.path) == []
    else :
        os.makedirs(args.path, exist_ok=False)
    # create meta dir
    os.makedirs(os.path.join(args.path, 'meta'), exist_ok=False)

    list_kv = []
    list_klen = []
    list_info = []
    # invert to original index 
    for ii in range(len(all_speaker_id)) :
        _path = os.path.join(args.path, all_key[ii]+'_spkvec.npz')
        list_kv.append(all_key[ii]+' '+_path)
        list_klen.append(all_key[ii]+' 1')
        list_info.append({'key':all_key[ii], 'feat':_path, 'feat_len':all_key[ii]})
        np.savez(_path, feat=all_embed[ii])

    list_kv = [list_kv[x] for x in np.argsort(sorted_feat_idx)]
    list_klen = [list_klen[x] for x in np.argsort(sorted_feat_idx)]
    list_info = [list_info[x] for x in np.argsort(sorted_feat_idx)]

    with open(os.path.join(args.path, 'meta', 'feat.scp'), 'w') as f :
        f.write('\n'.join(list_kv))
    with open(os.path.join(args.path, 'meta', 'feat_len.scp'), 'w') as f :
        f.write('\n'.join(list_klen))
    with open(os.path.join(args.path, 'meta', 'info.log'), 'w') as f :
        f.write(yaml.dump({'model':args.model}, default_flow_style=False))

    if args.dump :
        print(yaml.dump(list_info, default_flow_style=False))

    if args.vis is not None :
        from tsne import bh_sne
        embed_proj_tsne = bh_sne(all_embed, perplexity=10)
        # scatter plot #
        plt.scatter(embed_proj_tsne[:, 0], embed_proj_tsne[:, 1], c=all_speaker_id)
        DPI = 300
        plt.savefig(args.vis, dpi=DPI)
        print('Figure saved in {}'.format(args.vis), file=sys.stderr)
