
import sys
import time
import re

import numpy as np
import json

# pytorch #
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence as pack, pad_packed_sequence as unpack

# torchev #
from torchev.generator import generator_rnn, generator_attention
from torchev.custom import decoder
from torchev.utils.helper import torchauto, tensorauto

# utilbox #
from utilbox.config_util import ConfigParser


class ENCCNN_DECRNN_ATT_ASR(nn.Module) :
    def __init__(self, enc_in_size, dec_in_size, n_class,
            enc_fnn_sizes=[512], enc_fnn_act='tanh', enc_fnn_do=0.25,
            enc_cnn_channels=256, enc_cnn_ksize=[5, 5, 5, 5], enc_cnn_do=0.25,
            enc_cnn_strides=[1,1,1,1],
            enc_cnn_act='leaky_relu',
            dec_emb_size=64, dec_emb_do=0.0,
            dec_rnn_sizes=[512, 512], dec_rnn_cfgs={"type":"lstm"}, dec_rnn_do=0.25,
            dec_cfg={"type":"standard_decoder"}, 
            att_cfg={"type":"mlp"},
            ) :
        super(ENCCNN_DECRNN_ATT_ASR, self).__init__()

        self.enc_in_size = enc_in_size
        self.dec_in_size = dec_in_size
        self.n_class = n_class
        self.enc_fnn_sizes = enc_fnn_sizes
        self.enc_fnn_act = enc_fnn_act
        self.enc_fnn_do = ConfigParser.list_parser(enc_fnn_do, len(enc_fnn_sizes))
        self.enc_cnn_channels = enc_cnn_channels # use same size for highway #
        self.enc_cnn_ksize = enc_cnn_ksize
        self.enc_cnn_strides = enc_cnn_strides
        self.enc_cnn_do =  ConfigParser.list_parser(enc_cnn_do, len(enc_cnn_ksize))
        self.enc_cnn_act = enc_cnn_act

        self.dec_emb_size = dec_emb_size
        self.dec_emb_do = dec_emb_do
        self.dec_rnn_sizes = dec_rnn_sizes
        self.dec_rnn_cfgs = ConfigParser.list_parser(dec_rnn_cfgs, len(dec_rnn_sizes)) 
        self.dec_rnn_do = ConfigParser.list_parser(dec_rnn_do, len(dec_rnn_sizes))
        self.dec_cfg = dec_cfg
        self.att_cfg = att_cfg
        
        # modules #
        # init encoder #
        self.enc_fnn = nn.ModuleList()
        prev_size = enc_in_size
        for ii in range(len(enc_fnn_sizes)) :
            self.enc_fnn.append(nn.Linear(prev_size, enc_fnn_sizes[ii]))
            prev_size = enc_fnn_sizes[ii]

        self.enc_cnn = nn.ModuleList()
        self.use_pad1 = []
        # batch x ndim x seq x 1#
        for ii in range(len(enc_cnn_ksize)) :
            self.enc_cnn.append(nn.Conv2d(prev_size, enc_cnn_channels, 
                kernel_size=(self.enc_cnn_ksize[ii], 1), stride=(self.enc_cnn_strides[ii], 1), 
                padding=((self.enc_cnn_ksize[ii]-1)//2, 0)))
            self.use_pad1.append(True if self.enc_cnn_ksize[ii]%2 == 0 else False)
            prev_size = enc_cnn_channels

        final_enc_size = prev_size
        # init position embedding function #
        self.pos_emb = nn.Linear(1, final_enc_size)

        # init decoder #
        self.dec_emb = nn.Embedding(self.dec_in_size, dec_emb_size, padding_idx=None)
        prev_size = dec_emb_size
        _dec_rnn_cfgs = ConfigParser.list_parser(dec_rnn_cfgs, len(dec_rnn_sizes))
        for ii in range(len(dec_rnn_sizes)) :
            _type = _dec_rnn_cfgs[ii]['type']
            if re.match('stateful.*cell', _type) is None :
                _dec_rnn_cfgs[ii]['type'] = 'stateful_{}cell'.format(_type)
        # TODO : dec_cfg #
        self.dec = decoder.StandardDecoder(att_cfg, final_enc_size, dec_emb_size, 
                dec_rnn_sizes, _dec_rnn_cfgs, dec_rnn_do)
        self.pre_softmax = nn.Linear(self.dec.output_size, n_class)
        pass 

    def get_config(self) :
        # TODO
        return {'class':str(self.__class__),
                'enc_in_size':self.enc_in_size,
                'dec_in_size':self.dec_in_size,
                'n_class':self.n_class,
                'enc_fnn_sizes':self.enc_fnn_sizes,
                'enc_fnn_act':self.enc_fnn_act,
                'enc_fnn_do':self.enc_fnn_do,
                'enc_cnn_channels':self.enc_cnn_channels,
                'enc_cnn_strides':self.enc_cnn_strides,
                'enc_cnn_ksize':self.enc_cnn_ksize,
                'enc_cnn_do':self.enc_cnn_do,
                'enc_cnn_act':self.enc_cnn_act,
                'dec_emb_size':self.dec_emb_size,
                'dec_emb_do':self.dec_emb_do,
                'dec_rnn_sizes':self.dec_rnn_sizes,
                'dec_rnn_cfgs':self.dec_rnn_cfgs,
                'dec_rnn_do':self.dec_rnn_do,
                'dec_cfg':self.dec_cfg,
                'att_cfg':self.att_cfg,
                }
        pass

    def encode(self, input, src_len=None) :
        """
        input : (batch x max_src_len x in_size)
        mask : (batch x max_src_len)
        """
        batch, max_src_len, in_size = input.size()

        if src_len is None :
            src_len = [max_src_len] * batch
        res = input.view(batch * max_src_len, in_size)
        enc_fnn_act = getattr(F, self.enc_fnn_act)
        for ii in range(len(self.enc_fnn)) :
            res = F.dropout(enc_fnn_act(self.enc_fnn[ii](res)), self.enc_fnn_do[ii], self.training)
            pass
        # res = batch * max_src_len x ndim #
        res = res.view(batch, max_src_len, res.size(1)).transpose(1,2).unsqueeze(3)
        # res = batch x ndim x src_len x 1 #
        enc_cnn_act = getattr(F, self.enc_cnn_act)
        for ii in range(len(self.enc_cnn)) :
            if self.use_pad1[ii] :
                res = F.pad(res, (0, 0, 0, 1))
            res = self.enc_cnn[ii](res)
            res = enc_cnn_act(res)
            src_len = [x//self.enc_cnn_strides[ii] for x in src_len]
            pass
        res = res.squeeze(3).transpose(1, 2) # batch x src_len x ndim #
        # add position embedding #
        _pos_arr = np.arange(0, res.size(1)).astype('float32') # src_len #
        _pos_arr = np.repeat(_pos_arr[np.newaxis, :], batch, 0) # batch x src_len #
        _pos_arr /= np.array(src_len)[:, np.newaxis] # divide for relative position #
        _pos_arr = tensorauto(self, torch.from_numpy(_pos_arr))
        _pos_var = Variable(_pos_arr.view(batch * _pos_arr.size(1), 1))
        # TODO : absolute or relative position #
        res_pos = self.pos_emb(_pos_var)
        res_pos = res_pos.view(batch, _pos_arr.size(1), -1)
        ctx = res + res_pos # TODO : sum or concat ? #
        # create mask if required #
        if src_len is not None :
            ctx_mask = torchauto(self).FloatTensor(batch, ctx.size(1)).zero_()
            for ii in range(batch) :
                ctx_mask[ii, 0:src_len[ii]] = 1.0
            ctx_mask = Variable(ctx_mask)
        else : 
            ctx_mask = None
        self.dec.set_ctx(ctx, ctx_mask)

    def reset(self) :
        self.dec.reset()

    def decode(self, y_tm1, mask=None) :
        assert y_tm1.dim() == 1, "batchsize only"
        res = self.dec_emb(y_tm1)
        if self.dec_emb_do > 0.0 :
            res = F.dropout(res, self.dec_emb_do, self.training)
        res = self.dec(res, mask)
        return self.pre_softmax(res['dec_output']), res 
    pass
