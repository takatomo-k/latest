
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
from torchev.generator import generator_rnn, generator_attention, generator_act_fn, generator_act_module
from torchev.custom import decoder
from torchev.utils.helper import torchauto
from torchev.utils.mask_util import generate_seq_mask
from torchev.nn.modules import LayerNorm

# utilbox #
from utilbox.config_util import ConfigParser


class EncRNNDecRNNAtt(nn.Module) :
    def __init__(self, enc_in_size, dec_in_size, dec_out_size,
            enc_fnn_sizes=[512], enc_fnn_act='LeakyReLU', enc_fnn_do=0.25,
            enc_rnn_sizes=[256, 256, 256], enc_rnn_cfgs={"type":"lstm", "bi":True}, enc_rnn_do=0.25,
            downsampling=[False, True, True],
            dec_emb_size=256, dec_emb_do=0.25, dec_emb_tied_weight=True, 
            # tying weight from char/word embedding with softmax layer
            dec_rnn_sizes=[512, 512], dec_rnn_cfgs={"type":"lstm"}, dec_rnn_do=0.25,
            dec_cfg={"type":"standard_decoder"}, 
            att_cfg={"type":"mlp"},
            use_layernorm=False,
            ) :
        super().__init__()

        self.enc_in_size = enc_in_size
        self.dec_in_size = dec_in_size
        self.dec_out_size = dec_out_size
        self.enc_fnn_sizes = enc_fnn_sizes
        self.enc_fnn_act = enc_fnn_act
        self.enc_fnn_do = ConfigParser.list_parser(enc_fnn_do, len(enc_fnn_sizes))
        self.enc_rnn_sizes = enc_rnn_sizes
        self.enc_rnn_cfgs = enc_rnn_cfgs
        self.enc_rnn_do =  ConfigParser.list_parser(enc_rnn_do, len(enc_rnn_sizes))
        self.downsampling = ConfigParser.list_parser(downsampling, len(enc_rnn_sizes))

        self.dec_emb_size = dec_emb_size
        self.dec_emb_do = dec_emb_do
        self.dec_emb_tied_weight = dec_emb_tied_weight
        self.dec_rnn_sizes = dec_rnn_sizes
        self.dec_rnn_cfgs = ConfigParser.list_parser(dec_rnn_cfgs, len(dec_rnn_sizes)) 
        self.dec_rnn_do = ConfigParser.list_parser(dec_rnn_do, len(dec_rnn_sizes))
        self.dec_cfg = dec_cfg
        self.att_cfg = att_cfg

        self.use_layernorm = use_layernorm
        if self.use_layernorm == True :
            raise ValueError("LayerNorm is not implemented yet")

        # modules #
        # init encoder #
        prev_size = enc_in_size
        _tmp = []
        for ii in range(len(enc_fnn_sizes)) :
            _tmp.append(nn.Linear(prev_size, enc_fnn_sizes[ii]))
            if use_layernorm :
                _tmp.append(LayerNorm(enc_fnn_sizes[ii]))
            _tmp.append(generator_act_module(enc_fnn_act))
            _tmp.append(nn.Dropout(p=self.enc_fnn_do[ii]))
            prev_size = enc_fnn_sizes[ii]
        self.enc_fnn_lyr = nn.Sequential(*_tmp)

        self.enc_rnn_lyr = nn.ModuleList()
        _enc_rnn_cfgs = ConfigParser.list_parser(enc_rnn_cfgs, len(enc_rnn_sizes))
        for ii in range(len(enc_rnn_sizes)) :
            _rnn_cfg = {}
            _rnn_cfg['type'] = _enc_rnn_cfgs[ii]['type']
            _rnn_cfg['args'] = [prev_size, enc_rnn_sizes[ii], 1, True, True, 0, _enc_rnn_cfgs[ii]['bi']]
            self.enc_rnn_lyr.append(generator_rnn(_rnn_cfg))
            prev_size = enc_rnn_sizes[ii] * (2 if _enc_rnn_cfgs[ii]['bi'] else 1)
        final_enc_size = prev_size
        # init decoder #
        self.dec_emb_lyr = nn.Embedding(self.dec_in_size, dec_emb_size, padding_idx=None)
        prev_size = dec_emb_size
        _dec_rnn_cfgs = ConfigParser.list_parser(dec_rnn_cfgs, len(dec_rnn_sizes))
        for ii in range(len(dec_rnn_sizes)) :
            _type = _dec_rnn_cfgs[ii]['type']
            if re.match('stateful.*cell', _type) is None :
                _dec_rnn_cfgs[ii]['type'] = 'stateful_{}cell'.format(_type)
        # TODO : dec_cfg #
        assert 'type' in dec_cfg, "decoder type need to be defined"
        if dec_cfg['type'] == 'standard_decoder' :
            _tmp_dec_cfg = dict(dec_cfg)
            del _tmp_dec_cfg['type'] #
            self.dec_att_lyr = decoder.StandardDecoder(att_cfg=att_cfg, ctx_size=final_enc_size, in_size=dec_emb_size, 
                    rnn_sizes=dec_rnn_sizes, rnn_cfgs=_dec_rnn_cfgs, rnn_do=dec_rnn_do, **_tmp_dec_cfg)
        else :
            raise NotImplementedError("decoder type {} is not found".format(dec_cfg['type']))
        self.dec_presoftmax_lyr = nn.Linear(self.dec_att_lyr.output_size, dec_out_size)
        if dec_emb_tied_weight :
            assert dec_out_size == dec_in_size and self.dec_emb_lyr.embedding_dim == self.dec_presoftmax_lyr.in_features
            self.dec_presoftmax_lyr.weight = self.dec_emb_lyr.weight
        pass 

    def get_config(self) :
        # TODO
        return {'class':str(self.__class__),
                'enc_in_size':self.enc_in_size,
                'dec_in_size':self.dec_in_size,
                'dec_out_size':self.dec_out_size,
                'enc_fnn_sizes':self.enc_fnn_sizes,
                'enc_fnn_act':self.enc_fnn_act,
                'enc_fnn_do':self.enc_fnn_do,
                'enc_rnn_sizes':self.enc_rnn_sizes,
                'enc_rnn_cfgs':self.enc_rnn_cfgs,
                'enc_rnn_do':self.enc_rnn_do,
                'downsampling':self.downsampling,
                'dec_emb_size':self.dec_emb_size,
                'dec_emb_do':self.dec_emb_do,
                'dec_emb_tied_weight':self.dec_emb_tied_weight,
                'dec_rnn_sizes':self.dec_rnn_sizes,
                'dec_rnn_cfgs':self.dec_rnn_cfgs,
                'dec_rnn_do':self.dec_rnn_do,
                'dec_cfg':self.dec_cfg,
                'att_cfg':self.att_cfg,
                'use_layernorm':self.use_layernorm
                }
    
    @property
    def state(self) :
        return (self.dec_att_lyr.state, )

    @state.setter
    def state(self, value) :
        self.dec_att_lyr.state = value[0]

    def encode(self, input, src_len=None) :
        """
        input : (batch x max_src_len x in_size)
        mask : (batch x max_src_len)
        """
        batch, max_src_len, in_size = input.size()

        if src_len is None :
            src_len = [max_src_len] * batch
        res = input.view(batch * max_src_len, in_size)
        
        res = self.enc_fnn_lyr(res)

        res = res.view(batch, max_src_len, -1)
        for ii in range(len(self.enc_rnn_lyr)) :
            res = pack(res, src_len, batch_first=True)
            res = self.enc_rnn_lyr[ii](res)[0] # get h only #
            res,_ = unpack(res, batch_first=True)
            res = F.dropout(res, self.enc_rnn_do[ii], self.training)
            if self.downsampling[ii] == True :
                res = res[:, 1::2]
                src_len = [x // 2 for x in src_len]
                pass
        ctx = res
        # create mask if required #
        if src_len is not None :
            ctx_mask = Variable(generate_seq_mask(src_len, self, max_len=ctx.size(1)))
        else : 
            ctx_mask = None
        self.dec_att_lyr.set_ctx(ctx, ctx_mask)

    def reset(self) :
        self.dec_att_lyr.reset()

    def decode(self, y_tm1, mask=None) :
        assert y_tm1.dim() == 1, "batchsize only"
        res = self.dec_emb_lyr(y_tm1)
        if self.dec_emb_do > 0.0 :
            res = F.dropout(res, self.dec_emb_do, self.training)
        res = self.dec_att_lyr(res, mask)
        return self.dec_presoftmax_lyr(res['dec_output']), res 
