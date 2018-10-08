
import sys
import time

import numpy as np
import json

# pytorch #
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils.rnn import pack_padded_sequence as pack, pad_packed_sequence as unpack

from torch.autograd import Variable

# torchev #
from torchev.generator import generator_rnn

# utilbox #
from utilbox.config_util import ConfigParser

class FNN_RNN_IVEC_CTC(nn.Module) :
    def __init__(self, in_size, n_class, fnn_sizes, rnn_sizes, 
            do_fnn=0.0, do_rnn=0.0, 
            downsampling=None,
            fnn_act='tanh', rnn_cfgs='{"type":"lstm", "bi":true}',
            ivec_cfg={"type":"concat"}, ivec_dim=100,
            use_pack=False,
            train=True,
            ) :
        super(FNN_RNN_IVEC_CTC, self).__init__()

        self.in_size = in_size
        self.n_class = n_class
        self.fnn_sizes = fnn_sizes
        self.rnn_sizes = rnn_sizes
        self.downsampling = downsampling
        self.do_fnn = do_fnn if isinstance(do_fnn, list) and len(do_fnn) == len(fnn_sizes) else list(do_fnn)*len(self.fnn_sizes)
        self.do_rnn = do_rnn if isinstance(do_rnn, list) and len(do_rnn) == len(rnn_sizes) else list(do_rnn)*len(self.rnn_sizes)
        self.fnn_act = fnn_act
        self.rnn_cfgs = rnn_cfgs
        
        self.ivec_cfg = ivec_cfg
        self.ivec_dim = ivec_dim
        self.use_pack = use_pack
        
        # modules #
        self.fnn = nn.ModuleList()
        prev_size = in_size
        for ii in range(len(fnn_sizes)) :
            self.fnn.append(nn.Linear(prev_size, fnn_sizes[ii]))
            prev_size = fnn_sizes[ii]
        
        self.rnn = nn.ModuleList()
        _rnn_cfgs = ConfigParser.list_parser(rnn_cfgs, len(rnn_sizes))
        for ii in range(len(rnn_sizes)) :
            _rnn_cfg = {}
            _rnn_cfg['type'] = _rnn_cfgs[ii]['type']
            _rnn_cfg['args'] = [prev_size, rnn_sizes[ii], 1, True, False, 0, _rnn_cfgs[ii]['bi']]
            self.rnn.append(generator_rnn(_rnn_cfg))
             
            prev_size = rnn_sizes[ii] * (2 if _rnn_cfgs[ii]['bi'] else 1)

        self.pre_softmax = nn.Linear(prev_size, n_class)

        ### extension for ivec ###
        if ivec_cfg['type'] == 'concat' :
            self.aug_layer = nn.Linear(ivec_dim, fnn_sizes[0])
        elif ivec_cfg['type'] == 'aug_hid' : 
            """
            main_param : hid x in
            """
            _in_size = in_size
            _hid_size = fnn_sizes[0]
            self.aug_layer = nn.Linear(ivec_dim, _hid_size)
            self.scale = ivec_cfg.get('scale', 1.0)
            def fn_gen_params(ivec_feat) :
                return self.aug_layer(ivec_feat)
            def fn_aug_params(main_param, aug_param) :
                return main_param + self.scale * aug_param.t().expand_as(main_param)

        elif ivec_cfg['type'] == 'aug_in' :
            _in_size = in_size
            _hid_size = fnn_sizes[0]
            self.aug_layer = nn.Linear(ivec_dim, _in_size)
            self.scale = ivec_cfg.get('scale', 1.0)
            def fn_gen_params(ivec_feat) :
                return self.aug_layer(ivec_feat)
            def fn_aug_params(main_param, aug_param) :
                return main_param + self.scale * aug_param.expand_as(main_param)

        elif ivec_cfg['type'] == 'aug_lr' :
            _rank = ivec_cfg.get('rank', 3)
            _in_size = in_size
            _hid_size = fnn_sizes[0]
            self.aug_layer_in2lr = nn.Linear(ivec_dim, _in_size*_rank)
            self.aug_layer_hid2lr = nn.Linear(ivec_dim, _hid_size*_rank)
            self.scale = ivec_cfg.get('scale', 1.0)
            def fn_gen_params(ivec_feat) :
                _mat_in2lr = self.aug_layer_in2lr(ivec_feat)
                _mat_hid2lr = self.aug_layer_hid2lr(ivec_feat)
                _mat_in2lr = _mat_in2lr.view(_rank, _in_size)
                _mat_hid2lr = _mat_hid2lr.view(_hid_size, _rank)
                _mat_in2hid = torch.mm(_mat_hid2lr, _mat_in2lr)
                return _mat_in2hid
            def fn_aug_params(main_param, aug_param) :
                return main_param + self.scale * aug_param
        else :
            raise NotImplementedError
        if ivec_cfg['type'] != 'concat' :
            self.fn_gen_params = fn_gen_params
            self.fn_aug_params = fn_aug_params
        pass

    def forward(self, x, src_len=None, ivec_feat=None) :
        """
        x : (batch x seq x ndim) 
        mask : (batch x seq)
        """
        batchsize, seqlen, ndim = x.size()
        
        if src_len is None :
            src_len = [seqlen] * batchsize

        ### FNN ###
        # apply augmentation first layer #
        # TODO : dynamic layer # 
        assert self.ivec_dim >= ivec_feat.size(1)
        ivec_feat = ivec_feat[:, 0:self.ivec_dim]
        if self.ivec_cfg['type'] == 'concat' :
            # calculate h from ivec #
            res_ivec = self.aug_layer(ivec_feat)
            res_ivec = res_ivec.unsqueeze(1).expand(batchsize, seqlen, res_ivec.size(1))
            res_ivec = res_ivec.contiguous().view(seqlen * batchsize, -1)
            
            res_main = x.contiguous().view(seqlen * batchsize, ndim)
            res_main = self.fnn[0](res_main)

            res = res_main + res_ivec
        else :
            _res_list = []
            for ii in range(batchsize) :
                _aug_param = self.fn_gen_params(ivec_feat[ii:ii+1])
                _main_aug_param = self.fn_aug_params(self.fnn[0].weight, _aug_param)
                _main_bias = self.fnn[0].bias
                res_ii = F.linear(x[ii], _main_aug_param, _main_bias)
                _res_list.append(res_ii)
            res = torch.stack(_res_list)
            pass
        res = getattr(F, self.fnn_act)(res)
        res = F.dropout(res, self.do_fnn[0], training=self.training)
        prev_size = res.size(1)

        for ii in range(1, len(self.fnn_sizes)) :
            res = getattr(F, self.fnn_act)(self.fnn[ii](res))
            res = F.dropout(res, self.do_fnn[ii], training=self.training)
         
        ### RNN ###
        # convert shape for RNN #
        res = res.view(batchsize, seqlen, -1)
        for ii in range(len(self.rnn_sizes)) :
            # AVOID pack, slow !!! #
            if self.use_pack :
                res = pack(res, src_len, batch_first=True)
                res = self.rnn[ii](res)[0] # get h only #
                res,_ = unpack(res, batch_first=True)
            else :
                res = self.rnn[ii](res)[0] # get h only #
            
            if self.downsampling[ii] == True :
                res = res[:, 1::2]
                src_len = [x // 2 for x in src_len]
                pass

            res = F.dropout(res, self.do_rnn[ii], training=self.training)

        ### PRE SOFTMAX ###
        batchsize, seqlen_final, ndim_final = res.size()
        res = res.view(seqlen_final * batchsize, ndim_final)

        res = self.pre_softmax(res)
        res = res.view(batchsize, seqlen_final, -1) 
        res = res.transpose(1, 0)
        return res, Variable(torch.IntTensor(src_len))


    def get_config(self) :
        return {'class':str(self.__class__),
                'in_size':self.in_size,
                'n_class':self.n_class,
                'fnn_sizes':self.fnn_sizes,
                'rnn_sizes':self.rnn_sizes,
                'do_fnn':self.do_fnn,
                'do_rnn':self.do_rnn,
                'downsampling':self.downsampling,
                'fnn_act':self.fnn_act,
                'rnn_cfgs':self.rnn_cfgs,
                'ivec_cfg':self.ivec_cfg,
                'ivec_dim':self.ivec_dim,
                'use_pack':self.use_pack
                }

    pass
