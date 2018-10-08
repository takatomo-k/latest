
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
from torchev.generator import generator_rnn, generator_attention, generator_act_fn
from torchev.custom import decoder
from torchev.utils.helper import torchauto, tensorauto
from torchev.nn.modules import Conv2dEv
from torchev.custom import GatedConv2dLinearUnit

# utilbox #
from utilbox.config_util import ConfigParser


class ENCCNNRNN_DECRNN_ATT_ASR(nn.Module) :
    def __init__(self, enc_in_size, dec_in_size, n_class,
            enc_cnn_sizes=[80, 25, 10, 5], enc_cnn_act='leaky_relu', enc_cnn_stride=[4, 2, 1, 1], enc_cnn_do=0.0, enc_cnn_filter=256,
            enc_cnn_gated=[False, False, False, False], use_bn=False,
            enc_nin_filter=[128,128],
            enc_rnn_sizes=[256, 256, 256], enc_rnn_cfgs={"type":"lstm", "bi":True}, enc_rnn_do=0.25,
            downsampling=None,
            dec_emb_size=64, dec_emb_do=0.0,
            dec_rnn_sizes=[512, 512], dec_rnn_cfgs={"type":"lstm"}, dec_rnn_do=0.25,
            dec_cfg={"type":"standard_decoder"}, 
            att_cfg={"type":"mlp"},
            ) :
        super(ENCCNNRNN_DECRNN_ATT_ASR, self).__init__()

        self.enc_in_size = enc_in_size
        self.dec_in_size = dec_in_size
        self.n_class = n_class
        self.enc_cnn_sizes = enc_cnn_sizes
        self.enc_cnn_act = enc_cnn_act
        self.enc_cnn_gated = ConfigParser.list_parser(enc_cnn_gated, len(enc_cnn_sizes))
        self.enc_cnn_stride = enc_cnn_stride
        self.enc_cnn_filter = ConfigParser.list_parser(enc_cnn_filter, len(enc_cnn_sizes))
        self.enc_cnn_do = ConfigParser.list_parser(enc_cnn_do, len(enc_cnn_sizes))
        self.use_bn = use_bn
        self.enc_nin_filter = enc_nin_filter

        self.enc_rnn_sizes = enc_rnn_sizes # kernel size #
        self.enc_rnn_cfgs = enc_rnn_cfgs
        self.enc_rnn_do =  ConfigParser.list_parser(enc_rnn_do, len(enc_rnn_sizes))

        self.downsampling = ConfigParser.list_parser(downsampling, len(enc_rnn_sizes))

        self.dec_emb_size = dec_emb_size
        self.dec_emb_do = dec_emb_do
        self.dec_rnn_sizes = dec_rnn_sizes
        self.dec_rnn_cfgs = ConfigParser.list_parser(dec_rnn_cfgs, len(dec_rnn_sizes)) 
        self.dec_rnn_do = ConfigParser.list_parser(dec_rnn_do, len(dec_rnn_sizes))
        self.dec_cfg = dec_cfg
        self.att_cfg = att_cfg

        # modules #
        # init encoder #
        self.enc_cnn = nn.ModuleList()
        self.enc_cnn_bn = nn.ModuleList()
        prev_size = enc_in_size 
        prev_ch = 1
        for ii in range(len(enc_cnn_sizes)) :
            if self.enc_cnn_gated[ii] :
                _cnn_lyr = GatedConv2dLinearUnit 
            else :
                _cnn_lyr = Conv2dEv
            self.enc_cnn.append(_cnn_lyr(prev_ch, self.enc_cnn_filter[ii], (self.enc_cnn_sizes[ii], 1), 
                stride=(self.enc_cnn_stride[ii], 1), padding='valid', dilation=1))
            self.enc_cnn_bn.append(nn.BatchNorm2d(self.enc_cnn_filter[ii]))
            prev_size = enc_cnn_sizes[ii]
            prev_ch = self.enc_cnn_filter[ii]
        
        self.enc_nin = nn.ModuleList()
        for ii in range(len(enc_nin_filter)) :
            self.enc_nin = self.enc_nin.append(nn.Conv2d(prev_ch, enc_nin_filter[ii], [1, 1]))
            prev_ch = enc_nin_filter[ii]
        self.enc_raw_enc = nn.ModuleList([self.enc_cnn, self.enc_cnn_bn, self.enc_nin])
        prev_size = prev_ch # global pooling after conv #
        self.enc_rnn = nn.ModuleList()
        _enc_rnn_cfgs = ConfigParser.list_parser(enc_rnn_cfgs, len(enc_rnn_sizes))
        for ii in range(len(enc_rnn_sizes)) :
            _rnn_cfg = {}
            _rnn_cfg['type'] = _enc_rnn_cfgs[ii]['type']
            _rnn_cfg['args'] = [prev_size, enc_rnn_sizes[ii], 1, True, True, 0, _enc_rnn_cfgs[ii]['bi']]
            self.enc_rnn.append(generator_rnn(_rnn_cfg))
            prev_size = enc_rnn_sizes[ii] * (2 if _enc_rnn_cfgs[ii]['bi'] else 1)
        final_enc_size = prev_size
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

    # def init_conv_dct(self, weight) :
        # out_ch, in_ch, height, width = weight.size()
        # assert width == 1
        # # template #
        # kernel_dct = torch.FloatTensor(weight.size()).zero_()
        # basis = torch.arange(0, height)
        # for oo in range(out_ch) :
            # freq = (oo+1) / out_ch
            # for ii in range(in_ch) :
                # kernel_dct[oo, ii, :, 0] = torch.cos(np.pi / height * basis * freq)
        # weight.data = tensorauto(self, kernel_dct)
        # weight.requires_grad = False
        # pass

    @property
    def state(self) :
        return (self.dec.state, )

    @state.setter
    def state(self, value) :
        self.dec.state = value[0]

    def get_config(self) :
        # TODO
        return {'class':str(self.__class__),
                'enc_in_size':self.enc_in_size,
                'dec_in_size':self.dec_in_size,
                'n_class':self.n_class,
                'enc_cnn_sizes':self.enc_cnn_sizes,
                'enc_cnn_act':self.enc_cnn_act,
                'enc_cnn_gated':self.enc_cnn_gated,
                'enc_cnn_filter':self.enc_cnn_filter,
                'enc_cnn_stride':self.enc_cnn_stride,
                'enc_cnn_do':self.enc_cnn_do,
                'use_bn':self.use_bn,
                'enc_nin_filter':self.enc_nin_filter,
                'enc_rnn_sizes':self.enc_rnn_sizes,
                'enc_rnn_cfgs':self.enc_rnn_cfgs,
                'enc_rnn_do':self.enc_rnn_do,
                'downsampling':self.downsampling,
                'dec_emb_size':self.dec_emb_size,
                'dec_emb_do':self.dec_emb_do,
                'dec_rnn_sizes':self.dec_rnn_sizes,
                'dec_rnn_cfgs':self.dec_rnn_cfgs,
                'dec_rnn_do':self.dec_rnn_do,
                'dec_cfg':self.dec_cfg,
                'att_cfg':self.att_cfg,
                }
        pass

    def encode_raw(self, input, src_len=None) :
        batch, max_src_len, in_size = input.size()

        if src_len is None :
            src_len = [max_src_len] * batch
        res = input.view(batch * max_src_len, 1, in_size, 1)
        enc_cnn_act = generator_act_fn(self.enc_cnn_act)
        # apply conv #
        for ii in range(len(self.enc_cnn)) :
            res = F.dropout(enc_cnn_act(self.enc_cnn[ii](res)), self.enc_cnn_do[ii], self.training)
            if self.use_bn :
                res = self.enc_cnn_bn[ii](res)
                pass
            pass

        # apply NiN #
        for ii in range(len(self.enc_nin)) :
            res = enc_cnn_act(self.enc_nin[ii](res))
            
        final_h, final_w = res.size()[2:]
        res = F.avg_pool2d(res, (final_h, final_w)) # (batch * seq_len) x ch x 1 x 1 #
        res = res.unsqueeze(2).unsqueeze(2) # (batch * seq_len) x ch #
        res = res.view(batch, max_src_len, -1)
        return res
        pass

    def encode(self, input, src_len=None) :
        """
        input : (batch x max_src_len x in_size)
        mask : (batch x max_src_len)
        """
        batch, max_src_len, in_size = input.size()
        
        # apply raw -> feat #
        res = self.encode_raw(input, src_len)

        for ii in range(len(self.enc_rnn)) :
            res = pack(res, src_len, batch_first=True)
            res = self.enc_rnn[ii](res)[0] # get h only #
            res,_ = unpack(res, batch_first=True)
            res = F.dropout(res, self.enc_rnn_do[ii], self.training)
            if self.downsampling[ii] == True :
                res = res[:, 1::2]
                src_len = [x // 2 for x in src_len]
                pass
        ctx = res
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
