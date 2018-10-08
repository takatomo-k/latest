"""
Tacotron model with prediction only for Mel-spectrogram

Mel -> Linear is handled by inverter.py

The reason is the inverter shouldn't be optimized togather with the core, 
but only after the core model converged
"""

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
from torch.nn import init

# torchev #
from torchev.generator import generator_rnn, generator_attention, generator_act_fn, generator_act_module
from torchev.custom import decoder
from torchev.custom.composite import cbhg
from torchev.utils.helper import torchauto, tensorauto
from torchev.utils.mask_util import generate_seq_mask
from torchev.nn import Conv1dEv

# utilbox #
from utilbox.config_util import ConfigParser
from enum import Enum

class TacotronType(Enum) :
    SINGLE_SPEAKER = 'single_speaker'
    MULTI_SPEAKER = 'multi_speaker'

class TacotronV1Core(nn.Module) :
    TYPE = TacotronType.SINGLE_SPEAKER
    # Ref : Tacotron (Google Brain)
    def __init__(self, enc_in_size, dec_in_size, dec_out_size,
            enc_emb_size=256, enc_emb_do=0.0,
            enc_prenet_size=[256, 128], enc_prenet_do=[0.5, 0.5], enc_prenet_fn='leaky_relu',
            dec_prenet_size=[256, 128], dec_prenet_do=[0.5, 0.5], dec_prenet_fn='leaky_relu',
            dec_rnn_sizes=[256, 256], dec_rnn_cfgs={"type":"lstm"}, dec_rnn_do=0.0,
            dec_cfg={"type":"standard_decoder"}, 
            att_cfg={"type":"mlp"},
            dec_core_gen_size=[512], dec_core_gen_fn='leaky_relu', dec_core_gen_do=0.0,
            # CBHG # 
            enc_cbhg_cfg={},
            # FRAME ENDING #
            dec_bern_end_size=[256], dec_bern_end_fn='LeakyReLU', dec_bern_end_do=0.0,
            # OPTIONAL #
            dec_in_range=None
            ) :
        """
        Args:
            enc_in_size : size of vocab
            dec_in_size : input (mel) dim size
            dec_out_size : output (mel) dim size (usually same as dec_in_size)
            dec_in_range : 
                pair of integer [x, y] \in [0, dec_in_size], 
                all dims outside this pair will be masked as 0
                in Tacotron paper, they only use last time-step instead of all group
        """
        super().__init__()
        self.enc_in_size = enc_in_size
        self.dec_in_size = dec_in_size
        self.dec_out_size = dec_out_size # mel spec dim size
        self.enc_emb_size = enc_emb_size 
        self.enc_emb_do = enc_emb_do
        self.enc_prenet_size = enc_prenet_size
        self.enc_prenet_do = ConfigParser.list_parser(enc_prenet_do, len(enc_prenet_size))
        self.enc_prenet_fn = enc_prenet_fn
        self.dec_prenet_size = dec_prenet_size
        self.dec_prenet_do = ConfigParser.list_parser(dec_prenet_do, len(dec_prenet_size))
        self.dec_prenet_fn = dec_prenet_fn
        self.dec_rnn_sizes = dec_rnn_sizes
        self.dec_rnn_cfgs = dec_rnn_cfgs
        self.dec_rnn_do = dec_rnn_do
        self.dec_core_gen_size = dec_core_gen_size
        self.dec_core_gen_fn = dec_core_gen_fn
        self.dec_core_gen_do = ConfigParser.list_parser(dec_core_gen_do, len(dec_core_gen_size))
        self.dec_cfg = dec_cfg
        self.att_cfg = att_cfg

        # FRAME ENDING #
        self.dec_bern_end_size = dec_bern_end_size
        self.dec_bern_end_fn = dec_bern_end_fn
        self.dec_bern_end_do = ConfigParser.list_parser(dec_bern_end_do)

        # OPTIONAL #
        self.dec_in_range = dec_in_range
        if self.dec_in_range is not None :
            assert isinstance(self.dec_in_range, (list, tuple)) \
                    and len(self.dec_in_range) == 2

        # CBHG config #
        self.enc_cbhg_cfg = ConfigParser.item_parser(enc_cbhg_cfg)


        self.enc_emb_lyr = nn.Embedding(enc_in_size, enc_emb_size)
        # init enc prenet #
        self.enc_prenet_lyr = nn.ModuleList()
        prev_size = enc_emb_size
        for ii in range(len(self.enc_prenet_size)) :
            self.enc_prenet_lyr.append(nn.Linear(prev_size, enc_prenet_size[ii]))
            prev_size = enc_prenet_size[ii]
        # init enc middle #
        self.enc_core_lyr = cbhg.CBHG1d(prev_size, **enc_cbhg_cfg)
        # init dec prenet #
        self.dec_prenet_lyr = nn.ModuleList()
        prev_size = dec_in_size if self.dec_in_range is None else ((self.dec_in_range[-1] or 0) - (self.dec_in_range[-2] or 0))
        for ii in range(len(self.dec_prenet_size)) :
            self.dec_prenet_lyr.append(nn.Linear(prev_size, dec_prenet_size[ii]))
            prev_size = dec_prenet_size[ii]

        # init dec rnn #
        _dec_rnn_cfgs = ConfigParser.list_parser(dec_rnn_cfgs, len(dec_rnn_sizes))
        for ii in range(len(dec_rnn_sizes)) :
            _type = _dec_rnn_cfgs[ii]['type']
            if re.match('stateful.*cell', _type) is None :
                _dec_rnn_cfgs[ii]['type'] = 'stateful_{}cell'.format(_type)
        # TODO : dec_cfg #
        final_enc_size = self.enc_core_lyr.output_size
        self.dec_att_lyr = decoder.StandardDecoder(att_cfg, final_enc_size, prev_size, 
                dec_rnn_sizes, dec_rnn_cfgs, dec_rnn_do)
        
        # init decoder layer melspec generator #
        prev_size = self.dec_att_lyr.output_size
        self.dec_core_gen_lyr = nn.ModuleList()
        for ii in range(len(self.dec_core_gen_size)) :
            self.dec_core_gen_lyr.append(nn.Linear(prev_size, self.dec_core_gen_size[ii])) 
            prev_size = self.dec_core_gen_size[ii]
        self.dec_core_gen_lyr.append(nn.Linear(prev_size, self.dec_out_size))

        # init decoder frame ending predictor #
        # p(t=STOP | dec_hid[t], y[t]) #
        _tmp = []
        prev_size = self.dec_att_lyr.output_size + self.dec_out_size
        for ii in range(len(dec_bern_end_size)) :
            _tmp.append(nn.Linear(prev_size, self.dec_bern_end_size[ii]))
            _tmp.append(generator_act_module(self.dec_bern_end_fn))
            _tmp.append(nn.Dropout(p=self.dec_bern_end_do[ii]))
            prev_size = self.dec_bern_end_size[ii]
        _tmp.append(nn.Linear(prev_size, 1))
        # output is logit, not transformed into sigmoid #
        self.dec_bernoulli_end_lyr = nn.Sequential(*_tmp)

    def get_config(self) :
        # TODO
        return {'class':str(self.__class__),
                'enc_in_size' : self.enc_in_size,
                'dec_in_size' : self.dec_in_size,
                'dec_out_size' : self.dec_out_size,
                'enc_emb_size' : self.enc_emb_size, 
                'enc_emb_do' : self.enc_emb_do,
                'enc_prenet_size' : self.enc_prenet_size,
                'enc_prenet_do' : self.enc_prenet_do,
                'enc_prenet_fn' :self.enc_prenet_fn,
                'dec_prenet_size' : self.dec_prenet_size,
                'dec_prenet_do' : self.dec_prenet_do,
                'dec_prenet_fn' : self.dec_prenet_fn,
                'dec_rnn_sizes' : self.dec_rnn_sizes,
                'dec_rnn_cfgs' : self.dec_rnn_cfgs,
                'dec_rnn_do' : self.dec_rnn_do,
                'dec_core_gen_size' : self.dec_core_gen_size,
                'dec_core_gen_do' : self.dec_core_gen_do,
                'dec_core_gen_fn' : self.dec_core_gen_fn,
                'dec_cfg' : self.dec_cfg,
                'att_cfg' : self.att_cfg,
                # FINAL FRAME #
                'dec_bern_end_size' : self.dec_bern_end_size,
                'dec_bern_end_do' : self.dec_bern_end_do,
                'dec_bern_end_fn' : self.dec_bern_end_fn,
                # CBHG config #
                'enc_cbhg_cfg':self.enc_cbhg_cfg,
                # OPTIONAL #
                'dec_in_range':self.dec_in_range,
                }

    def reset(self) :
        self.dec_att_lyr.reset()

    def mask_dec_feat(self, y_tm1) :
        res = y_tm1
        # OPTIONAL : apply input dimension mask if available
        if self.dec_in_range is not None :
            res = res[:, slice(*self.dec_in_range)]
        return res

    def encode(self, input, src_len=None) :
        """
        input : (batch x max_src_len)
        mask : (batch x max_src_len)
        """
        batch, max_src_len = input.size()

        if src_len is None :
            src_len = [max_src_len] * batch
        res = self.enc_emb_lyr(input) # batch x max_src_len x emb_dim #
        res = F.dropout(res, self.enc_emb_do, training=self.training)
        res = res.view(batch * max_src_len, -1) 
        for ii in range(len(self.enc_prenet_lyr)) :
            res = self.enc_prenet_lyr[ii](res)
            res = generator_act_fn(self.enc_prenet_fn)(res)
            res = F.dropout(res, p=self.enc_prenet_do[ii], training=self.training)
        res = res.view(batch, max_src_len, -1)
        res = self.enc_core_lyr(res, src_len)
        
        ctx = res

        if src_len is not None :
            ctx_mask = Variable(generate_seq_mask(src_len, self, max_len=ctx.size(1)))
        else : 
            ctx_mask = None
        
        self.ctx = ctx
        self.ctx_mask = ctx_mask
        self.src_len = src_len

        self.dec_att_lyr.set_ctx(ctx, ctx_mask)

    def reset(self) :
        self.dec_att_lyr.reset()

    def mask_dec_feat(self, y_tm1) :
        res = y_tm1
        # OPTIONAL : apply input dimension mask if available
        if self.dec_in_range is not None :
            res = res[:, slice(*self.dec_in_range)]
        return res

    def decode(self, y_tm1, mask=None) :
        """
        Return:
            res_first : core (Mel) prediction
            res_dec : decoder RNN Attention result
            res_bern_end : final frame prediction
        """
        assert y_tm1.dim() == 2, "batchsize x dec_in_size ( 1 timestep only)"

        res = y_tm1

        # OPTIONAL #
        res = self.mask_dec_feat(res)

        for ii in range(len(self.dec_prenet_lyr)) :
            res = self.dec_prenet_lyr[ii](res)
            res = generator_act_fn(self.dec_prenet_fn)(res)
            res = F.dropout(res, self.dec_prenet_do[ii], training=self.training)

        # compute decoder rnn #
        res_dec = self.dec_att_lyr(res, mask)
        res = res_dec['dec_output']
        # generate mel-spec prediction #
        res_first = res
        for ii in range(len(self.dec_core_gen_lyr)) :
            res_first = self.dec_core_gen_lyr[ii](res_first)
            if ii != len(self.dec_core_gen_lyr) - 1 : # if not last layer, apply act_fn & dropout
                res_first = generator_act_fn(self.dec_core_gen_fn)(res_first)
                res_first = F.dropout(res_first, self.dec_core_gen_do[ii], training=self.training)
        res_bern_end = self.dec_bernoulli_end_lyr(torch.cat([res_first, res_dec['dec_output']], 1).detach()) # stop gradient produce better result
        return res_first, res_dec, res_bern_end
    pass

class TacotronV1CoreMSpkContinuous(TacotronV1Core) :
    """
    Tacotron Multispeaker with continuous speaker vector (can be extracted from Deep Speaker or i-vector)
    """
    TYPE = TacotronType.MULTI_SPEAKER
    def __init__(self, speaker_emb_dim=256, 
            speaker_proj_size=[512], speaker_proj_fn='none', speaker_proj_do=0.0,
            speaker_integrate_fn='none', speaker_emb_scale=1.0,
            *args, **kwargs) :
        super().__init__(*args, **kwargs)
        self.speaker_emb_dim = speaker_emb_dim
        self.speaker_proj_size = speaker_proj_size
        self.speaker_proj_fn = speaker_proj_fn
        self.speaker_proj_do = ConfigParser.list_parser(speaker_proj_do, n=len(speaker_proj_size))
        self.speaker_emb_scale = speaker_emb_scale # scalar x spk_vector 
        # speaker_integrate_fn applied before non-linearity on decoder layer
        self.speaker_integrate_fn = speaker_integrate_fn 

        _tmp = []
        prev_size = speaker_emb_dim
        for ii in range(len(self.speaker_proj_size)) :
            _tmp.append(nn.Linear(prev_size, self.speaker_proj_size[ii]))
            _tmp.append(generator_act_module(self.speaker_proj_fn))
            _tmp.append(nn.Dropout(self.speaker_proj_do[ii]))
            prev_size = self.speaker_proj_size[ii]

        self.speaker_proj_lyr = nn.Sequential(*_tmp) 
        self.speaker_module_lyr = nn.Module()
        
        # speaker proj -> decoder prenet (last layer) #
        self.speaker_module_lyr.add_module('dec_proj_prenet_lyr', nn.Linear(prev_size, self.dec_prenet_lyr[-1].out_features))

        # speaker proj -> decoder regression core (first layer) #
        assert len(self.dec_core_gen_lyr) >= 1, "dec_core_gen_lyr must have atleast 1 layer"
        self.speaker_module_lyr.add_module('dec_proj_core_gen_lyr', nn.Linear(prev_size, self.dec_core_gen_lyr[0].out_features))
        pass
    
    def reset(self) :
        super().reset()
        self.speaker_vector = None

    def set_aux_info(self, aux_info) :
        assert 'speaker_vector' in aux_info
        self.speaker_vector = self.speaker_proj_lyr(aux_info['speaker_vector']) \
                * self.speaker_emb_scale

    def get_config(self) :
        _config = super().get_config()
        _config['speaker_emb_dim'] = self.speaker_emb_dim
        _config['speaker_proj_size'] = self.speaker_proj_size
        _config['speaker_proj_fn'] = self.speaker_proj_fn
        _config['speaker_proj_do'] = self.speaker_proj_do
        _config['speaker_emb_scale'] = self.speaker_emb_scale
        _config['speaker_integrate_fn'] = self.speaker_integrate_fn
        return _config

    def decode(self, y_tm1, mask=None) :
        """
        Return:
            res_first : core (Mel) prediction
            res_dec : decoder RNN Attention result
            res_bern_end : final frame prediction
        """
        assert y_tm1.dim() == 2, "batchsize x dec_in_size ( 1 timestep only)"
        assert self.speaker_vector is not None, "set speaker vector into with method set_aux_info"
        assert self.speaker_vector.shape[0] == y_tm1.shape[0] == self.ctx.shape[0], "batch size is different"

        res = y_tm1

        # OPTIONAL #
        res = self.mask_dec_feat(res)

        for ii in range(len(self.dec_prenet_lyr)) :
            res = self.dec_prenet_lyr[ii](res)
            if ii == len(self.dec_prenet_lyr)-1 : # last prenet layer
                res_spk = self.speaker_module_lyr.dec_proj_prenet_lyr(self.speaker_vector)
                res_spk = generator_act_fn(self.speaker_integrate_fn)(res_spk)
                res += res_spk
            res = generator_act_fn(self.dec_prenet_fn)(res)
            res = F.dropout(res, self.dec_prenet_do[ii], training=self.training)

        # compute decoder rnn #
        res_dec = self.dec_att_lyr(res, mask)
        res = res_dec['dec_output']
        # generate mel-spec prediction #
        res_first = res
        
        for ii in range(len(self.dec_core_gen_lyr)) :
            res_first = self.dec_core_gen_lyr[ii](res_first)

            if ii == 0 :
                # integrate speaker info #
                res_spk = self.speaker_module_lyr.dec_proj_core_gen_lyr(self.speaker_vector)
                res_spk = generator_act_fn(self.speaker_integrate_fn)(res_spk)
                res_first = res_first + res_spk

            if ii != len(self.dec_core_gen_lyr) - 1 : # if not last layer, apply act_fn & dropout
                res_first = generator_act_fn(self.dec_core_gen_fn)(res_first)
                res_first = F.dropout(res_first, self.dec_core_gen_do[ii], training=self.training)

        # predict frame ending #
        res_bern_end = self.dec_bernoulli_end_lyr(torch.cat([res_first, res_dec['dec_output']], 1).detach()) # stop gradient produce better result
        return res_first, res_dec, res_bern_end

############## TACOTRON 2 ################

class TacotronV2Core(nn.Module) :
    TYPE = TacotronType.SINGLE_SPEAKER
    """
    Simpler Tacotron (without CBHG module)
    """
    def __init__(self, enc_in_size, dec_in_size, dec_out_size,
            enc_emb_size=256, enc_emb_do=0.0,
            enc_conv_sizes=[5, 5, 5], enc_conv_filter=[256, 256, 256], 
            enc_conv_do=0.25, enc_conv_fn='LeakyReLU',

            enc_rnn_sizes=[256], enc_rnn_cfgs={"type":"lstm", 'bi':True}, enc_rnn_do=0.2,
            
            dec_prenet_size=[256, 256], dec_prenet_fn='leaky_relu', dec_prenet_do=0.25,

            dec_rnn_sizes=[512, 512], dec_rnn_cfgs={"type":"lstm"}, dec_rnn_do=0.2,

            dec_proj_size=[512, 512], dec_proj_fn='leaky_relu', dec_proj_do=0.25,

            dec_bern_end_size=[256], dec_bern_end_do=0.0, dec_bern_end_fn='LeakyReLU',

            dec_cfg={"type":"standard_decoder"}, 
            att_cfg={"type":"mlp_history", "kwargs":{"history_conv_ksize":[2, 4, 8]}}, # location sensitive attention
            # OPTIONAL #
            dec_in_range=None,
            use_bn=False, # Tacotron V2 default activate BatchNorm
            use_ln=False, # Use layer-normalization on feedforward
            ) :
        """
        Tacotron V2 
        
        Decoder generates 2 outputs mel + linear spec, use main for conditional input next step
        Args:
            enc_in_size : size of vocab
            dec_in_size : input (mel) dim size
            dec_out_size : output (mel/linear) dim size (usually same as dec_in_size)
            dec_in_range : 
                pair of integer [x, y] \in [0, dec_in_size], 
                all dims outside this pair will be masked as 0
                in Tacotron paper, they only use last time-step instead of all group
        """
        super().__init__()
        self.enc_in_size = enc_in_size
        self.dec_in_size = dec_in_size
        self.dec_out_size = dec_out_size # output projection -> mel/linear spec #

        self.enc_emb_size = enc_emb_size 
        self.enc_emb_do = enc_emb_do

        self.enc_conv_sizes = enc_conv_sizes
        self.enc_conv_filter = enc_conv_filter
        self.enc_conv_do = ConfigParser.list_parser(enc_conv_do, len(enc_conv_sizes))
        self.enc_conv_fn = enc_conv_fn
        
        self.enc_rnn_sizes = enc_rnn_sizes
        self.enc_rnn_do = ConfigParser.list_parser(enc_rnn_do, len(enc_rnn_sizes))
        self.enc_rnn_cfgs = ConfigParser.list_parser(enc_rnn_cfgs, len(enc_rnn_sizes))

        self.dec_prenet_size = dec_prenet_size
        self.dec_prenet_do = ConfigParser.list_parser(dec_prenet_do, len(dec_prenet_size))
        self.dec_prenet_fn = dec_prenet_fn
        self.dec_rnn_sizes = dec_rnn_sizes
        self.dec_rnn_cfgs = ConfigParser.list_parser(dec_rnn_cfgs, len(dec_rnn_sizes))
        self.dec_rnn_do = ConfigParser.list_parser(dec_rnn_do, len(dec_rnn_sizes))

        self.dec_proj_size = dec_proj_size
        self.dec_proj_fn = dec_proj_fn
        self.dec_proj_do = ConfigParser.list_parser(dec_proj_do, len(dec_proj_size))

        self.dec_bern_end_size = dec_bern_end_size
        self.dec_bern_end_do = ConfigParser.list_parser(dec_bern_end_do, len(dec_bern_end_size))
        self.dec_bern_end_fn = dec_bern_end_fn
        
        self.dec_cfg = dec_cfg
        self.att_cfg = att_cfg
        self.use_bn = use_bn
        self.use_ln = use_ln
        if use_ln == True :
            raise ValueError("Layer Normalization is not supported yet!")

        # OPTIONAL #
        self.dec_in_range = dec_in_range
        if self.dec_in_range is not None :
            assert isinstance(self.dec_in_range, (list, tuple)) \
                    and len(self.dec_in_range) == 2
        ### FINISH ###
        
        # init emb layer
        self.enc_emb_lyr = nn.Embedding(enc_in_size, enc_emb_size)

        # init enc conv #
        _tmp = []
        prev_size = enc_emb_size
        for ii in range(len(self.enc_conv_sizes)) :
            _tmp.append(Conv1dEv(prev_size, self.enc_conv_filter[ii], 
                self.enc_conv_sizes[ii], padding='same'))
            _tmp.append(generator_act_module(self.enc_conv_fn))
            if self.use_bn :
                _tmp.append(nn.BatchNorm1d(self.enc_conv_filter[ii]))
            _tmp.append(nn.Dropout(p=self.enc_conv_do[ii]))
            prev_size = self.enc_conv_filter[ii]
        self.enc_conv_lyr = nn.Sequential(*_tmp)
        
        # init enc rnn #
        self.enc_rnn_lyr = nn.ModuleList() 
        _enc_rnn_cfgs = ConfigParser.list_parser(enc_rnn_cfgs, len(enc_rnn_sizes))
        for ii in range(len(self.enc_rnn_sizes)) :
            _rnn_cfg = {}
            _rnn_cfg['type'] = _enc_rnn_cfgs[ii]['type']
            _rnn_cfg['args'] = [prev_size, enc_rnn_sizes[ii], 1, True, True, 0, _enc_rnn_cfgs[ii]['bi']]
            self.enc_rnn_lyr.append(generator_rnn(_rnn_cfg))
            prev_size =  enc_rnn_sizes[ii]

        # init dec prenet #
        _tmp = []
        prev_size = dec_in_size if self.dec_in_range is None else ((self.dec_in_range[-1] or 0) - (self.dec_in_range[-2] or 0))
        for ii in range(len(self.dec_prenet_size)) :
            _tmp.append(nn.Linear(prev_size, self.dec_prenet_size[ii]))
            prev_size = self.dec_prenet_size[ii]

        self.dec_prenet_lyr = nn.ModuleList(_tmp)
        
        # init dec rnn #
        _dec_rnn_cfgs = ConfigParser.list_parser(dec_rnn_cfgs, len(dec_rnn_sizes))
        for ii in range(len(dec_rnn_sizes)) :
            _type = _dec_rnn_cfgs[ii]['type']
            if re.match('stateful.*cell', _type) is None :
                _dec_rnn_cfgs[ii]['type'] = 'stateful_{}cell'.format(_type)

        final_enc_size = self.enc_rnn_lyr[-1].hidden_size * (2 if self.enc_rnn_lyr[-1].bidirectional else 1)
        assert 'type' in dec_cfg, "decoder type need to be defined"
        if dec_cfg['type'] == 'standard_decoder' :
            _tmp_dec_cfg = dict(dec_cfg)
            del _tmp_dec_cfg['type'] #
            self.dec_att_lyr = decoder.StandardDecoder(att_cfg=att_cfg,
                    ctx_size=final_enc_size, in_size=prev_size, 
                    rnn_sizes=dec_rnn_sizes, rnn_cfgs=dec_rnn_cfgs, rnn_do=dec_rnn_do,
                    **_tmp_dec_cfg)

        # init dec lin proj -> mel/linear-spec 
        prev_size = self.dec_att_lyr.out_features
        _tmp = []
        for ii in range(len(self.dec_proj_size)) :
            _tmp.append(nn.Linear(prev_size, self.dec_proj_size[ii]))
            prev_size = self.dec_proj_size[ii]
        _tmp.append(nn.Linear(prev_size, self.dec_out_size))
        self.dec_proj_lyr = nn.ModuleList(_tmp)

        # init dec bern end layer
        _tmp = []
        prev_size = self.dec_out_size + self.dec_att_lyr.out_features + (self.enc_rnn_lyr[-1].hidden_size * (2 if self.enc_rnn_lyr[-1].bidirectional else 1))
        for ii in range(len(self.dec_bern_end_size)) :
            _tmp.append(nn.Linear(prev_size, self.dec_bern_end_size[ii]))
            _tmp.append(generator_act_module(dec_bern_end_fn))
            _tmp.append(nn.Dropout(self.dec_bern_end_do[ii]))
            prev_size = self.dec_bern_end_size[ii]
            pass
        _tmp.append(nn.Linear(prev_size, 1))
        self.dec_bern_end_lyr = nn.Sequential(*_tmp)
        pass 

    def get_config(self) :
        # TODO
        return {
                'class':str(self.__class__),
                'enc_in_size':self.enc_in_size,
                'dec_in_size':self.dec_in_size,
                'dec_out_size':self.dec_out_size,

                'enc_emb_size':self.enc_emb_size, 
                'enc_emb_do':self.enc_emb_do,
                
                'enc_conv_sizes':self.enc_conv_sizes,
                'enc_conv_filter':self.enc_conv_filter,
                'enc_conv_do':self.enc_conv_do,
                'enc_conv_fn':self.enc_conv_fn,
                
                'enc_rnn_sizes':self.enc_rnn_sizes,
                'enc_rnn_do':self.enc_rnn_do,
                'enc_rnn_cfgs':self.enc_rnn_cfgs,

                'dec_prenet_size':self.dec_prenet_size,
                'dec_prenet_do':self.dec_prenet_do,
                'dec_prenet_fn':self.dec_prenet_fn,
                'dec_rnn_sizes':self.dec_rnn_sizes,
                'dec_rnn_cfgs':self.dec_rnn_cfgs,
                'dec_rnn_do':self.dec_rnn_do,

                'dec_proj_size':self.dec_proj_size,
                'dec_proj_fn':self.dec_proj_fn,
                'dec_proj_do':self.dec_proj_do,

                'dec_bern_end_size':self.dec_bern_end_size,
                'dec_bern_end_do':self.dec_bern_end_do,

                'dec_cfg':self.dec_cfg,
                'att_cfg':self.att_cfg,

                # OPTIONAL #
                'dec_in_range':self.dec_in_range,
                'use_bn':self.use_bn,
                'use_ln':self.use_ln,
                }

        pass

    def encode(self, input, src_len=None) :
        """
        input : (batch x max_src_len)
        mask : (batch x max_src_len)
        """
        batch, max_src_len = input.size()

        if src_len is None :
            src_len = [max_src_len] * batch

        mask_input = Variable(generate_seq_mask(src_len, self, max_len=input.shape[1]).unsqueeze(-1))

        res = self.enc_emb_lyr(input) # batch x src_len x emb_dim #
        res = res * mask_input
        res = res.transpose(1, 2) # batch x emb_dim x src_len #
        # apply enc conv 
        res = self.enc_conv_lyr(res) # batch x filter x src_len #
        res = res.transpose(1, 2) # batch x src_len x filter #
        res = res * mask_input

        # apply enc rnn
        for ii in range(len(self.enc_rnn_lyr)) :
            res = pack(res, src_len, batch_first=True)
            res = self.enc_rnn_lyr[ii](res)[0]
            res,_ = unpack(res, batch_first=True)
            if ii != len(self.enc_rnn_lyr)-1 :
                res = F.dropout(res, p=self.enc_rnn_do[ii], training=self.training)

        # save as context 
        ctx = res

        if src_len is not None :
            ctx_mask = mask_input.squeeze(-1)
        else : 
            ctx_mask = None
        
        self.ctx = ctx
        self.ctx_mask = ctx_mask
        self.src_len = src_len

        self.dec_att_lyr.set_ctx(ctx, ctx_mask)

    def reset(self) :
        self.dec_att_lyr.reset()

    def mask_dec_feat(self, y_tm1) :
        res = y_tm1
        # OPTIONAL : apply input dimension mask if available
        if self.dec_in_range is not None :
            res = res[:, slice(*self.dec_in_range)]
        return res

    def decode(self, y_tm1, mask=None) :
        assert y_tm1.dim() == 2, "batchsize x dec_in_size ( 1 timestep only)"

        res = y_tm1

        # OPTIONAL #
        res = self.mask_dec_feat(res)

        # apply dec prenet lyr #
        for ii in range(len(self.dec_prenet_lyr)) :
            res = self.dec_prenet_lyr[ii](res)
            res = generator_act_fn(self.dec_prenet_fn)(res)
            res = F.dropout(res, p=self.dec_prenet_do[ii], training=self.training)

        # apply dec att lyr #
        res_dec_att = self.dec_att_lyr(res, mask)
        res = res_dec_att['dec_output']

        # apply lin proj lyr #
        for ii in range(len(self.dec_proj_lyr)) :
            if ii != len(self.dec_proj_lyr)-1 :
                res = self.dec_proj_lyr[ii](res)
                res = generator_act_fn(self.dec_proj_fn)(res)
                res = F.dropout(res, p=self.dec_proj_do[ii], training=self.training)
            else :
                res = self.dec_proj_lyr[ii](res)
            
        # predict frame stopping #
        # input = spec + dec_att_output + att_context
        _bern_end_input = torch.cat([res, res_dec_att['dec_output'],
            res_dec_att['att_output']['expected_ctx']], dim=1)
        res_bern_end = self.dec_bern_end_lyr(_bern_end_input.detach()) # stop gradient

        return res, res_dec_att, res_bern_end
