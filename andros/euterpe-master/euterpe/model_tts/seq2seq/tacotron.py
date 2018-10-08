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

# utilbox #
from utilbox.config_util import ConfigParser


class TACOTRON(nn.Module) :
    # Ref : Tacotron (Google Brain)
    def __init__(self, enc_in_size, dec_in_size, dec_out_size, dec_out_post_size,
            enc_emb_size=256, enc_emb_do=0.0,
            enc_prenet_size=[256, 128], enc_prenet_do=[0.5, 0.5], enc_prenet_fn='leaky_relu',
            dec_prenet_size=[256, 128], dec_prenet_do=[0.5, 0.5], dec_prenet_fn='leaky_relu',
            dec_rnn_sizes=[256, 256], dec_rnn_cfgs={"type":"lstm"}, dec_rnn_do=0.0,
            dec_cfg={"type":"standard_decoder"}, 
            att_cfg={"type":"mlp"},
            # CBHG # 
            enc_cbhg_cfg={},
            dec_postnet_cbhg_cfg={},
            # OPTIONAL #
            dec_in_range=None
            ) :
        """
        Args:
            enc_in_size : size of vocab
            dec_in_size : input (mel) dim size
            dec_out_size : output (mel) dim size (usually same as dec_in_size)
            dec_out_post_size : output (linear) dim size
            dec_in_range : 
                pair of integer [x, y] \in [0, dec_in_size], 
                all dims outside this pair will be masked as 0
                in Tacotron paper, they only use last time-step instead of all group
        """
        super(TACOTRON, self).__init__()
        self.enc_in_size = enc_in_size
        self.dec_in_size = dec_in_size
        self.dec_out_size = dec_out_size # first output -> mel spec #
        self.dec_out_post_size = dec_out_post_size # second output -> raw spec #
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
        self.dec_cfg = dec_cfg
        self.att_cfg = att_cfg

        # OPTIONAL #
        self.dec_in_range = dec_in_range
        if self.dec_in_range is not None :
            assert isinstance(self.dec_in_range, (list, tuple)) \
                    and len(self.dec_in_range) == 2

        # CBHG config #
        self.enc_cbhg_cfg = ConfigParser.item_parser(enc_cbhg_cfg)
        self.dec_postnet_cbhg_cfg = ConfigParser.item_parser(dec_postnet_cbhg_cfg)


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
        
        # init dec regression melspec #
        self.dec_first_reg_lyr = nn.Linear(self.dec_att_lyr.output_size, self.dec_out_size)
        # init dec postnet #
        self.dec_postnet_lyr = cbhg.CBHG1d(self.dec_out_size, 
                conv_proj_filter=[256, dec_out_size], **dec_postnet_cbhg_cfg)
        # init dec regression rawspec #
        self.dec_second_reg_lyr = nn.Linear(self.dec_postnet_lyr.output_size, self.dec_out_post_size)
        pass 

    def get_config(self) :
        # TODO
        return {'class':str(self.__class__),
                'enc_in_size' : self.enc_in_size,
                'dec_in_size' : self.dec_in_size,
                'dec_out_size' : self.dec_out_size,
                'dec_out_post_size' : self.dec_out_post_size,
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
                'dec_cfg' : self.dec_cfg,
                'att_cfg' :self.att_cfg,
                # CBHG config #
                'enc_cbhg_cfg':self.enc_cbhg_cfg,
                'dec_postnet_cbhg_cfg':self.dec_postnet_cbhg_cfg,
                # OPTIONAL #
                'dec_in_range':self.dec_in_range,
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
        res = self.enc_emb_lyr(input) # batch x max_src_len x emb_dim #
        res = F.dropout(res, self.enc_emb_do, self.training)
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
        res_first = self.dec_first_reg_lyr(res)
        return res_first, res_dec 

    def decode_post(self, y_below, seq_len=None) :
        batch, max_seq_len, _ = y_below.size()
        res_second = self.dec_postnet_lyr(y_below, seq_len)
        res_second = res_second.contiguous().view(batch * max_seq_len, -1)
        res_second = self.dec_second_reg_lyr(res_second)
        res_second = res_second.contiguous().view(batch, max_seq_len, -1)
        return res_second
    pass

class TACOTRONBernoulliEnd(TACOTRON) :
    def __init__(self, dec_bern_end_size=[256], dec_bern_end_fn='Tanh', dec_bern_end_do=0.0,
            *args, **kwargs) :
        super(TACOTRONBernoulliEnd, self).__init__(*args, **kwargs)
        self.dec_bern_end_size = dec_bern_end_size
        self.dec_bern_end_fn = dec_bern_end_fn
        self.dec_bern_end_do = ConfigParser.list_parser(dec_bern_end_do)

        # p(t = frame stop | dec_hid[t], y[t]) #
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
        pass

    def get_config(self) :
        _config = super().get_config()
        _config['dec_bern_end_size'] = self.dec_bern_end_size
        _config['dec_bern_end_do'] = self.dec_bern_end_do
        _config['dec_bern_end_fn'] = self.dec_bern_end_fn
        return _config

    def decode(self, y_tm1, mask=None) :
        """
        additional output 
            res_bern_end 
        """
        res_first, res_dec = super(TACOTRONBernoulliEnd, self).decode(y_tm1, mask)
        res_bern_end = self.dec_bernoulli_end_lyr(torch.cat([res_first, res_dec['dec_output']], 1))
        return res_first, res_dec, res_bern_end

        pass

class TACOTRONBernoulliEndAdapter(TACOTRONBernoulliEnd) :
    def __init__(self, init_adapter=True, *args, **kwargs) :
        super(TACOTRONBernoulliEndAdapter, self).__init__(*args, **kwargs)
        raise NotImplementedError("this class is abandoned")
        if init_adapter :
            self.init_adapter_lyr()
        pass

    def init_adapter_lyr(self) :
        # add adapter layer from prenet -> RNN #
        self.dec_adapter_lyr = nn.Module()
        ### OLD ###
        # self.dec_adapter_lyr.add_module('prenet_to_rnn', 
                # nn.Linear(self.dec_prenet_lyr[-1].out_features, self.dec_prenet_lyr[-1].out_features))
        # self.dec_adapter_lyr.add_module('rnn_to_first_reg', 
                # nn.Linear(self.dec_att_lyr.out_features, self.dec_att_lyr.out_features))
        # self.dec_adapter_lyr.add_module('postnet_to_second_reg', 
                # nn.Linear(self.dec_postnet_lyr.out_features, self.dec_postnet_lyr.out_features))
        self.dec_adapter_lyr.add_module('prenet_to_rnn', 
                nn.Linear(self.dec_prenet_lyr[-1].out_features, self.dec_prenet_lyr[-1].out_features))
        self.dec_adapter_lyr.add_module('first_reg', 
                nn.Linear(self.dec_out_size, self.dec_out_size))
        self.dec_adapter_lyr.add_module('second_reg',
                nn.Linear(self.dec_out_post_size, self.dec_out_post_size))
        # init identity matrix and zero bias #
        init.eye(self.dec_adapter_lyr.prenet_to_rnn.weight)
        init.eye(self.dec_adapter_lyr.first_reg.weight)
        init.eye(self.dec_adapter_lyr.second_reg.weight)
        init.constant(self.dec_adapter_lyr.prenet_to_rnn.bias, 0)
        init.constant(self.dec_adapter_lyr.first_reg.bias, 0)
        init.constant(self.dec_adapter_lyr.second_reg.bias, 0)
        pass

    def decode(self, y_tm1, mask=None) :
        assert y_tm1.dim() == 2, "batchsize x dec_in_size ( 1 timestep only)"

        res = y_tm1
        for ii in range(len(self.dec_prenet_lyr)) :
            res = self.dec_prenet_lyr[ii](res)
            res = generator_act_fn(self.dec_prenet_fn)(res)
            res = F.dropout(res, self.dec_prenet_do[ii], training=self.training)
        ### APPLY ADAPTER PRENET -> RNN ###
        res = self.dec_adapter_lyr.prenet_to_rnn(res)
        res = generator_act_fn(self.dec_prenet_fn)(res)
        res = F.dropout(res, self.dec_prenet_do[-1], training=self.training)
        ###

        # compute decoder rnn #
        res_dec = self.dec_att_lyr(res, mask)
        res = res_dec['dec_output']

        ### APPLY ADAPTER RNN -> FIRST REG ###
        # res = self.dec_adapter_lyr.rnn_to_first_reg(res)
        # res = self.dec_att_lyr.ctx_proj_fn_act(res)
        # res = F.dropout(res, self.dec_att_lyr.do[-1], self.training)
        ###
        res_first = self.dec_first_reg_lyr(res)
        ### APPLY ADAPTER AFTER FIRST_REG ###
        res_first = self.dec_adapter_lyr.first_reg(res_first)
        ###

        # compute sigmoid layer #
        res_bern_end = self.dec_bernoulli_end_lyr(torch.cat([res_first, res_dec['dec_output']], 1))
        return res_first, res_dec, res_bern_end

    def decode_post(self, y_below, seq_len=None) :
        batch, max_seq_len, _ = y_below.size()
        res_second = self.dec_postnet_lyr(y_below, seq_len)
        res_second = res_second.contiguous().view(batch * max_seq_len, -1)
        ### APPLY ADAPTER POSTNET -> SECOND REG ###
        # res_second = self.dec_adapter_lyr.postnet_to_second_reg(res_second)
        # res_second = F.tanh(res_second) # last CBHG act_fn is tanh #
        ###
        res_second = self.dec_second_reg_lyr(res_second)
        ### APPLY ADAPTER SECOND_REG #
        res_second = self.dec_adapter_lyr.second_reg(res_second)
        ###
        res_second = res_second.contiguous().view(batch, max_seq_len, -1)
        return res_second


class TACOTRONBernoulliEndMultiSpk(TACOTRONBernoulliEnd) :
    def __init__(self, map_spk2id, speaker_emb_dim=128, max_spk=-1, 
            speaker_act_fn='softsign', *args, **kwargs) :
        super(TACOTRONBernoulliEndMultiSpk, self).__init__(*args, **kwargs)
        self.map_spk2id = map_spk2id
        self.speaker_emb_dim = speaker_emb_dim
        self.max_spk = self.num_speaker if max_spk == -1 else max_spk
        self.speaker_act_fn = speaker_act_fn

        self.init_spk_lyr()
        pass

    def get_config(self) :
        _config = super(TACOTRONBernoulliEndMultiSpk, self).get_config()
        _config['map_spk2id'] = self.map_spk2id
        _config['speaker_emb_dim'] = self.speaker_emb_dim
        _config['max_spk'] = self.max_spk
        _config['speaker_act_fn'] = self.speaker_act_fn
        return _config

    @property
    def num_speaker(self) :
        return max(self.map_spk2id.values())+1

    def add_speaker(self, speaker) :
        if speaker in self.map_spk2id :
            raise ValueError("speaker already exist")
        else :
            self.map_spk2id[speaker] = self.num_speaker

    def get_speaker_emb(self, speaker_list) :
        speaker_id_list = [self.map_spk2id[x] for x in speaker_list]
        speaker_list_var = Variable(torchauto(self).LongTensor(speaker_id_list))
        # get embedding for each speaker #
        speaker_emb_var = self.spk_module_lyr.emb_lyr(speaker_list_var)
        return speaker_emb_var
    
    def init_spk_lyr(self) :
        self.spk_module_lyr = nn.Module()
        
        # speaker embedding #
        self.spk_module_lyr.add_module('emb_lyr', nn.Embedding(self.max_spk, self.speaker_emb_dim))

        # concat to encoder pre_net #
        self.spk_module_lyr.add_module('enc_lin_prenet_lyr', 
                nn.Linear(self.speaker_emb_dim, self.enc_prenet_lyr[-1].out_features))

        # concat to encoder h_t #
        self.spk_module_lyr.add_module('enc_lin_core_lyr', 
                nn.Linear(self.speaker_emb_dim, self.enc_core_lyr.out_features))

        # concat to decoder prenet #
        self.spk_module_lyr.add_module('dec_lin_prenet_lyr', 
                nn.Linear(self.speaker_emb_dim, self.dec_prenet_lyr[-1].out_features))

        # concat to decoder pre first regression #
        self.spk_module_lyr.add_module('dec_lin_pre_reg_first_lyr', 
                nn.Linear(self.speaker_emb_dim, self.dec_att_lyr.output_size))

        # concat to decoder pre second regression #
        self.spk_module_lyr.add_module('dec_lin_pre_reg_second_lyr', 
                nn.Linear(self.speaker_emb_dim, self.dec_postnet_lyr.out_features))

        # concat to decoder pre second regression #
        # TODO : DeepVoice 2 doesn't have this #

        # default DeepVoice 2 act fn for speaker information #
        pass

    def encode(self, input, input_aux, src_len=None) :
        """
        input : feat matrix
        input_aux : map contains additional info speaker embedding ID
        """
        batch, max_src_len = input.size()
        self.input_spk_emb = self.get_speaker_emb(input_aux['spk'])
        assert self.input_spk_emb.size(0) == batch

        if src_len is None :
            src_len = [max_src_len] * batch
        res = self.enc_emb_lyr(input) # batch x max_src_len x emb_dim #
        res = F.dropout(res, self.enc_emb_do, self.training)
        res = res.view(batch * max_src_len, -1) 
        for ii in range(len(self.enc_prenet_lyr)) :
            res = self.enc_prenet_lyr[ii](res)
            res = generator_act_fn(self.enc_prenet_fn)(res)
            res = F.dropout(res, p=self.enc_prenet_do[ii], training=self.training)
        res = res.view(batch, max_src_len, -1)

        ### SPK ###
        # res_spk = self.spk_enc_lin_prenet_lyr(input_spk_emb).unsqueeze(1).expand_as(
                # batch, max_src_len, self.spk_emb_lyr.embedding_dim)
        # res_spk = self.spk_act_fn(res_spk)
        # res = res + res_spk
        ###########

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
        pass

    def reset(self) :
        super(TACOTRONBernoulliEndMultiSpk, self).reset()
        self.input_spk_emb = None

    def decode(self, y_tm1, mask=None) :
        assert y_tm1.dim() == 2, "batchsize x dec_in_size ( 1 timestep only)"
        batch = y_tm1.size(0)
        res = y_tm1

        # OPTIONAL #
        res = self.mask_dec_feat(res)

        for ii in range(len(self.dec_prenet_lyr)) :
            res = self.dec_prenet_lyr[ii](res)
            if ii == len(self.dec_prenet_lyr) - 1 : # last layer #
                # concat speaker info #
                res_spk = self.spk_module_lyr.dec_lin_prenet_lyr(self.input_spk_emb) 
                res_spk = generator_act_fn(self.speaker_act_fn)(res_spk)
                res += res_spk
            res = generator_act_fn(self.dec_prenet_fn)(res)
            res = F.dropout(res, self.dec_prenet_do[ii], training=self.training)

        # compute decoder rnn #
        res_dec = self.dec_att_lyr(res, mask)
        res = res_dec['dec_output']

        # concat speaker info #
        res_spk = self.spk_module_lyr.dec_lin_pre_reg_first_lyr(self.input_spk_emb)
        res_spk = generator_act_fn(self.speaker_act_fn)(res_spk)
        res = res + res_spk
        
        res_first = self.dec_first_reg_lyr(res)

        res_bern_end = self.dec_bernoulli_end_lyr(torch.cat([res_first, res_dec['dec_output']], 1))
        return res_first, res_dec, res_bern_end 

    def decode_post(self, y_below, seq_len=None) :
        batch, max_seq_len, _ = y_below.size()
        res_second = self.dec_postnet_lyr(y_below, seq_len)
        res_second = res_second.contiguous().view(batch * max_seq_len, -1)
        
        # concat speaker info #
        # TODO : should we use this ? investigate without this first

        # res_spk = self.spk_module_lyr.dec_lin_pre_reg_second_lyr(self.input_spk_emb)
        # res_spk = res_spk.unsqueeze(1).expand(batch, max_seq_len, res_spk.size(-1))
        # res_spk = self.spk_act_fn(res_spk)
        # res_second += res_spk

        res_second = self.dec_second_reg_lyr(res_second)
        res_second = res_second.contiguous().view(batch, max_seq_len, -1)
        return res_second
    pass
