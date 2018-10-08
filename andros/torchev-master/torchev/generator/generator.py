from torch import nn
from torchev import nn as nnev
from torch.nn import functional as F
import torch
import torchev

"""
RNN generator
"""
def generator_rnn(config) :
    mod_type = config['type'].lower()
    mod_args = config.get('args', [])
    mod_kwargs = config.get('kwargs', {})

    if mod_type == 'lstm' :
        _lyr = nn.LSTM
    elif mod_type == 'gru' :
        _lyr = nn.GRU
    elif mod_type == 'rnn' :
        _lyr = nn.RNN
    elif mod_type == 'lstmcell' :
        _lyr = nn.LSTMCell
    elif mod_type == 'grucell' :
        _lyr = nn.GRUCell
    elif mod_type == 'rnncell' :
        _lyr = nn.RNNCell
    elif mod_type =='stateful_lstmcell' :
        _lyr = nnev.StatefulLSTMCell
    else :
        raise NotImplementedError("rnn class {} is not implemented/existed".format(mod_type))
    return _lyr(*mod_args, **mod_kwargs)

def generator_attention(config) :
    mod_type = config['type'].lower()
    mod_args = config.get('args', [])
    mod_kwargs = config.get('kwargs', {})
    if mod_type == 'bilinear' :
        _lyr = torchev.custom.attention.BilinearAttention
    elif mod_type == 'dot' :
        _lyr = torchev.custom.attention.DotProductAttention
    elif mod_type == 'mlp' :
        _lyr = torchev.custom.attention.MLPAttention
    elif mod_type == 'mlp_history' :
        _lyr = torchev.custom.attention.MLPHistoryAttention
    elif mod_type == 'gmm' :
        _lyr = torchev.custom.attention.GMMAttention
    elif mod_type == 'local_gmm_scorer' :
        _lyr = torchev.custom.attention.LocalGMMScorerAttention 
    elif mod_type == 'multihead_kvq' :
        _lyr = torchev.custom.attention.MultiheadKVQAttention
    else :
        raise NotImplementedError()
    return _lyr(*mod_args, **mod_kwargs)

def generator_act_fn(name) :
    act_fn = None
    if name is None or name.lower() in ['none', 'null'] :
        act_fn = (lambda x : x)
    else :
        try :
            act_fn = getattr(F, name)
        except AttributeError :
            act_fn = getattr(torch, name)
    return act_fn

def generator_act_module(name) :
    act_module = None
    if name is None or name.lower() in ['none', 'null'] :
        act_module = nnev.activation.Identity()
    else :
        act_module = getattr(torch.nn, name)()
    return act_module

def generator_optim(params, config) :
    opt_type = config['type']
    opt = getattr(torch.optim, opt_type)
    args = config.get('args', [])
    kwargs = config.get('kwargs', {})
    return opt(params, *args, **kwargs)


### SPECIFIC CLASS GENERATOR ###
def generator_bayes_rv(config) :
    raise ValueError("please use Pyro instead")
    mod_type = config['type'].lower()
    mod_args = config.get('args', [])
    mod_kwargs = config.get('kwargs', [])

    if mod_type == 'normal' :
        _obj = torchev.nn.modules.bayes.NormalRV
    elif mod_type == 'mix_normal' :
        _obj = torchev.nn.modules.bayes.MixtureNormalRV
    else :
        raise NotImplementedError("{} prior does not exist".format(mod_type))
    return _obj(*mod_args, **mod_kwargs)
