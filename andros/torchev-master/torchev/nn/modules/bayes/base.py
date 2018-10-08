import math

import torch
from torch.nn import Module
from torch.nn import Parameter, ParameterList
from torch.autograd import Variable
from torch.nn import functional as F
from numbers import Number
from .stats import *

def variable_wrapper(obj) :
    if isinstance(obj, Parameter) :
        return obj 
    else :
        return Variable(obj)
    

def inv_softplus(y) :
    return math.log(math.exp(y)-1)

class ProbabilisticModule(Module) :
    pass

class ProbabilisticParameter(ProbabilisticModule) :
    """
    TODO : implement something important
    """
    def __init__(self, *args, **kwargs) :
        super().__init__(*args, **kwargs)

    def sample(self) :
        raise NotImplementedError()
    pass

class RandomVariable(Module) :
    def __init__(self) :
        super().__init__()
        pass
    pass

class NormalRV(RandomVariable) :
    def __init__(self, mu=0, sigma=1) :
        super().__init__()
        self.mu = mu
        self.sigma = sigma
        pass

    def build(self, sizes, is_parameter=False) :
        _var_mu = torch.FloatTensor(*sizes).fill_(self.mu)
        _var_pre_sigma = torch.FloatTensor(*sizes).fill_(inv_softplus(self.sigma)) 
        if is_parameter :
            self._var_mu = Parameter(_var_mu)
            self._var_pre_sigma = Parameter(_var_pre_sigma)
        else :
            self.register_buffer('_var_mu', _var_mu)
            self.register_buffer('_var_pre_sigma', _var_pre_sigma)
        pass

    @property
    def var_mu(self) :
        return variable_wrapper(self._var_mu)

    @property
    def var_pre_sigma(self) :
        return variable_wrapper(self._var_pre_sigma)

    @property
    def var_sigma(self) :
        return F.softplus(self.var_pre_sigma)

    def sample(self) :
        if self.training :
            self._point = Normal.sample(self.var_mu, self.var_sigma)
        else :
            # no sampling in inference step (except for MC inference)
            self._point = self.var_mu + 0
        return self._point

    @property
    def point(self) :
        return self._point

    def logpdf(self, point=None) :
        if point is None :
            point = self._point
        return Normal.logpdf(point, self.var_mu, self.var_sigma)
    pass

class MixtureNormalRV(RandomVariable) :
    def __init__(self, mix=[0.5, 0.5], mu=[0., 0.], sigma=[1.0, 1.0]) :
        super().__init__()
        self.num_mix = len(mix)
        self.mix = mix
        self.mu = mu
        self.sigma = sigma
        pass

    def build(self, sizes, is_parameter=True) :
        sizes = [self.num_mix] + list(sizes)
        _var_pre_mix = torch.FloatTensor(*sizes)
        _var_mu = torch.FloatTensor(*sizes)
        _var_pre_sigma = torch.FloatTensor(*sizes)
        for ii in range(self.num_mix) :
            _var_pre_mix[ii].fill_(inv_softplus(self.mix[ii]))
            _var_mu[ii].fill_(self.mu[ii])
            _var_pre_sigma[ii].fill_(inv_softplus(self.sigma[ii]))
        if is_parameter :
            self._var_pre_mix = Parameter(_var_pre_mix)
            self._var_mu = Parameter(_var_mu)
            self._var_pre_sigma = Parameter(_var_pre_sigma)
        else :
            self.register_buffer('_var_pre_mix', _var_pre_mix)
            self.register_buffer('_var_mu', _var_mu)
            self.register_buffer('_var_pre_sigma', _var_pre_sigma)
        pass

    def sample(self) :
        raise NotImplementedError

    @property
    def var_mu(self) :
        return variable_wrapper(self._var_mu)

    @property
    def var_pre_sigma(self) :
        return variable_wrapper(self._var_pre_sigma)

    @property
    def var_sigma(self) :
        return F.softplus(self.var_pre_sigma)

    @property
    def var_pre_mix(self) :
        return variable_wrapper(self._var_pre_mix)

    def logpdf(self, point=None) :
        if point is None :
            point = self._point
        mix = F.normalize(F.softplus(self.var_pre_sigma), p=1, dim=-1)
        sigma = F.softplus(self.var_pre_sigma)
        return MixtureNormal.logpdf(point, mix, self.var_mu, sigma)
    pass
