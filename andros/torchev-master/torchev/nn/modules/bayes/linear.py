import math

import torch
from torch.nn import Module
from torch.nn.parameter import Parameter
from torch.nn import functional as F
from .base import *

class LinearBayes(ProbabilisticModule):

    def __init__(self, in_features, out_features, bias=True, 
            posterior_w=None, prior_w=None,
            posterior_b=None, prior_b=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.posterior_w = posterior_w
        self.prior_w = prior_w
        self.posterior_w.build((out_features, in_features), is_parameter=True)
        self.prior_w.build((1,), is_parameter=False)
        if bias:
            self.posterior_b = posterior_b
            self.prior_b = prior_b
            self.posterior_b.build((out_features,), is_parameter=True)
            self.prior_b.build((1,), is_parameter=False)
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        # WARNING : only work for Normal prior !!!
        stdv = 1. / math.sqrt(self.in_features + self.out_features)
        pass

    def forward(self, input):
        return F.linear(input, self.posterior_w.sample(), self.posterior_b.sample())

    def logp(self) :
        return (self.posterior_w.logpdf().sum() + 
                self.posterior_b.logpdf().sum())

    def logq(self) :
        return (self.prior_w.logpdf(self.posterior_w.point).sum() + 
                self.prior_b.logpdf(self.posterior_b.point).sum())

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + str(self.in_features) + ' -> ' \
            + str(self.out_features) + ')'
