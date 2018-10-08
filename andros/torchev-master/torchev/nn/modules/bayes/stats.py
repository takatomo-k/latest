import math

import torch
from torch.nn import Module
from torch.nn import Parameter, ParameterList
from torch.autograd import Variable
from torch.nn import functional as F

def logsumexp(tensors, dim=0) :
    if isinstance(tensors, list) :
        tensors = torch.stack(tensors)
    max_val = torch.max(tensors, dim=dim)[0]

    tensors = tensors - max_val
    output = torch.log(torch.sum(torch.exp(tensors), dim=dim))
    output = output + max_val
    return output
    pass

class Stat() :
    pass

class Normal(Stat) :
    """
    Normal RV function for diagonal Gaussian
    """
    @staticmethod
    def sample(mu, sigma) :
        eps = Variable(mu.data.new().resize_as_(mu.data).normal_(0, 1))
        return mu + sigma * eps
    
    @staticmethod
    def logpdf(x, mu, sigma, dim=None) :
        constant = math.log(2 * math.pi)
        return -0.5 * ((2*torch.log(sigma))
                        + ((x - mu) ** 2 / sigma ** 2)
                        + constant)
    @staticmethod
    def kl(mu_1, sigma_1, mu_2, sigma_2) :
        """ KL-divergence between 2 normal distribution """
        return 0.5 * (2*torch.log(sigma_2) - 2*torch.log(sigma_1) - 1 
                      + (sigma_1**2/sigma_2**2) + ((mu_2-mu_1)**2/(sigma_2**2)))

class MixtureNormal(Stat) :
    @staticmethod
    def logpdf(x, mixture, mu, sigma, dim=None) :
        n_mix = len(mixture)
        res = 0
        _tmp_sum_exp = [] 
        for ii in range(n_mix) :
            _tmp_sum_exp.append(torch.log(mixture[ii]) + Normal.logpdf(x, mu[ii], sigma[ii]))
        res = logsumexp(_tmp_sum_exp, dim=0)
        return res

    @staticmethod
    def sample(mixture, mu, sigma) :
        raise NotImplementedError()
    pass
