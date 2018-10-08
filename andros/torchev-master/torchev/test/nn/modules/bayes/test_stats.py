import pytest
import numpy as np
import scipy.stats as stats
import torch
from torch.autograd import Variable
from torchev.nn.modules.bayes import Normal, MixtureNormal

class TestNormal :
    def test_logpdf(self) :
        mu = np.array([[0., 1., 2.]], dtype='float32')
        sigma = np.array([[0.5, 0.7, 0.3]], dtype='float32')
        x = np.random.normal(mu, sigma, size=(2, 3)).astype('float32')
        truth_logpdf = stats.norm.logpdf(x, mu, sigma)
        pred_logpdf = Normal.logpdf(Variable(torch.from_numpy(x)), Variable(torch.from_numpy(mu).expand(*x.shape)), Variable(torch.from_numpy(sigma).expand(*x.shape)))

        assert np.allclose(truth_logpdf, pred_logpdf.data.numpy())
        assert np.allclose(truth_logpdf.sum(), pred_logpdf.data.sum())
        pass

    def test_kldiv(self) :
        mu = np.array([[0., 1., 2.]], dtype='float32')
        sigma = np.array([[0.5, 0.7, 0.3]], dtype='float32')
        x = np.random.normal(mu, sigma, size=(1, 3)).astype('float32')
        prior_mu = np.array([[0.1, 0.2, 0.3]], dtype='float32')
        prior_sigma = np.array([[0.1, 0.2, 0.3]], dtype='float32')
        truth_kldiv = (np.log(prior_sigma / sigma) + (sigma**2 + (mu - prior_mu)**2)/(2*prior_sigma**2) - 0.5)
        pred_kldiv = Normal.kl(
                Variable(torch.from_numpy(mu)), 
                Variable(torch.from_numpy(sigma)),
                Variable(torch.from_numpy(prior_mu)),
                Variable(torch.from_numpy(prior_sigma))
                )

        assert np.allclose(truth_kldiv, pred_kldiv.data.numpy())
        assert np.allclose(truth_kldiv.sum(), pred_kldiv.data.sum())

    def test_kldiv_single(self) :
        mu = np.array([[0., 1., 2.]], dtype='float32')
        sigma = np.array([[0.5, 0.7, 0.3]], dtype='float32')
        x = np.random.normal(mu, sigma, size=(1, 3)).astype('float32')
        prior_mu = 0.5
        prior_sigma = 0.9
        truth_kldiv = (np.log(prior_sigma / sigma) + (sigma**2 + (mu - prior_mu)**2)/(2*prior_sigma**2) - 0.5)
        pred_kldiv = Normal.kl(
                Variable(torch.from_numpy(mu)), 
                Variable(torch.from_numpy(sigma)),
                Variable(torch.FloatTensor([prior_mu])),
                Variable(torch.FloatTensor([prior_sigma]))
                )

        assert np.allclose(truth_kldiv, pred_kldiv.data.numpy())
        assert np.allclose(truth_kldiv.sum(), pred_kldiv.data.sum())
        pass

class TestMixtureNormal :
    def test_logpdf(self) :
        mix = np.array([0.4, 0.6], dtype='float32')
        mu = np.array([0., 1.], dtype='float32')
        sigma = np.array([0.5, 0.7], dtype='float32')
        x = np.random.normal(0.75, 0.3, size=(2, 3)).astype('float32')
        truth_logpdf = np.log(np.sum([mix[ii]*np.exp(stats.norm.logpdf(x, mu[ii], sigma[ii])) for ii in range(len(mix))], axis=0))
        pred_logpdf = MixtureNormal.logpdf(Variable(torch.from_numpy(x)), Variable(torch.from_numpy(mix)), Variable(torch.from_numpy(mu)), Variable(torch.from_numpy(sigma)))

        assert np.allclose(truth_logpdf, pred_logpdf.data.numpy())
        assert np.allclose(truth_logpdf.sum(), pred_logpdf.data.sum())
        pass

