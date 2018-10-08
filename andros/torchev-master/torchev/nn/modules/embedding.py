import math
import torch

from torch.nn import Module
from torch.nn import Parameter
from torch.autograd import Variable
from torch import nn
from torch.nn import functional as F
from ...utils.helper import torchauto, tensorauto, _auto_detect_cuda

from scipy.special import betainc

class PositionEmbedding(Module) :
    def __init__(self, embedding_dim, max_len=2048, pos_type='abs', 
            const_rel=3, const_abs=10**4):
        super().__init__()
        assert pos_type in ['abs', 'rel'], 'pos_type must be "abs" or "rel"'
        self.embedding_dim = embedding_dim
        assert self.embedding_dim % 2 == 0, "embedding dim must be divisible by 2"
        self.max_len = max_len
        self.pos_type = pos_type
        self.const_rel = const_rel
        self.const_abs = const_abs
        if pos_type == 'abs' :
            self._generate_sincos_position_abs(max_len)
        pass

    def _generate_sincos_position_abs(self, max_len) :
        pos_emb = torch.arange(0, max_len).unsqueeze(1).expand(max_len, self.embedding_dim)
        denominator = 1 / torch.pow(self.const_abs, torch.arange(0, self.embedding_dim, 1) / self.embedding_dim)
        pos_emb = pos_emb * denominator.expand_as(pos_emb)
        pos_emb[:, 0::2] = torch.sin(pos_emb[:, 0::2])
        pos_emb[:, 1::2] = torch.sin(pos_emb[:, 1::2])
        self.register_buffer('pos_emb', pos_emb)
        pass

    def _generate_power_rel(self, seq_len, device=0) :
        batch_size = len(seq_len)
        max_len = max(seq_len)
        seq_len_m1 = [x-1 for x in seq_len]
        pos_emb = torch.arange(0, max_len)[None, :, None].expand(batch_size, max_len, self.embedding_dim) / torch.FloatTensor(seq_len_m1)[:, None, None]
        power_dim = torch.linspace(1, self.const_rel, self.embedding_dim // 2)
        power_dim_rev = torch.linspace(self.const_rel, 1, self.embedding_dim // 2)
        power_dim = torch.cat([1/power_dim_rev, power_dim])
        power_dim = power_dim[None, None, :].expand(batch_size, max_len, self.embedding_dim)
        pos_emb = torch.pow(pos_emb, power_dim)
        return tensorauto(device, pos_emb)

    """
    def _generate_betacdf_rel(self, seq_len, device=0) :
        batch_size = len(seq_len)
        max_len = max(seq_len)
        seq_len_m1 = [x-1 for x in seq_len]
        pos_emb = np.tile(np.arange(0, max_len)[None, :, None], (batch_size, max_len, self.embedding_dim)) / np.array(seq_len_m1)[:, None, None]
        pos_emb = np.clip(pos_emb, 0, 1)
        for ii in range(0, embedding_dim//2) :
            pos_emb[:, :, ii] = betainc(ii+1, ii+1, pos_emb[:, :, ii])
            pos_emb[:, :, ii+embedding_dim//2] = betainc(1/(ii+1), 1/(ii+1), pos_emb[:, :, ii])
        pos_emb = torch.from_numpy(pos_emb)
        return tensorauto(device, pos_emb)
    """

    @property
    def out_features(self) :
        return self.embedding_dim

    def forward_abs(self, input) :
        assert input.dim() == 3, 'input shape must be [batch, seq_len, ndim]'
        batch, seq_len = input.size()[0:2] 
        assert seq_len < self.max_len, "input seq_len is larger than current embedding"
        pos_emb = self.pos_emb[0:seq_len]
        pos_emb = pos_emb.unsqueeze(0).expand(batch, seq_len, self.embedding_dim)
        pos_emb = Variable(pos_emb)
        return pos_emb

    def forward_rel(self, input, input_len) :
        return Variable(self._generate_power_rel(input_len, device=_auto_detect_cuda(input)))

    def forward(self, input, input_len=None) :
        if self.pos_type == 'abs' :
            return self.forward_abs(input)
        else :
            return self.forward_rel(input, input_len)
