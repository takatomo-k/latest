import torch,random
from torch import nn,optim
from torch.nn import functional as F

from modules.attention import *
import model_config as cf
from torch.nn.utils.rnn import pack_padded_sequence as pack, pad_packed_sequence as unpack

class asr_model(nn.Module):
    def __init__(self, enc_in_size, dec_out_size):
        super(asr_model,self).__init__()
        prev_size = enc_in_size
        _fnn = list()
        for n_units,drop in zip(cf.enc_fnn_sizes,cf.enc_fnn_drop) :
            _fnn.append(nn.Linear(prev_size, n_units))
            _fnn.append(nn.LeakyReLU())
            _fnn.append(nn.Dropout(p=drop))
            prev_size = n_units
        self.enc_fnn_lyr = nn.Sequential(*_fnn)
        self.enc_rnn_lyr = nn.ModuleList()
        for n_units,bi in zip(cf.enc_rnn_sizes,cf.enc_rnn_bi) :
            self.enc_rnn_lyr.append(nn.LSTM(prev_size,n_units,bias=True,batch_first=True,bidirectional=bi))
            prev_size = n_units * (2 if bi else 1)
        enc_out_size = prev_size
        #self.att_fn=DotProductAttention(True)
        self.att_fn=MLPAttention(enc_out_size,cf.dec_rnn_sizes,cf.dec_rnn_sizes)
        self.enc_dec_conection=nn.Linear(enc_out_size,cf.dec_rnn_sizes)
        self.dec_emb_lyr = nn.Embedding(dec_out_size, cf.dec_emb_size, padding_idx=None)
        prev_size = cf.dec_emb_size
        self.dec_rnn_lyr = nn.LSTMCell(cf.dec_emb_size,cf.dec_rnn_sizes)
        self.dec_concat_lyr=nn.Linear(cf.dec_rnn_sizes*2,cf.dec_emb_size)
        self.dec_out_lyr=nn.Linear(cf.dec_emb_size,dec_out_size)
        self.dec_out_lyr.weight = self.dec_emb_lyr.weight

        self.loss_fn=nn.CrossEntropyLoss(ignore_index=0)
        #self.loss_fn=nn.SmoothL1Loss()

        self.ignore_index=0
        self.optimizer=optim.Adam(self.parameters(), lr=cf.lr)
        self.cuda()

    def __call__(self,fbank,len=None,txt=None,key="word",cl_len=None):
        if cl_len is not None:
            cl_len=int(fbank.size(1)*(cl_len/10))
            fbank=fbank[:,:cl_len]
            for idx in range(fbank.size(0)):
                len[idx]=min(len[idx],cl_len)
        fbank=fbank.cuda()
        if self.training:
            len,perm_index=len.sort(0, descending=True)
            len=len.numpy()
            fbank,txt=fbank[perm_index],txt[perm_index]
            self.ctx,(self.h,self.c)=self.enc(fbank,len)
            output,score=self.dec(txt.cuda(),key=key)
            loss=self.update(output,txt.cuda())
            output.detach()
            return output.argmax(-1)[0].cpu().numpy(),txt[0].cpu().numpy(),loss
        else:
            with torch.no_grad():
                #import pdb; pdb.set_trace()
                self.ctx,(self.h,self.c)=self.enc(fbank)
                output,score=self.dec(key=key)
            return output.argmax(-1)[0].cpu().numpy(),score[0].cpu().numpy()

    def enc(self,fbank,len=None):
        res=self.enc_fnn_lyr(fbank)
        for rnn,skip,drop in zip(self.enc_rnn_lyr,cf.enc_rnn_skip,cf.enc_rnn_drop):
            if len is not None:
                res = pack(res, len, batch_first=True)
            res,(h,c)=rnn(res)
            if len is not None:
                res,_=unpack(res, batch_first=True)
                len=[x // skip for x in len]
            res = F.dropout(res[:,::skip],drop,self.training)
            #res=F.normalize(res,dim=-1)

        res=self.enc_dec_conection(res)
        if rnn.bidirectional:
            c=self.enc_dec_conection(torch.cat((c[0],c[1]),-1))
            h=self.enc_dec_conection(torch.cat((h[0],h[1]),-1))
        else:
            c=self.enc_dec_conection(c[0])
            h=self.enc_dec_conection(h[1])

        return res,(h,c)

    def dec(self,txt=None,mask=None,key="word"):
        batch_size,src_len,dim=self.ctx.size()
        dec_in=torch.ones((batch_size)).type(torch.LongTensor)*cf.SOS
        dec_in=dec_in.cuda()
        out_len=txt.size(1) if txt is not None else cf.maxlen[key]
        output,score=list(),list()

        for idx in range(out_len):
            emb=F.dropout(self.dec_emb_lyr(dec_in),cf.dec_emb_drop,self.training)
            self.h,self.c=self.dec_rnn_lyr(emb,(self.h,self.c))
            self.h=F.dropout(self.h,cf.dec_rnn_drop,self.training)
            context,_score=self.att_fn(self.ctx,self.h,mask)
            concat=self.dec_concat_lyr(torch.cat((self.h, context), -1))
            out=self.dec_out_lyr(concat)
            if txt is not None:
                dec_in=txt[:,idx]
            else:
                dec_in=out.argmax(-1)
                if idx>0 and dec_in.item()==cf.EOS:
                    break
            output.append(out)
            score.append(_score)
            #concat.append(_concat)
        try:
            output,score=torch.stack(output,1),torch.stack(score,1)
        except Exception as e:
            import pdb; pdb.set_trace()
        return output,score#,torch.stack(concat,1)

    def update(self,hyp,ref):
        loss=self.loss_fn(hyp.transpose(1,2),ref)
        loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(), 1.)
        self.optimizer.step()
        self.zero_grad()
        return loss.item()

    def save(self,path):
        torch.save(self.state_dict(),path)

    def load(self,path):
        self.load_state_dict(torch.load(path))
        self.cuda()

    def adjust_lr(self,lr):
         for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
def make_stop_targets(len):
    stop_targets = torch.ones((len.size(),))
    for i in range(len(stop_targets)):
        stop_targets[i, 0:audio_lengths[i] - 1] *= 0
    return stop_targets

def make_mask(len):
    return 0
    stop_targets = torch.ones((len.size(0),len.max()))
    for i in range(len(stop_targets)):
        stop_targets[i, 0:audio_lengths[i] - 1] *= 0
    return stop_targets

class LuongDecoder(nn.Module):
    def __init__(self, insize,hidden_size,out_size,drop):
        super(LuongDecoder, self).__init__()
        #self.cell=nn.GRUCell(insize,hidden_size)
        self.rnn_nets=nn.ModuleList()
        self.drops=nn.ModuleList()
        self.concat=nn.Linear(hidden_size*2,out_size)
        self.stop=nn.Linear(hidden_size*2,1)

        for d in drop:
            self.rnn_nets.append(nn.LSTM(insize,hidden_size,batch_first=True))
            self.drops.append(nn.Dropout(d))
            insize=hidden_size


    def forward(self,input,att_fn):
        #import pdb; pdb.set_trace()
        for rnn,drop in zip(self.rnn_nets,self.drops):
            input,self.hidden=rnn(input,self.hidden)
            input=drop(input)

        att_weights=att_fn(input,self.memory)
        context=att_weights.bmm(self.memory)
        #import pdb; pdb.set_trace()
        output=self.concat(torch.cat((input, context), -1))
        stop=self.stop(torch.cat((input, context), -1))
        return output,stop,att_weights,context
    def init(self,memory):
        #memory.requires_grad=True
        self.memory=memory
        self.hidden=(memory[:,-1,:].unsqueeze(0).contiguous(),memory[:,-1,:].unsqueeze(0).contiguous())
        #import pdb; pdb.set_trace()
