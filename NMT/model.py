import torch,random
from torch import nn,optim
from torch.nn import functional as F
from modules.attention import *
from torch.nn.utils.rnn import pack_padded_sequence as pack, pad_packed_sequence as unpack
enc_emb_size=256
enc_emb_drop=0.25
enc_rnn_sizes=[256, 256]
enc_rnn_bi=[True,True]
enc_rnn_skip=[1,1]
enc_rnn_drop=[0.0,0.25]
dec_emb_size=256
dec_emb_drop=0.25#en:0.8 ja:0.25
# tying weight from char/word embedding with softmax layer
dec_rnn_sizes=512
dec_rnn_drop =0.25#en:0.8 ja:0.25

lr=0.001
EOS=2
SOS=1
maxlen={'word':30,'char':100,'p_word':30,'p_char':100}
class nmt_model(nn.Module):
    def __init__(self, enc_in_size, dec_out_size):
        super(nmt_model,self).__init__()
        self.enc_emb_lyr = nn.Embedding(enc_in_size, enc_emb_size, padding_idx=None)
        prev_size = enc_emb_size
        self.enc_rnn_lyr = nn.ModuleList()
        for n_units,bi in zip(enc_rnn_sizes,enc_rnn_bi) :
            self.enc_rnn_lyr.append(nn.LSTM(prev_size,n_units,bias=True,batch_first=True,bidirectional=bi))
            prev_size = n_units * (2 if bi else 1)
        enc_out_size = prev_size
        self.att_fn=MLPAttention(prev_size,dec_rnn_sizes,dec_rnn_sizes)
        self.enc_dec_conection=nn.Linear(enc_out_size,dec_rnn_sizes)
        self.dec_emb_lyr = nn.Embedding(dec_out_size, dec_emb_size, padding_idx=None)
        prev_size = dec_emb_size
        self.dec_rnn_lyr = nn.LSTMCell(dec_emb_size,dec_rnn_sizes)
        self.dec_concat_lyr=nn.Linear(dec_rnn_sizes*2,dec_emb_size)
        self.dec_out_lyr=nn.Linear(dec_emb_size,dec_out_size)
        #self.dec_out_lyr.weight = self.dec_emb_lyr.weight
        self.loss_fn=nn.CrossEntropyLoss(ignore_index=0)

        self.optimizer=optim.Adam(self.parameters(), lr=lr)
        self.cuda()

    def __call__(self,src,len=None,trg=None,key="word"):
        #import pdb; pdb.set_trace()
        if self.training:
            len,perm_index=len.sort(0, descending=True)
            len=len.numpy()
            src,trg=src[perm_index].cuda(),trg[perm_index].cuda()
            ctx,(h,c)=self.enc(src,len)
            output,score=self.dec(ctx,h,c,trg,key=key,)
            loss=self.update(output,trg)
            output.detach()
            return output.argmax(-1)[0].cpu().numpy(),trg[0].cpu().numpy(),loss
            #return self.update(output,txt.cuda())
        else:
            with torch.no_grad():
                ctx,(h,c)=self.enc(src.cuda())
                output,score=self.dec(key=key)
            return output.argmax(-1)[0].cpu().numpy(),score[0].cpu().numpy()

    def enc(self,src,len=None):
        res=F.dropout(self.enc_emb_lyr(src),enc_emb_drop,self.training)
        h,c=None,None
        #import pdb; pdb.set_trace()
        for rnn,skip,drop in zip(self.enc_rnn_lyr,enc_rnn_skip,enc_rnn_drop):
            res = res if len is None else pack(res, len, batch_first=True)
            res,(h,c) =rnn(res)
            if len is not None:
            #    import pdb; pdb.set_trace()
                res,_=unpack(res, batch_first=True)
                len  =[x // skip for x in len]

            res = F.dropout(res[:,::skip],drop,self.training)
            #res=F.normalize(,dim=-1)
        res=self.enc_dec_conection(res)
        if rnn.bidirectional:
            c=self.enc_dec_conection(torch.cat((c[0],c[1]),-1))
            h=self.enc_dec_conection(torch.cat((h[0],h[1]),-1))
        else:
            c=self.enc_dec_conection(c[0])
            h=self.enc_dec_conection(h[1])
        return res,(h,c)

    def dec(self,ctx,h,c,trg=None,mask=None,key="word"):
        batch_size,src_len,dim=ctx.size()
        dec_in=torch.ones((batch_size)).type(torch.LongTensor)*SOS
        dec_in=dec_in.cuda()
        out_len=trg.size(1) if trg is not None else maxlen[key]
        output,score=list(),list()
        for idx in range(out_len):
            emb=F.dropout(self.dec_emb_lyr(dec_in),dec_emb_drop,self.training)
            h,c=self.dec_rnn_lyr(emb,(h,c))
            h=F.dropout(h,dec_rnn_drop,self.training)
            context,_score=self.att_fn(ctx,h,mask)
            concat=self.dec_concat_lyr(torch.cat((h, context), -1))
            out=self.dec_out_lyr(concat)
            #import pdb; pdb.set_trace()
            if trg is not None:
                dec_in=trg[:,idx]
            else:
                dec_in=out.argmax(-1)
                if idx>0 and dec_in.item()==EOS:
                    break
            output.append(out)
            score.append(_score)

        return torch.stack(output,1),torch.stack(score,1)

    def dec_one(self,dec_in,ctx,h,c,prev_context=None):
        emb=self.dec_emb_lyr(dec_in)
        emb=F.dropout(emb,dec_emb_drop,self.training)
        h,c=self.dec_rnn_lyr(emb,(h,c))
        h=F.dropout(h,dec_rnn_drop,self.training)
        if prev_context is not None:
            h=h+prev_context
        context,_score=self.att_fn(ctx,h,None)
        concat=self.dec_concat_lyr(torch.cat((h, context), -1))
        out=self.dec_out_lyr(concat)
        return out,_score,context

    def update(self,hyp,ref):
        #import pdb; pdb.set_trace()
        loss=self.loss_fn(hyp.transpose(1,2),ref)

        loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(), 1.)
        self.optimizer.step()
        self.zero_grad()
        return loss.item()
    def save(self,path):
        torch.save(self.state_dict(),path+self.__class__.__name__)

    def load(self,path):
        self.load_state_dict(torch.load(path+self.__class__.__name__))
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
    """docstring for [object Object]."""
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
