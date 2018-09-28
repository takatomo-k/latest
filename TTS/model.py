import torch,random
from collections import OrderedDict
from torch import nn,optim
from torch.nn import functional as F
from modules.attention import *
from modules.networks import *
from torch.nn.utils.rnn import pack_padded_sequence as pack, pad_packed_sequence as unpack
num_mel=80
num_linear=1025
emb_size=128
pre_hidden=128
dec_hidden=512
mel_drop=0.3
outputs_per_step=5
linear_hidden=512
hidden_size=512
griffin_lim_iters=100
lr=0.0001

class tts_model(nn.Module):
    def __init__(self,input_size,feat_size):
        super(tts_model,self).__init__()
        #Encoder
        self.embedding_size = emb_size
        self.embed = nn.Embedding(input_size, emb_size)
        self.emb_prenet = nn.Sequential(OrderedDict([
             ('fc1', nn.Linear(emb_size, pre_hidden*2)),
             ('relu1', nn.ReLU()),
             ('dropout1', nn.Dropout(0.5)),
             ('fc2', nn.Linear(pre_hidden*2, pre_hidden)),
             ('relu2', nn.ReLU()),
             ('dropout2', nn.Dropout(0.5)),
        ]))
        self.cbhg = CBHG(pre_hidden)

        #Attention
        self.att_fn=MLPAttention(pre_hidden*2,dec_hidden)
        #Mel Decoder
        self.feat_size=feat_size
        self.mel_prenet = nn.Sequential(OrderedDict([
             ('fc1', nn.Linear(feat_size, pre_hidden*2)),
             ('relu1', nn.ReLU()),
             ('dropout1', nn.Dropout(0.5)),
             ('fc2', nn.Linear(pre_hidden*2, pre_hidden)),
             ('relu2', nn.ReLU()),
             ('dropout2', nn.Dropout(0.5)),
        ]))

        self.attn_gru = nn.LSTMCell(pre_hidden, dec_hidden,dec_hidden)
        self.gru1 = nn.LSTMCell(dec_hidden, dec_hidden)
        self.gru2 = nn.LSTMCell(dec_hidden, dec_hidden)

        self.attn_projection = nn.Linear(dec_hidden+(pre_hidden* 2), dec_hidden)
        self.mel_out = nn.Linear(dec_hidden, feat_size * outputs_per_step)
        self.stop= nn.Linear(dec_hidden,outputs_per_step)
        """
        #Linear decoder
        self.postcbhg = CBHG(linear_hidden,
                             K=8,
                             projection_size=feat_size,
                             is_post=True)
        self.out_linear = nn.Linear(hidden_size * 2,
                                num_linear)
        """
        #optimizer
        self.sigmoid=nn.Sigmoid()
        self.loss_fn=nn.SmoothL1Loss()
        self.loss_stop=nn.BCELoss()
        self.optimizer=optim.Adam(self.parameters(), lr=lr)
        #self.n_priority_freq = int(3000 / (16000* 0.5) * num_linear)
        self.cuda()

    def __call__(self,txt,mel_input=None,mel_len=None):
        with torch.set_grad_enabled(self.training):
            #import pdb; pdb.set_trace()
            ctx=self.enc(txt.cuda())
            mel_out,stops,score,_=self.dec(ctx,mel_input)
            #linear_out=self.dec_linear(mel_out)
            if self.training:
                loss=self.update(mel_out,mel_input.cuda(),stops,make_stop_targets(mel_len,mel_input.size(1)))
                return mel_out,loss
            else:
                return mel_out,score


    def enc(self,txt):

        _input=self.embed(txt)
        _input=self.emb_prenet(_input)
        ctx=self.cbhg(_input)
        return ctx

    def enc_softmax(self,_input):
        batch_size=_input.size(0)
        in_size,emb_size=self.embed.weight.size()
        _input=F.softmax(_input,-1)
        _input=torch.bmm(_input,self.embed.weight.unsqueeze(0).expand(batch_size,in_size,emb_size))
        _input=self.emb_prenet(_input)
        ctx=self.cbhg(_input)
        return ctx

    def init_dec(self,batch_size):
        dec_in       =torch.ones(batch_size, self.feat_size).cuda()
        query,cell   =torch.zeros(batch_size, self.attn_gru.hidden_size).cuda(),torch.zeros(batch_size, self.attn_gru.hidden_size).cuda()
        gru1_h,gru1_c=torch.zeros(batch_size, self.gru1.hidden_size).cuda(),torch.zeros(batch_size, self.gru1.hidden_size).cuda()
        gru2_h,gru2_c=torch.zeros(batch_size, self.gru2.hidden_size).cuda(),torch.zeros(batch_size, self.gru2.hidden_size).cuda()
        return dec_in,query,cell,gru1_h,gru1_c,gru2_h,gru2_c

    def dec(self,ctx,mel_input=None):
        if mel_input is not None:
            mel_input=F.pad(mel_input,(0,0,0,outputs_per_step-(mel_input.size(1)%outputs_per_step)), mode='constant',value=0)
        out_len=800 if mel_input is None else mel_input.size(1)
        output=list()
        stops  =list()
        score  =list()
        dec_in,query,cell,gru1_h,gru1_c,gru2_h,gru2_c=self.init_dec(ctx.size(0))
        for idx in range(0,out_len,outputs_per_step):
            _input=self.mel_prenet(dec_in)
            query,cell=self.attn_gru(_input,(query,cell))
            expected_ctx, p_ctx=self.att_fn(ctx,query)
            query=F.dropout(query,0.5,self.training)
            # Residual GRU
            gru1_input = self.attn_projection(torch.cat([query, expected_ctx], -1))
            gru1_h,gru1_c = self.gru1(gru1_input, (gru1_h,gru1_c))
            gru2_input = gru1_input + gru1_h+gru1_c

            gru2_h,gru2_c = self.gru2(gru2_input, (gru2_h,gru2_c))
            bf_out = gru2_input + gru2_h+gru2_c

            # Output
            out = self.mel_out(bf_out.squeeze(0)).view(-1, outputs_per_step, self.feat_size)
            stop = self.sigmoid(self.stop(bf_out.squeeze(0)).view(-1, outputs_per_step, 1 ))

            if not self.training or random.random()<0.2:
                dec_in=out[:,-1]
                if (not self.training) and stop[:,-1]>=0.7 :
                    break
            else:
                dec_in=mel_input[:,idx]
            output.append(out)
            stops.append(stop)
            score.append(p_ctx)

        return torch.cat(output,1),torch.cat(stops,1),torch.cat(score,1),mel_input

    def dec_one(self,dec_in,ctx,query,cell,gru1_h,gru1_c,gru2_h,gru2_c):
        _input=self.mel_prenet(dec_in)
        query,cell=self.attn_gru(_input,(query,cell))
        expected_ctx, p_ctx=self.att_fn(ctx,query)
        query=F.dropout(query,0.5,self.training)
        # Residual GRU
        gru1_input = self.attn_projection(torch.cat([query, expected_ctx], -1))
        gru1_h,gru1_c = self.gru1(gru1_input, (gru1_h,gru1_c))
        gru2_input = gru1_input + gru1_h+gru1_c

        gru2_h,gru2_c = self.gru2(gru2_input, (gru2_h,gru2_c))
        bf_out = gru2_input + gru2_h+gru2_c

        # Output
        out = self.mel_out(bf_out.squeeze(0)).view(-1, outputs_per_step, self.feat_size)
        stop = self.sigmoid(self.stop(bf_out.squeeze(0)).view(-1, outputs_per_step, 1 ))
        return out,stop,p_ctx,query,cell,gru1_h,gru1_c,gru2_h,gru2_c

    def update(self,mel_hyp,mel_ref,stop_hyp,stop_ref):
        #import pdb; pdb.set_trace()
        loss=self.loss_stop(stop_hyp.squeeze(-1),stop_ref)+self.loss_fn(mel_hyp,mel_ref)#+0.5*self.loss_fn(linear_hyp,linear_ref)+0.5*self.loss_fn(linear_hyp[:,:self.n_priority_freq],linear_ref[:,:self.n_priority_freq])
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

def make_stop_targets(len,maxlen):
    stop_targets = torch.ones((len.size(0),maxlen))#.type(torch.LongTensor)
    for i in range(len.size(0)):
        stop_targets[i, 0:len[i] - 1] *= 0
    return stop_targets.cuda()
