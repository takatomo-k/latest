import torch,random,os,sys
from torch import nn,optim
from torch.nn import functional as F
from TTS.model import tts_model
from NMT.model import nmt_model
from torch.nn.utils.rnn import pack_padded_sequence as pack, pad_packed_sequence as unpack

outputs_per_step=5
lr=0.001
EOS=2
class nmtts_model(nn.Module):
    def __init__(self,input_size,output_size,feat_size):
        super(nmtts_model,self).__init__()
        self.feat_size=feat_size

        self.nmt=nmt_model(input_size,output_size)
        self.tts=tts_model(output_size,feat_size)

        self.loss_fn=nn.CrossEntropyLoss(ignore_index=0)
        self.loss_fn=nn.SmoothL1Loss()
        self.loss_stop=nn.BCELoss()
        self.optimizer=optim.Adam(self.parameters(), lr=lr)
        self.cuda()

    def __call__(self,src,src_len=None,trg=None,trg_len=None,mel=None,mel_len=None,key="word"):
        if self.training:
            _,perm_index=src_len.sort(0, descending=True)
            src,src_len=src[perm_index],src_len[perm_index]
            trg,trg_len=trg[perm_index],trg_len[perm_index]
            mel,mel_len=mel[perm_index],mel_len[perm_index]

            nmt_out,nmt_score=self.forward_nmt(src,src_len,trg,key)
            mel_out,stops,tts_score,mel_ref=self.forward_tts(nmt_out,mel)
            #import pdb; pdb.set_trace()
            tts_loss=self.update(mel_out,mel_ref,stops,make_stop_targets(mel_len,mel_ref.size(1)))
            nmt_loss=F.nll_loss(nmt_out.transpose(1,2),trg.cuda(),ignore_index=0).item()

            return nmt_loss,tts_loss
        else:
            with torch.no_grad():
                nmt_out=self.forward_nmt(src,src_len,trg,key)
                mel_out,stops,_=self.forward_softmax_tts(nmt_out,mel)
                return mel_out,nmt_out.argmax(-1)

    def forward_nmt(self,src,src_len,trg,key):
        if self.training:
            trg,src_len=trg.cuda(),src_len.numpy()
        ctx,(h,c)=self.nmt.enc(src.cuda(),src_len)
        dec_in=torch.ones((ctx.size(0))).type(torch.LongTensor).cuda()
        out_len=trg.size(1) if trg is not None else 100
        out,score,context=list(),list(),list()
        _context=None
        for idx in range(out_len):
            _out,_score,_context=self.nmt.dec_one(dec_in,ctx,h,c,_context)
            if trg is not None:
                dec_in=trg[:,idx]
            else:
                dec_in=out.argmax(-1)
                if idx>0 and dec_in.item()==EOS:
                    break
            score.append(_score)
            context.append(_context)

            #import pdb; pdb.set_trace()
            gumbel=False
            if gumbel:
                out.append(F.softmax(_out,-1))
            else:
                out.append(F.softmax(_out,-1))

        return torch.stack(out,1),torch.stack(score,1)


    def forward_tts(self,txt,mel):
        if self.training:
            mel=F.pad(mel,(0,0,0,outputs_per_step-(mel.size(1)%outputs_per_step)), mode='constant',value=0).cuda()
        ctx=self.tts.enc(txt.argmax(-1).cuda())
        out_len=800 if mel is None else mel.size(1)
        out,stop,score=list(),list(),list()

        dec_in,query,cell,gru1_h,gru1_c,gru2_h,gru2_c=self.tts.init_dec(ctx.size(0))
        for idx in range(0,out_len,outputs_per_step):
            _out,_stop,_score,query,cell,gru1_h,gru1_c,gru2_h,gru2_c=self.tts.dec_one(dec_in,ctx,query,cell,gru1_h,gru1_c,gru2_h,gru2_c)
            if not self.training or random.random()<0.2:
                dec_in=_out[:,-1]
                if (not self.training) and _stop[:,-1]>=0.7 :
                    break
            else:
                dec_in=mel[:,idx]
            out.append(_out)
            stop.append(_stop)
            score.append(_score)
        return torch.cat(out,1),torch.cat(stop,1),torch.cat(score,1),mel

    def forward_softmax_tts(self,txt,mel):
        if self.training:
            mel=mel.cuda()
        ctx=self.tts.enc_softmax(txt.cuda())
        mel_out,stops,score,mel_ref=self.tts.dec(ctx,mel)
        return mel_out,stops,mel_ref



    def update(self,mel_hyp,mel_ref,stop_hyp,stop_ref):
        #import pdb; pdb.set_trace()
        loss=self.loss_stop(stop_hyp.squeeze(-1),stop_ref)+self.loss_fn(mel_hyp,mel_ref)#+0.5*self.loss_fn(linear_hyp,linear_ref)+0.5*self.loss_fn(linear_hyp[:,:self.n_priority_freq],linear_ref[:,:self.n_priority_freq])
        loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(), 1.)
        self.optimizer.step()
        self.zero_grad()
        return loss.item()

    def save(self,key=""):
        torch.save(self.state_dict(),self.model_path+key)

    def load(self,src,trg,src_key,trg_key,btec,key=""):
        self.model_path=os.path.join('./LOG',src+"2"+trg,btec)
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        self.model_path=os.path.join(self.model_path,src_key+"2"+trg_key+key)
        try:
            self.load_state_dict(torch.load(self.model_path))
            print("LOADMODEL:",self.model_path)
        except:
            nmt_path=os.path.join('../NMT/NMT/LOG',src+"2"+trg,btec,src_key+"2"+trg_key)
            tts_path=os.path.join('../TTS/TTS/LOG',trg,btec,trg_key)
            self.nmt.load(nmt_path)
            self.tts.load(tts_path)

        self.cuda()
    def adjust_lr(self,epoch):
         for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr/epoch

def make_stop_targets(len,maxlen):
    stop_targets = torch.ones((len.size(0),maxlen))#.type(torch.LongTensor)
    for i in range(len.size(0)):
        stop_targets[i, 0:len[i] - 1] *= 0
    return stop_targets.cuda()
