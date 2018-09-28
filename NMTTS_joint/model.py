import torch,random,os,sys
from torch import nn,optim
from torch.nn import functional as F
from TTS.model import tts_model
from NMT.model import nmt_model
from torch.nn.utils.rnn import pack_padded_sequence as pack, pad_packed_sequence as unpack
lr=0.001
class nmtts_model(nn.Module):
    def __init__(self,input_size,output_size,feat_size):
        super(nmtts_model,self).__init__()
        self.nmt=nmt_model(input_size,output_size)
        self.tts=tts_model(output_size,feat_size)
        self.loss_fn_nmt=nn.CrossEntropyLoss(ignore_index=0)
        self.loss_fn=nn.SmoothL1Loss()
        self.loss_stop=nn.BCELoss()
        self.optimizer=optim.Adam(self.parameters(), lr=lr)
        #self.n_priority_freq = int(3000 / (16000* 0.5) * cf.num_linear)
        self.cuda()

    def __call__(self,src,src_len=None,trg=None,trg_len=None,mel=None,mel_len=None,key="word"):
        #import pdb; pdb.set_trace()
        if self.training:
            _,perm_index=src_len.sort(0, descending=True)
            src,src_len=src[perm_index],src_len[perm_index]
            trg,trg_len=trg[perm_index],trg_len[perm_index]
            mel,mel_len=mel[perm_index],mel_len[perm_index]

            nmt_out=self.forward_nmt(src,src_len,trg,key)
            mel_out,stops,mel_ref=self.forward_softmax_tts(nmt_out,mel)

            nmt_loss,tts_loss=self.update(mel_out,mel_ref,stops,make_stop_targets(mel_len,mel_ref.size(1)),nmt_out,trg.cuda())

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
        output,score=self.nmt.dec(ctx,h,c,trg,key=key)
        return output


    def forward_tts(self,txt,mel):
        if self.training:
            mel=mel.cuda()
        ctx=self.tts.enc_softmax(txt.argmax(-1).cuda())
        mel_out,stops,score,mel_ref=self.tts.dec(ctx,mel)
        return mel_out,stops,mel_ref

    def forward_softmax_tts(self,txt,mel):
        if self.training:
            mel=mel.cuda()
        ctx=self.tts.enc_softmax(txt.cuda())
        mel_out,stops,score,mel_ref=self.tts.dec(ctx,mel)
        return mel_out,stops,mel_ref

    def update(self,mel_hyp,mel_ref,stop_hyp,stop_ref,nmt_hyp,nmt_ref):
        nmt_loss=self.loss_fn_nmt(nmt_hyp.transpose(1,2),nmt_ref)
        tts_loss=self.loss_stop(stop_hyp.squeeze(-1),stop_ref)+self.loss_fn(mel_hyp,mel_ref)
        loss=nmt_loss+tts_loss
        loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(), 1.)
        self.optimizer.step()
        self.zero_grad()
        return nmt_loss.item(),tts_loss.item()

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
            self.cuda()
        except:
            pass
            #nmt_path=os.path.join('../NMT/NMT/LOG',src+"2"+trg,btec,src_key+"2"+trg_key)
            #tts_path=os.path.join('../TTS/TTS/LOG',trg,btec,trg_key)
            #self.nmt.load(nmt_path)
            #self.tts.load(tts_path)

    def adjust_lr(self,epoch):
         for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr/epoch

def make_stop_targets(len,maxlen):
    stop_targets = torch.ones((len.size(0),maxlen))#.type(torch.LongTensor)
    for i in range(len.size(0)):
        stop_targets[i, 0:len[i] - 1] *= 0
    return stop_targets.cuda()
