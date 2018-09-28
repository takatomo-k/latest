import sys,os,argparse
sys.path.append(os.path.dirname(os.path.abspath(__file__)) +'/../')
from torch.utils.data import DataLoader
from model import nmtts_model
from dataset import NMTTSDataset,collate_fn
from tqdm import tqdm
import torch.nn.functional as F

revese_vocab=dict()
def set_reverse_vocab(vocab):
    for key in vocab:
        revese_vocab.update({vocab[key]:key})

def save_txt(hyp,ref,path):
    hyp_txt=list()
    ref_txt=list()
    for h in hyp:
        if h==2:
            break
        if revese_vocab[h]!=" ":
            hyp_txt.append(revese_vocab[h])
    for r in ref:
        if r==2:
            break
        if revese_vocab[r]!=" ":
            ref_txt.append(revese_vocab[r])
    with open(path,"w")as f:
        f.write(" ".join(hyp_txt)+"|"+" ".join(ref_txt))

def mean_squared_error(hyp,ref):
    length=max(hyp.size(0),ref.size(0))
    hyp=F.pad(hyp,(0,0,0,length-hyp.size(0)), mode='constant',value=0)
    ref=F.pad(ref,(0,0,0,length-ref.size(0)), mode='constant',value=0)
    return F.mse_loss(hyp,ref,size_average=True).item()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s','--src', type=str, help='Src language')
    parser.add_argument('-t','--trg', type=str, help='Src language')
    parser.add_argument('-sk','--src_segment', type=str, help='Src language',default="word")
    parser.add_argument('-tk','--trg_segment', type=str, help='Src language',default="word")
    parser.add_argument('-b','--batch', type=int, help='Src language',default=16)
    parser.add_argument('-e','--epoch', type=int, help='Src language',default=1000)
    parser.add_argument('-n','--num_workers', type=int, help='Src language',default=16)
    parser.add_argument('-g','--gpu', type=int, help='Src language',default=0)
    parser.add_argument('--btec', type=str, help='Src language',default="ALL")
    args = parser.parse_args()
    src_key,trg_key=args.src_segment,args.trg_segment
    #import pdb; pdb.set_trace()
    dataset=NMTTSDataset(args.src,args.trg,key=args.btec)
    dataset.sort(trg_key)
    set_reverse_vocab(dataset.trg_vocab[trg_key])
    model=nmtts_model(len(dataset.src_vocab[src_key]),len(dataset.trg_vocab[trg_key]),dataset.mean.shape[0])
    min_loss=100
    #model.load(args.src,args.trg,src_key,trg_key,args.btec)

    for epoch in range(1,args.epoch):
        #train
        model.train()
        dataloader= DataLoader(dataset, batch_size=args.batch,
        shuffle= True, collate_fn=collate_fn, drop_last= True, num_workers=args.num_workers)
        pbar=tqdm(enumerate(dataloader),total=len(dataloader))
        nmt_loss,tts_loss=0,0
        for idx,data in pbar:
            src,src_len=data['src_'+src_key],data['src_'+src_key+'_len']
            trg,trg_len=data['trg_'+trg_key],data['trg_'+trg_key+'_len']
            mel,mel_len=data['feat'],data['feat_len']
            Nloss,Tloss=model(src,src_len,trg,trg_len,mel,mel_len,trg_key)

            nmt_loss+=Nloss
            tts_loss+=Tloss
            if (idx+1)%100==0:
                pbar.set_description(str(tts_loss/(idx+1))[:5])

        model.save()
        if epoch>5:
            model.adjust_lr(epoch)
        #dev
        pbar=tqdm(enumerate(dataset.dev),total=len(dataset.dev))
        model.train(False)
        tts_loss=0
        for idx,label in pbar:
            data=dataset.feature_extraction(dataset.src[label],dataset.trg[label])

            tts_hyp,nmt_hyp=model(data['src_'+src_key].unsqueeze(0),key=trg_key)
            tts_hyp=tts_hyp[0].cpu()
            ref=data['feat']#.transpose(0,1)
            tts_loss+=mean_squared_error(tts_hyp,ref)
            if idx<10:
                dataset.gen_wav(tts_hyp.numpy(),"./"+args.trg+str(epoch)+"_"+str(idx)+".wav")
                save_txt(nmt_hyp[0].cpu().numpy(),data['trg_'+trg_key].numpy(),"./"+args.trg+str(epoch)+"_"+str(idx)+".txt")

            #ref=data['trg_'+trg_key][0].numpy()
            #loss+=wer(hyp,ref)
            pbar.set_description(str(tts_loss/(idx+1))[:5])

        if min_loss>tts_loss:
            model.save("_BEST")
            min_loss=tts_loss

    model.load(model_path+"_BEST")
    dataset.update_state("test")
    dataloader= DataLoader(dataset, batch_size=1,
    shuffle= False, collate_fn=collate_fn, drop_last= False, num_workers=args.num_workers)
    pbar=tqdm(enumerate(dataloader),total=len(dataloader))

    model.train(False)
    loss=0
    for idx,data in pbar:
        hyp,score=model(data['src_'+src_key],data['src_'+src_key+'_len'],key=trg_key)
        ref=data['trg_'+trg_key][0].numpy()
        loss+=wer(hyp,ref)
        pbar.set_description(str(loss/(idx+1)))
