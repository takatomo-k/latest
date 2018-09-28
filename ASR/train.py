import sys,os,argparse
sys.path.append(os.path.dirname(os.path.abspath(__file__)) +'/../')
from torch.utils.data import DataLoader
from model import asr_model
from dataset import ASRDataset,collate_fn
from tqdm import tqdm
import Levenshtein

#from utils import eval
revese_vocab=dict()

def set_reverse_vocab(vocab):
    for key in vocab:
        revese_vocab.update({vocab[key]:key})

def make_stop_targets(batch_size,length, audio_lengths):
    stop_targets = torch.ones((batch_size, length)).type(torch.FloatTensor)
    for i in range(len(stop_targets)):
        stop_targets[i, 0:audio_lengths[i] - 1] *= 0
    return stop_targets

def wer(hyp,ref):
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

    import pdb; pdb.set_trace()
    Levenshtein.distance(string1, string2)
    loss=eval.wer(hyp,ref)
    print(" ".join(hyp_txt)+"|"+" ".join(ref_txt))
    return loss

if __name__ == '__main__':
    #import pdb; pdb.set_trace()
    parser = argparse.ArgumentParser()
    parser.add_argument('-s','--src', type=str, help='Src language')
    parser.add_argument('-b','--batch', type=int, help='Src language',default=16)
    parser.add_argument('-e','--epoch', type=int, help='Src language',default=1000)
    parser.add_argument('-n','--num_workers', type=int, help='Src language',default=16)
    parser.add_argument('-k','--segment', type=str, help='Src language',default="word")
    parser.add_argument('-g','--gpu', type=int, help='Src language',default=0)
    parser.add_argument('--btec', type=str, help='Src language',default="ALL")
    args = parser.parse_args()
    dataset=ASRDataset(args.src,key=args.btec)
    dataset.sort(args.segment)
    set_reverse_vocab(dataset.vocab[args.segment])
    model=asr_model(dataset.mean.shape[0],len(revese_vocab))
    model_path=os.path.join('./ASR/LOG',args.src,args.btec)
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    model_path=os.path.join(model_path,args.segment)
    min_loss=100
    try:
        model.load(model_path)
    except:
        pass

    for epoch in range(1,args.epoch):
        #train
        
        model.train()
        dataloader= DataLoader(dataset, batch_size=args.batch,
        shuffle= epoch!=1, collate_fn=collate_fn, drop_last= True, num_workers=args.num_workers)
        pbar=tqdm(enumerate(dataloader),total=len(dataloader))
        loss=0
        for idx,data in pbar:
            hyp,ref,_loss=model(data['feat'],data['feat_len'],data[args.segment],args.segment)
            loss+=_loss
            if (idx+1)%100==0:
            #    _=wer(hyp,ref)
            #loss+=model(data['fbank'],data['fbank_len'],data[args.segment],args.segment)
                pbar.set_description(str(loss/(100))[:5])
                loss=0
        model.save(model_path)
        if epoch>5:
            model.adjust_lr(0.001/epoch)
        #dev
        pbar=tqdm(enumerate(dataset.dev),total=len(dataset.dev))
        model.train(False)
        loss=0
        for idx,label in pbar:
            #try:
            data=dataset.feature_extraction(dataset.data[label])
            hyp,score=model(data['feat'].unsqueeze(0),key=args.segment)
            #ref=data[args.segment][0].numpy()
            #loss+=wer(hyp,ref)
            #pbar.set_description(str(loss/(idx+1))[:5])
            if idx==100:
                break
        #print(str(loss/(idx+1))[:5])
            #except:
            #    pass
        #if min_loss>loss:
            #model.save(model_path+"_BEST")
            #min_loss=loss

    model.load(model_path+"_BEST")
    dataset.update_state("test")
    dataloader= DataLoader(dataset, batch_size=1,
    shuffle= False, collate_fn=collate_fn, drop_last= False, num_workers=args.num_workers)
    pbar=tqdm(enumerate(dataloader),total=len(dataloader))

    model.train(False)
    loss=0
    for idx,data in pbar:
        hyp,score=model(data['fbank'],data['fbank_len'],key=args.segment)
        ref=data[args.segment][0].numpy()
        loss+=wer(hyp,ref,dataset.src_inv_vocab[args.segment])
        pbar.set_description(str(loss/(idx+1)))
