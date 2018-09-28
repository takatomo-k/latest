import sys,os,argparse,scipy,librosa
sys.path.append(os.path.dirname(os.path.abspath(__file__)) +'/../')
from torch.utils.data import DataLoader
#from TTS.model import tts_model
from TTS.dataset import TTSDataset,collate_fn
from NMT.dataset import NMTDataset,collate_fn
from tqdm import tqdm
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s','--src', type=str, help='Src language')
    parser.add_argument('-t','--trg', type=str, help='Src language')
    parser.add_argument('-b','--batch', type=int, help='Src language',default=16)
    parser.add_argument('-e','--epoch', type=int, help='Src language',default=1000)
    parser.add_argument('-n','--num_workers', type=int, help='Src language',default=16)
    parser.add_argument('-k','--segment', type=str, help='Src language',default="word")
    parser.add_argument('-g','--gpu', type=int, help='Src language',default=0)
    parser.add_argument('--btec', type=str, help='Src language',default="ALL")
    args = parser.parse_args()
    import pdb; pdb.set_trace()
    tts_dataset=TTSDataset(args.src,key=args.btec)
    nmt_dataset=NMTDataset(args.src,args.trg,key=args.btec)
    import pdb; pdb.set_trace()
    tts_dataset.sort(args.segment)
    model=tts_model(len(dataset.vocab[args.segment]),dataset.mean.shape[0])
    model_path=os.path.join('./TTS',"LOG",args.src,args.btec)
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
        skip=0
        for idx,data in pbar:
            #import pdb; pdb.set_trace()
            if data[args.segment].size(1)>100:
                skip+=1
                continue
            hyp,_loss=model(data[args.segment],data['feat'],data["feat_len"])
            loss+=_loss
            if (idx-skip+1)%100==0:
                pbar.set_description(str(loss/(100))[:5])
                loss=0
        model.save(model_path)
        if epoch>5:
            model.adjust_lr(0.001/epoch)
        else:
            continue
        #dev
        pbar=tqdm(enumerate(dataset.dev),total=len(dataset.dev))
        model.train(False)
        loss=0
#        import pdb; pdb.set_trace()
        for idx,label in pbar:
            data=dataset.feature_extraction(dataset.data[label])
            hyp,score=model(data[args.segment].unsqueeze(0))
            dataset.gen_wav(hyp[0].cpu().numpy(),"./"+str(epoch)+"_"+str(idx)+".wav")
            if idx>10:
                break
            #pbar.set_description(str(loss/(idx+1))[:5])
            #except:
            #    pass
        #if min_loss>loss:
        #    model.save(model_path+"_BEST")
        #    min_loss=loss

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
