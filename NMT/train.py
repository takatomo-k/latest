import sys,os,argparse
sys.path.append(os.path.dirname(os.path.abspath(__file__)) +'/../')
from torch.utils.data import DataLoader
from model import nmt_model
from dataset import NMTDataset,collate_fn
from tqdm import tqdm

revese_vocab=dict()
def set_reverse_vocab(vocab):
    for key in vocab:
        revese_vocab.update({vocab[key]:key})

def make_stop_targets(batch_size,length, audio_lengths):
    stop_targets = torch.ones((batch_size, length)).type(torch.FloatTensor)
    for i in range(len(stop_targets)):
        stop_targets[i, 0:audio_lengths[i] - 1] *= 0
    return stop_targets

if __name__ == '__main__':
    #import pdb; pdb.set_trace()
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

    dataset=NMTDataset(args.src,args.trg,key=args.btec)
    dataset.sort(trg_key)
    set_reverse_vocab(dataset.trg_vocab[trg_key])
    model=nmt_model(len(dataset.src_vocab[src_key]),len(dataset.trg_vocab[trg_key]))
    model_path=os.path.join('./NMT/LOG',args.src+"2"+args.trg,args.btec)
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    model_path=os.path.join(model_path,src_key+"2"+trg_key)
    min_loss=100
    try:
        model.load(model_path)
    except:
        pass

    for epoch in range(1,args.epoch):
        #train
        model.train()
        dataloader= DataLoader(dataset, batch_size=args.batch,
        shuffle= True, collate_fn=collate_fn, drop_last= True, num_workers=args.num_workers)
        pbar=tqdm(enumerate(dataloader),total=len(dataloader))
        loss=0
        for idx,data in pbar:
            hyp,ref,_loss=model(data['src_'+src_key],data['src_'+src_key+'_len'],data['trg_'+trg_key],trg_key)
            loss+=_loss
            if (idx+1)%100==0:
                #_=wer(hyp,ref)
            #loss+=model(data['fbank'],data['fbank_len'],data[args.segment],args.segment)
                pbar.set_description(str(loss/(idx+1))[:5])

        model.save(model_path)
        if epoch>5:
            model.adjust_lr(0.001/epoch)
        else:
            continue
        #dev
        pbar=tqdm(enumerate(dataset.dev),total=len(dataset.dev))
        model.train(False)
        loss=0
        for idx,label in pbar:
            data=dataset.feature_extraction(dataset.src[label],dataset.trg[label])
            hyp,score=model(data['src_'+src_key].unsqueeze(0),key=trg_key)
            #ref=data['trg_'+trg_key][0].numpy()
            #loss+=wer(hyp,ref)
            #pbar.set_description(str(loss/(idx+1))[:5])

        if min_loss>loss:
            model.save(model_path+"_BEST")
            min_loss=loss

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
