import sys,os,argparse,torch,collections,tqdm,random
sys.path.append(os.path.dirname(os.path.abspath(__file__)) +'/../')
from torch.utils.data import Dataset,Sampler
import numpy as np
from utils.common import *
from torch.nn.utils.rnn import pad_sequence
from torch.nn.functional import pad

class RandomBatchSampler(Sampler):
    r"""Samples elements randomly, without replacement.

    Arguments:
        data_source (Dataset): dataset to sample from
    """

    def __init__(self, data_source,batch_size):
        self.data_source = data_source
        self.batch_size = batch_size
    def __iter__(self):
        size=len(self.data_source)-len(self.data_source)%self.batch_size
        data=np.array(list(range(size)))
        data=data.reshape(-1,self.batch_size)
        np.random.shuffle(data)
        data=data.flatten().tolist()
        if size<len(self.data_source):
            data.extend(list(range(size,len(self.data_source))))
        #print(data)
        return iter(data)

    def __len__(self):
        return len(self.data_source)

class NMTDataset(Dataset):
    """docstring for [object Object]."""
    def __init__(self,src_lang,trg_lang,key='ALL'):
        super(NMTDataset, self).__init__()
        self.src,self.src_vocab=load_data(os.path.join("../TEXT",key,src_lang,"clean.txt"))
        self.trg,self.trg_vocab=load_data(os.path.join("../TEXT",key,trg_lang,"clean.txt"))
        self.test,self.dev,self.train=self.cleaner()

    def cleaner(self):
        self.src=get_uniq_list(self.src)
        self.src,self.trg=get_pairwise_list(self.src,self.trg)
        return get_split_list(self.src)
    def sort(self,key):
        self.train=sorted(self.train,key=lambda x: len(self.trg[x][key]))

    def __len__(self):
        return len(self.train)

    def __getitem__(self, idx):
        return self.feature_extraction(self.src[self.train[idx]],self.trg[self.train[idx]])

    def feature_extraction(self,src,trg):
        ret=dict()
        for segment in src.keys():
            ids=list()
            for w in src[segment]:
                if w in self.src_vocab[segment]:
                    ids.append(self.src_vocab[segment][w])
                else:
                    ids.append(self.src_vocab[segment]['UNK'])
            ids.append(self.src_vocab[segment]['EOS'])
            ret.update({"src_"+segment:torch.from_numpy(np.array(ids)).type(torch.LongTensor)})

        for segment in trg.keys():
            ids=list()
            for w in trg[segment]:
                if w in self.trg_vocab[segment]:
                    ids.append(self.trg_vocab[segment][w])
                else:
                    ids.append(self.trg_vocab[segment]['UNK'])
            ids.append(self.trg_vocab[segment]['EOS'])
            ret.update({"trg_"+segment:torch.from_numpy(np.array(ids)).type(torch.LongTensor)})
        return ret



"""
class CustomDataset(Dataset):
    def __init__(self, src_lang,trg_lang=None):
        super(CustomDataset, self).__init__()
        self.src_lang=src_lang
        self.src_train,self.src_dev,self.src_test=self.load_data(src_lang)
        self.trg_train,self.trg_dev,self.trg_test=self.load_data(trg_lang)
        self.pairwise()
        self.trg_vocab,self.trg_inv_vocab=self.get_vocab(self.trg_train)
        self.src_vocab,self.src_inv_vocab=self.get_vocab(self.src_train)
        self.train_list=self.sort([k for k in self.src_train.keys()])
        self.dev_list=[k for k in self.src_dev.keys()]
        self.test_list=[k for k in self.src_test.keys()]
        self.data_list=self.train_list
        self.train_list=self.exists(self.train_list)
        self.dev_list=self.exists(self.dev_list)
        self.test_list=self.exists(self.test_list)
        #self.train_list=self.reduce_i()

    def sort(self,list_):
        if self.trg_train is not None:
            list_=sorted(list_, key=lambda x:(self.trg_train[x]['dur'],self.src_train[x]['dur']),reverse=True)
        else:
            list_=sorted(list_, key=lambda x:(self.src_train[x]['dur']),reverse=True)
        return list_

    def reduce_i(self):
        #import pdb; pdb.set_trace()
        ret=list()
        memory=dict()
        limit=10000
        for label in self.train_list:
            if self.src_train[label]["char"][0] not in memory:
                memory.update({self.src_train[label]["char"][0]:1})
            else:
                memory[self.src_train[label]["char"][0]]+=1
            if memory[self.src_train[label]["char"][0]]<=limit:
                ret.append(label)

        return ret
        pass

    def pairwise(self):
        if self.trg_train is None:
            self.src_train=common.uniq(self.src_train)
            self.src_dev=common.uniq(self.src_dev)
            self.src_test=common.uniq(self.src_test)
        else:
            self.src_train,self.trg_train=common.pairwise(self.src_train,self.trg_train)
            self.src_dev,self.trg_dev=common.pairwise(self.src_dev,self.trg_dev)
            self.src_test,self.trg_test=common.pairwise(self.src_test,self.trg_test)

    def exists(self,this_list):
        ret=list()
        for label in tqdm.tqdm(this_list):
            if label in self.src_train:
                fbank_path=self.src_train[label]['audio'].replace(".mp3",".npy").replace("mp3","fbank")
            elif label in self.src_dev:
                fbank_path=self.src_dev[label]['audio'].replace(".mp3",".npy").replace("mp3","fbank")
            elif label in self.src_test:
                fbank_path=self.src_test[label]['audio'].replace(".mp3",".npy").replace("mp3","fbank")

            #fbank_path=fbank_path.replace(".npy",".fbank.npy")
            if os.path.exists(fbank_path):
                ret.append(label)
                try:
                    torch.from_numpy(np.load(fbank_path)).type(torch.FloatTensor)
                    ret.append(label)
                except:
                    os.remove(fbank_path)
        return ret

    def get_vocab(self,data):
        if data is None:
            return None,None
        return text_utils.make_vocab(data)

    def load_data(self,lang):
        if lang is None:
            return None,None,None
        ret=list()
        data_list=["train","dev","test"]
        data_path="./data/scp"
        for d in data_list:
            path=os.path.join(data_path,lang+"_"+d)
            ret.append(common.cut_by_length(common.load_pickle(path)))
        return ret

    def save(self,path):
        common.save_pickle(path,self)

    def load(self,path):
        self=common.load_pickle(path)

    def set_feat_extract_fnc(self,func):
        self.feat_func=func

    def update_state(self,state):
        if state =="train":
            self.data_list=self.train_list
        elif state =="dev":
            self.data_list=self.dev_list
        elif state=="test":
            self.data_list=self.test_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        label=self.data_list[idx]
        if label in self.src_train:
            src=self.src_train[label]
            trg=self.trg_train[label] if self.trg_train is not None else None
        elif label in self.src_dev:
            src=self.src_dev[label]
            trg=self.trg_dev[label] if self.trg_dev is not None else None
        elif label in self.src_test:
            src=self.src_test[label]
            trg=self.trg_test[label] if self.trg_test is not None else None

        return self.feat_func(src,trg,self.src_vocab,self.trg_vocab)
"""
def collate_fn(batch):
    if isinstance(batch[0], collections.Mapping):
        data=dict()
        for key in batch[0].keys():
                feat,lengths=pad_batch(batch,key)
                data.update({key:feat,key+'_len':lengths})
        return data

    raise TypeError(('batch must contain tensors, numbers, dicts or lists; found {}'
                     .format(type(batch[0]))))


def _pad_data(x, length):
    _pad=0
    if x.dim()==1 :
        return pad(x, (0, length - x.shape[0]), mode='constant',value=_pad)
    else:
        return pad(x,(0,0,0,length - x.shape[0]), mode='constant',value=_pad)

def pad_batch(batch,key):
    feat= [d[key] for d in batch]
    lengths = [x.size(0) for x in feat]
    if len(batch)>1:
        max_len = max(lengths)
        return torch.stack([_pad_data(x, max_len) for x in feat]),torch.from_numpy(np.asarray(lengths))
    return feat[0].unsqueeze(0),np.asarray(lengths)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s','--src_lang', type=str, help='Src language')
    parser.add_argument('-t','--trg_lang', type=str, help='Src language')

    args = parser.parse_args()
    dataset=CustomDataset(args.src_lang)
    dataset=PairwiseDataset(args.src_lang,args.trg_lang)
