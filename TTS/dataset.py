import sys,os,argparse,torch,collections,tqdm,random
sys.path.append(os.path.dirname(os.path.abspath(__file__)) +'/../')
from utils.common import *
from utils.audio import *

from torch.utils.data import Dataset
import numpy as np
from torch.nn.utils.rnn import pad_sequence
from torch.nn.functional import pad

DECLEN=1000
random.seed(a=1, version=1)

class TTSDataset(Dataset):
    def __init__(self,lang,key='ALL'):
        super(TTSDataset, self).__init__()
        self.path=path=os.path.join('/project/nakamura-lab08/Work/takatomo-k/dataset/BTEC/speech/',lang,'world')
        self.mean,self.std,self.mgc_size,self.lf0_size,self.bap_size=None,None,None,None,None
        self.data,self.vocab=load_data(os.path.join('../TEXT',key,lang,'clean.txt'))
        self.test,self.dev,self.train=self.cleaner()
        self.get_mean()

    def set_feat_path(self):
        for key in self.data:
            self.data[key].update({"feat":os.path.join(self.path,key+".npy")})

    def cleaner(self,func_list):
        self.set_feat_path()
        self.data=get_uniq_list(self.data)
        self.data=check_file(self.data)
        return get_split_list(self.data)

    def __len__(self):
        return len(self.train)

    def sort(self,key):
        self.train=sorted(self.train,key=lambda x: len(self.data[x][key]),reverse=True)

    def __getitem__(self, idx):
        return self.feature_extraction(self.data[self.train[idx]])

    def feature_extraction(self,data):
        ret=dict()
        for segment in data.keys():
            if segment=='feat':
                feat=self.normalize(np.load(data['feat']))
                ret.update({'feat':torch.from_numpy(feat).type(torch.FloatTensor)})
            else:
                ids=list()
                for w in data[segment]:
                    if w in self.vocab[segment]:
                        ids.append(self.vocab[segment][w])
                    else:
                        ids.append(self.vocab[segment]['UNK'])
                ids.append(self.vocab[segment]['EOS'])
                ret.update({segment:torch.from_numpy(np.array(ids)).type(torch.LongTensor)})
        return ret

    def get_mean(self):
        num=1000
        for idx in tqdm.tqdm(range(num)):
            feat=np.load(self.data[self.train[idx]]["feat"])
            if self.mean is None:
                #feature,self.mgc_size,self.lf0_size,self.bap_size=get_feature(self.path.replace("world","wav",label+".wav"))
                self.mean=feat.mean(axis=0)
                self.std =feat.std(axis=0)
            else:
                self.mean+=feat.mean(axis=0)
                self.std +=feat.std(axis=0)
        self.mean/=num
        self.std/=num

    def normalize(self,feature):
        return normalize(feature,self.mean,self.std)

    def gen_wav(self,feature,path):
        waveform=gen_waveform(feature, self.mean, self.std)
        write_wav(waveform,path)

    def denormalize(self,feature):
        denormalize(feature,self.mean,self.std)

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
    parser.add_argument('-k','--key', type=str, help='Src language')

    args = parser.parse_args()
    dataset=TTSDataset(args.src_lang,args.key)
    import pdb; pdb.set_trace()
    tmp=dataset.__getitem__(1)

    dataset.gen_wav(tmp['feat'].numpy(),"./test.wav")
