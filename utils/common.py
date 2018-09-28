import pickle,random,tqdm,os,sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) +'/../')
random.seed(a=1, version=1)
PAD,SOS,EOS,UNK=0,1,2,3
key_list=['label','word','p_word','pos']

def get_split_list(data):
    _list=[i for i in data.keys()]
    random.shuffle(_list)
    return _list[0:500],_list[500:1500],_list[1500:-1]

def get_pairwise_list(src,trg):
    _src,_trg=dict(),dict()
    for key in src.keys():
        if key in trg:
            _src.update({key:src[key]})
            _trg.update({key:trg[key]})
    return _src,_trg

def get_uniq_list(data):
    _data=dict()
    uniq=dict()
    for key in data.keys():
        text=" ".join(data[key]["word"])
        if text not in uniq:
            uniq.update({text:0})
            _data.update({key:data[key]})
    return _data

def get_data_list(data):
    list=data.keys()
    random.shuffle(list)
    return train[0:len(list)-1500],dev[len(list)-1500:len(list)-500],test[len(list)-500:len(list)]

def make_vocab(data):
    vocab=dict()
    for label in data.keys():
        for key in data[label]:
            if key not in vocab:
                vocab.update({key:{'PAD':PAD,'SOS':SOS,'EOS':EOS,'UNK':UNK}})
            for w in data[label][key]:
                if w not in vocab[key]:
                    vocab[key].update({w:len(vocab[key])})
    return vocab

def load_data(path):
    data =dict()
    for line in open(path).readlines():
        for idx,value in enumerate(line.strip().split('|')):
            if key_list[idx]=="label":
                label=value
                data.update({label:dict()})
            elif key_list[idx]=="pos":
                pass
            else:
                #word
                data[label].update({key_list[idx]:value.split()})
                #char
                data[label].update({key_list[idx].replace("word","char"):list(value.replace(" ",""))})
    return data, make_vocab(data)

def check_file(data):
    _data=dict()
    for label in tqdm.tqdm(data.keys()):
        if not os.path.exists(data[label]["feat"]):
            pass
        elif os.path.getsize(data[label]["feat"])==0:
            pass
        else:
            _data.update({label:data[label]})
    return _data
