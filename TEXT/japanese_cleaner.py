import os,sys,argparse
from glob import glob

parser = argparse.ArgumentParser()
parser.add_argument('-i','--input', type=str, help='Src language')
parser.add_argument('-o','--out', type=str, help='Src language')
parser.add_argument('-k','--key', type=str, help='Src language',default=None)

args = parser.parse_args()
if args.key is None:
    txt_list=open(args.input).readlines()
else:
    txt_list=list()
    for line in open(args.input).readlines():
        if args.key in line:
            txt_list.append(line)

PAD,SOS,EOS,UNK=0,1,2,3
new_list=list()
flg=True

def update(keys,vocab,count):
    for key in keys:
        if key in vocab:
            count[key]+=1
        else:
            vocab.update({key:len(vocab)})
            count.update({key:1})


def check_count(keys,count,num=3):
    for key in keys:
        if count[key]<=num:
            return True
    return False

while flg:
    word_vocab={'PAD':PAD,'SOS':SOS,'EOS':EOS,'UNK':UNK}
    word_count={'PAD':1,'SOS':1,'EOS':1,'UNK':1}
    char_vocab={'PAD':PAD,'SOS':SOS,'EOS':EOS,'UNK':UNK}
    char_count={'PAD':1,'SOS':1,'EOS':1,'UNK':1}
    p_word_vocab={'PAD':PAD,'SOS':SOS,'EOS':EOS,'UNK':UNK}
    p_word_count={'PAD':1,'SOS':1,'EOS':1,'UNK':1}
    p_char_vocab={'PAD':PAD,'SOS':SOS,'EOS':EOS,'UNK':UNK}
    p_char_count={'PAD':1,'SOS':1,'EOS':1,'UNK':1}
    pos_vocab={'PAD':PAD,'SOS':SOS,'EOS':EOS,'UNK':UNK}
    pos_count={'PAD':1,'SOS':1,'EOS':1,'UNK':1}

    for txt in txt_list:
        txt=txt.strip()
        lable,word,p_word,pos=txt.split("|")
        char=list(word.replace(" ",""))
        p_char=list(p_word.replace(" ",""))
        word=word.split()
        p_word=p_word.split()
        pos=pos.split()
        #txt=list(txt.replace(" ",""))
        update(word,word_vocab,word_count)
        update(char,char_vocab,char_count)
        update(p_word,p_word_vocab,p_word_count)
        update(p_char,p_char_vocab,p_char_count)
        update(pos,pos_vocab,pos_count)


    new_list=list()
    for txt in txt_list:
        txt=txt.strip()
        lable,word,p_word,pos=txt.split("|")
        char=list(word.replace(" ",""))
        p_char=list(p_word.replace(" ",""))
        word=word.split()
        p_word=p_word.split()
        pos=pos.split()

        if check_count(word,word_count) or check_count(char,char_count,5):
            pass
        elif check_count(p_word,p_word_count) or check_count(p_char,p_char_count,100):
            pass
        elif "零" in txt:
            pass
        else:
            new_list.append(txt)

    if len(new_list)==len(txt_list):
        flg=False
    else:
        print(str(len(txt_list))+" -> "+str(len(new_list)))
        txt_list=new_list
        new_list=list()

with open(args.out+"clean.txt","w") as f:
    for line in txt_list:
        f.write(line+"\n")

with open(args.out+"word","w") as f:
    for key in sorted(word_vocab.keys(),key=lambda x: word_count[x]):
        f.write(key+" "+str(word_vocab[key])+" "+str(word_count[key])+"\n")
with open(args.out+"p_word","w") as f:
    for key in sorted(p_word_vocab.keys(),key=lambda x: p_word_count[x]):
        f.write(key+" "+str(p_word_vocab[key])+" "+str(p_word_count[key])+"\n")

with open(args.out+"char","w") as f:
    for key in sorted(char_vocab.keys(),key=lambda x: char_count[x]):
        f.write(key+" "+str(char_vocab[key])+" "+str(char_count[key])+"\n")
with open(args.out+"p_char","w") as f:
    for key in sorted(p_char_vocab.keys(),key=lambda x: p_char_count[x]):
        f.write(key+" "+str(p_char_vocab[key])+" "+str(p_char_count[key])+"\n")

"""
import pdb; pdb.set_trace()
with open(args.out,"w") as f:
    for key in sorted(vocab.keys(),key=lambda x: count[x]):
        f.write(key+" "+str(vocab[key])+" "+str(count[key])+"\n")
"""
