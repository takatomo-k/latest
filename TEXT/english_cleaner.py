import os,sys,argparse,re
from glob import glob

parser = argparse.ArgumentParser()
parser.add_argument('-i','--input', type=str, help='Src language')
parser.add_argument('-o','--out', type=str, help='Src language')
parser.add_argument('-k','--key', type=str, help='Src language',default=None)
args = parser.parse_args()
abbreviations = [(re.compile('\\b%s\\.' % x[0], re.IGNORECASE), x[1]) for x in [
  ('mrs', 'misess'),
  ('ms', 'miss'),
  ('mr', 'mister'),
  ('dr', 'doctor'),
  ('st', 'saint'),
  ('co', 'company'),
  ('jr', 'junior'),
  ('maj', 'major'),
  ('gen', 'general'),
  ('drs', 'doctors'),
  ('rev', 'reverend'),
  ('lt', 'lieutenant'),
  ('hon', 'honorable'),
  ('sgt', 'sergeant'),
  ('capt', 'captain'),
  ('esq', 'esquire'),
  ('ltd', 'limited'),
  ('col', 'colonel'),
  ('ft', 'fort'),
]]

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

def expand_abbreviations(text):
  for regex, replacement in abbreviations:
    text = re.sub(regex, replacement, text)

  text=text.replace("'"," '").replace("?"," ?").replace("!"," !")
  text=text.replace(",","").replace(".","").replace("'","").replace(":","").replace(";","").replace("-"," ").replace("\"","")
  while "  " in text:
      text=text.replace("  "," ")
  return text

tmp=list()
for line in txt_list:
    lable,word=line.strip().split("|")
    word=expand_abbreviations(word.lower())
    line=lable+"|"+word
    tmp.append(line)
txt_list=tmp
#import pdb; pdb.set_trace()

while flg:
    word_vocab={'PAD':PAD,'SOS':SOS,'EOS':EOS,'UNK':UNK}
    word_count={'PAD':1,'SOS':1,'EOS':1,'UNK':1}
    char_vocab={'PAD':PAD,'SOS':SOS,'EOS':EOS,'UNK':UNK}
    char_count={'PAD':1,'SOS':1,'EOS':1,'UNK':1}

    for txt in txt_list:
        txt=txt.strip()
        lable,word=txt.split("|")
        char=list(word.replace(" ",""))
        word=word.split()
        update(word,word_vocab,word_count)
        update(char,char_vocab,char_count)



    new_list=list()
    for txt in txt_list:
        txt=txt.strip()
        lable,word=txt.split("|")
        char=list(word.replace(" ",""))
        word=word.split()

        if check_count(word,word_count) or check_count(char,char_count,5):
            pass
        else:
            new_list.append(txt)

    if len(new_list)==len(txt_list):
        flg=False
    else:
        print(str(len(new_list))+" vs "+str(len(txt_list)))
        txt_list=new_list
        new_list=list()

with open(args.out+"clean.txt","w") as f:
    for line in txt_list:
        f.write(line+"\n")

with open(args.out+"word","w") as f:
    for key in sorted(word_vocab.keys(),key=lambda x: word_count[x]):
        f.write(key+" "+str(word_vocab[key])+" "+str(word_count[key])+"\n")

with open(args.out+"char","w") as f:
    for key in sorted(char_vocab.keys(),key=lambda x: char_count[x]):
        f.write(key+" "+str(char_vocab[key])+" "+str(char_count[key])+"\n")

"""
import pdb; pdb.set_trace()
with open(args.out,"w") as f:
    for key in sorted(vocab.keys(),key=lambda x: count[x]):
        f.write(key+" "+str(vocab[key])+" "+str(count[key])+"\n")
"""
