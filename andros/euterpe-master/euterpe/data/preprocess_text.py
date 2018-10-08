import json
import re
import sys

"""
preprocessing wsj 

ref : https://github.com/rizar/attention-lvcsr/blob/master/exp/wsj/write_hdf_dataset.sh

"""

re_all = re.compile("[^a-zA-Z\'\-\.\~ ]") # remove unused char
re_space = re.compile(' +') # remove double space 

def preprocess_text_to_char(raw_text) :
    text = raw_text.strip().lower()
    text = text.replace('<noise>', '~')
    text = text.replace('`', "'")
    text = re_all.sub('', text)
    text = re_space.sub('*', text)
    text = list(text)
    for ii, ch in enumerate(text) :
        if ch == '~' :
            text[ii] = '<noise>'
        elif ch == '*' :
            text[ii] = '<spc>'
        pass
    return ' '.join(text)
    pass

if __name__ == '__main__' : 
    """
    usage : %run data/preprocess_text.py /home/is/andros-tj/rsch/asr/kaldi/egs/wsj/s5/data/test_eval92/text
    """

    re_key_text = re.compile('(^[^\s]+) (.+)$')
    path = sys.argv[1]
    res = []
    with open(path) as f :
        all_texts = [x.strip() for x in f.readlines()]
        for ii in range(len(all_texts)) :
            _key, _text = re_key_text.findall(all_texts[ii])[0]
            _text = preprocess_text_to_char(_text)
            # res.append('%s %s\n'%(_key, _text))
            print('%s %s'%(_key, _text))
        pass
    # with open(path+'_char', 'w') as f :
        # f.writelines(res)
    pass
