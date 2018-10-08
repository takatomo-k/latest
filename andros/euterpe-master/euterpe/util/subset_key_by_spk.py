import os
import argparse
from utilbox.regex_util import regex_key_val

def parse() :
    parser = argparse.ArgumentParser()
    parser.add_argument('--utt2spk', type=str)
    parser.add_argument('--spk', type=str)
    return parser.parse_args()
    pass

if __name__ == '__main__':
    args = parse()    
    list_kv = regex_key_val.findall(open(os.path.abspath(args.utt2spk)).read())
    list_spk = set(open(args.spk).read().split('\n'))
    list_subset = [x[0] for x in list_kv if x[1] in list_spk]
    list_subset = sorted(list_subset)
    for k in list_subset :
        print(k)
    pass

