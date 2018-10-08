import sys
import pandas as pd
import argparse
import numpy as np
import json
from collections import OrderedDict

def parse() :
    args = argparse.ArgumentParser()
    args.add_argument('--utt2spk')
    args.add_argument('--spk_list', default=None, help='subset data contains only spk_list')
    args.add_argument('--key_list', default=None, help='subset data contains only key_list')
    args.add_argument('--split_ratio', nargs='+', type=int)
    args.add_argument('--split_name', nargs='+')
    args.add_argument('--random', default=123)
    return args.parse_args()

if __name__ == '__main__' :
    args = parse()
    np.random.seed(args.random)
    utt2spk = json.load(open(args.utt2spk))
    # filter utt2spk, remove everything not in key_list #
    print('Original lines: {}'.format(len(utt2spk)))
    if args.key_list is not None :
        _key_set = [x.strip() for x in open(args.key_list).readlines()]
        _key_set = set(utt2spk.keys()).difference(set(_key_set))
        for kk in _key_set :
            del utt2spk[kk]
        print('After filter by key lines: {}'.format(len(utt2spk)))
    if args.spk_list is not None :
        spk_list = [x.strip() for x in open(args.spk_list).readlines()]
    else :
        spk_list = list(set(utt2spk.values()))

    spk2utt = OrderedDict()
    for k,v in utt2spk.items() :
        if v not in spk2utt :
            spk2utt[v] = [k]
        else :
            spk2utt[v].append(k)

    # create new things
    split_ratio = args.split_ratio
    assert sum(split_ratio) == 100
    split_name = args.split_name
    assert len(split_ratio) == len(split_name)
    set_spk = [[] for _ in range(len(split_name))]
    for spk_ii in spk_list :
        utt_ii = list(spk2utt[spk_ii])
        np.random.shuffle(utt_ii)

        for ii in range(len(split_name)) :
            start = int(sum(split_ratio[0:ii]) / 100 * len(utt_ii))
            end = int(sum(split_ratio[0:ii+1]) / 100 * len(utt_ii))
            set_spk[ii].extend(utt_ii[start:end])
            pass
        pass 
    for set_name, set_ii in zip(split_name, set_spk) :
        with open(set_name, 'w') as f :
            f.write('\n'.join(set_ii))
    pass
