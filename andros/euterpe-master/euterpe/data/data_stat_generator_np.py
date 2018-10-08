import tables
import sys
import re
import argparse
import os
import shutil
import time
import pickle
import numpy as np
from tqdm import tqdm
from utilbox import data_util
from sklearn import preprocessing
from utilbox.regex_util import regex_key_val

def parse() :
    parser = argparse.ArgumentParser(description='params')
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--sample', type=int, default=-1, help='sample statistics')
    return parser.parse_args()
    pass

if __name__ == '__main__' :
    args = parse()
    folder = os.path.dirname(args.data_path)
    scaled_filename = os.path.join(folder, 'feat_stat.pkl')
    kv_list = regex_key_val.findall(open(args.data_path).read())
    k_list, v_list = zip(*kv_list)
    if args.sample == -1 : # use all #
        sample_idx = list(range(0, len(kv_list)))
    else :
        sample_idx = np.random.choice(len(kv_list), size=args.sample, replace=False)
    # calculate max, min, mean, std #
    minmax = preprocessing.MinMaxScaler()
    meanstd = preprocessing.StandardScaler()
    for ii in tqdm(sample_idx, ncols=60) :
        feat_ii = np.load(v_list[ii])['feat']
        minmax.partial_fit(feat_ii)
        meanstd.partial_fit(feat_ii)
    stat_array = {'min':minmax.data_min_, 'max':minmax.data_max_, 
            'mean':meanstd.mean_, 'std':meanstd.scale_}
    pickle.dump(stat_array, open(scaled_filename, 'wb'))
    print('===FINISH===')
    pass
