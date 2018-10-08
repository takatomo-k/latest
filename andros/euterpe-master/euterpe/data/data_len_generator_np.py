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
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Pool

from utilbox.regex_util import regex_key_val

def parse() :
    parser = argparse.ArgumentParser(description='params')
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--ncpu', type=int, default=16)
    return parser.parse_args()
    pass

if __name__ == '__main__' :
    args = parse()
    
    def get_feat_length(path) :
        return np.load(path)['feat'].shape[0]

    folder = os.path.dirname(args.data_path)
    kv_list = regex_key_val.findall(open(args.data_path).read())
    k_list, v_list = zip(*kv_list)

    with Pool(args.ncpu) as executor :
        # calculate max, min, mean, std #
        output_result = executor.map(get_feat_length, v_list)
    
    output_file = open(os.path.join(folder, '{}_len{}'.format(*os.path.splitext(args.data_path))), 'w')
    for k, v in zip(k_list, output_result) :
        output_file.write('{} {}\n'.format(k, v))
    output_file.close()
    pass
