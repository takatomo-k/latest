import tables
import sys
import re
import argparse
import os
import shutil
import time
import pickle
import numpy as np
from sklearn import preprocessing
from functools import partial
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import cpu_count
from multiprocessing.pool import Pool
from tqdm import tqdm
from utilbox.regex_util import regex_key_val
from utilbox.regex_util import regex_key

def parse() :
    parser = argparse.ArgumentParser(description='params')
    parser.add_argument('--data_path', type=str, required=True, help='file with information <key> <feature.npz>')
    parser.add_argument('--set', type=str, default=None, help='subset the data based on providet key set')
    parser.add_argument('--output', type=str, help='folder path for output')
    parser.add_argument('--scaler_type', type=str, default=None, choices=['meanstd'])
    parser.add_argument('--use_scaler', type=str, default=None)
    return parser.parse_args()
    pass

def fn_scaling_feat(in_path, out_path, scaler) :
    obj = np.load(in_path)
    obj = dict(obj)
    obj['feat'] = scaler.transform(obj['feat'])
    # TODO : flexible np.floatx
    obj['feat'] = obj['feat'].astype('float32')

    out_path = os.path.join(out_path, str(obj['key'])+'.npz')
    np.savez(out_path, **obj)
    return str(obj['key']), out_path
    pass


CHUNKSIZE = 500

if __name__ == '__main__' :
    args = parse()
    assert (args.scaler_type is not None) 

    if os.path.exists(args.output) :
        assert os.path.isdir(args.output), "output must be a folder"
    else :
        os.makedirs(args.output, mode=0o755, exist_ok=False)
        os.makedirs(os.path.join(args.output, 'meta'), mode=0o755, exist_ok=False)

    kv_list = open(args.data_path).read()
    kv_list = regex_key_val.findall(kv_list)

    # filter if key set exist #
    if args.set is not None :
        subset = regex_key.findall(open(args.set).read())
        subset = set(subset)
        total_len = len(kv_list)
        kv_list = [x for x in kv_list if x[0] in subset]
        print('[info] select {}/{} from dataset'.format(len(kv_list), total_len))
        
    k_list, v_list = zip(*kv_list) 

    executor = ProcessPoolExecutor(max_workers=cpu_count())
    if args.use_scaler is None :
        if args.scaler_type == 'meanstd' :
            scaler = preprocessing.StandardScaler(with_mean=True, with_std=True)
        else :
            raise ValueError()
        print('=== START : FITTING SCALER ===')
        for ii in tqdm(list(range(0, len(kv_list), CHUNKSIZE)), ncols=60) :
            # TODO : parallelize
            list_exec = list(map(np.load, v_list[ii:ii+CHUNKSIZE]))
            tmp_np = np.concatenate([item['feat'] for item in list_exec])
            scaler.partial_fit(tmp_np)
    else :
        print('=== START : LOAD EXISTING SCALER ===')
        scaler = pickle.load(open(args.use_scaler, 'rb'))

    # create new scaled file #
    print("=== START : SCALING & COPY FILE ===")
    file_kv = open(os.path.join(args.output, 'meta', 'feat.scp'), 'w')
    list_exec = []
    for ii in tqdm(list(range(0, len(kv_list))), ncols=60) :
        list_exec.append(executor.submit(partial(fn_scaling_feat, v_list[ii], args.output, scaler)))
        pass
    list_exec = [item.result() for item in list_exec]
    for item in list_exec :
        file_kv.write('{} {}\n'.format(*item))
    file_kv.flush()
    print("=== FINISH : SCALING FILE ===")
    if args.use_scaler is None :
        scaler_filename = 'feat_%s'%(args.scaler_type)+'.scaler'
        scaler_filename = os.path.join(args.output, 'meta', scaler_filename)
        pickle.dump(scaler, open(scaler_filename, 'wb'))
        print("=== FINISH : SERIALIZED SCALER ===")
