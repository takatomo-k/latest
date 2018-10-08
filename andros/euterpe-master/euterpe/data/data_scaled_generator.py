import tables
import sys
import re
import argparse
import os
import shutil
import time
import pickle
from sklearn import preprocessing

def parse() :
    parser = argparse.ArgumentParser(description='params')
    parser.add_argument('--table_path', type=str)
    parser.add_argument('--scaler_type', type=str, default=None, choices=['meanstd'])
    parser.add_argument('--use_scaler', type=str, default=None)
    return parser.parse_args()
    pass

CHUNKSIZE = 50000

if __name__ == '__main__' :
    args = parse()
    assert (args.scaler_type is not None) 
    filename, ext = os.path.splitext(args.table_path)
    scaled_filename = filename+'_%s'%(args.scaler_type)+ext
    # os.system("cp %s %s"%(args.table_path, scaled_filename))
    shutil.copy(args.table_path, scaled_filename)
    print("=== FINISH DUPLICATE FILE ===")
    hdf5_file = tables.open_file(scaled_filename, mode='r+')
    if args.use_scaler is None :
        if args.scaler_type == 'meanstd' :
            scaler = preprocessing.StandardScaler(with_mean=True, with_std=True)
        else :
            raise ValueError()
        for ii in range(0, hdf5_file.root.feat.shape[0], CHUNKSIZE) :
            scaler.partial_fit(hdf5_file.root.feat[ii:ii+CHUNKSIZE])
    else :
        scaler = pickle.load(open(args.use_scaler, 'rb'))

    for ii in range(0, hdf5_file.root.feat.shape[0], CHUNKSIZE) :
        hdf5_file.root.feat[ii:ii+CHUNKSIZE] = scaler.transform(hdf5_file.root.feat[ii:ii+CHUNKSIZE])
        pass

    hdf5_file.close()
    print("=== FINISH : SCALING FILE ===")
    if args.use_scaler is None :
        scaler_filename = filename+'_%s'%(args.scaler_type)+'.scaler'
        pickle.dump(scaler, open(scaler_filename, 'wb'))
        print("=== FINISH : SERIALIZED SCALER ===")
    pass
