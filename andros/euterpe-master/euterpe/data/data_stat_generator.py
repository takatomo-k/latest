import tables
import sys
import re
import argparse
import os
import shutil
import time
import pickle
from utilbox import data_util
from sklearn import preprocessing

def parse() :
    parser = argparse.ArgumentParser(description='params')
    parser.add_argument('--table_path', type=str)
    return parser.parse_args()
    pass

if __name__ == '__main__' :
    args = parse()
    filename, ext = os.path.splitext(args.table_path)
    scaled_filename = filename+'_stat.pkl'
    storage = tables.open_file(args.table_path)
    stat_array = data_util.stat_array(storage.root.feat[:])
    storage.close()
    pickle.dump(stat_array, open(scaled_filename, 'wb'))
    print('===FINISH===')
    pass
