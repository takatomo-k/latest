import numpy as np

import argparse
from fastdtw import fastdtw
from concurrent.futures import ProcessPoolExecutor
from utilbox.regex_util import regex_key_val
from scipy.spatial.distance import euclidean
from tqdm import tqdm

def parse() :
    parser = argparse.ArgumentParser(description='python wrapper for Kaldi\'s computer-wer')
    parser.add_argument('--ref', required=True, help='reference transcription  (with key)')
    parser.add_argument('--hyp', required=True, help='hypothesis transcription (with key)')
    return parser.parse_args()
    pass

if __name__ == '__main__':
    args = parse()
    list_kv_ref = regex_key_val.findall(open(args.ref).read())
    list_kv_hyp = regex_key_val.findall(open(args.hyp).read())

    list_key_ref = [x[0] for x in list_kv_ref]
    list_key_hyp = [x[0] for x in list_kv_hyp]
    assert list_key_ref == list_key_hyp, "the keys between refs & hyps are not same"
    dict_kv_ref = dict(list_kv_ref)
    dict_kv_hyp = dict(list_kv_hyp)
    total_dist = 0
    total_count = 0
    total_len = 0
    for kk in tqdm(list_key_hyp, ncols=50) :
        v_ref = dict_kv_ref[kk]
        v_hyp = dict_kv_hyp[kk]
        _feat_ref = np.load(v_ref)['feat']
        _feat_hyp = np.load(v_hyp)['feat']
        _dist = fastdtw(_feat_ref, _feat_hyp, dist=euclidean)[0]
        total_dist += _dist
        total_count += 1
        total_len += len(_feat_ref)
    print('DIST: {:.3f} [ LEN {} | COUNT {} ]'.format(total_dist, total_len, total_count))
