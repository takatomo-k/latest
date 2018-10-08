import os
import argparse
# from pyxdameraulevenshtein import damerau_levenshtein_distance, normalized_damerau_levenshtein_distance
import editdistance
from utilbox.regex_util import regex_key_val


def parse() :
    parser = argparse.ArgumentParser(description='python wrapper for Kaldi\'s computer-wer')
    parser.add_argument('--ref', required=True, help='reference transcription  (with key)')
    parser.add_argument('--hyp', required=True, help='hypothesis transcription (with key)')
    return parser.parse_args()
    pass

if __name__ == '__main__' :
    args = parse()
    list_kv_ref = regex_key_val.findall(open(args.ref).read())
    list_kv_hyp = regex_key_val.findall(open(args.hyp).read())

    list_key_ref = [x[0] for x in list_kv_ref]
    list_key_hyp = [x[0] for x in list_kv_hyp]
    assert list_key_ref == list_key_hyp, "the keys between refs & hyps are not same"
    list_val_ref = [x[1].split() for x in list_kv_ref]
    list_val_hyp = [x[1].split() for x in list_kv_hyp]
    total_edist = 0
    total_len = 0
    for v_ref, v_hyp in zip(list_val_ref, list_val_hyp) :
        total_edist += editdistance.eval(v_ref, v_hyp)
        total_len += len(v_ref)
    print('%WER: {:.3f} [ {} / {} ]'.format(total_edist / total_len * 100, total_edist, total_len))
