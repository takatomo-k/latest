import os
import sys
import subprocess
import numpy as np
import re

KALDI_PATH = '/home/is/andros-tj/rsch/asr/kaldi'
KALDI_FEATBIN = 'src/featbin'
DEFAULT_CONF = 'egs/timit/s5/conf'
DEFAULT_FBANK_CONF = 'fbank.conf'
DEFAULT_MFCC_CONF = 'mfcc.conf'

os.environ['PATH'] += ':{}'.format(os.path.join(KALDI_PATH, KALDI_FEATBIN))

def feat_string_to_obj(raw_str) :
    re_utt = re.compile('^([A-Za-z0-9\_]+)\s', flags=re.MULTILINE)
    re_feat = re.compile('\[([^\]\[]*)\]', flags=re.MULTILINE)
    re_text = re.compile('^[A-Za-z0-9\_]+\s([A-Za-z0-9\_ ]+)$')
    utt_list = re_utt.findall(raw_str)
    def str2mat(ss) :
        ss = ss.strip().split('\n')
        ss = [si.split() for si in ss]
        return np.array(ss, dtype='float32')
    feat_list = list(map(str2mat, re_feat.findall(raw_str)))
    assert len(feat_list) == len(utt_list)
    return utt_list, feat_list
    pass

def compute_fbank_feats(wavscp, config=os.path.join(KALDI_PATH, DEFAULT_CONF, DEFAULT_FBANK_CONF)) :
    compute_feat_cmd = "compute-fbank-feats --config={} scp:{} ark:- 2> /dev/null | copy-feats --compress=true ark:- ark,t:- 2> /dev/null"
    featscp = subprocess.check_output(compute_feat_cmd.format(config, wavscp), shell=True)
    return feat_string_to_obj(featscp)
    pass

def compute_mfcc_feats(wavscp, config=os.path.join(KALDI_PATH, DEFAULT_CONF, DEFAULT_MFCC_CONF)) :
    compute_feat_cmd = "compute-mfcc-feats --config={} scp:{} ark:- 2> /dev/null | copy-feats --compress=true ark:- ark,t:- 2> /dev/null"
    featscp = subprocess.check_output(compute_feat_cmd.format(config, wavscp), shell=True)
    return feat_string_to_obj(featscp)
    pass

def copy_feats_scp(scp) :
    copy_feats_cmd = "copy-feats scp:{} ark,t:- 2> /dev/null"
    featscp = (subprocess.getoutput(copy_feats_cmd.format(scp)))
    return feat_string_to_obj(featscp)
