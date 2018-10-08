from utilbox import kaldi_util

### SPEAKER INFORMATION DATA ###
def gen_utt2spk_dict(filepath) :
    """
    filepath : path to utt2spk
    """
    with open(filepath) as ff :
        utt2spk = map(lambda x : x.strip(), ff.readlines())
        utt2spk = dict(map(lambda x : x.split(), utt2spk))
    return utt2spk
    pass

def gen_spk2utt_dict(filepath) :
    """
    filepath : path to spk2utt
    """
    def _gen_kv(x) :
        return (x[0], x[1:])
    with open(filepath) as ff :
        spk2utt = map(lambda x : x.strip(), ff.readlines())
        spk2utt = dict(map(lambda x : _gen_kv(x.split()), spk2utt))
    return spk2utt
    pass

def gen_ivector_dict(scppath) :
    """
    """
    uttids, ivectors = kaldi_util.copy_feats_scp(scppath)
    for ii in range(len(ivectors)) :
        ivectors[ii] = ivectors[ii].flatten()
    return dict(zip(uttids, ivectors))
