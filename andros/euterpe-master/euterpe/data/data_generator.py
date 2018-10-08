import warnings
import os
import numpy as np
import tables
# import multiprocessing
import pathos.multiprocessing as multiprocessing
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import tempfile
from scipy.io import wavfile
import python_speech_features as pyspfeat
import librosa
import librosa.filters
from scipy import signal
from euterpe.common.helper import TacotronHelper

from tqdm import tqdm
from scipy import stats
from utilbox import world_vocoder_util, kaldi_util, signal_util

############
### UTIL ###
############

def linear2mu(signal, mu=255) :
    """
    -1 < signal < 1 
    """
    assert signal.max() <= 1 and signal.min() >= -1
    y = np.sign(signal) * np.log(1.0 + mu * np.abs(signal))/np.log(mu+1)
    return y

def mu2linear(mu_signal, mu=255) :
    assert mu_signal.max() <=1 and mu_signal >= -1
    y = np.sign(mu_signal)*(1.0/mu)*((1+mu)**(np.abs(mu_signal))-1) 
    return y

def feat_sil_from_stat(feat_stat, q=0.025) :
    return stats.norm.ppf(q, loc=feat_stat['mean'], scale=feat_stat['std'])

def feat_sil_start_end(feat_mat, threshold=0.04, window_size=20, step_size=1) :
    # calculate mean of stddev
    mat_len = feat_mat.shape[0]
    std = []
    for ii in range(window_size, mat_len-window_size-1, step_size) :
        std.append((feat_mat[ii-window_size:ii+window_size+1, :].std().mean(), ii))
    # get first and last #
    start_idx = 0
    end_idx = mat_len
    for std_ii, pos_ii in std :
        if std_ii > threshold :
            start_idx = pos_ii
            break
    for std_ii, pos_ii in std[::-1] :
        if std_ii > threshold :
            end_idx = pos_ii
            break
    return start_idx, end_idx

def list_feat_sil_strip(list_feat_mat, threshold=0.04, window_size=20) :
    startend = [feat_sil_start_end(x, threshold=threshold, window_size=window_size) for x in list_feat_mat]
    return [list_feat_mat[ii][x:y] for ii, (x, y) in enumerate(startend)]

def group_feat_timestep(arr, group=4) :
    assert arr.ndim == 2 
    len_arr, ndim = arr.shape
    len_arr_group = (len_arr+group-1)//group
    arr_group = np.zeros((len_arr_group, ndim*group), dtype='float32')
    for ii in range(len_arr_group) :
        curr_feat = arr[ii*group:(ii+1)*group]
        if curr_feat.shape[0] < group :
            remaining_curr_feat = [curr_feat[-1]] * (group - curr_feat.shape[0])
            remaining_curr_feat = np.stack(remaining_curr_feat, axis=0)
            curr_feat = np.concatenate([curr_feat, remaining_curr_feat], axis=0)
            pass
        arr_group[ii] = curr_feat.reshape(1, ndim*group)
    return arr_group

### FUNC FOR NPZ -- CREATE & SAVE ###
def save_arr(key, feat, path, compress=False) :
    if compress :
        np.savez_compressed(path, feat=feat, key=key)
    else :
        np.savez(path, feat=feat, key=key)

def gen_and_save_arr(key, wav_file, path, cfg, compress=False) :
    feat = generate_feat_opts(wav_file, cfg, None, None)
    # TODO : flexible np.floatx
    feat = feat.astype(np.float32)
    save_arr(key, feat, path, compress=compress)
    return key, path


##################
### ALL IN ONE ###
##################

def generate_feat_opts(path=None,  
        cfg={'pkg':'pysp', 'type':'logfbank', 'nfilt':40, 'delta':2},
        signal=None, rate=16000) :
    cfg = dict(cfg)
    if cfg['pkg'] == 'pysp' : # python_speech_features #
        if signal is None :
            rate, signal = wavfile.read(path)

        if cfg['type'] == 'logfbank' :
            feat_mat = pyspfeat.base.logfbank(signal, rate, nfilt=cfg.get('nfilt', 40))
        elif cfg['type'] == 'mfcc' :
            feat_mat = pyspfeat.base.mfcc(signal, rate,
                    numcep=cfg.get('nfilt', 26)//2, nfilt=cfg.get('nfilt', 26))
        elif cfg['type'] == 'wav' :
            feat_mat = pyspfeat.base.sigproc.framesig(signal, 
                    frame_len=cfg.get('frame_len', 400), 
                    frame_step=cfg.get('frame_step', 160))
        else :
            raise NotImplementedError("feature type {} is not implemented/available".format(cfg['type']))
            pass
        # delta #
        comb_feat_mat = [feat_mat]
        delta = cfg['delta']
        if delta > 0 :
            delta_feat_mat = pyspfeat.base.delta(feat_mat, 2)
            comb_feat_mat.append(delta_feat_mat)
        if delta > 1 :
            delta2_feat_mat = pyspfeat.base.delta(delta_feat_mat, 2)
            comb_feat_mat.append(delta2_feat_mat)
        if delta > 2 :
            raise NotImplementedError("max delta is 2, larger than 2 is not normal setting")
        return np.hstack(comb_feat_mat)
    elif cfg['pkg'] == 'rosa' :
        if signal is None :
            signal, rate = librosa.core.load(path, sr=cfg['sample_rate'])

        assert rate == cfg['sample_rate'], "sample rate is different with current data"

        if cfg.get('preemphasis', None) is not None :
            # signal = np.append(signal[0], signal[1:] - cfg['preemphasis']*signal[:-1])
            signal = signal_util.preemphasis(x, self.cfg['preemphasis'])

        if cfg.get('pre', None) == 'meanstd' :
            signal = (signal - signal.mean()) / signal.std()
        elif cfg.get('pre', None) == 'norm' :
            signal = (signal - signal.min()) / (signal.max() - signal.min()) * 2 - 1 

        # raw feature
        if cfg['type'] == 'wav' :
            if cfg.get('post', None) == 'mu' :
                signal = linear2mu(signal)
            
            feat_mat = pyspfeat.base.sigproc.framesig(signal, 
                    frame_len=cfg.get('frame_len', 400), 
                    frame_step=cfg.get('frame_step', 160))
            return feat_mat
        # spectrogram-based feature 
        raw_spec = signal_util.rosa_spectrogram(signal, n_fft=cfg['nfft'], hop_length=cfg.get('winstep', None), win_length=cfg.get('winlen', None))[0]
        if cfg['type'] in ['logmelfbank', 'melfbank'] :
            mel_spec = signal_util.rosa_spec2mel(raw_spec, nfilt=cfg['nfilt'])
            if cfg['type'] == 'logmelfbank' :
                return np.log(mel_spec)
            else :
                return mel_spec
        elif cfg['type'] == 'lograwfbank' :
            return np.log(raw_spec) 
        elif cfg['type'] == 'rawfbank' :
            return raw_spec
        else :
            raise NotImplementedError()
    elif cfg['pkg'] == 'taco' :
        # SPECIAL FOR TACOTRON #
        tacohelper = TacotronHelper(cfg)
        if signal is None :
            signal = tacohelper.load_wav(path)

        assert len(signal) != 0, ('file {} is empty'.format(path))

        try :
            if cfg['type'] == 'raw' :
                feat = tacohelper.spectrogram(signal).T
            elif cfg['type'] == 'mel' :
                feat = tacohelper.melspectrogram(signal).T
            else :
                raise NotImplementedError()
        except :
            import ipdb; ipdb.set_trace()
            pass
        return feat
    elif cfg['pkg'] == 'world' :
        if path is None :
            with tempfile.NamedTemporaryFile() as tmpfile :
                wavfile.write(tmpfile.name, rate, signal)
                logf0, bap, mgc = world_vocoder_util.world_analysis(tmpfile.name, cfg['mcep'])
        else :
            logf0, bap, mgc = world_vocoder_util.world_analysis(path, cfg['mcep'])
        
        vuv, f0, bap, mgc = world_vocoder_util.world2feat(logf0, bap, mgc)

        # ignore delta, avoid curse of dimensionality #
        return vuv, f0, bap, mgc 
    else :
        raise NotImplementedError()
        pass

#################################
### PYTHON_SPEECH_FEAT + ROSA ###
#################################

### HDF5 FILE
def generate_feat_standard_table_multi(key_list, wav_list, output_path, 
        cfg={'pkg':'standard', 'type':'logfbank', 'nfilt':40, 'delta':2}, ncpu=16, block=256) :
    assert len(key_list) == len(wav_list)
    assert cfg['pkg'] in ['pysp', 'rosa', 'taco']
    
    NUM_UTTS = len(key_list)
    
    # pytables @
    hdf5_path = output_path
    hdf5_file = tables.open_file(hdf5_path, mode='w')
    filters = tables.Filters(complevel=5, complib='blosc')
    
    # sample only #
    feat_ii = generate_feat_opts(wav_list[0], cfg=cfg)
    warnings.warn('feat saved on float16 datatype')
    feat_storage = hdf5_file.create_earray(hdf5_file.root, 'feat',
            tables.Atom.from_dtype(np.dtype(np.float16)),
            filters=filters,
            shape=(0, feat_ii.shape[-1])
            )
    key_storage = hdf5_file.create_earray(hdf5_file.root, 'key', 
            tables.StringAtom(itemsize=128),
            filters=filters,
            shape=(0, )
            )
    pos_storage = hdf5_file.create_earray(hdf5_file.root, 'pos',
            tables.Atom.from_dtype(np.dtype(np.int)),
            shape=(0, 2),
            filters=filters)
    start_idx = end_idx = 0
    pool = multiprocessing.Pool(processes=ncpu)
    for ii in tqdm(list(range(0, len(key_list), block))) :
        feat_ii_block = pool.starmap(generate_feat_opts, [(wav_list[idx], cfg, None, None) for idx in range(ii, min(ii+block, len(key_list)))] )
        for idx, feat_ii in enumerate(feat_ii_block) :
            feat_storage.append(feat_ii)
            # add idx start, end
            end_idx = start_idx + feat_ii.shape[0]
            pos_storage.append(np.array([[start_idx, end_idx]]))
            start_idx = end_idx
            key_storage.append(np.array([key_list[ii + idx]], dtype='S'))
        pass

    hdf5_file.close()
    pass

### NUMPY FILE
def generate_feat_standard_npfile(key_list, wav_list, output_path,
        cfg={'pkg':'standard', 'type':'logfbank', 'nfilt':40, 'delta':2}, ncpu=16, block=256, compress=False) :
    assert len(key_list) == len(wav_list)
    assert len(set(key_list)) == len(set(wav_list)), "number of key & wav unique set is not same"
    assert cfg['pkg'] in ['pysp', 'rosa', 'taco']
    if os.path.exists(output_path) :
        assert os.path.isdir(output_path), "output_path must be a folder"
        assert os.listdir(output_path) == [], "folder must be empty"
    else :
        os.makedirs(output_path, mode=0o755, exist_ok=False)
        os.makedirs(os.path.join(output_path, 'meta'), mode=0o755, exist_ok=False)

    NUM_UTTS = len(key_list)
    # pool = multiprocessing.Pool(processes=ncpu)
    executor = ProcessPoolExecutor(max_workers=ncpu)
    list_execs = []
    file_kv = open(os.path.join(output_path, 'meta', 'feat.scp'), 'w')

    for ii in range(len(key_list)) :
        path_ii = os.path.join(output_path, key_list[ii]+'.npz')
        list_execs.append(executor.submit(partial(gen_and_save_arr, key_list[ii], wav_list[ii], path_ii, cfg=cfg, compress=compress))) 
    list_result = []
    for item in tqdm(list_execs) :
        list_result.append(item.result())
    for item in list_result :
        file_kv.write('{} {}\n'.format(*item))
    file_kv.flush()
    file_kv.close()
    pass

#####################
### WORLD FEATURE ###
#####################

class FeatInfo(tables.IsDescription) :
    feat = tables.StringCol(30, pos=0)
    dim = tables.Int32Col(pos=1)
    pass

def generate_feat_world_table_multi(key_list, wav_list, output_path,
        cfg={'pkg':'world', 'mcep':59}, ncpu=16, block=256
        ) :
    assert len(key_list) == len(wav_list)
    assert cfg['pkg'] == 'world'
    NUM_UTTS = len(key_list)

    # pytables @
    hdf5_path = output_path
    hdf5_file = tables.open_file(hdf5_path, mode='w')
    filters = tables.Filters(complevel=5, complib='blosc')
    
    # sample only #
    rate, signal = wavfile.read(wav_list[0])
    feat_ii = generate_feat_opts(path=wav_list[0], cfg=cfg)
    vuv_ii, f0_ii, bap_ii, mgc_ii = feat_ii 
    # import ipdb; ipdb.set_trace()
    feat_storage = hdf5_file.create_earray(hdf5_file.root, 'feat',
            tables.Atom.from_dtype(np.dtype(np.float32)),
            filters=filters,
            shape=(0, sum([feat_ii[ii].shape[-1] for ii in range(len(feat_ii))]))
            )
    # save important information #
    # vuv, f0, bap, mgc #
    feat_info = hdf5_file.create_table(hdf5_file.root, 'feat_info',
            FeatInfo)
    feat_info_row = feat_info.row
    feat_info_row['feat'] = 'vuv'
    feat_info_row['dim'] = vuv_ii.shape[-1]
    feat_info_row.append()
    
    feat_info_row['feat'] = 'f0'
    feat_info_row['dim'] = f0_ii.shape[-1]
    feat_info_row.append()
   
    feat_info_row['feat'] = 'bap'
    feat_info_row['dim'] = bap_ii.shape[-1]
    feat_info_row.append()

    feat_info_row['feat'] = 'mgc'
    feat_info_row['dim'] = mgc_ii.shape[-1]
    feat_info_row.append()
    ### end ###

    key_storage = hdf5_file.create_earray(hdf5_file.root, 'key', 
            tables.StringAtom(itemsize=128),
            filters=filters,
            shape=(0, )
            )
    pos_storage = hdf5_file.create_earray(hdf5_file.root, 'pos',
            tables.Atom.from_dtype(np.dtype(np.int)),
            shape=(0, 2),
            filters=filters)
    start_idx = end_idx = 0

    pool = multiprocessing.Pool(processes=ncpu)
    for ii in tqdm(list(range(0, len(key_list), block))) :
        feat_ii_block = pool.starmap(generate_feat_opts, [(wav_list[idx], cfg, None, None) for idx in range(ii, min(ii+block, len(key_list)))] )
        for idx, feat_ii in enumerate(feat_ii_block) :
            feat_ii = np.hstack(feat_ii)
            feat_storage.append(feat_ii)
            # add idx start, end
            end_idx = start_idx + feat_ii.shape[0]
            pos_storage.append(np.array([[start_idx, end_idx]]))
            start_idx = end_idx
            key_storage.append(np.array([key_list[ii + idx]], dtype='S'))
        pass

    hdf5_file.close()
    pass

#####################
### WORLD VOCODER ###
#####################

### NUMPY FILE
def generate_feat_world_npfile(key_list, wav_list, output_path,
        cfg={'pkg':'world', 'mcep':59}, ncpu=16, block=256, compress=False) :
    assert len(key_list) == len(wav_list)
    assert len(set(key_list)) == len(set(wav_list)), "number of key & wav unique set is not same"
    assert cfg['pkg'] in ['standard', 'rosa']
    if os.path.exists(output_path) :
        assert os.path.isdir(output_path), "output_path must be a folder"
        assert os.listdir(output_path) == [], "folder must be empty"
    else :
        os.makedirs(output_path, mode=0o755, exist_ok=False)
        os.makedirs(os.path.join(output_path, 'meta'), mode=0o755, exist_ok=False)

    NUM_UTTS = len(key_list)
    # pool = multiprocessing.Pool(processes=ncpu)
    executor = ProcessPoolExecutor(max_workers=ncpu)
    list_execs = []
    file_kv = open(os.path.join(output_path, 'meta', 'feat.scp'), 'w')

    for ii in range(len(key_list)) :
        path_ii = os.path.join(output_path, key_list[ii]+'.npz')
        list_execs.append(executor.submit(partial(gen_and_save_arr, key_list[ii], wav_list[ii], path_ii, cfg=cfg, compress=compress))) 
    list_result = []
    for item in tqdm(list_execs) :
        list_result.append(item.result())
    for item in list_result :
        file_kv.write('{} {}\n'.format(*item))
    file_kv.flush()
    file_kv.close()
    pass


### KALDI ###
def generate_feat_kaldi_table_multi(key_list, wav_list, output_path, 
        cfg={'pkg':'kaldi'}, ncpu=16, block=8192) :
    assert len(key_list) == len(wav_list)
    assert cfg['pkg'] == 'standard'
    # split scp #
    tmpdir = tempfile.TemporaryDirectory()
    idx = 0
    for ii in range(0, len(key_list), block) :
        with open(os.path.join(tmpdir, 'tmp_feat_{}.scp'.format(idx)), 'w') as f :
            f.writelines(['{} {}\n'.format(key_list[jj], wav_list[jj]) for jj in range(ii, ii+block)])
            pass
        idx += 1
    NUM_UTTS = len(key_list)
    
    # pytables @
    hdf5_path = output_path
    hdf5_file = tables.open_file(hdf5_path, mode='w')
    filters = tables.Filters(complevel=5, complib='blosc')
    
    # sample only #
    rate, signal = wavfile.read(wav_list[0])
    feat_ii = generate_feat_opts(signal=signal, rate=rate, cfg=cfg)
    
    feat_storage = hdf5_file.create_earray(hdf5_file.root, 'feat',
            tables.Atom.from_dtype(np.dtype(np.float32)),
            filters=filters,
            shape=(0, feat_ii.shape[-1])
            )
    key_storage = hdf5_file.create_earray(hdf5_file.root, 'key', 
            tables.StringAtom(itemsize=128),
            filters=filters,
            shape=(0, )
            )
    pos_storage = hdf5_file.create_earray(hdf5_file.root, 'pos',
            tables.Atom.from_dtype(np.dtype(np.int)),
            shape=(0, 2),
            filters=filters)
    start_idx = end_idx = 0
    pool = multiprocessing.Pool(processes=ncpu)
    for ii in range(0, len(key_list), block) :
        feat_ii_block = pool.starmap(generate_feat_opts, [(wav_list[idx], cfg, None, None) for idx in range(ii, min(ii+block, len(key_list)))] )
        for idx, feat_ii in enumerate(feat_ii_block) :
            feat_storage.append(feat_ii)
            # add idx start, end
            end_idx = start_idx + feat_ii.shape[0]
            pos_storage.append(np.array([[start_idx, end_idx]]))
            start_idx = end_idx
            key_storage.append(np.array([key_list[ii + idx]], dtype='S'))
        pass

    hdf5_file.close()

    # delete tmpdir #
    tmpdir.cleanup()
    pass

#################
### GENERATOR ###
#################

import sys
import re
import argparse
import json
import time

def parse() :
    parser = argparse.ArgumentParser(description='params')
    parser.add_argument('--wav_scp', type=str)
    parser.add_argument('--output', type=str)
    parser.add_argument('--cfg', type=str, default='config/feat/logfbank_f40_d2.json')
    parser.add_argument('--ncpu', type=int, default=32)
    parser.add_argument('--compress', action='store_true', default=False)
    parser.add_argument('--type', type=str, choices=['h5', 'npz'], default='h5')
    return parser.parse_args()
    pass

if __name__ == '__main__' :
    args = parse() 
    wav_scp = args.wav_scp
    output_path = args.output
    cfg = json.load(open(args.cfg))
    re_key_wav = re.compile('(^[^\s]+) (.+)$')
    wav_list = []
    key_list = []
    with open(wav_scp) as f :
        lines = [x.strip() for x in f.readlines()]
        for ii in range(len(lines)) :
            _key, _wav = re_key_wav.findall(lines[ii])[0]
            wav_list.append(_wav)
            key_list.append(_key)
        pass
    start_time = time.time()
    if cfg['pkg'] in ['pysp', 'rosa', 'taco'] :
        if args.type == 'h5' : 
            generate_feat_standard_table_multi(key_list, wav_list, output_path=output_path, cfg=cfg, ncpu=args.ncpu)
        elif args.type == 'npz' :
            generate_feat_standard_npfile(key_list, wav_list, output_path=output_path, cfg=cfg, ncpu=args.ncpu, compress=args.compress)
    elif cfg['pkg'] == 'world' :
        generate_feat_world_table_multi(key_list, wav_list, output_path=output_path, cfg=cfg, ncpu=args.ncpu)
    elif cfg['pkg'] == 'kaldi' :
        generate_feat_kaldi_table(key_list, wav_list, output_path=output_path, cfg=cfg, ncpu=args.ncpu)
    else :
        raise ValueError("pkg name is not included")
        pass

    print("Time elapsed %.2f secs"%(time.time() - start_time))
    print("=== FINISH ===")
    pass
