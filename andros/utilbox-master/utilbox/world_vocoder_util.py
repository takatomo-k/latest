import os
import subprocess
import tempfile
from os.path import join
import numpy as np
import librosa

WORLD_DIR = '/home/is/andros-tj/rsch/spsyn/external/merlin/tools/bin/WORLD/'
SPTK_DIR = '/home/is/andros-tj/rsch/spsyn/external/merlin/tools/bin/SPTK-3.9/'

analysis_cmd = os.path.join(WORLD_DIR, 'analysis')
x2x_cmd = join(SPTK_DIR, 'x2x')
sopr_cmd = join(SPTK_DIR, 'sopr')
mcep_cmd = join(SPTK_DIR, 'mcep')
synth_cmd = join(WORLD_DIR, 'synth')
mgc2sp_cmd = join(SPTK_DIR, 'mgc2sp')

F0_FILE = 'f0.d'
F0A_FILE = 'f0a.d'
LF0_FILE = 'lf0.d'
SP_FILE = 'sp.d'
AP_FILE = 'ap.d'
BAP_FILE = 'bap.d'
MGC_FILE = 'mgc.d'
REWAV_FILE = 'resyn.wav'
# rate 16000 
fs=16000
nffthalf=1024 
alpha=0.58
mcep_dim=59
order=4

def world_analysis(wav_path, mcep_dim=mcep_dim) :
    with tempfile.TemporaryDirectory() as tmpdir :
        ### extract f0, sp, ap ###
        subprocess.check_output('{} {} {} {} {}'.format(analysis_cmd, wav_path, join(tmpdir, F0_FILE), join(tmpdir, SP_FILE), join(tmpdir, AP_FILE)), shell=True)
        ### convert f0 to lf0 ###
        subprocess.check_output('{} +da {} > {}'.format(x2x_cmd, join(tmpdir, F0_FILE), join(tmpdir, F0A_FILE)), shell=True)
        subprocess.check_output('{} +af {} | {} -magic 0.0 -LN -MAGIC -1.0E+10 > {}'.format(x2x_cmd, join(tmpdir, F0A_FILE), sopr_cmd, join(tmpdir, LF0_FILE)), shell=True)
        ### convert sp to mgc ###
        subprocess.check_output('{0} +df {1} | {2} -R -m 32768.0 | {3} -a {4} -m {5} -l {6} -e 1.0E-8 -j 0 -f 0.0 -q 3 > {7}'.format(
            x2x_cmd, join(tmpdir, SP_FILE), sopr_cmd, 
            mcep_cmd, alpha, mcep_dim, 
            nffthalf, join(tmpdir, MGC_FILE)), shell=True)
        ### convert ap to bap ###
        subprocess.check_output('{0} +df {1} > {2} '.format(
            x2x_cmd, join(tmpdir, AP_FILE),  
            join(tmpdir, BAP_FILE)), shell=True)
        ### remove ###
        subprocess.check_output('rm {} {} {} {} -rf'.format(join(tmpdir, AP_FILE), join(tmpdir, SP_FILE), join(tmpdir, F0_FILE), join(tmpdir, F0A_FILE)), shell=True)
        ### retrieve features ###
        logf0_mat = np.fromfile(join(tmpdir, LF0_FILE), dtype='float32')
        bap_mat = np.fromfile(join(tmpdir, BAP_FILE), dtype='float32')
        mgc_mat = np.fromfile(join(tmpdir, MGC_FILE), dtype='float32').reshape(-1, mcep_dim+1)
        pass

    logf0_mat = logf0_mat.reshape(-1, 1)
    bap_mat = bap_mat.reshape(-1, 1)
    return logf0_mat, bap_mat, mgc_mat
    pass

def world2feat(logf0_mat, bap_mat, mgc_mat) :
    # interpolate f0 #
    f0_interpolate, vuv = interpolate_f0(np.exp(logf0_mat))
    logf0_interpolate = np.log(f0_interpolate)
    return vuv, logf0_interpolate, bap_mat, mgc_mat
    pass

def feat2world(vuv, logf0_interpolate, bap_mat, mgc_mat) :
    logf0_mat = np.array(logf0_interpolate, dtype='float32')
    logf0_mat[vuv < 0.5] = -1.0E+10 
    return logf0_mat, bap_mat, mgc_mat

def interpolate_f0(f0_raw) :
    
    f0_interpolate = np.array(f0_raw) # copy
    f0_len = f0_raw.size
    # voice / unvoice #
    vuv = np.zeros_like(f0_raw)
    vuv[f0_raw > 0] = 1.0
    
    # interpolate #
    idx_nonz = np.where(f0_raw > 0.0)[0]
    if idx_nonz[0] > 0 :
        f0_interpolate[0:idx_nonz[0]] = f0_raw[idx_nonz[0]]
    if idx_nonz[-1] < f0_len-1 :
        f0_interpolate[idx_nonz[-1]:] = f0_raw[idx_nonz[-1]]
    for ii in range(0, len(idx_nonz)-1) :
        start, end = idx_nonz[ii:ii+2]
        width = end - start
        if width > 1 :
            filler = np.full(width-1, f0_raw[start], dtype=np.float)
            grad = (f0_raw[end] - f0_raw[start])/width
            filler += np.arange(1, width) * grad
            f0_interpolate[start+1:end, 0] = filler
    return f0_interpolate, vuv
    pass

def world_synthesis(logf0_mat, bap_mat, mgc_mat, mcep_dim=mcep_dim) :
    
    with tempfile.TemporaryDirectory() as tmpdir :
        with open(join(tmpdir, LF0_FILE), 'wb') as f :
            logf0_mat.astype('float32').tofile(f)
        with open(join(tmpdir, MGC_FILE), 'wb') as f :
            mgc_mat.astype('float32').tofile(f)
        with open(join(tmpdir, BAP_FILE), 'wb') as f :
            bap_mat.astype('float32').tofile(f)

        ### convert lf0 to f0 ###
        subprocess.check_output('{0} -magic -1.0E+10 -EXP -MAGIC 0.0 {1} | {2} +fa > {3}'.format(
            sopr_cmd, join(tmpdir, LF0_FILE), x2x_cmd, join(tmpdir, F0A_FILE)), shell=True)
        
        subprocess.check_output('{0} +ad {1} > {2}'.format(
            x2x_cmd, join(tmpdir, F0A_FILE), join(tmpdir, F0_FILE)), shell=True)
        ### convert mgc to sp ###
        subprocess.check_output('{0} -a {1} -g 0 -m {2} -l {3} -o 2 {4} | {5} -d 32768.0 -P | {6} +fd > {7}'.format(
            mgc2sp_cmd, alpha, mcep_dim, 
            nffthalf, join(tmpdir, MGC_FILE), sopr_cmd,
            x2x_cmd, join(tmpdir, SP_FILE)
            ), shell=True)
        ### convert bap to ap ###
        subprocess.check_output('{} +fd {} > {}'.format(
            x2x_cmd, join(tmpdir, BAP_FILE), join(tmpdir, AP_FILE)), shell=True)
        ### convert to wav ###
        subprocess.check_output('{synth} {nffthalf} {fs} {f0} {sp} {ap} {wav}'.format(
            synth=synth_cmd, nffthalf=nffthalf, fs=fs, 
            f0=join(tmpdir, F0_FILE), sp=join(tmpdir, SP_FILE), ap=join(tmpdir, AP_FILE), 
            wav=join(tmpdir, REWAV_FILE)
            ), shell=True)
        signal, rate = librosa.load(join(tmpdir, REWAV_FILE), sr=None)
        pass
    return rate, signal
