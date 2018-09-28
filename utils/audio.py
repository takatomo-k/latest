import sys,os,argparse,torch,threading,subprocess
import numpy as np
from config import audio_world_config
from multiprocessing import Pool,cpu_count
import numpy as np
from nnmnkwii import paramgen
from nnmnkwii.datasets import FileSourceDataset, FileDataSource
from nnmnkwii import preprocessing as P
from nnmnkwii.io import hts
from nnmnkwii.frontend import merlin as fe
import pysptk
import pyworld
from scipy.io import wavfile

def get_feature(wav_path,preprocessing=False,getsize=False):
    fs, x = wavfile.read(wav_path)
    x = x.astype(np.float64)
    if audio_world_config.use_harvest:
        f0, timeaxis = pyworld.harvest(
            x, fs, frame_period=audio_world_config.frame_period,
            f0_floor=audio_world_config.f0_floor, f0_ceil=audio_world_config.f0_ceil)
    else:
        f0, timeaxis = pyworld.dio(
            x, fs, frame_period=audio_world_config.frame_period,
            f0_floor=audio_world_config.f0_floor, f0_ceil=audio_world_config.f0_ceil)
        f0 = pyworld.stonemask(x, f0, timeaxis, fs)
    spectrogram = pyworld.cheaptrick(x, f0, timeaxis, fs)
    aperiodicity = pyworld.d4c(x, f0, timeaxis, fs)
    bap = pyworld.code_aperiodicity(aperiodicity, fs)

    alpha = pysptk.util.mcepalpha(fs)
    mgc = pysptk.sp2mc(spectrogram, order=audio_world_config.mgc_dim, alpha=alpha)
    f0 = f0[:, None]
    lf0 = f0.copy()
    nonzero_indices = np.nonzero(f0)
    lf0[nonzero_indices] = np.log(f0[nonzero_indices])
    if audio_world_config.use_harvest:
        # https://github.com/mmorise/World/issues/35#issuecomment-306521887
        vuv = (aperiodicity[:, 0] < 0.5).astype(np.float32)[:, None]
    else:
        vuv = (lf0 != 0).astype(np.float32)
    lf0 = P.interp1d(lf0, kind=audio_world_config.f0_interpolation_kind)

    # Parameter trajectory smoothing
    if audio_world_config.mod_spec_smoothing:
        hop_length = int(fs * (audio_world_config.frame_period * 0.001))
        modfs = fs / hop_length
        mgc = P.modspec_smoothing(
            mgc, modfs, cutoff=audio_world_config.mod_spec_smoothing_cutoff)

    mgc = P.delta_features(mgc, audio_world_config.windows)
    lf0 = P.delta_features(lf0, audio_world_config.windows)
    bap = P.delta_features(bap, audio_world_config.windows)


    features = np.hstack((mgc, lf0, vuv, bap))
    if preprocessing:
        out_path=wav_path.replace(".wav","").replace("wav","world")
        np.save(out_path,features)
    elif getsize:
        feature,mgc.shape[0],lf0.shape[0],bap.shape[0]
    else:
        return features
def normalize(feature,mean,std):
    return P.scale(feature, mean,std)

def gen_waveform(y_predicted, Y_mean, Y_std, post_filter=False, coef=1.4,
                 fs=16000, mge_training=False):
    alpha = pysptk.util.mcepalpha(fs)
    fftlen = fftlen = pyworld.get_cheaptrick_fft_size(fs)
    frame_period = audio_world_config.frame_period

    # Generate parameters and split streams
    mgc, lf0, vuv, bap = gen_parameters(y_predicted, Y_mean, Y_std, mge_training)

    if post_filter:
        mgc = merlin_post_filter(mgc, alpha, coef=coef)

    spectrogram = pysptk.mc2sp(mgc, fftlen=fftlen, alpha=alpha)
    aperiodicity = pyworld.decode_aperiodicity(bap.astype(np.float64), fs, fftlen)
    f0 = lf0.copy()
    f0[vuv < 0.5] = 0
    f0[np.nonzero(f0)] = np.exp(f0[np.nonzero(f0)])

    generated_waveform = pyworld.synthesize(f0.flatten().astype(np.float64),
                                            spectrogram.astype(np.float64),
                                            aperiodicity.astype(np.float64),
                                            fs, frame_period)
    # Convert range to int16

    # return features as well to compare natural/genearted later
    return generated_waveform#, mgc, lf0, vuv, bap


def write_wav(waveform,path,fs=16000):
    wavfile.write(path, fs, waveform.astype(np.int16))
def gen_parameters(y_predicted, Y_mean, Y_std, mge_training=True):
    mgc_dim, lf0_dim, vuv_dim, bap_dim = audio_world_config.stream_sizes

    mgc_start_idx = 0
    lf0_start_idx = mgc_dim
    vuv_start_idx = lf0_start_idx + lf0_dim
    bap_start_idx = vuv_start_idx + vuv_dim

    windows = audio_world_config.windows

    #ty = "acoustic"

    # MGE training
    if mge_training:
        # Split acoustic features
        mgc = y_predicted[:, :lf0_start_idx]
        lf0 = y_predicted[:, lf0_start_idx:vuv_start_idx]
        vuv = y_predicted[:, vuv_start_idx]
        bap = y_predicted[:, bap_start_idx:]

        # Perform MLPG on normalized features
        mgc = paramgen.mlpg(mgc, np.ones(mgc.shape[-1]), windows)
        lf0 = paramgen.mlpg(lf0, np.ones(lf0.shape[-1]), windows)
        bap = paramgen.mlpg(bap, np.ones(bap.shape[-1]), windows)

        # When we use MGE training, denormalization should be done after MLPG.
        mgc = P.inv_scale(mgc, Y_mean[:mgc_dim // len(windows)],
                          Y_std[:mgc_dim // len(windows)])
        lf0 = P.inv_scale(lf0, Y_mean[lf0_start_idx:lf0_start_idx + lf0_dim // len(windows)],
                          Y_std[lf0_start_idx:lf0_start_idx + lf0_dim // len(windows)])
        bap = P.inv_scale(bap, Y_mean[bap_start_idx:bap_start_idx + bap_dim // len(windows)],
                          Y_std[bap_start_idx:bap_start_idx + bap_dim // len(windows)])
        vuv = P.inv_scale(vuv, Y_mean[vuv_start_idx], Y_std[vuv_start_idx])
    else:
        # Denormalization first
        y_predicted = P.inv_scale(y_predicted, Y_mean, Y_std)

        # Split acoustic features
        mgc = y_predicted[:, :lf0_start_idx]
        lf0 = y_predicted[:, lf0_start_idx:vuv_start_idx]
        vuv = y_predicted[:, vuv_start_idx]
        bap = y_predicted[:, bap_start_idx:]

        # Perform MLPG
        Y_var = Y_std * Y_std
        mgc = paramgen.mlpg(mgc, Y_var[:lf0_start_idx], windows)
        lf0 = paramgen.mlpg(lf0, Y_var[lf0_start_idx:vuv_start_idx], windows)
        bap = paramgen.mlpg(bap, Y_var[bap_start_idx:], windows)
    return mgc, lf0, vuv, bap
