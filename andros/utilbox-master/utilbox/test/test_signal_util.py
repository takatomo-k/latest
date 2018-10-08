# import numpy as np
# import librosa
# import os

# from ..signal_util import rosa_spectrogram, rosa_inv_spectrogram, mel2spec, spec2mel

# WAV_PATH = '/home/is/andros-tj/rsch/misc/utilbox/utilbox/test/BTEC1jpn00100040en.wav'
# HOP_LENGTH = 200
# WIN_LENGTH = 800
# NFFT = 1024
# def draw(matrix, filename) :
    # import matplotlib as mpl;
    # mpl.use('Agg')
    # from pylab import plt
    # plt.imshow(matrix.T)
    # plt.savefig(filename)

# def test_raw_spectrogram_and_inv() :
    # signal, rate = librosa.load(WAV_PATH, None)

    # raw_spec = rosa_spectrogram(signal, n_fft=NFFT, hop_length=HOP_LENGTH, win_length=WIN_LENGTH)
    # inv_raw_spec = rosa_inv_spectrogram(raw_spec[0], n_fft=NFFT, hop_length=HOP_LENGTH, win_length=WIN_LENGTH)
    # print('raw recon error : %f'%(np.sum((signal[0:len(inv_raw_spec)]-inv_raw_spec)**2)))
    # librosa.output.write_wav(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'inv_raw_spec.wav'), inv_raw_spec, rate)

# def test_mel_spectrogram_and_inv() :
    # signal, rate = librosa.load(WAV_PATH, None)

    # raw_spec = rosa_spectrogram(signal, n_fft=NFFT, hop_length=HOP_LENGTH, win_length=WIN_LENGTH)
    # print("raw_spec shape : {}".format(raw_spec[0].shape))
    # mel_fbank, inv_mel_fbank = get_mel_inv_filterbank(nfft=NFFT)
    # mel_spec = spec2mel(raw_spec[0], mel_fbank)
    # print("mel_spec shape : {}".format(mel_spec.shape))
    # draw(np.log(mel_spec), 'plot_mel_spec.png')
    # inv_mel_spec = mel2spec(mel_spec, inv_mel_fbank)
    # inv_raw_spec = rosa_inv_spectrogram(inv_mel_spec, n_fft=NFFT, hop_length=HOP_LENGTH, win_length=WIN_LENGTH)
    
    # print('mel recon error : %f'%(np.sum((signal[0:len(inv_raw_spec)]-inv_raw_spec)**2)))
    # librosa.output.write_wav(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'inv_raw_spec.wav'), inv_raw_spec, rate)

