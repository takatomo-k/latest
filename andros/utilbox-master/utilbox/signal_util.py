import librosa
import numpy as np
from scipy import signal
np.random.seed(123)

"""
source : https://github.com/madebyollin/acapellabot/blob/master/conversion.py
"""
def zero_guard(x) :
    return np.where(x < np.finfo(np.float32).eps, np.finfo(np.float32).eps, x)

def rosa_spectrogram(signal, n_fft=512, hop_length=None, power=2, win_length=None) :
    spectrogram = librosa.stft(signal, n_fft, hop_length, win_length)
    phase = np.imag(spectrogram)
    magnitude = np.abs(spectrogram)
    sqr_magnitude = np.power(magnitude, power)
    sqr_magnitude = zero_guard(sqr_magnitude)
    return sqr_magnitude.T, phase.T
    pass

def rosa_inv_spectrogram(spectrogram, n_fft=512, hop_length=None, win_length=None, power=2,
         phase_iter=50) :
    spectrogram = spectrogram.T
    magnitude = np.power(spectrogram, 1/power)
    magnitude = np.power(magnitude, 1.5) # TODO : tacotron tricks #
    for ii in range(phase_iter) :
        if ii == 0 :
            recons = np.pi * np.random.random_sample(magnitude.shape) + 1j * (2 * np.pi * np.random.random_sample(magnitude.shape) - np.pi)
        else :
            recons = librosa.stft(signal, n_fft, hop_length, win_length)
        spectrum = magnitude * np.exp(1j * np.angle(recons))
        signal = librosa.istft(spectrum, hop_length, win_length)
    return signal

def rosa_spec2mel(sqr_magnitude, nfilt) :
    return librosa.feature.melspectrogram(S=sqr_magnitude.T, n_mels=nfilt).T

def preemphasis(x, coeff_preemph):
    return signal.lfilter([1, -coeff_preemph], [1], x)

def inv_preemphasis(x, coeff_preemph) :
    return signal.lfilter([1], [1, -coeff_preemph], x)

"""
from python_speech_features import base, sigproc
def get_mel_inv_filterbank(nfilt=40, nfft=512, samplerate=16000 ,lowfreq=0, highfreq=None) :
    fbank = base.get_filterbanks(nfilt, nfft, samplerate, lowfreq, highfreq)
    inv_fbank = np.linalg.pinv(fbank)
    return fbank, inv_fbank
    pass

def spec2mel(spec, mel_fbank) :
    mel_spec = np.dot(spec, mel_fbank.T)
    mel_spec = zero_guard(mel_spec)
    return mel_spec

def mel2spec(mel_spec, inv_mel_fbank) :
    inv_mel_spec = np.dot(mel_spec, inv_mel_fbank.T) 
    inv_mel_spec = zero_guard(inv_mel_spec)
    return inv_mel_spec
"""
