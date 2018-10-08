import numpy as np
import librosa
import librosa.filters
import soundfile
import math
from scipy import signal

#########################
### TACOTRON HELPER   ###
#########################
class TacotronHelper() :

    def __init__(self, cfg) :
        self.cfg = cfg
        pass

    def load_wav(self, path):
        return librosa.core.load(path, sr=self.cfg['sample_rate'])[0]


    def save_wav(self, wav, path):
        wav *= 32767 / max(0.01, np.max(np.abs(wav)))
        try :
            librosa.output.write_wav(path, wav.astype(np.int16), self.cfg['sample_rate'])
        except :
            soundfile.write(path, wav.astype(np.int16), self.cfg['sample_rate'])


    def preemphasis(self, x):
        return signal.lfilter([1, -self.cfg['preemphasis']], [1], x)


    def inv_preemphasis(self, x):
        return signal.lfilter([1], [1, -self.cfg['preemphasis']], x)


    def spectrogram(self, y):
        D = self._stft(self.preemphasis(y))
        S = self._amp_to_db(np.abs(D)) - self.cfg['ref_level_db']
        return self._normalize(S)


    def inv_spectrogram(self, spectrogram):
        '''Converts spectrogram to waveform using librosa'''
        S = self._db_to_amp(self._denormalize(spectrogram) + self.cfg['ref_level_db']) # Convert back to linear
        return self.inv_preemphasis(self._griffin_lim(S ** self.cfg['power'])) # Reconstruct phase

    def melspectrogram(self, y):
        D = self._stft(self.preemphasis(y))
        S = self._amp_to_db(self._linear_to_mel(np.abs(D))) - self.cfg['ref_level_db']
        return self._normalize(S)


    # DO NOT USE THIS #
    def find_endpoint(self, wav, threshold_db=-40, min_silence_sec=0.8):
        window_length = int(self.cfg['sample_rate'] * min_silence_sec)
        hop_length = int(window_length / 4)
        threshold = self._db_to_amp(threshold_db)
        for x in range(hop_length, len(wav) - window_length, hop_length):
            if np.max(wav[x:x+window_length]) < threshold:
                return x + hop_length
        return len(wav)


    def _griffin_lim(self, S):
        '''librosa implementation of Griffin-Lim
        Based on https://github.com/librosa/librosa/issues/434
        '''
        angles = np.exp(2j * np.pi * np.random.rand(*S.shape))
        S_complex = np.abs(S).astype(np.complex)
        y = self._istft(S_complex * angles)
        for i in range(self.cfg['griffin_lim_iters']):
            angles = np.exp(1j * np.angle(self._stft(y)))
            y = self._istft(S_complex * angles)
        return y

    def _stft(self, y):
        n_fft, hop_length, win_length = self._stft_parameters()
        return librosa.stft(y=y, n_fft=n_fft, hop_length=hop_length, win_length=win_length)


    def _istft(self, y):
        _, hop_length, win_length = self._stft_parameters()
        return librosa.istft(y, hop_length=hop_length, win_length=win_length)

    def _stft_parameters(self):
        n_fft = (self.cfg['num_freq'] - 1) * 2
        hop_length = int(self.cfg['frame_shift_ms'] / 1000 * self.cfg['sample_rate'])
        win_length = int(self.cfg['frame_length_ms'] / 1000 * self.cfg['sample_rate'])
        return n_fft, hop_length, win_length


    # Conversions:
    def _linear_to_mel(self, spectrogram):
        _mel_basis = None
        if _mel_basis is None:
            _mel_basis = self._build_mel_basis()
        return np.dot(_mel_basis, spectrogram)

    def _build_mel_basis(self):
        n_fft = (self.cfg['num_freq'] - 1) * 2
        return librosa.filters.mel(self.cfg['sample_rate'], n_fft, n_mels=self.cfg['num_mels'])

    def _amp_to_db(self, x):
        return 20 * np.log10(np.maximum(1e-5, x))

    def _db_to_amp(self, x):
        return np.power(10.0, x * 0.05)

    def _normalize(self, S):
        return np.clip((S - self.cfg['min_level_db']) / -self.cfg['min_level_db'], 0, 1)

    def _denormalize(self, S):
        return (np.clip(S, 0, 1) * -self.cfg['min_level_db']) + self.cfg['min_level_db']


####################################
### MULTISPEAKER TACOTRON HELPER ###
####################################
