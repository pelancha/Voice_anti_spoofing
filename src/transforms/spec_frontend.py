import torch
from torch import nn
import torchaudio
import numpy as np
import hydra

class SpectrogramFrontEnd(nn.Module):
    def __init__(self, sample_rate, n_fft, win_length_s, hop_length_s, window_fn, power, normalized, n_filter_banks):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_fft = n_fft 
        self.win_length = int(self.sample_rate * win_length_s)
        self.hop_length = int(self.sample_rate * hop_length_s)
        self.window_fn = hydra.utils.get_method(window_fn)
        self.power = power
        self.normalized = normalized
        
        self.n_filter_banks = n_filter_banks

        self.STFT = self._getSTFT()

        self.fc_layer = nn.Linear(self.n_fft // 2 + 1, self.n_filter_banks)
        self.fc_layer.weight.data = self._createFilterBank()

    def _getSTFT(self):
        STFT_CONFIG = {
            "n_fft": self.n_fft,
            "win_length": self.win_length,
            "hop_length": self.hop_length, 
            "window_fn": self.window_fn(self.win_length), 
            "power": self.power,
            "normalized": self.normalized,
        }
        
        STFT = torchaudio.transforms.Spectrogram(**STFT_CONFIG)

        return STFT
    
    def _getMagnitudeSpectrum(self, input_spec):
        magnitude_spectrogram = input_spec.abs()
        # power_spectrum = magnitude_spectrogram ** 2
        # log_power_spectrum = 10*torch.log10((power_spectrum/1) + 1e-9) #1=p0 by AES17; based on 2nd seminar of DLA
        return magnitude_spectrogram

    def _createFilterBank(self): 
        freqs = np.linspace(0, self.sample_rate / 2, int(self.n_fft // 2 + 1))
        f_min, f_max = 0.0, self.sample_rate / 2
        filter_edges = np.linspace(f_min, f_max, self.n_filter_banks + 2)
        
        filter_banks = np.zeros((self.n_filter_banks, len(freqs)))

        for i in range(1, self.n_filter_banks + 1):
            left = filter_edges[i - 1]
            center = filter_edges[i]
            right = filter_edges[i + 1]
            
            left_slope = (freqs - left) / (center - left)
            right_slope = (right - freqs) / (right - center)
            filter_banks[i - 1] = np.maximum(0, np.minimum(left_slope, right_slope))
            
        return torch.tensor(filter_banks, dtype = torch.float)

    def forward(self, input_audio):
        spectrogram = self.STFT(input_audio)
        magnitude_spectrogram = self._getMagnitudeSpectrum(spectrogram)
        
        magnitude_spectrogram = magnitude_spectrogram.squeeze(0).transpose(0,1)

        compressed_spec = self.fc_layer(magnitude_spectrogram)
        compressed_spec = compressed_spec.transpose(0,1).unsqueeze(0)
        return compressed_spec