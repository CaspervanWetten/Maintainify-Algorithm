# For audio loading testing:
# from pydub import AudioSegment
import numpy as np
import sys
from scipy.io import wavfile
from plotly.offline import init_notebook_mode
import plotly.graph_objs as go
import plotly
from scipy.io import wavfile
from tqdm import tqdm

import torch
import logging
import torchaudio
import torch.nn.functional as F
import torchaudio.transforms as T
import torchaudio.compliance.kaldi as ta_kaldi

dog_sound = "Data/dog2.wav"

fs_wav, data_wav = wavfile.read(dog_sound)

data_wav, or_sample_rate = torchaudio.load(dog_sound)

print(f"signal duration: {data_wav.shape[0] / or_sample_rate}")
np.set_printoptions(threshold=sys.maxsize)
# print(f'signal: {data_wav}')

#Skip alle stille momenten
# data_wav = data_wav[np.any(data_wav != 0, axis=1)]







print(f"1d data_wav: {data_wav}")
## K means the numpy array:
## ----------------------

# if data_wav.ndim == 2: # Als het een 2d array is (stereo), maak het een 1d array (mono)
#     data_wav = np.mean(data_wav, axis=1)


print(f"extracted features: {data_wav}")

def extract_features(waveform, or_sample_rate):
    sample_rate:    int     = 512 # The amount of individual frames that the soundbyte is converted 
    sample_rate = 512
    n_mels = 128
    n_fft = 256
    hop_length = 512 
    n_bits=8 # Used in 
    if waveform.shape[0] > 1: 
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    if or_sample_rate != sample_rate:
        resampler = T.Resample(orig_freq=or_sample_rate, new_freq=sample_rate)
        waveform = resampler(waveform)
    waveform = waveform / torch.max(torch.abs(waveform)) # Ik vraag me af of dit waarde heeft
    MS_transform = T.MelSpectrogram(sample_rate=sample_rate, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    mel_spec = MS_transform(waveform)
    # Turn it into a numpy array!
    mel_spec = mel_spec.squeeze().numpy().flatten()

    return mel_spec

def k_means(data_wav):
    max_iters = 1024
    vocab_size = 128
    convergence_tolerance = 1e-20
    centroids = np.random.choice(data_wav, size=vocab_size, replace=False)
    for _ in tqdm(range(max_iters), desc="committing K-means clustering"):
        distances = np.abs(data_wav[:, np.newaxis] - centroids)
        labels = np.argmin(distances, axis=1)
        new_centroids = np.array([data_wav[labels == k].mean() for k in range(vocab_size)])
        if np.all(np.abs(new_centroids - centroids) < convergence_tolerance):
            break
        centroids = new_centroids
    return centroids, labels

mel_spec = extract_features(data_wav, or_sample_rate)
centroids, labels = k_means(mel_spec)

print(f"labels: {labels}")

print()


print()

print(f"centroids: {centroids}")


print(f'number of points in each cluster: {np.bincount(labels)}')


