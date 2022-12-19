"""
Ben Hayes 2020

ECS7006P Music Informatics

Coursework 1: Beat Tracking

File: beat_tracking_tcn/utils/spectrograms.py

Descrption: Utility functions for computing and trimming mel spectrograms.
"""
import os

import librosa
import numpy as np

def create_spectrogram(
        file_path,
        n_fft,
        hop_length_in_seconds,
        n_mels):
    
    x, sr = librosa.load(file_path) # x size is 1425409 in samples
    hop_length_in_samples = int(np.floor(hop_length_in_seconds * sr)) # spectrogram size = floor((x length - n_fft) / hop_length) + 1
    spec = librosa.feature.melspectrogram(  #MJ: return: np.ndarray [shape=(…, n_mels, t)]
        x,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length_in_samples,
        n_mels=n_mels) # (81, 6480)
    mag_spec = np.abs(spec)

    return mag_spec

def create_spectrograms(
        audio_dir,
        spectrogram_dir,
        n_fft,
        hop_length_in_seconds,
        n_mels):

    for file in os.scandir(audio_dir):
        if os.path.splitext(file.name)[1] != '.wav':
            continue

        mag_spec = create_spectrogram(
            file.path,
            n_fft,
            hop_length_in_seconds,
            n_mels)
        np.save(os.path.join(spectrogram_dir,
                             os.path.splitext(file.name)[0]), mag_spec)
        print('Saved spectrum for {}'.format(file.name))

def trim_spectrogram(spectrogram, trim_size):
    output = np.zeros(trim_size)  #MJ: trim_size: shape = (81,3000)
    dim0_range = min(trim_size[0], spectrogram.shape[0]) #:MJ: freq bins
    dim1_range = min(trim_size[1], spectrogram.shape[1])  #MJ: sample points

    output[:dim0_range, :dim1_range] = spectrogram[:dim0_range, :dim1_range]
    return output
