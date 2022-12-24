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
    spec = librosa.feature.melspectrogram(  #MJ: return: np.ndarray of shape=(B, n_mels, T) = (B,81, 3350), say
        x,
        sr=sr,
        n_fft=n_fft, # length of the FFT window
        hop_length=hop_length_in_samples, #number of samples between successive frames.
        n_mels=n_mels) # (81, 6480)
    mag_spec = np.abs(spec)

    return mag_spec

#MJ: melspectorgram: https://medium.com/analytics-vidhya/understanding-the-mel-spectrogram-fca2afa2ce53

#Studies have shown that humans do not perceive frequencies on a linear scale.
# We are better at detecting differences in lower frequencies than higher frequencies.
# For example, we can easily tell the difference between 500 and 1000 Hz,
# but we will hardly be able to tell a difference between 10,000 and 10,500 Hz, 
# even though the distance between the two pairs are the same.

# In 1937, Stevens, Volkmann, and Newmann proposed a unit of pitch such that equal distances in pitch 
# sounded equally distant to the listener. This is called the mel scale.
# We perform a mathematical operation on frequencies to convert them to the mel scale.


# https://librosa.org/doc/latest/generated/librosa.feature.melspectrogram.html:
# librosa.feature.melspectrogram(*, y=None, sr=22050, S=None, n_fft=2048, hop_length=512, win_length=None, window='hann', center=True, pad_mode='constant', power=2.0, **kwargs)
# win_length==None: win_length = n_fft. Each frame of audio is windowed by window()=hann()
# center==True:  the signal x is padded so that frame t is centered at y[t * hop_length]. By default, STFT uses zero padding.
# ==> the signal y is padded so that frame D[:, t] is centered at y[t * hop_length]
# Frames here correspond to short windows of the signal (y), 
# each separated by hop_length = 512 samples. librosa uses centered frames, 
# so that the kth frame is centered around sample k * hop_length.
#==> returns: S = np.ndarray [shape=(…, n_mels, t)]

###The above is equivalent to:
#     D = np.abs(librosa.stft(y))**2
#     S = librosa.feature.melspectrogram(S=D, sr=sr)
# A mel spectrogram is a spectrogram where the frequencies are converted to the mel scale.

#==> returns:  Dnp.ndarray [shape=(…, 1 + n_fft/2, n_frames), dtype=dtype] =(1+1024, 3000)

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
