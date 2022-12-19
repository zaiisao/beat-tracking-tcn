"""
Ben Hayes 2020

ECS7006P Music Informatics

Coursework 1: Beat Tracking

File: beat_tracking_tcn/beat_tracker.py

Descrption: The main entry point function for the beat tracker. This can be
imported as follows:

>>> from beat_tracking_tcn.beat_tracker import beatTracker

Then it can be invoked like so:

>>> beats, downbeats = beatTracker(path_to_audio_file)
"""
import os
import pickle

from madmom.features import DBNBeatTrackingProcessor
import torch

from beat_tracking_tcn.models.beat_net import BeatNet
from beat_tracking_tcn.utils.spectrograms import create_spectrogram,\
                                                 trim_spectrogram


def load_checkpoint(model, checkpoint_file):
    """
    Restores a model to a given checkpoint, but loads directly to CPU, allowing
    model to be run on non-CUDA devices.
    """    
    model.load_state_dict(
        torch.load(checkpoint_file, map_location=torch.device('cpu')))


# Some important constants that don't need to be command line params
FFT_SIZE = 2048
HOP_LENGTH_IN_SECONDS = 0.01
HOP_LENGTH_IN_SAMPLES = 220  #MJ; beat-transformer uses 1024?
N_MELS = 81
TRIM_SIZE = (81, 3000)
SR = 22050

# Paths to checkpoints distributed with the beat tracker. It's possible to
# call the below functions with custom checkpoints also.
DEFAULT_CHECKPOINT_PATH = os.path.join(
        os.path.dirname(__file__),
        'checkpoints/default_checkpoint.torch')
DEFAULT_DOWNBEAT_CHECKPOINT_PATH = os.path.join(
        os.path.dirname(__file__),
        'checkpoints/default_downbeat_checkpoint.torch')


# Prepare the models
model = BeatNet()
model.eval()

downbeat_model = BeatNet(downbeats=True)
downbeat_model.eval()

# Prepare the post-processing dynamic Bayesian networks, courtesy of madmom.
# MJ: from Davies & Boek, 2019: Given the beat activation function produced by our TCN has the same
# temporal resolution and target structure as in [Boek & Krebs 2014], we directly
# reuse this existing DBN together with the default parameters
# given in [Krebs & Boek 2015]: a tempo range of 55–215 beats per minute, and
# the transition-λ, which aims to control the ability of the model
# to react to tempo changes, at a value of 100.


dbn = DBNBeatTrackingProcessor(
    min_bpm=55,
    max_bpm=215,
    transition_lambda=100,
    fps=(SR / HOP_LENGTH_IN_SAMPLES),  #MJ: IN beat-transformer: 44100 / 1024 = 43 frames/sec
    online=True)

downbeat_dbn = DBNBeatTrackingProcessor(
    min_bpm=10,
    max_bpm=75,
    transition_lambda=100,
    fps=(SR / HOP_LENGTH_IN_SAMPLES),
    online=True)


def beat_activations_from_spectrogram(
    spectrogram,
    checkpoint_file=None,
    downbeats=True):
    """
    Given a spectrogram, use the TCN model to compute a beat activation
    function.
    """

    # Load the appropriate checkpoint
    if checkpoint_file is not None:
        load_checkpoint(
            downbeat_model if downbeats else model,  #MJ: downbeat_model = BeatNet(downbeats=True)
            checkpoint_file)
    else:
        load_checkpoint(
            downbeat_model if downbeats else model,
            DEFAULT_DOWNBEAT_CHECKPOINT_PATH
                if downbeats else DEFAULT_CHECKPOINT_PATH)

    # Speed up computation by skipping torch's autograd
    with torch.no_grad():
        # Convert to torch tensor if necessary
        if type(spectrogram) is not torch.Tensor:
            spectrogram_tensor = torch.from_numpy(spectrogram)\
                                    .unsqueeze(0)\
                                    .unsqueeze(0)\
                                    .float()
        else:
            # Otherwise use the spectrogram as-is
            spectrogram_tensor = spectrogram

        # Forward the spectrogram through the model. Note there are no size
        # restrictions here, as the model is fully convolutional. 
        return downbeat_model(spectrogram_tensor).numpy() if downbeats\
               else model(spectrogram_tensor).numpy()

#MJ: Detect beats by DBN
def predict_beats_from_spectrogram(
        spectrogram,
        checkpoint_file=None,
        downbeats=True):
  
    raw_activations =\
        beat_activations_from_spectrogram(
            spectrogram,
            checkpoint_file,
            downbeats).squeeze()

    # Perform independent post-processing for downbeats
    if downbeats:
        beat_activations = raw_activations[0]  #MJ: Probabilities that beats occur at frame t. frame interval=0.023 sec= 23ms
                                               # 44100/1024 = 43 frames/s; 1/43 = 0.023 s/frame
        downbeat_activations = raw_activations[1] #MJ: Probabilities that  downbeats occur at frame.

        dbn.reset()
        
        #MJ: https://dida.do/blog/beat-tracking-with-deep-neural-networks:
        # Post-processing
# Generally, the metrical structure of a musical piece builds upon a hierarchy of approximate regular pulses.
# To exploit this sequential structure, a probabilistic dynamic model is used to result in 
# a more robust estimation of the beat instants and to correctly operate under rhythmic fluctuations, 
# such as ritardando and accelarando or metric modulations.

# To this end, a DBN is employed which jointly infers the period and phase of the beat.
# More precisely, the post-processing stage constitutes a hidden Markov model (HMM), 
# in which a sequence of hidden variables that represent the beat structure of an audio piece
# is inferred from the sequence of observed beat activations. The probabilistic state space consists of two hidden variables, namely the beat period, i.e. the reciprocal of the tempo, 
# and the position inside the beat period (Krebs, 2015).

# Given a sequence of beat activations 
# , the goal of the hidden Markov model is to identify the most probable hidden state sequence 
# . For every time frame the hidden state is defined as (phi_t, tau_t), 
#  where phi_t  denotes the position inside a beat period, and tau_t
#  refers to the length of the current beat period measured in frames. 
#  Due to the principle to use exactly one state per frame to indicate the position inside the beat period,
#  the number of position states corresponds exactly to length of the current beat period 
# . Usually, the tempo 
#  of a musical piece is measured in beats per minute (BPM), and therefore needs to be converted 
#  to the beat period measured in frames, by
 
        #MJ:   Detect the beats in the given activation function with Viterbi  decoding.
        predicted_beats = dbn.process_offline(beat_activations.squeeze())
        downbeat_dbn.reset()
        predicted_downbeats =\
            downbeat_dbn.process_offline(downbeat_activations.squeeze())

        return predicted_beats, predicted_downbeats
    else: #MJ: beat 
        beat_activations = raw_activations
        dbn.reset()
        predicted_beats = dbn.process_offline(beat_activations.squeeze())
        return predicted_beats


def beatTracker(input_file, checkpoint_file=None, downbeats=True):
    """
    Our main entry point — load an audio file, create a spectrogram and predict
    a list of beat times from it.
    """    
    mag_spectrogram = create_spectrogram(
            input_file,
            FFT_SIZE,
            HOP_LENGTH_IN_SECONDS,
            N_MELS).T
    
    return predict_beats_from_spectrogram(
        mag_spectrogram,
        checkpoint_file,
        downbeats)