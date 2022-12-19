"""
Ben Hayes 2020

ECS7006P Music Informatics

Coursework 1: Beat Tracking

File: scripts/create_spectrograms.py
Description: Process a folder of audio files to create a folder of mel
             spectrograms.
"""

import librosa
import numpy as np
import os
import sys

#MJ: sys.path:
# the first item of this list, path[0], is the directory containing the script that was used
# to invoke the Python interpreter.

# If the script directory is not available (e.g. if the interpreter is invoked interactively
#  or if the script is read from standard input), path[0] is the empty string, 
# which directs Python to search modules in the current directory first.

#Add directly the current working directory to sys.path:
sys.path.append('/home/yeol/moon/beat-tracking-tcn')
#sys.path.append('/home/yeol/moon/beat-tracking-tcn/beat_tracking_tcn')
print(f' sys.path=\n{sys.path}')

from argparse import ArgumentParser

from beat_tracking_tcn.utils.spectrograms import create_spectrograms


def parse_args():
    """Parse command line arguments using argparse module"""

    parser = ArgumentParser(
        description="Process a folder of audio files and output a folder of " +
                    "mel spectrograms as NumPy dumps")

    parser.add_argument(
        "audio_directory",
        type=str
    )
    parser.add_argument(
        "output_directory",
        type=str
    )
    parser.add_argument(
        "-f",
        "--fft_size",
        type=int,
        default=2048,
        help="Size of the FFT (default=2048)"
    )
    parser.add_argument(
        "-l",
        "--hop_length",
        type=float,
        default=0.01,
        help="Hop length in seconds (default=0.01)"
    )
    parser.add_argument(
        "-n",
        "--n_mels",
        type=int,
        default=81,
        help="Number of Mel bins (default=81)"
    )

    return parser.parse_args()



#MJ: package install errors for numba and librosa: https://github.com/librosa/librosa/issues/1160
if __name__ == '__main__':
    args = parse_args()

    create_spectrograms(
        args.audio_directory,
        args.output_directory,
        args.fft_size,
        args.hop_length,
        args.n_mels)
