"""
Ben Hayes 2020

ECS7006P Music Informatics

Coursework 1: Beat Tracking

File: beat_tracking_tcn/datasets/ballroom_dataset.py
Description: A PyTorch dataset class representing the ballroom dataset.
"""

import glob
import torch
from torch.utils.data import Dataset

import os
import numpy as np


class BallroomDataset(Dataset):
    """
    A PyTorch Dataset wrapping the ballroom dataset for beat detection tasks

    Provides mel spectrograms and a vector of beat annotations per spectrogram
    frame as per the Davies & Böck paper (that is, with two frames of 0.5
    either side of the beat's peak.

    Requires dataset to be preprocessed to mel spectrograms using the provided
    script.
    """

    def __init__(
            self,
            spectrogram_dir,
            label_dir,
            dataset_name,
            subset="train",
            validation_fold=None,
            sr=22050,
            hop_size_in_seconds=0.01,  #MJ: = 10ms
            trim_size=(81, 3000), 
            #MJ: 1/100 s/f => 100 frames/sec * 30 sec = 3000 frames. Obtain 81 freq bins over a window of 1/100 s/f= 10ms /frame
            #MJ: Davies and Boek, 2019: BLSTM:
            #  Three such spectrograms are calculated at a fixed hop size of 10 ms 
            # with increasing window sizes of 23.2 ms, 46.4 ms and 92.9 ms
            
            #MJ: TCN: As the initial input representation we use a single log magnitude spectrogram
            # with a hop size of 10 ms and a window size of 46.4 ms (2048 samples/ FFT_SIZE = 2048). 
            # A logarithmic grouping of frequency bins with 12 bands per octave provides an input representation 
            # with a total of 81 frequency bands from 30 Hz up to 17 kHz,
            
            downbeats=False):
        """
        Initialise the dataset object.

        Parameters:
            spectrogram_dir: directory holding spectrograms as NumPy dumps
            label_dir: directory containing labels as NumPy dumps

        Keyword Arguments:
            sr (=22050): Sample rate to use when converting annotations to
                         spectrogram frame indices
            hop_size_in_seconds (=0.01): Mel spectrogram hop size
            trim_size (=(81,3000)): Dimensions to trim spectrogram down to.
                                    Should match input dimensions of network.
        """
        self.spectrogram_dir = spectrogram_dir
        self.label_dir = label_dir
        self.dataset_name = dataset_name
        self.subset = subset
        self.validation_fold = validation_fold
        self.data_names = self._get_data_list()

        self.sr = 22050
        self.hop_size = int(np.floor(hop_size_in_seconds * 22050))  #MJ: hop_size in audio sample unit which is 220
        self.trim_size = trim_size # JA: Shape is (81, 3000)

        self.downbeats = downbeats

    def __len__(self):
        """Overload len() calls on object."""
        return len(self.data_names)

    def __getitem__(self, i):
        """Overload square bracket indexing on object"""
        raw_spec, raw_beats = self._load_spectrogram_and_labels(i) #MJ = spectrogram, beat_vector
        x, y = self._trim_spec_and_labels(raw_spec, raw_beats)

        if self.downbeats: #MJ: y =(3000,2) => (2,3000) = (ch, time)
            y = y.T

        return {
            'spectrogram': torch.from_numpy(
                    np.expand_dims(x.T, axis=0)).float(),
            'target': torch.from_numpy(y[:3000].astype('float64')).float(),
        }

    def get_name(self, i):
        """Fetches name of datapoint specified by index i"""
        return self.data_names[i]

    def get_ground_truth(self, i, quantised=True, downbeats=False):
        """
        Fetches ground truth annotations for datapoint specified by index i

        Parameters:
            i: Index signifying which datapoint to fetch truth for

        Keyword Arguments:
            quantised (=True): Whether to return a quantised grount truth
        """

        return self._get_quantised_ground_truth(i, downbeats)\
            if quantised else self._get_unquantised_ground_truth(i, downbeats)

    def _trim_spec_and_labels(self, spec, labels):
        """
        Trim spectrogram matrix and beat label vector to dimensions specified
        in self.trim_size. Returns tuple of trimmed NumPy arrays

        Parameters:
            spec: Spectrogram as NumPy array
            labels: Labels as NumPy array
        """

        x = np.zeros(self.trim_size)
        if not self.downbeats:
            y = np.zeros(self.trim_size[1])
        else: #MJ: downbeat
            y = np.zeros((self.trim_size[1], 2))

        to_x = self.trim_size[0]
        to_y = min(self.trim_size[1], spec.shape[1])

        x[:to_x, :to_y] = spec[:, :to_y]
        y[:to_y] = labels[:to_y]

        return x, y

    def _get_data_list(self):
        """Fetches list of datapoints in label directory"""

        names = []
        if self.validation_fold is None:
            label_dir_entries = os.scandir(self.label_dir)
            if self.dataset_name == "beatles":
                label_dir_entries = glob.glob(os.path.join(self.label_dir, "**", "*.txt"), recursive=True)

            for entry in label_dir_entries:
                _, file_extension = os.path.splitext(entry)
                if file_extension == ".folds":
                    continue

                if self.dataset_name == "rwc_popular":
                    # JA: rwc popular dataset is named .BEAT.TXT.
                    # calling splitext only once will remove the .TXT but not the preceding .BEAT.
                    filename = os.path.splitext(entry)[0]
                    filename = os.path.splitext(filename)[0]
                else:
                    filename = os.path.splitext(entry)[0]

                names.append(filename)
        else:
            fold_files = glob.glob(os.path.join(self.label_dir, "*.folds"))  #MJ: /mount/beat-tracking/ballroom/label/???.folds etc
            fold_file = fold_files[0]  #MJ: len(fold_files) = 1

            k = 8 # JA: k = 8 is the standard in beat tracking

            with open(fold_file, 'r') as fp:
                lines = fp.readlines()

                for line in lines:
                    line = line.strip('\n')
                    line = line.replace('\t', ' ')
                    audio_filename, fold_number = line.split(' ')
                    audio_filename_start = len(self.dataset_name) + 1 # Each line in a .folds file starts with "DATASET_"

                    validation_fold = self.validation_fold
                    test_fold = (validation_fold + 1) % k
                    is_valid_and_training = \
                        self.subset == "train" \
                        and validation_fold != int(fold_number) \
                        and test_fold != int(fold_number) #MJ: is the file for training

                    is_valid_and_validation = \
                        self.subset == "val" and \
                        validation_fold == int(fold_number) #MJ: is the file for  validation or testing????

                    is_valid_and_test = \
                        self.subset == "test" \
                        and test_fold == int(fold_number)

                    if is_valid_and_training or is_valid_and_validation or is_valid_and_test:
                        audio_file_path = os.path.join(self.spectrogram_dir, audio_filename[audio_filename_start:] + ".npy")
                        if not os.path.isfile(audio_file_path): #MJ: audio_file_path is not a file? If recursive is true, the pattern “**” will match any files and zero or more directories and subdirectories. If the pattern is followed by an os.sep, only directories and subdirectories match.
                            audio_file_paths = glob.glob(os.path.join(self.spectrogram_dir, "**", audio_filename[audio_filename_start:] + ".npy"), recursive=True)
                            if len(audio_file_paths) > 0:
                                audio_file_path = audio_file_paths[0]

                        if os.path.isfile(audio_file_path):
                            names.append(os.path.splitext(audio_file_path)[0])
                        else:
                            print(f"{audio_file_path} not found; skipping")

        return names

    def _text_label_to_float(self, text):
        """Exracts beat time from a text line and converts to a float"""
        allowed = '1234567890. \t'
        filtered = ''.join([c for c in text if c in allowed])
        if '\t' in filtered:
            t = filtered.rstrip('\n').split('\t')
        else:
            t = filtered.rstrip('\n').split(' ')
        
        if self.dataset_name == "rwc_popular":
            time_sec = int(t[0]) / 100.0
            beat = 1 if int(t[2]) == 384 else 2
            return time_sec, beat
        elif self.dataset_name == "beatles":
            if len(t) == 3:
                return float(t[0]), float(t[2])
            elif len(t) == 2:
                return float(t[0]), float(t[1])
            else:
                raise NotImplementedError
        else:
            return float(t[0]), float(t[1])
    
    def _get_beat_label_file_extension(self):
        if self.dataset_name == "ballroom":
            return ".beats"
        elif self.dataset_name == "hainsworth":
            return ".txt"
        elif self.dataset_name == "rwc_popular":
            return ".BEAT.TXT"
        elif self.dataset_name == "beatles":
            return ".txt"

    def _get_quantised_ground_truth(self, i, downbeats):
        """
        Fetches the ground truth (time labels) from the appropriate
        label file. Then, quantises it to the nearest spectrogram frames in
        order to allow fair performance evaluation.
        """

        file_extension = self._get_beat_label_file_extension()
        with open(
                os.path.join(self.label_dir, self.data_names[i] + file_extension),
                'r') as f:

            beat_times = []

            for line in f:
                time, index = self._text_label_to_float(line)
                if not downbeats:
                    beat_times.append(time * self.sr)
                else:
                    if index == 1:
                        beat_times.append(time * self.sr)
        quantised_times = []

        for time in beat_times:
            spec_frame = int(time / self.hop_size)
            quantised_time = spec_frame * self.hop_size / self.sr
            quantised_times.append(quantised_time)

        return np.array(quantised_times)

    def _get_unquantised_ground_truth(self, i, downbeats):
        """
        Fetches the ground truth (time labels) from the appropriate
        label file.
        """

        file_extension = self._get_beat_label_file_extension()
        with open(
                os.path.join(self.label_dir, self.data_names[i] + file_extension),
                'r') as f:
            
            beat_times = []

            for line in f:
                time, index = self._text_label_to_float(line)
                if not downbeats:
                    beat_times.append(time)
                else:
                    if index == 1:
                        beat_times.append(time)

        return np.array(beat_times)

    def _load_spectrogram_and_labels(self, i): #MJ: i = index to an audio's spectrogram
        """
        Given an index for the data name array, return the contents of the
        corresponding spectrogram and label dumps.
        """
        data_name = os.path.basename(self.data_names[i])

        file_extension = self._get_beat_label_file_extension()
        with open(
                os.path.join(self.label_dir, data_name + file_extension), #MJ: Media-106011.beats
                'r') as f:

            beat_floats = []
            beat_indices = []
            
            for line in f:
                parsed = self._text_label_to_float(line)
                beat_floats.append(parsed[0])  #MJ: target beat time location in sec
                beat_indices.append(parsed[1]) #MJ: 1, 2,3,4; 1,2,34; etc
                
            beat_times = np.array(beat_floats) * self.sr  #MJ: beat_times= beat locations in audio sample unit

            if self.downbeats:
                downbeat_times = self.sr * np.array(
                    [t for t, i in zip(beat_floats, beat_indices) if i == 1])

        spectrogram =\
            np.load(os.path.join(self.spectrogram_dir, data_name + '.npy')) # (81, 3022) will be trimmed to (81, 3000)
        if not self.downbeats:
            beat_vector = np.zeros(spectrogram.shape[-1])  #MJ: spectrogram: shape = (81,3000)
                                                           #MJ: spectrogram.shape[-1] = 3000
        else:
            beat_vector = np.zeros((spectrogram.shape[-1], 2))  #MJ: beat_vector: shape = (3000,2)

        for time in beat_times:  #MJ: beat_times= beat locations in audio sample unit
            spec_frame =\
                min(int(time / self.hop_size), beat_vector.shape[0] - 1) #MJ: beat_vector.shape[0]=3000
                
            #MJ: Davies & Boek, 2019: Following the strategy of [14] for onset detection, 
            # we widen the temporal activation region around the annotations to include two adjacent temporal 
            # frames on either side of each quantised beat location and weight them with a value of 0.5
            # during training:    
            for n in range(-2, 3): #MJ: n in [-2,2]
                if 0 <= spec_frame + n < beat_vector.shape[0]:
                    if not self.downbeats:
                        beat_vector[spec_frame + n] = 1.0 if n == 0 else 0.5 #MJ: beat_vector=[0.5, 0.5, 1, 0.5, 0.5]
                    else:
                        beat_vector[spec_frame + n, 0] = 1.0 if n == 0 else 0.5  #MJ: Downbeat  is a beat and downbeat at the same time
        
        if self.downbeats:
            for time in downbeat_times:
                spec_frame =\
                    min(int(time / self.hop_size), beat_vector.shape[0] - 1)
                for n in range(-2, 3):
                    if 0 <= spec_frame + n < beat_vector.shape[0]:
                        beat_vector[spec_frame + n, 1] = 1.0 if n == 0 else 0.5


        return spectrogram, beat_vector
