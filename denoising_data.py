import numpy as np
import librosa
import glob2
import torch
import itertools
from torch.utils.data import Dataset, DataLoader
import random
import pdb

class DenoisingDataset(Dataset):
    def __init__(self, speaker_path, noise_path, duration = 4, sr = 16000,
                 snr = 0, random_start = True, num_speakers = 5,
                 noise_set = None, speech_split = 8, noise_split = 4,
                 transform = None):
        '''
        Parameters
        * speaker_path : (string) path to speaker files
        * noise_path : (string) path to noise files
        * duration : (float) minimum duration in seconds
        * sr : (int) sampling rate
        * snr : (float) the log signal to noise ratio
        * random_start : (boolean) if true starts sample speech and noise from a
          random offset, otherwise starts sample at the beginning of the recording
        * num_speakers : (int) selects number of speakers in training and
          validation sets
        * noise_set : (list of strings) list of noise files. These noise files
          should be in noise_path but should not include the noise_path in the filename
          (possibly changed due to clunky interface)
        * speech_split : (int between 0 and 10) the proportion of files allocated
          to training.
        * transform : (function object) transformation to apply to the input data.
        '''
        assert speech_split >= 0 and speech_split <= 10
        assert noise_split >= 0 and noise_split <= len(noise_set)
        self.duration = duration
        self.sr = sr
        self.snr = snr
        self.random_start = random_start
        self.transform = transform

        speakers = glob2.glob(speaker_path + '/*')
        random.shuffle(speakers)
        speakers = speakers[:num_speakers - 1]
        speech_files = []
        for speaker in speakers:
            speech_files.extend(glob2.glob(speaker + '/*.wav'))
            train_speech = speech_files[:speech_split]
            val_speech = speech_files[speech_split:]

        train_speech= [f for f in train_speech \
                        if librosa.core.get_duration(filename=f) > self.duration]

        val_speech = [f for f in val_speech \
                        if librosa.core.get_duration(filename=f) > self.duration]

        if not noise_set:
            noise_files = glob2.glob(noise_path + '/*.wav')
        else:
            noise_files = [noise_path + '/' + noise for noise in noise_set]

        noise_files = [f for f in noise_files \
                       if librosa.core.get_duration(filename=f) > self.duration]
        random.shuffle(noise_files)
        train_noise = noise_files[:noise_split]
        val_noise = noise_files[noise_split:]

        self.train = list(itertools.product(train_speech, train_noise))
        self.val = list(itertools.product(val_speech, val_noise))

    def __len__(self):
        return len(self.train)

    def _getmix(self, sfile, nfile):
        soffset = 0
        noffset = 0
        if self.random_start:
            soffset = np.random.random() * (librosa.core.get_duration(filename=sfile) - self.duration)
            noffset = np.random.random() * (librosa.core.get_duration(filename=nfile) - self.duration)

        # Read files
        saudio, _ = librosa.core.load(sfile, sr=self.sr, duration=self.duration, offset=soffset,
                                   res_type='kaiser_fast')
        naudio, _ = librosa.core.load(nfile, sr=self.sr, duration=self.duration, offset=noffset,
                                   res_type='kaiser_fast')

        # Deal with stereo noises
        if len(naudio.shape) >= 2:
            naudio = naudio[0] + naudio[1]

        # normalize and mix signals
        saudio = saudio / np.std(saudio)
        naudio = naudio / np.std(naudio)
        mixture = saudio + naudio

        if self.transform:
            mixture, _ = self.transform(mixture)
            saudio, _ = self.transform(saudio)
            naudio, _ = self.transform(naudio)

        return mixture, saudio, naudio,

    def __getitem__(self, i):
        # Notation
        # s = speech
        # n = noise
        sfile, nfile = self.train[i]
        mixture, saudio, _ = self._getmix(sfile, nfile)

        return torch.FloatTensor(mixture), torch.FloatTensor(saudio)

    def getvalset(self):
        mixes = []
        targets = []
        for sfile, nfile in self.val:
            mixture, saudio, naudio = self._getmix(sfile, nfile)
            mixes.append(torch.FloatTensor(mixture))
            targets.append(torch.FloatTensor(np.stack((saudio, naudio), axis = 0)))
        return mixes, targets

def main():
    speaker_path = '/media/data/timit-wav/train/dr1'
    noise_path = '/media/data/noises-16k'
    noise_set = ['babble-16k.wav', 'street-16k.wav', 'car-16k.wav',
                 'restaurant-16k.wav', 'subway-16k.wav']
    dataset = DenoisingDataset(speaker_path, noise_path, noise_set = noise_set)
    print('Length: ', len(dataset))

    # output validation set
    output = np.array([])
    features, targets = dataset.getvalset()
    for feature, target in zip(features, targets):
        feature = feature.numpy()
        target = target.numpy()
        feature = feature / (1.1 * np.max(feature))
        target = target[0] / (1.1 * np.max(target))
        output = np.append(output, (feature, target))
    librosa.output.write_wav('results/denoise_examples.wav', output, 16000, norm = True)

if __name__ == '__main__':
    main()
