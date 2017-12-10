import numpy as np
import librosa
import glob2
import torch
import itertools
from torch.utils.data import Dataset, DataLoader
import random
import pdb

class DenoisingDataset(Dataset):
    def __init__(self, speaker_path, noise_path, duration = None, sr = 16000,
                 snr = 0, random_start = True, speaker_set = None,
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
        * transform : (function object) transformation to apply to the input data.
        '''
        self.duration = duration
        self.sr = sr
        self.snr = snr
        self.random_start = random_start
        self.transform = transform

        if speaker_set is None:
            speakers = glob2.glob(speaker_path + '/*')
        else:
            speakers = [speaker_path + '/' + speaker for speaker in speaker_set]
        random.shuffle(speakers)
        speech = []
        for speaker in speakers:
            files = glob2.glob(speaker + '/*.wav')
            speech.extend(files)

        if duration is not None:
            speech = [f for f in speech \
                     if librosa.core.get_duration(filename=f) > self.duration]

        if not noise_set:
            noise_files = glob2.glob(noise_path + '/*.wav')
        else:
            noise_files = [noise_path + '/' + noise for noise in noise_set]
        random.shuffle(noise_files)

        self.mixes = list(itertools.product(speech, noise_files))

    def __len__(self):
        return len(self.mixes)

    def _getmix(self, sfile, nfile):
        soffset = 0
        noffset = 0
        sduration = librosa.core.get_duration(filename=sfile)
        nduration = librosa.core.get_duration(filename=nfile)
        if self.random_start and self.duration is not None:
            soffset = np.random.random() * (sduration - self.duration)
            sduration = min(sduration, self.duration)

        if self.random_start:
            noffset = np.random.random() * (nduration - sduration)

        # Read files
        saudio, _ = librosa.core.load(sfile, sr=self.sr, duration=self.duration, offset=soffset,
                                   res_type='kaiser_fast')
        naudio, _ = librosa.core.load(nfile, sr=self.sr, duration=sduration, offset=noffset,
                                   res_type='kaiser_fast')

        # Deal with stereo noises
        if len(naudio.shape) >= 2:
            naudio = naudio[0] + naudio[1]

        # normalize and mix signals
        saudio = saudio / np.std(saudio)
        naudio = naudio / np.std(naudio)
        mixture = saudio + naudio

        if self.transform:
            mixture = self.transform(mixture)
            saudio = self.transform(saudio)
            naudio = self.transform(naudio)

        return mixture, saudio, naudio

    def __getitem__(self, i):
        # Notation
        # s = speech
        # n = noise
        sfile, nfile = self.mixes[i]
        mixture, saudio, naudio = self._getmix(sfile, nfile)

        return torch.FloatTensor(mixture), torch.FloatTensor(saudio), torch.FloatTensor(naudio)

def main():
    speaker_path = '/media/data/timit-wav/train/dr1'
    noise_path = '/media/data/noises-16k'
    train_noise = ['babble-16k.wav', 'street-16k.wav', 'car-16k.wav',
                 'restaurant-16k.wav', 'subway-16k.wav']
    val_noise = ['bus-16k.wav', 'airport-16k.wav']
    speaker_set = ['fcjf0', 'fdml0', 'fetb0', 'mcpm0', 'mdpk0']


    trainset = DenoisingDataset(speaker_path, noise_path, noise_set=train_noise,
                                speaker_set=speaker_set)
    valset = DenoisingDataset(speaker_path, noise_path, noise_set=val_noise,
                              speaker_set=speaker_set)
    print('Train Length: ', len(trainset))
    print('Validation Length: ', len(valset))

    # output validation set
    output = np.array([])
    for i in range(len(valset)):
        feature, target, _ = valset[i]
        feature = feature.numpy()
        target = target.numpy()
        output = np.append(output, (feature, target))
    librosa.output.write_wav('results/denoise_examples.wav', output, 16000, norm = True)

if __name__ == '__main__':
    main()
