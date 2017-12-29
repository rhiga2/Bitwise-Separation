import numpy as np
import librosa
import glob2
import torch
import itertools
from torch.utils.data import Dataset, DataLoader
import random
import pdb

class DenoisingDataset(Dataset):
    def __init__(self, speeches, noises, sr = 16000,
                 snr = 0, random_start = True, transform = None):
        '''
        Parameters
        * speaker_path : (string) path to speaker files
        * noise_path : (string) path to noise files
        * duration : (float) minimum duration in seconds
        * sr : (int) sampling rate
        * snr : (float) the log signal to noise ratio
        * random_start : (boolean) if true starts sample noise from a
          random offset, otherwise starts sample at the beginning of the recording
        * num_speakers : (int) selects number of speakers in training and
          validation sets
        * noise_set : (list of strings) list of noise files. These noise files
          should be in noise_path but should not include the noise_path in the filename
          (possibly changed due to clunky interface)
        * transform : (function object) transformation to apply to the input data.
        '''
        self.sr = sr
        self.snr = snr
        self.random_start = random_start
        self.transform = transform
        self.mixes = list(itertools.product(speeches, noises))

    def __len__(self):
        return len(self.mixes)

    def _getmix(self, sfile, nfile):
        soffset = 0
        noffset = 0
        sduration = librosa.core.get_duration(filename=sfile)
        nduration = librosa.core.get_duration(filename=nfile)

        if self.random_start:
            noffset = np.random.random() * (nduration - sduration)

        # Read files
        saudio, _ = librosa.core.load(sfile, sr=self.sr, mono=True,
                                      duration=self.duration, offset=soffset,
                                      res_type='kaiser_fast')
        naudio, _ = librosa.core.load(nfile, sr=self.sr, mono=True,
                                      duration=self.duration, offset=noffset,
                                      res_type='kaiser_fast')

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

        return mixture, saudio, naudio

def get_speech_files(speaker_path, num_speakers = 7, num_train = 8, num_val = 1):
    assert num_train + num_val <= 10 # Assumes each speaker has 10 sentences
    train_speeches = []
    val_speeches = []
    test_speeches = []
    speakers = []
    dialects = glob2.glob(speaker_path + '/*')
    for dialect in dialects:
        speakers.extend(glob2.glob(dialect + '/*'))

    sample = speakers
    if not num_speakers:
        sample = random.sample(speakers, num_speakers)
    for speaker in sample:
        files = glob2.glob(speaker + '/*.wav')
        random.shuffle(files)
        train_speeches.extend(files[: num_train])
        val_speeches.extend(files[num_train : num_train + num_val])
        test_speeches.extend(files[num_train + num_val :])
    return train_speeches, val_speeches, test_speeches

def get_noise_files(noise_path, num_noises = 10, num_train = 6, num_val = 2):
    assert num_train + num_val <= num_noises
    noises = glob2.glob(noise_path)
    sample = random.sample(noises, num_noises)
    train_noises = noises[: num_train]
    val_noises = noises[num_train : num_train + num_val]
    test_noises = noises[num_train + num_val :]
    return train_noises, val_noises, test_noises

def main():
    speaker_path = '/media/data/timit-wav/train'
    noise_path = '/media/data/noises-16k'
    train_noise = ['babble-16k.wav', 'street-16k.wav', 'car-16k.wav',
                 'restaurant-16k.wav', 'subway-16k.wav']
    val_noise = ['bus-16k.wav', 'airport-16k.wav']


    # get training sentences, validation sentences, and testing sentences
    train_speeches, val_speeches, test_speeches = get_speech_files(speaker_path)
    train_noises, val_noises, test_noises = get_noise_files(noise_path)

    trainset = DenoisingDataset(train_sentences, train_noises)
    valset = DenoisingDataset(val_sentences, val_noises)
    testset = DenoisingDataset(test_sentences, test_noises)
    print('Train Length: ', len(trainset))
    print('Validation Length: ', len(valset))
    print('Test Length: ', len(testset))

    # output validation set
    output = np.array([])
    for i in range(len(valset)):
        feature, target, _ = valset[i]
        output = np.append(output, (feature, target))
    librosa.output.write_wav('results/denoise_examples.wav', output, 16000, norm = True)

if __name__ == '__main__':
    main()
