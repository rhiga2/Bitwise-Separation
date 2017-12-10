import numpy as np
import librosa
import scipy.signal as signal
from denoising_data import DenoisingDataset
import matplotlib.pyplot as plt
import pdb

class PulseDensityModulation(object):
    def __init__(self, old_sr, new_sr):
        self.old_sr = old_sr
        self.new_sr = new_sr

    def __call__(self, x):
        '''
        Converts pulse code modulation to pulse density modulation.
        I would like to have a less sequential transformation.
        More details: https://en.wikipedia.org/wiki/Pulse-density_modulation
        '''
        x = x / np.max(np.abs(x))
        x = librosa.core.resample(x, self.old_sr, self.new_sr, res_type = 'scipy')
        y = []
        qe = 0 # quantization error
        for i in range(x.shape[0]):
            if x[i] >= qe:
                y.append(1)
            else:
                y.append(-1)
            qe = y[i] - x[i] + qe
        y = (np.array(y) + 1) // 2
        return np.array(y).astype(np.float32)

class PulseCodingModulation(object):
    def __init__(self, downsample_factor = 64):
        self.downsample_factor = downsample_factor

    def __call__(self, x):
        '''
        Converts pulse density modulation to pulse coding modulation.
        '''
        y = 2 * x - 1
        y = signal.decimate(y, self.downsample_factor, ftype = 'fir', zero_phase = True)
        # for i in range(2):
        #     y = signal.decimate(y, 8, 8, ftype = 'iir', zero_phase = True)
        return y.astype(np.float32)


def main():
    sr = 16000
    oversample = 64

    pdm = PulseDensityModulation(sr, oversample * sr)
    pcm = PulseCodingModulation(oversample)
    speaker_path = '/media/data/timit-wav/train/dr1'
    noise_path = '/media/data/noises-16k'
    noise_set = ['babble-16k.wav', 'street-16k.wav', 'car-16k.wav',
                 'restaurant-16k.wav', 'subway-16k.wav']
    dataset = DenoisingDataset(speaker_path, noise_path, duration=3,
    num_speakers=5, noise_set=noise_set, transform=pdm)
    print('Length Training Set: ', len(dataset))
    print('Get Validation Set: ', dataset.lenval())

    # Test PDM-PCM conversion
    mixture, saudio = dataset[0]
    mixture = mixture.numpy()
    recovered_mix = pcm(mixture)
    librosa.output.write_wav('results/recovered.wav', recovered_mix, sr, norm = True)

    for i in range(len(dataset)):
        mixture, speech = dataset[i]
        mixture = mixture.numpy().astype(bool)
        speech = speech.numpy().astype(bool)
        np.savez('/media/data/bitwise_pdm/train%d' % (i,), mixture=mixture,
                 speech=speech)

    i = 0
    mixtures, targets = dataset.getvalset()
    print(len(mixtures), len(targets))
    for mixture, target in zip(mixtures, targets):
        mixture = mixture.numpy().astype(bool)
        target = target.numpy().astype(bool)
        np.savez('/media/data/bitwise_pdm/val%d' % (i,), mixture=mixture,
                 target=target)
        i += 1

if __name__ == '__main__':
    main()
