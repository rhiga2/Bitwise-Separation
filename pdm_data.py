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
    def __init__(self, downsample_factor = 64, symmetric=False):
        self.downsample_factor = downsample_factor
        self.symmetric = symmetric

    def __call__(self, x):
        '''
        Converts pulse density modulation to pulse coding modulation.
        '''
        if not self.symmetric:
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
    train_noise = ['babble-16k.wav', 'street-16k.wav', 'car-16k.wav',
                 'restaurant-16k.wav', 'subway-16k.wav']
    val_noise = ['bus-16k.wav', 'airport-16k.wav']
    speaker_set = ['fcjf0', 'fdml0', 'fetb0', 'mcpm0', 'mdpk0']

    trainset = DenoisingDataset(speaker_path, noise_path, duration=3,
    speaker_set=speaker_set, noise_set=train_noise, transform=pdm)
    valset = DenoisingDataset(speaker_path, noise_path, duration=None,
    speaker_set=speaker_set, noise_set=val_noise, transform=pdm)
    print('Length Training Set: ', len(trainset))
    print('Get Validation Set: ', len(valset))

    # Test PDM-PCM conversion
    mixture, saudio = trainset[0]
    mixture = mixture.numpy()
    recovered_mix = pcm(mixture)
    librosa.output.write_wav('results/recovered.wav', recovered_mix, sr, norm = True)

    for i in range(len(trainset)):
        mixture, speech, noise = trainset[i]
        mixture = mixture.astype(bool)
        speech = speech.astype(bool)
        noise = noise.astype(bool)
        np.savez('/media/data/bitwise_pdm/train%d' % (i,), mixture=mixture,
                 speech=speech, noise=noise)

    for i in range(len(valset)):
        mixture, speech, noise = trainset[i]
        mixture = mixture.astype(bool)
        speech = speech.astype(bool)
        noise = noise.astype(bool)
        np.savez('/media/data/bitwise_pdm/val%d' % (i,), mixture=mixture,
                 speech=speech, noise=noise)

if __name__ == '__main__':
    main()
