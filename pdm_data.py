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
        return np.array(y)

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
        return y


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
    num_speakers=12, noise_set=noise_set, transform=pdm)
    print('Length Training Set: ', len(dataset))
    print('Length Validation Set: ', len(valset))

    # Test PDM-PCM conversion
    mixture, saudio = dataset[0]
    mixture = mixture.numpy()
    recovered_mix = pcm(pdm_mix)
    librosa.output.write_wav('results/recovered.wav', recovered_mix, sr, norm = True)

    for i, mixture, speech in enumerate(dataset):
        np.savez('/media/data/bitwise_pdm/train%d' % (i,), mixture=mixture,
                 speech=speech)

    valset = dataset.getvalset()
    for i, data in enumerate(valset):
        mixture, speech, noise = data
        np.savez('/media/data/bitwise_pdm/val%d' % (i,), mixture=mixture,
                 speech=speech, noise=noise)

if __name__ == '__main__':
    main()
