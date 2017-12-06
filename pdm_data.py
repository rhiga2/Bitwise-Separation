import numpy as np
import librosa
import scipy.signal as signal
from denoising_data import DenoisingDataset

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
        x = librosa.core.resample(x, self.old_sr, self.new_sr)
        y = []
        qe = [0] # quantization error
        for i in range(x.shape[0]):
            if x[i] >= qe[i]:
                y.append(1)
            else:
                y.append(-1)
            qe.append(y[i] - x[i] + qe[i])
        y = (np.array(y) + 1) // 2
        return np.array(y)

class PulseCodingModulation(object):
    def __init__(self, downsample_factor):
        self.downsample_factor = downsample_factor

    def __call__(self, x):
        '''
        Converts pulse density modulation to pulse coding modulation.
        '''
        x = 2 * x - 1
        y = signal.decimate(x, self.downsample_factor, ftype = 'fir')
        return y


def main():
    speaker_path = '/media/data/timit-wav/train/dr1'
    noise_path = '/media/data/noises-16k'
    noise_set = ['babble-16k.wav', 'street-16k.wav', 'car-16k.wav',
                 'restaurant-16k.wav', 'subway-16k.wav']
    dataset = DenoisingDataset(speaker_path, noise_path, noise_set = noise_set)
    print('Length: ', len(dataset))

    # Test PDM-PCM conversion
    pdm = PulseDensityModulation(16000, 1024000)
    pcm = PulseCodingModulation(64)
    mixture, saudio = dataset[0]
    mixture = mixture.numpy()
    pdm_mix = pdm(mixture)
    recovered_mix = pcm(pdm_mix)
    librosa.output.write_wav('results/sample_pdm.wav', recovered_mix, 16000, norm = True)
