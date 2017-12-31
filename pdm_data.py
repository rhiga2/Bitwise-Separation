import numpy as np
import librosa
import scipy.signal as signal
from denoising_data import DenoisingDataset
import matplotlib.pyplot as plt
import mir_eval
import pdb
import numpy as np
import matplotlib.pyplot as plt
from deltasigma import *

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

class PCM2PDM(object):
    def __init__(self, old_sr, os = 64, order = 2):
        self.old_sr = old_sr
        self.os = os
        self.ntf = synthesizeNTF(order, os, 0)

    def __call__(self, pcm):
        u = pcm[np.floor(np.arange( 0, len(pcm), 1/self.os)).astype(int)]
        return simulateDSM(u, self.ntf)[0]

class PDM2PCM(object):
    def __init__(self, os = 64):
        self.os = os
        self.firwin = signal.firwin(2 * os, 1 / os)

    def __call__(self, pdm):
        os = self.os
        return signal.convolve(pdm, self.firwin)[os + os // 2 : -os : os]

class PulseCodingModulation(object):
    def __init__(self, downsample_factor = 64, symmetric=False):
        self.downsample_factor = downsample_factor
        self.symmetric = symmetric

    def __call__(self, x):
        '''
        Converts pulse density modulation to pulse coding modulation.
        '''
        y = x
        if not self.symmetric:
            y = 2 * y - 1
        y = signal.decimate(y, self.downsample_factor, ftype = 'fir', zero_phase = True)
        #for i in range(2):
        #    y = signal.decimate(y, 8, 8, ftype = 'iir', zero_phase = True)
        return y.astype(np.float32)

def test1():
    sr = 32
    os = 32
    pcm2pdm = PCM2PDM(sr)
    pdm2pcm = PDM2PCM()
    pcm_t = np.linspace(0, 7, 7 * sr)
    pdm_t = np.linspace(0, 7, 7 * sr * os)
    pcm = np.sin(pcm_t)
    pdm = pcm2pdm(pcm)
    recovered_pcm = pdm2pcm(pdm)
    print('PCM Shape: ', pcm.shape)
    print('PDM Shape: ', pdm.shape)
    print('Recovered PCM Shape: ', recovered_pcm.shape)
    plt.plot(pcm_t, pcm)
    plt.plot(pdm_t, pdm)
    plt.plot(pcm_t, recovered_pcm)
    plt.show()

def test2():
    sr = 16000
    pcm2pdm = PCM2PDM(sr)
    pdm2pcm = PDM2PCM()
    pcm, sr = librosa.core.load('./results/partita_original.wav', sr = 16000)
    pcm = pcm / np.max(np.abs(pcm))
    pdm = pcm2pdm(pcm)
    new_pcm = pdm2pcm(pdm)
    librosa.output.write_wav('./results/partita_recovered.wav', new_pcm, sr, norm=True)

def main():
    sr = 16000
    oversample = 32

    pdm = PulseDensityModulation(sr, oversample * sr)
    pcm = PulseCodingModulation(oversample)
    speaker_path = '/media/data/timit-wav/train'
    noise_path = '/media/data/noises-16k'

    train_speeches, val_speeches, test_speeches = get_speech_files(speaker_path, 7)
    train_noises, val_noises, test_noises = get_noise_files(noise_path)
    trainset = DenoisingDataset(train_speeches, train_noises, transform=pdm)
    valset = DenoisingDataset(val_speeches, val_noises, transform=pdm)
    testset = DenoisingDataset(test_speeches, test_noises, transform=pdm)

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
        mixture, speech, noise = valset[i]
        mixture = mixture.astype(bool)
        speech = speech.astype(bool)
        noise = noise.astype(bool)
        np.savez('/media/data/bitwise_pdm/val%d' % (i,), mixture=mixture,
                 speech=speech, noise=noise)

    for i in range(len(testset)):
        mixture, speech, noise = testset[i]
        mixture = mixture.astype(bool)
        speech = speech.astype(bool)
        noise = noise.astype(bool)
        np.savez('/media/data/bitwise_pdm/test%d' % (i,), mixture=mixture,
                 speech=speech, noise=noise)

if __name__ == '__main__':
    test2()
