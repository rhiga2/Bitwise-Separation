import numpy as np
import librosa
import scipy.signal as signal
import denoising_data
from denoising_data import DenoisingDataset
import matplotlib.pyplot as plt
import mir_eval
import pdb
import numpy as np
import matplotlib.pyplot as plt
from deltasigma import *

class PCM2PDM(object):
    def __init__(self, old_sr, os = 64, order = 2, symmetric_output = False):
        self.old_sr = old_sr
        self.os = os
        self.ntf = synthesizeNTF(order, os, 0)
        self.symmetric_output = symmetric_output

    def __call__(self, pcm):
        u = pcm[np.floor(np.arange( 0, len(pcm), 1/self.os)).astype(int)]
        pdm = simulateDSM(u, self.ntf)[0]
        if not self.symmetric_output:
            return (pdm + 1) // 2
        return pdm

class PDM2PCM(object):
    def __init__(self, os = 64, symmetric_input = False):
        self.os = os
        self.firwin = signal.firwin(2 * os, 1 / os)
        self.symmetric_input = symmetric_input

    def __call__(self, pdm):
        os = self.os
        if not self.symmetric_input:
            pdm = 2 * pdm - 1
        if len(pdm.shape) == 1:
            firwin = self.firwin
            output = signal.convolve(pdm, firwin)
            output = output[os + os // 2 : -os : os]
        else:
            firwin = np.expand_dims(self.firwin, 0)
            output = signal.convolve(pdm, firwin)
            output = output[:, os + os // 2 : -os : os]
        return output

def test1():
    sr = 32
    os = 64
    pcm2pdm = PCM2PDM(sr, os)
    pdm2pcm = PDM2PCM(os)
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
    pcm, sr = librosa.core.load('./results/partita_original.wav', sr=16000)
    pdm = pcm2pdm(pcm)
    print('PCM Shape: ', pcm.shape)
    print('PDM Shape: ', pdm.shape)
    new_pcm = pdm2pcm(pdm)
    librosa.output.write_wav('./results/partita_recovered.wav', new_pcm, sr, norm=True)

def test3():
    sr = 16000
    os = 64

    pcm2pdm = PCM2PDM(sr, os)
    pdm2pcm = PDM2PCM(os)

    speaker_path = '/media/data/timit-wav/train'
    noise_path = '/media/data/noises-16k'
    train_speeches, val_speeches, test_speeches = denoising_data.get_speech_files(speaker_path, 7, 7, 2)
    train_noises, val_noises, test_noises = denoising_data.get_noise_files(noise_path, 12, 6, 3)
    trainset = DenoisingDataset(train_speeches, train_noises, transform=pcm2pdm)

    pdb.set_trace()
    mixture, speech, noise = trainset[0]
    recovered_mix = pdm2pcm(mixture)
    recovered_speech = pdm2pcm(speech)
    recovered = np.append(recovered_mix, recovered_speech)
    librosa.output.write_wav('results/recovered.wav', recovered, sr)

    trainset = DenoisingDataset(train_speeches, train_noises)
    mixture, speech, noise = trainset[0]
    sample = np.append(mixture, speech)
    librosa.output.write_wav('results/sample.wav', sample, sr, norm=True)

def main():
    sr = 16000
    os = 64

    pcm2pdm = PCM2PDM(sr, os)
    pdm2pcm = PDM2PCM(os)

    speaker_path = '/media/data/timit-wav/train'
    noise_path = '/media/data/noises-16k'

    train_speeches, val_speeches, test_speeches = denoising_data.get_speech_files(speaker_path, 7, 7, 2)
    train_noises, val_noises, test_noises = denoising_data.get_noise_files(noise_path, 12, 6, 3)
    trainset = DenoisingDataset(train_speeches, train_noises, transform=pcm2pdm)
    valset = DenoisingDataset(val_speeches, val_noises, transform=pcm2pdm)
    testset = DenoisingDataset(test_speeches, test_noises, transform=pcm2pdm)

    print('Length Training Set: ', len(trainset))
    print('Get Validation Set: ', len(valset))

    # Test PDM-PCM conversion
    mixture, speech, noise = trainset[0]
    recovered_mix = pdm2pcm(mixture)
    librosa.output.write_wav('results/recovered.wav', recovered_mix, sr, norm=True)

    lengths = []
    for i in range(len(trainset)):
        mixture, speech, noise = trainset[i]
        mixture = mixture.astype(bool)
        speech = speech.astype(bool)
        noise = noise.astype(bool)
        np.savez('/media/data/bitwise_pdm/train%d' % (i,), mixture=mixture,
                 speech=speech, noise=noise)
        lengths.append(speech.shape[0])
    print('Min Length: ', min(lengths))

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
    test3()
