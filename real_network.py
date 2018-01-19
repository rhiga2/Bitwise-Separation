import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable

import argparse
import glob2
import tqdm
import mir_eval
import pdm_data
import librosa
import visdom
import pdb

'''
Real network trained for denoising
'''
class BitwiseDataset(Dataset):
    def __init__(self, pattern, length=None, hop=256):
        self.files = glob2.glob(pattern)
        self.length = length
        self.stride = stride

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        dfile = self.files[i]
        data = np.load(dfile)
        mix = data['mixture'].astype(np.float32)
        speech = data['speech'].astype(np.float32)
        noise = data['noise'].astype(np.float32)
        if self.length:
            start = np.random.randint(0, mix.shape[0] - self.length)
            mix = mix[start : self.length + start]
            speech = speech[start : self.length + start]
            noise = noise[start : self.length + start]
        elif self.stride:
            length = (len(mix) // hop) * hop
            mix = mix[:length]
            speech = speech[:length]
            noise = noise[:length]
        return {'noise': noise, 'speech' : mix, 'mixture' : 2 * mix - 1}

class SeparationNetwork(nn.Module):
    def __init__(self, transform_size=1024, num_channels=3,
                 hidden_sizes=[512, 512], hop=256, dropout=0., activation=F.relu,
                 fft_init=True):
        super(SeparationNetwork, self).__init__()
        self.transform_size = transform_size
        self.hop = hop
        self.activation = activation

        # Convolutional transform initialized to FFT
        self.transform1d = nn.Conv1d(1, transform_size, transform_size, stride=hop)
        if fft_init:
            params = list(self.transform1d.parameters())
            window = np.sqrt(np.hanning(transform_size))
            fft = np.fft.fft(np.eye(transform_size))
            fft = np.vstack((np.real(fft[:int(transform_size/2),:]), np.imag(fft[:int(transform_size/2),:])))
            params[0].data = torch.FloatTensor(window * fft).unsqueeze(1)

        self.conv_bn1 = nn.BatchNorm1d(transform_size)
        self.smooth = nn.Conv2d(1, num_channels, 7, stride=1, padding=3)
        self.conv_bn2 = nn.BatchNorm2d(num_channels)

        # Fully connected layers
        self.linear1 = nn.Linear(num_channels * transform_size, 2 * transform_size)
        self.linear_bn1 = nn.BatchNorm1d(2 * transform_size)
        self.linear2 = nn.Linear(2 * transform_size, transform_size)

        # Convolutional transpose initialized to inverse FFT
        self.conv_transpose = nn.ConvTranspose1d(transform_size, 1, transform_size, stride=hop)
        if fft_init:
            params = list(self.conv_transpose.parameters())
            params[0].data = torch.FloatTensor(window * np.linalg.pinv(fft).T).unsqueeze(1)

    def forward(self, x):
        # (batch, 1, time)
        x = x.unsqueeze(1)
        x = self.activation(self.transform1d(x))
        x = self.conv_bn1(x)

        # (batch, transform, frame) to (batch, 1, transform, frame)
        x = x.unsqueeze(1)
        x = self.activation(self.smooth(x))
        x = self.conv_bn2(x)

        # (batch, channels, transform, frames) to (batch, frames, channels * transform)
        batch_size, _, _, frames = x.size()
        x = x.permute(0, 3, 1, 2).contiguous().view(batch_size, frames, -1)
        x = self.activation(self.linear1(x))
        x = x.permute(0, 2, 1).contiguous()
        x = self.linear_bn1(x)
        x = x.permute(0, 2, 1).contiguous()
        x = self.linear2(x)

        # (batch, frames, transform) to (batch, transform, frames)
        x = x.permute(0, 2, 1).contiguous()
        x = self.conv_transpose(x)
        x = x.squeeze(1)
        return x

class SignalDistortionRatio(nn.Module):
    def __init__(self, l1_penalty=0, epsilon = 2e-7):
        super(SignalDistortionRatio, self).__init__()
        self.epsilon = epsilon

    def forward(self, prediction, target):
        sdr = -torch.mean(prediction * target)**2 / (torch.mean(prediction**2) + self.epsilon)
        return sdr

def plot_loss_and_metrics(train_loss, val_loss, sdr, sar, sir, vis,
    output_period = 10):
    loss = [
        # Training sLoss
        dict(x=list(range(0, output_period * len(train_loss), output_period)),
             y=train_loss, name='Training Loss', hoverinfo='name+y+lines',
             line=dict(width=1), mode='lines', type='scatter'),

        # Validation loss
        dict(x=list(range(0, output_period * len(val_loss), output_period)),
             y=val_loss, name='Validation Loss', hoverinfo='name+y+lines',
             line=dict(width=1), mode='lines', type='scatter')
    ]
    loss_layout = dict(
        showlegend=True,
        legend=dict(orientation='h', y=1.1, bgcolor='rgba(0,0,0,0)'),
        margin=dict(r=30, b=40, l=50, t=50),
        font=dict(family='Bell Gothic Std'),
        xaxis=dict(autorange=True, title='Epochs'),
        yaxis=dict(autorange=True, title='Loss'),
        title='Losses',
    )
    vis._send(dict(data=loss, layout=loss_layout, win='Loss', eid='Ryley BNN'))

    # BSS_EVAL plots
    bss = [
        # SDR
        dict(x=list(range(0, output_period * len(sdr), output_period)),
             y=sdr, name='SDR', hoverinfo='name+y+lines', line=dict(width=1),
             mode='lines', type='scatter'),

        # SIR
        dict(x=list(range(0, output_period * len(sir), output_period)),
             y=sir, name='SIR', hoverinfo='name+y+lines', line=dict( width=1),
             mode='lines', type='scatter'),
        # SAR
        dict(x=list(range(0, output_period * len(sar), output_period)),
             y=sar, name='SAR', hoverinfo='name+y+lines', line=dict(width=1),
             mode='lines', type='scatter'),
    ]
    bss_layout = dict(
        showlegend=True,
        legend=dict(orientation='h', y=1.05, bgcolor='rgba(0,0,0,0)'),
        margin=dict(r=30, b=40, l=50, t=50),
        font=dict(family='Bell Gothic Std'),
        xaxis=dict(autorange=True, title='Epochs'),
        yaxis=dict(autorange=True, title='dB'),
        title='BSS_EVAL'
    )

    vis._send(dict(data=bss, layout=bss_layout, win='BSS', eid='Ryley BNN'))

def evaluate(speech, speech_estimate, noise, noise_estimate):
    references = np.concatenate((speech, noise), axis=0)
    estimates = np.concatenate((speech_estimate, noise_estimate), axis=0)
    sdr, sir, sar, _ = mir_eval.separation.bss_eval_sources(references, estimates,
    compute_permutation=False)
    return sdr[0], sir[0], sar[0]

def main():
    parser = argparse.ArgumentParser(description='Bitwise Network')
    parser.add_argument('--epochs', '-e', type=int, default=10000,
                        help='Number of epochs')
    parser.add_argument('--learningrate', '-lr', type=float, default=1e-4,
                        help='Learning Rate')
    parser.add_argument('--batchsize', '-b', type=int, default=4,
                        help='Batch Size')
    parser.add_argument('--weightdecay', '-wd', type=float, default=0,
                        help='L2 Regularization Constant')
    args = parser.parse_args()

    datapath = '/media/data/bitwise_pdm'

    # Dataset
    trainset = BitwiseDataset(datapath + '/train*.npz', length=1644544)
    valset = BitwiseDataset(datapath + '/val*.npz')
    trainloader = DataLoader(trainset, batch_size=args.batchsize, shuffle=True)
    valloader = DataLoader(valset, batch_size=1, shuffle=True)

    # Instantiate Network
    net = SeparationNetwork()
    net = torch.nn.DataParallel(net, device_ids=[0])
    net = net.cuda().float()

    # Instantiate progress bar
    progress_bar = tqdm.trange(args.epochs)
    pdm2pcm = pdm_data.PDM2PCM()

    # Instantiate optimizer and loss
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad,
                                 net.parameters()), lr=args.learningrate,
                                 weight_decay=args.weightdecay)

    criterion = nn.BCEWithLogitsLoss()

    # Instantiate Visdom
    vis = visdom.Visdom(port=5800)

    sdr, sir, sar = (0, 0, 0)
    train_loss_history = []
    val_loss_history = []
    val_sdr_history = []
    val_sar_history = []
    val_sir_history = []
    output_period = 1
    for epoch in progress_bar:
        train_loss = 0
        net.train()
        for batch_count, batch in enumerate(trainloader):
            x = Variable(batch['mixture'])
            y = Variable(batch['speech'])
            estimates = net(x).cpu()

            loss = criterion(estimates, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.data.cpu().float().numpy()[0]

        train_loss = train_loss / (batch_count + 1)

        if (epoch + 1) % output_period == 0:
            val_loss = 0
            val_sdr, val_sir, val_sar = (0, 0, 0)
            net.eval()
            for batch_count, batch in enumerate(valloader):
                x = Variable(batch['mixture'])
                y = Variable(batch['speech'])
                estimates = net(x).cpu()

                loss = criterion(estimates, y)
                val_loss += loss.data.cpu().float().numpy()[0]
                mixture =  pdm2pcm(batch['mixture'].numpy() > 0)
                speech = pdm2pcm(batch['speech'].numpy())
                pred = estimates.data.cpu().float().numpy() > 0
                speech_estimate = pdm2pcm(pred)
                noise = asym_pdm2pcm(batch['noise'].numpy())
                new_sdr, new_sir, new_sar = evaluate(speech, speech_estimate,
                                                     noise, mixture - speech_estimate)
                val_sdr += new_sdr
                val_sir += new_sir
                val_sar += new_sar

            val_loss = val_loss / (batch_count + 1)
            val_sdr = val_sdr / (batch_count + 1)
            val_sir = val_sir / (batch_count + 1)
            val_sar = val_sar / (batch_count + 1)

            train_loss_history.append(train_loss)
            val_loss_history.append(val_loss)
            val_sdr_history.append(val_sdr)
            val_sar_history.append(val_sar)
            val_sir_history.append(val_sir)

            output = np.append(mixture, speech)
            output = np.append(output, speech_estimate)
            librosa.output.write_wav('results/sample_output.wav', output, 16000)

            plot_loss_and_metrics(train_loss_history, val_loss_history,
                                  val_sdr_history, val_sar_history, val_sir_history, vis,
                                  output_period)

        progress_bar.set_description('L:%.3f P:%.1f/%.1f/%.1f' % \
              (train_loss, val_sdr, val_sir, val_sar))

if __name__ == '__main__':
    main()
