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
import visdom
import pdb

'''
Real network trained for denoising
'''
class BitwiseDataset(Dataset):
    def __init__(self, pattern):
        self.files = glob2.glob(pattern)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        dfile = self.files[i]
        data = np.load(dfile)
        mix = data['mixture'].astype(np.float32)
        speech = data['speech'].astype(np.float32)
        noise = data['noise'].astype(np.float32)
        return {'noise': noise, 'speech': speech, 'mixture' : mix}

class SeparationNetwork(nn.Module):
    def __init__(self, transform_size=1024, num_channels=5,
                 hidden_sizes=[512, 512], hop=256, dropout=0., activation=F.relu):
        super(SeparationNetwork, self).__init__()
        self.transform_size = transform_size
        self.hop = hop
        self.activation = activation

        # Double transform
        self.transform1d = nn.Conv1d(1, transform_size, transform_size, stride=hop)
        self.conv_bn1 = nn.BatchNorm1d(transform_size)
        self.smooth = nn.Conv2d(1, num_channels, 7, stride=1, padding=3)
        self.conv_bn2 = nn.BatchNorm2d(num_channels)

        # Fully connected layers
        self.linear1 = nn.Linear(num_channels * transform_size, 2 * transform_size)
        # self.linear_bn1 = nn.BatchNorm1d(2 * transform_size)
        self.linear2 = nn.Linear(2 * transform_size, transform_size)
        self.conv_transpose = nn.ConvTranspose1d(transform_size, 1, transform_size, stride=hop)

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
        # h = self.linear_bn1(h)
        x = self.linear2(x)

        # (batch, frames, transform) to (batch, transform, frames)
        x = x.permute(0, 2, 1).contiguous()
        x = self.conv_transpose(x)
        x = x.squeeze(1)
        return x

def evaluate(speech, speech_estimate, noise, noise_estimate):
    references = np.stack(speech, noise)
    estimates = np.stack(speech_estimate, noise_estimate)
    sdr, sir, sar = mir_eval.separation.bss_eval_sources(references, estimates,
    compute_permutation=False)
    return sdr[0], sir[0], sar[0]

def main():
    parser = argparse.ArgumentParser(description='Bitwise Network')
    parser.add_argument('--epochs', '-e', type=int, default=10000,
                        help='Number of epochs')
    parser.add_argument('--learningrate', '-lr', type=float, default=1e-4,
                        help='Learning Rate')
    parser.add_argument('--batchsize', '-b', type=int, default=16,
                        help='Batch Size')
    args = parser.parse_args()

    datapath = '/media/data/bitwise_pdm'

    # Dataset
    trainset = BitwiseDataset(datapath + '/train*.npz')
    valset = BitwiseDataset(datapath + '/val*.npz')
    trainloader = DataLoader(trainset, batch_size = args.batchsize, shuffle = True)
    valloader = DataLoader(valset, batch_size=1, shuffle=True)

    # Instantiate Network
    net = SeparationNetwork()
    net = torch.nn.DataParallel( net, device_ids=[0,1])
    net = net.cuda().half()

    # Instantiate progress bar
    progess_bar = tqdm.trange(args.epochs)
    pcm = pdm_data.PulseCodingModulation()

    # Instantiate optimizer and loss
    optimizer = torch.optim.Adam( filter(lambda p: p.requires_grad, net.parameters()), lr=args.learningrate)
    criterion = nn.MSELoss()

    # Instantiate Visdom
    vis = visdom.Visdom(port=5800)

    for epoch in progress_bar:
        train_loss = 0
        net.train()
        for batch_count, batch in enumerate(trainloader):
            x = Variable(batch['mixture'].cuda().half())
            y = Variable(batch['speech'].cuda().half())
            est = net(x)

            loss = criterion(est, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.data.cpu().float().numpy()[0]

        train_loss = train_loss / batch_count

        if (epoch + 1) % 10 == 0:
            val_loss = 0
            net.eval()
            for batch_count, batch in enumerate(valloader):
                batch = valset[i]
                x = Variable(batch['mixture'].cuda().half())
                y = Variable(batch['speech'].cuda.half())
                est = net(x)

                loss = criterion(est, y)
                val_loss += loss.data.cpu().float().numpy()[0]
                mixture =  pcm(batch['mixture'].numpy())
                speech = pcm(batch['speech'].numpy())
                speech_estimate = pcm(est.data.cpu().float().numpy())
                noise = pcm(batch['noise'].numpy())
                noise_estimate = mixture - speech_estimate
                sdr, sir, sar = evaluate(speech, speech_estimate, noise, noise_estimate)
                print('Validation Metrics: ', sdr, sir, sar)

        progress_bar.set_description('L:%.3f P:%.1f/%.1f/%.1f' % \
              (val_loss / batch_count, sdr, sir, sar))

        # win = vis.line([sdr, sir, sar], update='append')

if __name__ == '__main__':
    main()
