import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import argparse
import glob2
import tqdm
import mir_eval
import pdm_data
import numpy as np

'''
Real network trained for denoising
'''
class BitwiseDataset(Dataset):
    def __init__(self, path):
        self.trainfiles = glob2.glob(path + '/train*.npz')
        self.valfiles = glob2.glob(path + '/val*.npz')

    def __len__(self):
        return len(self.trainfiles)

    def lenval(self):
        return len(self.valfiles)

    def __getitem__(self, i):
        dfile = self.trainfiles[i]
        data = np.load(dfile)
        mix = data['mixture'].astype(float)
        speech = data['speech'].astype(float)
        return {'targets': torch.FloatTensor(speech),
                'features' : torch.FloatTensor(mix)}

    def getvalset(self):
        mixes = []
        targets = []
        for f in self.valfiles:
            data = np.load(f)
            mixes.append(torch.FloatTensor(data['mixture'].astype(float)))
            targets.append(torch.FloatTensor(data['target'].astype(float)))
        return mixes, targets

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
        self.linear_bn1 = nn.BatchNorm1d(2 * transform_size)
        self.linear2 = nn.Linear(2 * transform_size, transform_size)
        self.conv_transpose = nn.ConvTranspose1d(transform_size, 1, transform_size, stride=hop)

    def forward(self, X):
        # (batch, 1, time)
        x = x.unsqueeze(1)
        h = self.activation(self.conv1d(x))
        h = self.conv_bn1(h)

        # (batch, transform, frame) to (batch, 1, transform, frame)
        h = h.unsqueeze(1)
        h = self.activation(self.smooth(h))
        h = self.conv_bn2(H)

        # (batch, channels, transform, frames) to (batch, frames, channels * transform)
        batch_size, _, _, frames = h.size()[0]
        h = h.permute(0, 3, 1, 2).view(batch_size, frames, -1).contiguous()
        h = self.activation(self.linear1(h))
        h = self.linear_bn1(h)
        h = self.linear2(h)

        # (batch, frames, transform) to (batch, transform, frames)
        h = h.permute(0, 2, 1).contiguous()
        y = self.conv_transpose(h)
        y = y.squeeze(1)
        return y

def evaluate(mixture, speech, speech_estimate, noise):
    noise_estimate = mixture - speech_estimate
    references = np.stack(speech, noise)
    estimates = np.stack(speech_estimate, noise_estimate)
    sdr, sir, sar = mir_eval.separation.bss_eval_sources(references, estimates,
    compute_permutation=False)
    return sdr[0], sir[0], sar[0]

def main():
    parser = argparse.ArgumentParser( description='Bitwise Network')
    parser.add_argument('--epochs', '-e', type=int, default=10000,
                        help='Number of epochs')
    parser.add_argument('--learningrate', '-lr', type=float, default=1e-4,
                        help='Learning Rate')
    parser.add_argument('--batchsize', '-b', type=int, default=16,
                        help='Batch Size')
    args = parser.parse_args()

    # Dataset
    dataset = BitwiseDataset('/media/data/bitwise_pdm')
    dataloader = DataLoader(dataset, batch_size = args.batchsize, shuffle = True)

    # Instantiate Network
    net = SeparationNetwork()
    net = torch.nn.DataParallel( net, device_ids=[0,1])
    net = net.cuda()

    # Instantiate progress bar
    # progess_bar = tqdm.trange(args.epochs)
    pcm = pdm_data.PulseCodingModulation()

    # Instantiate optimizer and loss
    torch.optim.Adam( filter(lambda p: p.requires_grad, net.parameters()), lr=args.learningrate)
    criterion = nn.MSELoss()
    mixtures, targets = dataset.getvalset()

    for epoch in range(args.epochs):
        train_loss = 0
        net.train()
        for batch_count, batch in enumerate(dataloader):
            x = Variable(batch['features'])
            y = Variable(batch['targets'])
            est = net(x)

            loss = criterion(est, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.data.numpy()[0]

        # print('Train Loss ': train_loss)

        if (epoch + 1) % 10 == 0:
            val_loss = 0
            net.eval()
            for i in range(dataset.lenval()):
                x = Variable(mixtures[i])
                y = Variable(targets[i][0])
                est = net(x)

                loss = criterion(est, y)
                val_loss += loss.data.numpy()[0]
                mixture =  pcm(x.data.numpy()[0])
                speech = pcm(y.data.numpy()[0])
                speech_estimate = pcm(est.data.numpy()[0])
                noise = pcm(targets[i][1].data.numpy()[0])
                sdr, sir, sar = evaluate(mixture, speech, speech_estimate, noise)
                print('Validation Metrics: ', sdr, sir, sar)

if __name__ == '__main__':
    main()
