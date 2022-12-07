import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from matplotlib import pyplot as plt


class Generator(nn.Module):
    # 생성모델 정의
    def __init__(self):
        super(Generator, self).__init__()

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(100, 128, normalize=False),
            *block(128, 256, normalize=False),
            *block(256, 512, normalize=False),
            *block(512, 258, normalize=False),
            nn.Linear(258, 166),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.model(x)
        return x


class Discriminator(nn.Module):
    # 식별모델 정의
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(166, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        validity = self.model(x)

        return validity


def extract(v):
    return v.data.storage().tolist()


def stats(d):
    return [np.mean(d), np.std(d)]


df = pd.read_csv("./train.csv")
# hyper parameter
G = Generator()
D = Discriminator()
loss = nn.BCELoss()
epochs = 10000
d_optimizer = optim.Adam(D.parameters(), lr=1e-3, betas=(0.9, 0.1))
g_optimizer = optim.Adam(G.parameters(), lr=1e-3, betas=(0.9, 0.1))

for epoch in range(epochs):

    # define label (feature의 개수만큼 label 생성)
    # true_label = torch.FloatTensor(df[:1].size, 1).fill_(1.0)
    # fake_label = torch.FloatTensor((df[:1].size, 1)).fill_(1.0)

    true_label = torch.ones(166)
    fake_label = torch.zeros(166)

    true_data = torch.tensor(df.iloc[1, ].values).float()

    # -----------------
    #  Train Generator
    # -----------------

    g_optimizer.zero_grad()

    # noise as generator input
    noise = Variable(torch.rand(100))

    generate_data = G(noise)
    loss_generator = loss(D(generate_data), torch.ones(1))
    loss_generator.backward()
    g_optimizer.step()

    # ---------------------
    #  Train Discriminator
    # ---------------------

    d_optimizer.zero_grad()

    real_loss = loss(D(true_data), torch.ones(1))
    fake_loss = loss(D(generate_data.detach()), torch.zeros(1))

    loss_discriminator = (real_loss+fake_loss) / 2

    loss_discriminator.backward()
    d_optimizer.step()

    print(f'Discriminator loss: {loss_discriminator}, Generator loss: {loss_generator}')
