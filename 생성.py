import os
import pickle
import matplotlib.pyplot as plt
import torch
import torch.optim as optim 
from torch import nn
import pandas as pd
import numpy as np
from torch.optim import lr_scheduler

class Discriminator(nn.Module):
  def __init__(self):
    super(Discriminator, self).__init__()
    self.input = 2
    self.hidden = 672
    self.output= 1
    self.model = nn.Sequential(
        nn.Linear(self.input, self.hidden), # input size, hidden size
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(self.hidden, int(self.hidden/2)),
        nn.ReLU(),
        nn.Linear(int(self.hidden/2),self.output), # hidden size, output size
        nn.Sigmoid()
    ).to(device)

  def forward(self, x):
    #print(x.size())
    x = self.model(x)
    return x

class Generator(nn.Module):
  def __init__(self):
    super(Generator, self).__init__()
    self.input = 2
    self.output = 2
    self.hidden = 128
    self.model = nn.Sequential(
        nn.Linear(self.input, self.hidden), # input size, hidden size
        nn.LeakyReLU(0.2),
        nn.Dropout(0.3),
        nn.Linear(self.hidden, int(self.hidden/2)),
        nn.LeakyReLU(0.2),
        nn.Linear(int(self.hidden/2),self.output), # hidden size, output size
        nn.Tanh()
    ).to(device)

  def forward(self, x):
    #print('generator',x.size())
    x = self.model(x)
    return x

# 모델 정의
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

generator = Generator().to(device)
discriminator = Discriminator().to(device)

g_optim = optim.Adam(generator.parameters(), lr=2e-4)
d_optim = optim.Adam(discriminator.parameters(), lr=2e-4)

criterion = nn.BCELoss().to(device)
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer=g_optim, mode='min', verbose=True, patience=10, factor=0.5)

import time
n_epochs = 1000
sample_interval = 100 # 몇 번의 배치마다 결과를 출력할 것인가
noise = 1
start_time = time.time()
newdata = []
for epoch in range(n_epochs):
  dftonp = test.iloc[:,:2].values
  nptonn = torch.from_numpy(dftonp).float()

  real = torch.cuda.FloatTensor(nptonn.size(0), 1).fill_(1.0) # 진짜
  fake = torch.cuda.FloatTensor(nptonn.size(0), 1).fill_(0.0) # 가짜

  real_data = nptonn.cuda()
  
  g_optim.zero_grad()
  z1 = torch.normal(mean=13.5, std=8.083764, size=(nptonn.shape[0], noise)).cuda() # random noise
  z2 = torch.normal(mean=11.500000, std=6.927343, size=(nptonn.shape[0], noise)).cuda() # random noise
  z = torch.cat([z1,z2], dim=1) #[M, N+N, K]

  #print('noise',z.size())
  generated_dis = generator(z) # create distribution
  generated_dis_value = generated_dis.detach().cpu()

  # criterion.cuda()
  g_loss =  criterion(discriminator(generated_dis), real) # calculate generator loss
  
  # update generator
  g_loss.backward()
  g_optim.step()

  # update discriminator
  real_loss = criterion(discriminator(real_data), real)
  fake_loss = criterion(discriminator(generated_dis.detach()), fake)
  d_loss = (real_loss + fake_loss) / 2

  if  g_loss < 0.2 :
    newdata.append(generated_dis)

  d_loss.backward()
  d_optim.step()

  print(f"[Epoch {epoch}/{n_epochs}] [D loss: {d_loss.item():.6f}] [G loss: {g_loss.item():.6f}] [Elapsed time: {time.time() - start_time:.2f}s]")
