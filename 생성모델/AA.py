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
    self.input = 11
    self.output= 1
    self.model = nn.Sequential(
        nn.Linear(self.input, 1024), # input size, hidden size
        nn.LeakyReLU(0.2),
        nn.Dropout(0.2),
        nn.Linear(1024, 512),
        nn.LeakyReLU(0.2),
        nn.Linear(512, 256),
        nn.LeakyReLU(0.2),
        nn.Dropout(0.2),
        nn.Linear(256, 128),
        nn.LeakyReLU(0.2),
        nn.Linear(128, 64),
        nn.Linear(64, self.output), # hidden size, output size
        nn.Sigmoid()
    ).to(device)

  def forward(self, x):
    x = self.model(x)
    return x

class Generator(nn.Module):
  def __init__(self):
    super(Generator, self).__init__()
    self.features = 11 # 피쳐수
    self.output = 11 # 데이터수
    self.model = nn.Sequential(
        nn.Linear(self.features, 64), # input size, hidden size
        nn.LeakyReLU(0.2),
        nn.Dropout(0.2),
        nn.Linear(64, 128),
        nn.LeakyReLU(0.2),
        nn.Linear(128, self.output), # hidden size, output size
        nn.Tanh()
    ).to(device)

  def forward(self, x):
    x = self.model(x)
    return x

# 모델 정의
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

generator = Generator().to(device)
discriminator = Discriminator().to(device)

g_optim = optim.Adam(generator.parameters(), lr=1e-4)
d_optim = optim.Adam(discriminator.parameters(), lr=1e-4)

criterion = nn.BCELoss().to(device)
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer=g_optim, mode='min', verbose=True, patience=10, factor=0.5)

import time
n_epochs = 1000
#sample_interval = 100 # 몇 번의 배치마다 결과를 출력할 것인가
noise = 11
start_time = time.time()
newdata = []
testdata = []
for epoch in range(n_epochs):
  #test = pd.read_csv('drive/MyDrive/dacon/gan/상추/train_input/CASE_01.csv')
  #dftonp = test.iloc[:,:2].values
  test = np.array([[0,10,0,10,0,10,0,10,0,10,0]])
  nptonn = torch.from_numpy(test).float()

  real = torch.cuda.FloatTensor(nptonn.size(0), 1).fill_(1.0) # 진짜
  fake = torch.cuda.FloatTensor(nptonn.size(0), 1).fill_(0.0) # 가짜

  real_data = nptonn.cuda()
  #print(real_data.size())
  g_optim.zero_grad()
  z1 = torch.normal(mean=0, std=1, size=(nptonn.shape[0], noise)).cuda() # random noise
  z2 = torch.normal(mean=0, std=1, size=(nptonn.shape[0], noise)).cuda() # random noise
  z = torch.cat([z1,z2], dim=1) #[M, N+N, K]
  #z = real_data

  #print('noise',z.size())
  generated_dis = generator(z1) # create distribution
  generated_dis_value = generated_dis.detach().cpu()
  #print(generated_dis.detach().size())
  # criterion.cuda()
  g_loss =  criterion(discriminator(generated_dis), real) # calculate generator loss
  
  # update generator
  g_loss.backward()
  g_optim.step()

  # update discriminator
  real_loss = criterion(discriminator(real_data), real)
  r_score = discriminator(real_data).mean()
  fake_loss = criterion(discriminator(generated_dis.detach()), fake)
  g_score = discriminator(generated_dis).mean()
  d_loss = (real_loss + fake_loss) / 2

  if  g_score > 0.99 :
    newdata.append(generated_dis)
    testdata.append(real_data)

  d_loss.backward()
  d_optim.step()

  print(f"[Epoch {epoch}/{n_epochs}] [D loss: {d_loss.item():.6f}] [G loss: {g_loss.item():.6f}] [FAKE loss: {fake_loss.item():.6f}] ",
        g_score, r_score)
