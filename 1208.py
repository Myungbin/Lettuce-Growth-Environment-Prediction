import os
import pickle
import matplotlib.pyplot as plt
import torch
import torch.optim as optim 
from torch import nn
import pandas as pd
import numpy as np
from torch.optim import lr_scheduler

# 노이즈 -> 피쳐수만큼
# generator input -> 노이즈가 들어감
# generator ouput -> real data x 형태로 나와야 함.
# discriminator input -> real data x 가 들어감
# discriminator output -> 1

class Discriminator(nn.Module):
  def __init__(self):
    super(Discriminator, self).__init__()
    self.input = 9 
    self.output= 1
    self.model = nn.Sequential(
        nn.Linear(self.input, 16), # input size, hidden size
        #nn.LeakyReLU(0.2),
        nn.Sigmoid(),
        nn.Dropout(0.3),
        nn.Linear(16, 64),
        #nn.LeakyReLU(0.2),
        nn.Sigmoid(),
        nn.Linear(64, self.output), # hidden size, output size
        nn.Sigmoid()
    ).to(device)

  def forward(self, x):
    #print(x.size())
    x = self.model(x)
    return x

class Generator(nn.Module):
  def __init__(self):
    super(Generator, self).__init__()
    self.features = 9 # 피쳐수
    self.output = 9  # 데이터수
    self.model = nn.Sequential(
        nn.Linear(self.features, 16), # input size, hidden size
        nn.Sigmoid(),
        nn.Dropout(0.2),
        nn.Linear(16, 16),
        #nn.LeakyReLU(0.2),
        nn.Linear(16,self.output), # hidden size, output size
        nn.Sigmoid()
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
noise = 9
start_time = time.time()
newdata = []
testdata = []
#print(real_data)
for epoch in range(n_epochs):

  test = day0.iloc[:28].values
  
  nptonn = torch.from_numpy(newday0).float()

  real = torch.cuda.FloatTensor(nptonn.size(0), 1).fill_(1.0) # 
  fake = torch.cuda.FloatTensor(nptonn.size(0), 1).fill_(0.0) # 

  real_data = nptonn.cuda()
  g_optim.zero_grad()

  z0 = torch.normal(mean=11.5, std=6.9, size=(nptonn.shape[0], 1)).cuda() # random noise
  z1 = torch.normal(mean=25.78, std=4.3, size=(nptonn.shape[0], 1)).cuda() # random noise
  z2 = torch.normal(mean=54.91, std=12.2, size=(nptonn.shape[0], 1)).cuda() # random noise
  z3 = torch.normal(mean=533.833, std=144.1, size=(nptonn.shape[0], 1)).cuda() # random noise
  z4 = torch.normal(mean=1.273, std=0.932, size=(nptonn.shape[0], 1)).cuda() # random noise
  z5 = torch.normal(mean=430.600, std=491.308, size=(nptonn.shape[0], 1)).cuda() # random noise
  z6 = torch.normal(mean=6765.408, std=9450.28, size=(nptonn.shape[0], 1)).cuda() # random noise
  z7 = torch.normal(mean=1309.564, std=2653.722, size=(nptonn.shape[0], 1)).cuda() # random noise
  z8 = torch.normal(mean=856.852, std=1938.17, size=(nptonn.shape[0], 1)).cuda() # random noise
  zz = torch.normal(mean=0, std=1, size=(nptonn.shape[0], noise)).cuda()

  z = torch.cat([z0,z1,z2,z3,z4,z5,z6,z7,z8], dim=1) #[M, N+N, K]
  #print(z.size())

  generated_dis = generator(z) # create distribution
  #print(generated_dis.size())
  #print(real_data.size())
  generated_dis_value = generated_dis.detach().cpu()
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

  newdata.append(generated_dis_value)

  d_loss.backward()
  d_optim.step()

  print(f"[Epoch {epoch}/{n_epochs}] [D loss: {d_loss.item():.6f}] [G loss: {g_loss.item():.6f}] [FAKE loss: {fake_loss.item():.6f}] ",
        g_score, r_score)
