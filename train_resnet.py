import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, models, transforms
from torch.optim import lr_scheduler
from time import time
from dataloader import *

def train_resnet(dataset, epochs=10, batch_size=16, lr=1e-5):

  model = models.resnet18(pretrained=True)
  num_feat = model.fc.in_features
  model.fc = nn.Linear(num_feat, 8)
  model = model.to('cuda')

  crit = nn.CrossEntropyLoss()
  opt = optim.Adam(model.parameters(), lr=lr)
  loader = DataLoader(train, batch_size=batch_size, shuffle=True)

  model.train()

  start = time()
  losses = []
  for epoch in range(epochs):
    for batch, (image, label) in enumerate(loader):
      image = image.to('cuda')
      label = label.float().to('cuda')

      opt.zero_grad()

      out = model(image)

      loss = crit(out, label)
      loss.backward()
      opt.step() 
    elapsed = time() - start
    losses.append(loss.item())
    print(f'epoch: {epoch+1} -- loss: {loss.item():.4f} -- elapsed: {elapsed:.2f} s')

  return losses, model