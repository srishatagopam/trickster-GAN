import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, models, transforms
from torch.optim import lr_scheduler
from time import time
from dataloader import *

class ResNet18():
  def __init__(self, dir='./data/'):
    self.train_dir = dir + 'train/'
    self.test_dir = dir + 'test/'
    self.train = ImageDataset(self.train_dir)
    self.test = ImageDataset(self.test_dir)
    self.num_class = len(self.train)
    self.resnet18 = self.makemodel()
    self.utkresnet18 = self.loadmodel()

  def train_resnet(self, epochs=10, batch_size=16, lr=1e-5):
    model = model.to('cuda')

    crit = nn.CrossEntropyLoss()
    opt = optim.Adam(model.parameters(), lr=lr)
    loader = DataLoader(self.train, batch_size=batch_size, shuffle=True)

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

  def predict(self, model, image, label):
    with torch.no_grad():
      image = image.to('cuda')
      label = label.float().to('cuda')

      out = resnet18(image)
      _, pred = torch.max(out, 1)

    return self.train.getclass(pred.item())

  def makemodel(self):
    model = models.resnet18(pretrained=True)
    num_feat = model.fc.in_features
    model.fc = nn.Linear(num_feat, 8)
    return model

  def loadmodel(self):
    model = self.makemodel()
    model.load_state_dict(torch.load('resnet18_utk.pt'))
    model.eval()
    return model
