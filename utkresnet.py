import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, models, transforms
from torch.optim import lr_scheduler
from time import time
from sklearn.preprocessing import OneHotEncoder

from dataloader import *


class ResNet18():
    def __init__(self, dir='./data/', saved='resnet18_utk.pt', train=False, epochs=10, batch_size=16, lr=1e-5):
        super(ResNet18, self).__init__()
        self.train_dir = dir + 'train/'
        self.test_dir = dir + 'test/'
        self.train = ImageDataset(self.train_dir)
        self.test = ImageDataset(self.test_dir)
        self.num_class = len(self.train)
        self.batch_size = batch_size

        self.device = torch.device('cuda')
        self.resnet18 = self.makemodel()
        self.trained_model_weights = saved
        self.utkresnet18 = self.loadmodel()
        self.losses = None if not train else self.train_resnet(epochs, batch_size, lr)

    def train_model(self, epochs=10, lr=1e-4):
        self.resnet18 = self.resnet18.to(self.device)

        crit = nn.CrossEntropyLoss()
        opt = optim.Adam(self.resnet18.parameters(), lr=lr)
        loader = DataLoader(self.train, batch_size=self.batch_size, shuffle=True)
        
        self.resnet18.to(self.device)
        self.resnet18.train()
        

        start = time()
        losses = []
        for epoch in range(epochs):
            for batch, (image, label) in enumerate(loader):
                image = image.to(self.device)
                label = label.to(self.device)

                opt.zero_grad()

                out = self.resnet18(image)
                _, preds = torch.max(out, 1)
                loss = crit(out, label)
                loss.backward()
                opt.step()
            elapsed = time() - start
            losses.append(loss.item())
            print(f'epoch: {epoch + 1} -- loss: {loss.item():.4f} -- elapsed: {elapsed:.2f} s')

        return losses

    def test_model(self, model):
        correct = 0
        total = len(self.test)

        def getclass(x):
            x = x.tolist()
            key = x.index(1)
            return key

        loader = DataLoader(self.test, batch_size=self.batch_size, shuffle=True)
        model.eval()
        model.to('cuda')
        with torch.no_grad():
            for batch, (image, label) in enumerate(loader):
                image = image.to('cuda')
                label = label.cpu().numpy()
                out = model(image)
                _, preds = torch.max(out, 1)
                preds = preds.detach().cpu().numpy()
                correct += np.sum(np.equal(preds, label))

        print(f'Acc: {100 * correct / total}%')

    def predict(self, model, image):
        model.to('cuda')
        model.eval()
        with torch.no_grad():
            image = image.to('cuda')

            out = model(image)
            _, pred = torch.max(out, 1)

        return pred

    def makemodel(self):
        model = models.resnet18(pretrained=True)
        #     for param in model.parameters():
        #         param.requires_grad = False
        num_feat = model.fc.in_features
        model.fc = nn.Linear(num_feat, 8)
        return model

    def savemodel(self, fname):
        torch.save(self.resnet18.state_dict(), fname)

    def loadmodel(self):
        model = self.makemodel()
        model.load_state_dict(torch.load(self.trained_model_weights))
        return model
