import torch
import random
import glob
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
from torchvision.transforms import ToTensor, ToPILImage, Resize, Compose
from itertools import islice
from sklearn.preprocessing import OneHotEncoder


class ImageDataset(Dataset):
    def __init__(self, dir, lim=None):
        self.counts = [0] * 8
        self.dir = dir
        self.lim = lim
        self.gen = glob.iglob(dir + '*')
        self.classes = {
            (0, 0): 'male child',
            (0, 1): 'female child',
            (1, 0): 'young male',
            (1, 1): 'young female',
            (2, 0): 'middle age male',
            (2, 1): 'middle age female',
            (3, 0): 'elderly male',
            (3, 1): 'elderly female'
        }
        self.keys = list(self.classes.keys())
        self.counts = {i: 0 for i in self.keys}
        self.transform = Compose([
            ToTensor(),
            Resize(128)
        ])
        self.data = self.populate()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        fname = data[0]
        age, gender = data[1], data[2]

        image = self.transform(Image.open(fname))
        label = self.keys.index((age, gender))
        return image, label

    def getlabel(self, c):
        val = list(self.classes.values()).index(c)
        return np.array([0 if i != val else 1 for i in range(8)])

    def getclass(self, idx):
        onehot = idx.type(torch.int).tolist()
        key = self.keys[onehot.index(1)]
        return self.classes[key]

    def populate(self):
        data = []
        for fname in self.gen:
            classinfo = fname[len(self.dir):].split('_')
            age, gender = int(classinfo[0]), int(classinfo[1])
            agebin = 0
            if 0 <= age <= 18:
                agebin = 0
            elif 18 < age <= 40:
                agebin = 1
            elif 40 < age <= 65:
                agebin = 2
            else:
                agebin = 3

            if (agebin in [0, 1, 2, 3]) and (gender in [0, 1]) and (agebin, gender) in self.keys:
                data.append((fname, agebin, gender))
                self.counts[(agebin, gender)] += 1
        #print(self.counts)
        if self.lim is None:
            return data
        else:
            random.shuffle(data)
            return data[:self.lim]
