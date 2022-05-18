import torch
import glob
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
from itertools import islice

class ImageDataset(Dataset):
  def __init__(self, dir):
    self.dir = dir
    self.gen = glob.iglob(dir + '*')
    self.data = []
    for fname in self.gen:
      self.data.append(fname)
    self.classes = {
        (0, 0) : 'male child',
        (0, 1) : 'female child',
        (1, 0) : 'young male',
        (1, 1) : 'young female',
        (2, 0) : 'middle age male',
        (2, 1) : 'middle age female',
        (3, 0) : 'eldery male',
        (3, 1) : 'eldery female'
    }
    self.keys = list(self.classes.keys())

  def __len__(self):
    return len(self.data)

  def __getitem__(self, idx):
    fname = self.data[idx]
    image = read_image(fname)
    label = self.getlabel(fname)
    return image, label

  def getlabel(self, fname):
    age, gender, _, _ = fname[len(self.dir):].split('_')
    age, gender = int(age), int(gender)
    agebin = 0
    if 0 <= age <= 18: agebin = 0
    elif 18 < age <= 40: agebin = 1
    elif 40 < age <= 65: agebin = 2
    else: agebin = 3

    categorical = self.classes[(agebin, gender)]
    val = list(self.classes.values()).index(categorical)
    return np.array([0 if i != val else 1 for i in range(8)])

  def getclass(self, onehot):
    onehot = onehot.type(torch.int).tolist()
    key = self.keys[onehot.index(1)]
    return self.classes[key]
