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

  def __len__(self):
    return len(self.data)

  def __getitem__(self, idx):
    fname = self.data[idx]
    image = read_image(fname)
    label = self.getclass(fname)
    return image, label

  def getclass(self, fname):
    classes = {
        (0, 0) : 'chd_male',
        (0, 1) : 'chd_female',
        (1, 0) : 'yng_male',
        (1, 1) : 'yng_female',
        (2, 0) : 'mid_male',
        (2, 1) : 'mid_female',
        (3, 0) : 'eld_male',
        (3, 1) : 'eld_female'
    }
    age, gender, _, _ = fname[len(self.dir):].split('_')
    age, gender = int(age), int(gender)
    agebin = 0
    if 0 <= age <= 18: agebin = 0
    elif 18 < age <= 40: agebin = 1
    elif 40 < age <= 65: agebin = 2
    else: agebin = 3

    categorical = classes[(agebin, gender)]
    val = list(classes.values()).index(categorical)
    return np.array([0 if i != val else 1 for i in range(8)])
