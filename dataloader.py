import torch
import glob
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
from torchvision.transforms import ToTensor
from itertools import islice

class ImageDataset(Dataset):
  def __init__(self, dir):
    self.dir = dir
    self.gen = glob.iglob(dir + '*')
    self.data = self.populate()
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
    self.transform = ToTensor()

  def __len__(self):
    return len(self.data)

  def __getitem__(self, idx):
    data = self.data[idx]
    fname = data[0]
    age, gender = data[1], data[2]

    image = self.transform(Image.open(fname))
    label = self.getlabel(self.classes[(age, gender)])
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
      if 0 <= age <= 18: agebin = 0
      elif 18 < age <= 40: agebin = 1
      elif 40 < age <= 65: agebin = 2
      else: agebin = 3

      if (agebin in range(4)) and (gender in range(1)):
        data.append((fname, agebin, gender))
    return data
