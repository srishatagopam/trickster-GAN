import torch
import glob
import numpy as np
from torch.utils.data import Dataset
from torchvision.io import read_image
from itertools import islice

class ImageDataset(Dataset):
  def __init__(self, dir):
    self.dir = dir
    self.gen = glob.iglob(dir)

  def __len__(self):
    return sum(1 for _ in self.gen)

  def __getitem__(self, idx):
    fname = next(islice(self.gen, idx, None))
    image = read_image(fname)
    label = self.getclass(fname)
    return image, label

  def getclass(fname):
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
    age, gender, _, _ = fname.split('_')
    agebin = 0
    if 0 <= age <= 18: agebin = 0
    elif 18 < age <= 40: agebin = 1
    elif 40 < age <= 65: agebin = 2
    else: agebin = 3

    categorical = classes[(agebin, gender)]
    val = classes.values().index(categorical)
    return np.array([0 if i != val else 1 for i in range(8)])
