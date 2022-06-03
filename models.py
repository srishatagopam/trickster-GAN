import torch
import torch.nn as nn


def PGD(model, image, target, device='cuda', n_iter=5, alpha=1e-3, eps=0.6):
  crit = nn.CrossEntropyLoss()
  model.eval()
  if not isinstance(target, torch.Tensor): target = torch.Tensor([target]).type(torch.int64)
  pert = (-2*eps * torch.rand(image.size()) + eps).to(device); pert.requires_grad=True

  image = image.to(device)
  target = target.to(device)

  for _ in range(n_iter):
    out = model(image + pert)
    loss = crit(out, target)
    loss.backward()
    pert.data = (pert + alpha * torch.sign(pert.grad)).clamp(-eps, eps)
  return pert

class Generator(nn.Module):
  def __init__(self):
    super(Generator, self).__init__()
    self.encoder = Encoder()
    self.resblock1 = ResBlock(64, 3, 1, 1)
    self.resblock2 = ResBlock(64, 3, 1, 1)
    self.decoder = Decoder()
    
  def forward(self, x):
    out = self.encoder(x)
    # N, _, H, W = out.shape
    # out = torch.flatten(out)
    # concat = torch.concat([out, target])
    # out = nn.Linear(concat.shape[0], 8*3*128*128)(concat)
    # out = out.view(8, 3, 128, 128)
    out = self.resblock1(out)
    out = self.resblock2(out)
    out = self.decoder(out)
    return out

class Discriminator(nn.Module):
  def __init__(self, channels=3):
    super(Discriminator, self).__init__()
    self.encode1 = EncodeBlock(3, 8, 7, 2, 3)
    self.encode2 = EncodeBlock(8, 16, 3, 2, 1)
    self.conv = nn.Conv2d(16, 32, 3, 2, 1)
    self.dropout = nn.Dropout(0.2)
    self.fc1 = nn.Linear(32*16*16, 128)
    self.fc2 = nn.Linear(128, 8)
    self.fc3 = nn.Linear(128, 1)
    self.sig = nn.Sigmoid()

  def forward(self, x):
    out = self.encode1(x)
    out = self.encode2(out)
    out = self.conv(out)
    out = self.dropout(out)
    out = torch.flatten(out, 1)
    out = self.fc1(out)
    class_prob = self.fc2(out)
    fake_prob = self.sig(self.fc3(out))

    return fake_prob, class_prob

class Encoder(nn.Module):
  def __init__(self, channels=3):
    super(Encoder, self).__init__()
    self.channels = channels
    self.encode1 = EncodeBlock(self.channels, 16, 7, 2, 3)
    self.encode2 = EncodeBlock(16, 32, 3, 2, 1)
    self.encode3 = EncodeBlock(32, 64, 3, 2, 1)

  def forward(self, x):
    out = self.encode1(x)
    out = self.encode2(out)
    out = self.encode3(out)
    return out

class EncodeBlock(nn.Module):
  def __init__(self, in_channels, out_channels, kernel, stride, padding):
    super(EncodeBlock, self).__init__()
    self.in_channels = in_channels
    self.out_channels = out_channels
    self.kernel = kernel
    self.stride = stride
    self.padding = padding

    self.conv = nn.Conv2d(self.in_channels, self.out_channels, self.kernel, self.stride, self.padding)
    self.norm = nn.BatchNorm2d(self.out_channels)
    self.relu = nn.LeakyReLU(0.3)
    self.dropout = nn.Dropout(0.2)

  def forward(self, x):
    out = self.conv(x)
    out = self.relu(out)
    out = self.dropout(out)
    out = self.norm(out)
    return out

class ResBlockTrain(nn.Module):
  def _init__(self, channels=64):
    super(ResBlockTrain, self).__init__()
    self.channels = channels
    self.resblock1 = ResBlock(self.channels, 3, 1, 1)
    self.resblock2 = ResBlock(self.channels, 3, 1, 1)

  def forward(self, x):
    out = self.resblock1(x)
    out = self.resblock2(out)
    return out


class ResBlock(nn.Module):
  def __init__(self, channels=64, kernel=3, stride=1, padding=1):
    super(ResBlock, self).__init__()
    self.channels = channels
    self.kernel = kernel
    self.stride = stride
    self.padding = padding

    # if first: 
    #   self.conv1 = nn.Conv2d(self.channels+1, self.channels, 7, self.stride, 3)
    # else:
    self.conv1 = nn.Conv2d(self.channels, self.channels, self.kernel, self.stride, self.padding)
    self.conv2 = nn.Conv2d(self.channels, self.channels, self.kernel, self.stride, self.padding)
    self.norm1 = nn.BatchNorm2d(self.channels)
    self.norm2 = nn.BatchNorm2d(self.channels)
    self.relu = nn.LeakyReLU(0.3)
    self.dropout = nn.Dropout(0.2)

  def forward(self, x):
    out = self.conv1(x)
    out = self.relu(out)
    out = self.dropout(out)
    out = self.norm1(out)
    out = self.conv2(out)
    out = x + out
    out = self.relu(out)
    out = self.dropout(out)
    out = self.norm2(out)
    return out

class Decoder(nn.Module):
  def __init__(self, channels=3):
    super(Decoder, self).__init__()
    self.channels = channels
    self.decode1 = DecodeBlock(64, 32, 3, 2, 0)
    self.decode2 = DecodeBlock(32, 16, 3, 2, 0)
    self.decode3 = DecodeBlock(16, self.channels, 3, 2, 0)

  def forward(self, x):
    out = self.decode1(x)
    out = self.decode2(out)
    out = self.decode3(out)
    return out

class DecodeBlock(nn.Module):
  def __init__(self, in_channels, out_channels, kernel, stride, padding=0):
    super(DecodeBlock, self).__init__()
    self.in_channels = in_channels
    self.out_channels = out_channels
    self.kernel = kernel
    self.stride = stride
    self.padding = padding

    self.convT = nn.ConvTranspose2d(self.in_channels, self.out_channels, self.kernel, self.stride, self.padding)
    self.norm = nn.BatchNorm2d(self.out_channels)
    self.relu = nn.LeakyReLU(0.3)
    self.dropout = nn.Dropout(0.2)

  def forward(self, x):
    out = self.convT(x)
    out = self.relu(out)
    out = self.dropout(out)
    out = self.norm(out)
    return out
