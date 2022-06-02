import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn, optim
from time import time

from models import *
from dataloader import *

def train(G, D, target_model, data_path='./data/train/', device='cuda', batch_size=32, lr=1e-4, epochs=10, eps=0.3):
  G.train()
  D.train()

  G_opt = optim.Adam(G.parameters(), lr=lr)
  D_opt = optim.Adam(D.parameters(), lr=lr)
  train_set = ImageDataset(data_path)
  loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

  G_losses = []
  D_losses = []

  multi_crit = nn.CrossEntropyLoss()
  binary_crit = nn.BCELoss()

  start = time()

  for epoch in range(epochs):
    for batch, (image, label) in enumerate(loader):
      image = image.to(device)
      label = label.to(device)
      target = torch.from_numpy(np.random.randint(0,8,size=batch_size)).to(device)

      pert = G(image).clamp(-eps, eps)
      adv = PGD(target_model, image, target)

      # train discriminator
      D_opt.zero_grad()

      D_pert_bin, D_pert_multi = D(image + pert)
      D_clean_bin, D_clean_multi = D(image)
      D_adv_bin, D_adv_multi = D(image + adv)

      _, D_pert_pred = torch.max(D_pert_multi, 1)
      _, D_clean_pred = torch.max(D_clean_multi, 1)
      _, D_adv_pred = torch.max(D_adv_multi, 1)

      L_C_adv = multi_crit(D_adv_pred, label)
      L_C_adv.backward()
      L_C_pert = multi_crit(D_pert_pred, label)
      L_C_pert.backward()

      L_S_clean = binary_crit(D_clean_bin, torch.ones_like(D_clean_bin).to(device))
      L_S_pert = binary_crit(D_pert_bin, torch.zeros_like(D_pert_bin).to(device))
      L_S = L_S_clean + L_S_pert
      L_S.backward()

      D_loss = L_S + L_C_adv + L_C_pert

      D_opt.step()

      # train generator
      G_opt.zero_grad()

      pert = G(image).clamp(-eps, eps)
      adv = PGD(target_model, image, target)

      D_pert_bin, D_pert_multi = D(image + pert)
      D_clean_bin, D_clean_multi = D(image)
      D_adv_bin, D_adv_multi = D(image + adv)
      target_pert = target_model(image + pert)

      _, D_pert_pred = torch.max(D_pert_multi, 1)
      _, D_clean_pred = torch.max(D_clean_multi, 1)
      _, D_adv_pred = torch.max(D_adv_multi, 1)
      _, target_pert_pred = torch.max(target_pert, 1)

      L_target_pert = multi_crit(target_pert_pred, target)
      L_D_pert = multi_crit(D_pert_pred, target)

      L_S_clean = binary_crit(D_clean_bin, torch.ones_like(D_clean_bin).to(device))
      L_S_pert = binary_crit(D_pert_bin, torch.zeros_like(D_pert_bin).to(device))
      L_S = L_S_clean + L_S_pert

      G_loss = L_target_pert + L_D_pert - L_S
      G_loss.backward()

      G_opt.step()
    
    finish = time() - start
    G_losses.append(G_loss.item())
    D_losses.append(D_loss.item())
    print(f'epoch: {epoch+1}, gen_loss: {G_loss.item()},  dis_loss: {D_loss.item()}, elapsed: {finish:.2f} s')

  return G_losses, D_losses