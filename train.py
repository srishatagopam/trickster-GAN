from time import time
from torch import optim
from dataloader import *
from models import *


def train(G, D, target_model, data_path='./data/train/', device='cuda', batch_size=32, lr=1e-4, epochs=10, lim=10000):
    G.to(device)
    D.to(device)
    target_model.eval()
    target_model.to(device)

    G.train()
    D.train()

    G_opt = optim.Adam(G.parameters(), lr=lr)
    D_opt = optim.Adam(D.parameters(), lr=lr)
    train_set = ImageDataset(data_path, lim)
    loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)

    G_losses = []
    D_losses = []

    CELoss = nn.CrossEntropyLoss()
    BCELoss = nn.BCELoss()

    start = time()

    for epoch in range(epochs):
        for batch, (image, label) in enumerate(loader):
#             print(batch+1)
            image = image.to(device)
            label = label.to(device)
            target = torch.from_numpy(np.random.randint(0, 8, size=batch_size)).type(torch.long).to(device)

            # print(image.shape, target.shape)
            pert = Resize(128)(G(image))
            pert = torch.clamp(pert, 0, 1)
#             print('pert done')
            adv = PGD(target_model, image, target, device, n_iter=15)

#             print('pgd done')

            # train discriminator
            D_opt.zero_grad()

            D_pert_bin, D_pert_multi = D(image + pert)
            D_clean_bin, D_clean_multi = D(image)
            D_adv_bin, D_adv_multi = D(image + adv)

            L_C_adv = CELoss(D_adv_multi, label)
            # L_C_adv.backward()
            L_C_pert = CELoss(D_pert_multi, label)
            # L_C_pert.backward()

            L_S_clean = BCELoss(D_clean_bin, torch.ones_like(D_clean_bin).to(device))
            # L_S_clean.backward()
            L_S_pert = BCELoss(D_pert_bin, torch.zeros_like(D_pert_bin).to(device))
            # L_S_pert.backward()
            L_S = L_S_clean + L_S_pert

            D_loss = L_S + L_C_adv + L_C_pert
            D_loss.backward()

            D_opt.step()

#             print('done disc')


            # train generator
            G_opt.zero_grad()

            D_pert_bin, D_pert_multi = D(image + torch.clamp(pert.detach(), 0, 1))
            D_clean_bin, D_clean_multi = D(image)
            D_adv_bin, D_adv_multi = D(image + adv.detach())
            target_pert = target_model(image + torch.clamp(pert.detach(), 0, 1))

            L_target_pert = CELoss(target_pert, target)
            L_D_pert = CELoss(D_pert_multi, target)
            L_S_clean = BCELoss(D_clean_bin, torch.ones_like(D_clean_bin).to(device))
            L_S_pert = BCELoss(D_pert_bin, torch.zeros_like(D_pert_bin).to(device))
            L_S = L_S_clean + L_S_pert

            G_loss = L_target_pert + L_D_pert - L_S
            G_loss.backward()

            G_opt.step()

#             print('done gen')

        finish = time() - start
        G_losses.append(G_loss.item())
        D_losses.append(D_loss.item())
        print(f'epoch: {epoch + 1}, gen_loss: {G_loss.item()},  dis_loss: {D_loss.item()}, elapsed: {finish:.2f} s')

    return G_losses, D_losses
