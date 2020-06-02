# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import os
import torch
import itertools
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import MSELoss, L1Loss
from torch.autograd import Variable

from network import Generator, Discriminator
from dataset import ImageDataset
from utils import Buffer, save_model, log_images


# %%
# get the data loader
# rawdataset = ImageDataset(root, True)
# dataloader = DataLoader(rawdataset, batch_size=32, shuffle=True, num_workers=4)


# %%
"""
G12 is generator that learns to change image from 1 to 2
G21 is generator that learns to change image from 2 to 1
D1 is discriminator that differentiates fake generated by G21 from images of class 1
D2 is discriminator that differentiates fake generated by G12 from images of class 2

"""

G12 = Generator(3, 3)
G21 = Generator(3, 3)
D1 = Discriminator(3)
D2 = Discriminator(3)

# shift models to cuda if possible
if torch.cuda.is_available():
    G12.cuda()
    G21.cuda()
    D1.cuda()
    D2.cuda()


# %%
# optimizer and loss
LGAN = MSELoss()
LCYC = L1Loss()
LIdentity = L1Loss()

optimizer_G = Adam(itertools.chain(
    G12.parameters(), G21.parameters()), lr=0.001)
optimizer_D1 = Adam(D1.parameters(), lr=0.001)
optimizer_D2 = Adam(D2.parameters(), lr=0.001)


# %%
# train models
real_label = torch.full((32,), 1, device="cuda:0")
false_label = torch.full((32,), 0, device="cuda:0")
bufD1 = Buffer(50)
bufD2 = Buffer(50)

num_epochs = 100
learning_rate = 0.01
for epoch in range(num_epochs):
    for i, (realA, realB) in enumerate(dataloader):
        if torch.cuda.is_available():
            realA.cuda()
            realB.cuda()

        #------------ Generator 1->2 and 2->1 -------------#
        optimizer_G.zero_grad()

        fakeB = G12(realA)
        pred_fakeB = D2(fakeB)
        loss_GAN_G12 = LGAN(pred_fakeB, real_label)

        fakeA = G21(realB)
        pred_fakeA = D1(fakeA)
        loss_GAN_G21 = LGAN(pred_fakeA, real_label)

        # cyclic loss
        recA = G21(fakeB)
        Cl1 = LCYC(recA, realA)
        recB = G12(fakeA)
        Cl2 = LCYC(recB, realB)
        cyclic_loss = Cl1 + Cl2

        total_loss_G = loss_GAN_G12 + loss_GAN_G21 + 10*cyclic_loss
        total_loss_G.backward()

        optimizer_G.step()

        #------------ Discriminator 1 -------------#

        optimizer_D1.zero_grad()

        # real loss
        pred_real = D1(realA)
        loss_D1_real = LGAN(pred_real, real_label)

        # fake loss
        fakeA = bufD1.replay_fake(fakeA)
        pred_fake = D1(fakeA)
        loss_D1_fake = LGAN(pred_fake, false_label)

        loss_D_1 = loss_D1_fake + loss_D1_real
        loss_D_1.backward()
        optimizer_D1.step()

        #----------- Discriminator 2 --------------#

        optimizer_D2.zero_grad()

        # real loss
        pred_real = D2(realB)
        loss_D2_real = LGAN(pred_real, real_label)

        # fake loss
        fakeB = bufD2.replay_fake(fakeB)
        pred_fake = D2(fakeB)
        loss_D2_fake = LGAN(pred_fake, false_label)

        loss_D_2 = loss_D2_fake + loss_D2_real
        loss_D_2.backward()

        optimizer_D2.step()

        #------------------------------------------------#

        print(f"""
                total_loss_G = {total_loss_G}\n
                loss_G_GAN = {loss_GAN_G12 + loss_GAN_G21}\n
                loss_G_Cycl = {cyclic_loss}\n
                loss_D = {loss_D_1 + loss_D_2}
            """)

        log_losses({"total_loss_G": total_loss_G.item(),
                    "loss_G_GAN": (loss_GAN_G12 + loss_GAN_G21).item(),
                    "loss_G_Cycl": cyclic_loss.item(),
                    "loss_D": (loss_D_1 + loss_D_2).item()})

        log_images({"epoch": epoch, "batch": i, "images": {
                   "realA": realA, "fakeA": fakeA, "realB": realB, "fakeB": fakeB}})

    #------------------- save model -----------------#
    save_model(G12, epoch)
    save_model(G21, epoch)
    save_model(D1, epoch)
    save_model(D2, epoch)