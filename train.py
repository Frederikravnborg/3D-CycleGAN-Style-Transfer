"""
Training for CycleGAN
Code partly based on Aladdin Persson <aladdin.persson at hotmail dot com>
"""

import torch
import csv
import os
from datetime import datetime
from load_data import ObjDataset
from utils import save_checkpoint, load_checkpoint
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import config
from tqdm import tqdm
from torchvision.utils import save_image
from pointnet_model import Discriminator
from foldingnet_model import Generator


def train_fn(
    disc_M, disc_F, gen_F, gen_M, loader, opt_disc, opt_gen, l1, mse, d_scaler, g_scaler, epoch, currentTime
):
    M_reals = 0
    M_fakes = 0
    loop = tqdm(loader, leave=True) #Progress bar

    for idx, (female, male) in enumerate(loop):
        female = female.to(config.DEVICE).float()
        male = male.to(config.DEVICE).float()

        # Train Discriminators H and Z
        with torch.cuda.amp.autocast(): #Necessary for float16
            fake_male, _, _ = gen_M(female) #Creating fake input
            #fake_male = fake_male.transpose(2,1)
            # print(torch.transpose(male,1,2).to(torch.float32))
            D_M_real = disc_M(torch.transpose(male,1,2)) #Giving discriminator real input
            D_M_fake = disc_M(fake_male.detach()) #Giving discriminator fake input
            M_reals += D_M_real.mean().item()
            M_fakes += D_M_fake.mean().item()
            D_M_real_loss = mse(D_M_real, torch.ones_like(D_M_real)) #MSE of D_M_real, expect 1
            D_M_fake_loss = mse(D_M_fake, torch.zeros_like(D_M_fake)) #MSE of D_M_fake, expect 0
            D_M_loss = D_M_real_loss + D_M_fake_loss #Sum of loss
            print(D_M_real_loss)

            fake_female, _, _ = gen_F(male)
            #fake_female = fake_female.transpose(2,1)
            D_F_real = disc_F(torch.transpose(female,1,2))
            D_F_fake = disc_F(fake_female.detach())
            D_F_real_loss = mse(D_F_real, torch.ones_like(D_F_real))
            D_F_fake_loss = mse(D_F_fake, torch.zeros_like(D_F_fake))
            D_F_loss = D_F_real_loss + D_F_fake_loss

            # put it together
            D_loss = (D_M_loss + D_F_loss) / 2

        #Standard update of weights
        opt_disc.zero_grad()
        d_scaler.scale(D_loss).backward()
        d_scaler.step(opt_disc)
        d_scaler.update()

        # Train Generators H and Z
        with torch.cuda.amp.autocast(): #Necessary for float16
            # adversarial loss for both generators
            D_M_fake = disc_M(fake_male) #fake_male generated by gen_M
            D_F_fake = disc_F(fake_female) #fake_female generated by gen_F
            #adversarial loss for male
            loss_G_M = mse(D_M_fake, torch.ones_like(D_M_fake)) #Real = 1, trick discriminator
            #adversarial loss for female
            loss_G_F = mse(D_F_fake, torch.ones_like(D_F_fake)) #Real = 1, trick discriminator

            # cycle loss
            fake_male = fake_male.transpose(2,1)
            fake_female = fake_female.transpose(2,1)
            cycle_female, _, cycle_female_loss = gen_F(fake_male)
            cycle_male, _, cycle_male_loss = gen_M(fake_female)
            # cycle_female_loss = l1(female, cycle_female.transpose(2,1))
            # cycle_male_loss = l1(male, cycle_male.transpose(2,1))

            #cycle loss
            cycle_loss = (
                cycle_female_loss * config.LAMBDA_CYCLE
                + cycle_male_loss * config.LAMBDA_CYCLE
            )

            # add all losses together (full generator loss)
            G_loss = (
                loss_G_F
                + loss_G_M
                + cycle_loss
            )


        opt_gen.zero_grad()
        g_scaler.scale(G_loss).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()

        # if idx == idx:
        #    save_image(fake_male * 0.5 + 0.5, f"saved_images/male_{idx}.png")
        #    save_image(fake_female * 0.5 + 0.5, f"saved_images/female_{idx}.png")
        #save idx, D_loss, G_loss, mse, L1 in csv file
        with open(f'output/loss_{currentTime}.csv', 'a') as f: 
           f.write(f'{idx},{D_loss},{loss_G_M},{loss_G_F},{cycle_loss},{G_loss},{epoch}\n')
        
        #Update progress bar
        loop.set_postfix(M_real=M_reals / (idx + 1), M_fake=M_fakes / (idx + 1))


def main():
    #Initializing Discriminators and Generators
    disc_F = Discriminator().to(config.DEVICE)
    disc_M = Discriminator().to(config.DEVICE)
    gen_F = Generator().to(config.DEVICE)
    gen_M = Generator().to(config.DEVICE)
    opt_disc = optim.Adam(
        list(disc_M.parameters()) + list(disc_F.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999),
    )
    opt_gen = optim.Adam(
        list(gen_F.parameters()) + list(gen_M.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999),
    )

    L1 = nn.L1Loss() #Cycle consistensy loss and Identity loss
    mse = nn.MSELoss() #Adverserial loss

    if config.LOAD_MODEL: #True/False defined in config
        load_checkpoint(
            config.CHECKPOINT_GEN_M,
            gen_M,
            opt_gen,
            config.LEARNING_RATE,
        )
        load_checkpoint(
            config.CHECKPOINT_GEN_F,
            gen_F,
            opt_gen,
            config.LEARNING_RATE,
        )
        load_checkpoint(
            config.CHECKPOINT_CRITIC_M,
            disc_M,
            opt_disc,
            config.LEARNING_RATE,
        )
        load_checkpoint(
            config.CHECKPOINT_CRITIC_F,
            disc_F,
            opt_disc,
            config.LEARNING_RATE,
        )

    #Create dataset
    dataset = ObjDataset(
        root_male=config.TRAIN_DIR + "/male", 
        root_female=config.TRAIN_DIR + "/female",
        transform=config.transforms,
        n_points=config.N_POINTS
    )
    val_dataset = ObjDataset(
        root_male=config.VAL_DIR + "/male",
        root_female=config.VAL_DIR + "/female",
        transform=config.transforms,
        n_points=config.N_POINTS
    )

    loader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader( val_dataset, batch_size=config.BATCH_SIZE, shuffle=True, pin_memory=True)
    
    g_scaler = torch.cuda.amp.GradScaler() #Scaler to run in float 16, if removed we run in float 32
    d_scaler = torch.cuda.amp.GradScaler() #Scaler to run in float 16, if removed we run in float 32

    #create csv file to store losses
    currentDateAndTime = datetime.now()
    currentTime = currentDateAndTime.strftime("%m.%d.%H.%M.%S")

    with open(f'output/loss_{currentTime}.csv', 'w') as f: 
        f.write(f"meta:{config.TRAIN_DIR=},{config.BATCH_SIZE=},{config.NUM_EPOCHS=},{config.LEARNING_RATE=},{config.LAMBDA_IDENTITY=},{config.LAMBDA_CYCLE},{config.NUM_EPOCHS},{config.LOAD_MODEL=},{config.N_POINTS=}\n")
        f.write('idx,D_loss,G_M_loss,G_F_loss,cycle_loss,G_loss,epoch\n')


    for epoch in range(config.NUM_EPOCHS):
        train_fn(
            disc_M,
            disc_F,
            gen_F,
            gen_M,
            loader,
            opt_disc,
            opt_gen,
            L1,
            mse,
            d_scaler,
            g_scaler,
            epoch,
            currentTime
        )

        #Save model for every epoch 
        if config.SAVE_MODEL:
            save_checkpoint(gen_M, opt_gen, filename=config.CHECKPOINT_GEN_M)
            save_checkpoint(gen_F, opt_gen, filename=config.CHECKPOINT_GEN_F)
            save_checkpoint(disc_M, opt_disc, filename=config.CHECKPOINT_CRITIC_M)
            save_checkpoint(disc_F, opt_disc, filename=config.CHECKPOINT_CRITIC_F)


if __name__ == "__main__":
    main()