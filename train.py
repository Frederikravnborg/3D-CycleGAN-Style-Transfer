"""
Training for CycleGAN
Code partly based on Aladdin Persson <aladdin.persson at hotmail dot com>
"""

import torch
import os
from datetime import datetime
from load_data import ObjDataset
from utilities.utilities import save_checkpoint, load_checkpoint
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import config
from tqdm import tqdm
from pointnet_model import Discriminator
from foldingnet_model import Generator
import trimesh
from torchvision import transforms
import wandb

# start a new wandb run to track this script
wandb.init(
    # set the wandb project where this run will be logged
    project="Fagprojekt",
    
    # track hyperparameters and run metadata
    config={
    "BATCH_SIZE": config.BATCH_SIZE,
    "LEARNING_RATE": config.LEARNING_RATE,
    "TRAIN_GAN": config.TRAIN_GAN,
    "GAN_NUM_EPOCHS": config.GAN_NUM_EPOCHS,
    "LOAD_MODEL": config.LOAD_MODEL,
    "SAVE_MODEL": config.SAVE_MODEL,
    "DISC_WIDTH_REDUCER": config.DISC_WIDTH_REDUCER,
    "TRAIN_FOLD": config.TRAIN_FOLD,
    "FOLD_NUM_EPOCH": config.FOLD_NUM_EPOCH,
    "LOAD_FOLD_MODEL": config.LOAD_FOLD_MODEL,
    "FOLD_SHAPE": config.FOLD_SHAPE,
    "LAMBDA_CYCLE": config.LAMBDA_CYCLE,
    }
)

def train_fold(gen_M, gen_F, loader, opt_gen, g_scaler, epoch, folder_name):
    loop = tqdm(loader, leave=True) #Progress bar

    # loop through the data loader
    for idx, (female, male) in enumerate(loop):
        female = female.to(config.DEVICE).float()
        male = male.to(config.DEVICE).float()

        # Train Generators
        with torch.cuda.amp.autocast(): #Necessary for float16
            # Generate fake images
            fake_female, _, female_cycle_loss = gen_F(female)
            fake_male, _, male_cycle_loss = gen_M(male)

            # Compute generator losses
            cycle_loss = (
                (female_cycle_loss + male_cycle_loss) * config.LAMBDA_CYCLE
            )
        
        # update of weights
        opt_gen.zero_grad()  #compute zero gradients
        g_scaler.scale(cycle_loss).backward() #backpropagate
        g_scaler.step(opt_gen) #update weights
        g_scaler.update() #update scaler

        # save point clouds every SAVE_RATE iterations
        if config.FOLD_SAVE_OBJ and (epoch+1) % config.SAVE_RATE == 0 and idx == 0:

            female_vertices = fake_female[0].detach().cpu().numpy().transpose(1,0)
            fake_female = trimesh.Trimesh(vertices=female_vertices)
            fake_female.export(f"{folder_name}/epoch_{epoch}_female_{idx}.obj")
            wandb.log({f"FOLD_female_epoch_{epoch}": wandb.Object3D(female_vertices) })
            
            male_vertices = fake_male[0].detach().cpu().numpy().transpose(1,0)
            fake_male = trimesh.Trimesh(vertices=male_vertices)
            fake_male.export(f"{folder_name}/epoch_{epoch}_male_{idx}.obj")
            wandb.log({f"FOLD_male_epoch_{epoch}": wandb.Object3D(male_vertices) })

        # update progress bar
        loop.set_postfix(epoch=epoch, cycle_loss=cycle_loss.item())

# define the training function
def train_fn(
    disc_M, disc_F, gen_F, gen_M, loader, opt_disc, opt_gen, mse, d_scaler, g_scaler, epoch, currentTime, folder_name
):
    # initialize variables to keep track of discriminator outputs
    M_reals = 0
    M_fakes = 0
    loop = tqdm(loader, leave=True) #Progress barq

    # loop through the data loader
    for idx, (female, male) in enumerate(loop):
        # define data to fit either on cpu or gpu (DEVICE parameter)
        female = female.to(config.DEVICE).float()
        male = male.to(config.DEVICE).float()

        # Train Discriminators H and Z
        with torch.cuda.amp.autocast(): #Necessary for float16
            """  FEMALE -> MALE  """
            fake_male, _, _ = gen_M(female) #Creating fake input
            D_M_real = disc_M(torch.transpose(male,1,2)) #Giving discriminator real input
            D_M_fake = disc_M(fake_male.detach()) #Giving discriminator fake input
            M_reals += D_M_real.mean().item()
            M_fakes += D_M_fake.mean().item()
            # error between discriminator output and expected output
            D_M_real_loss = mse(D_M_real, torch.ones_like(D_M_real)) #MSE of D_M_real, expect 1
            D_M_fake_loss = mse(D_M_fake, torch.zeros_like(D_M_fake)) #MSE of D_M_fake, expect 0
            D_M_loss = D_M_real_loss + D_M_fake_loss #Sum of loss

            """  MALE -> FEMALE  """
            fake_female, _, _ = gen_F(male)
            #fake_female = fake_female.transpose(2,1)
            D_F_real = disc_F(torch.transpose(female,1,2))
            D_F_fake = disc_F(fake_female.detach())
            # error between discriminator output and expected output
            D_F_real_loss = mse(D_F_real, torch.ones_like(D_F_real)) #MSE of D_F_real, expect 1
            D_F_fake_loss = mse(D_F_fake, torch.zeros_like(D_F_fake)) #MSE of D_F_fake, expect 0
            D_F_loss = D_F_real_loss + D_F_fake_loss #Sum of loss

            # define total discriminator loss as the average of the two
            D_loss = (D_M_loss + D_F_loss) / 2 

        # update of weights
        opt_disc.zero_grad() #compute zero gradients
        d_scaler.scale(D_loss).backward() #backpropagate
        d_scaler.step(opt_disc) #update weights
        d_scaler.update() #update scaler

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

            # cycle loss scaled by lambda
            cycle_loss = (
                cycle_female_loss * config.LAMBDA_CYCLE
                + cycle_male_loss * config.LAMBDA_CYCLE
            )

            # add all generator losses together to obtain full generator loss
            G_loss = (
                loss_G_F
                + loss_G_M
                + cycle_loss
            )

        # update of weights
        opt_gen.zero_grad()  #compute zero gradients
        g_scaler.scale(G_loss).backward() #backpropagate
        g_scaler.step(opt_gen) #update weights
        g_scaler.update() #update scaler

        # save point clouds every SAVE_RATE iterations
        if config.SAVE_OBJ and (epoch+1) % config.SAVE_RATE == 0 and idx == 0:

            fake_female_vertices = fake_female[0].detach().cpu().numpy()
            fake_female = trimesh.Trimesh(vertices=fake_female_vertices)
            fake_female.export(f"{folder_name}/epoch_{epoch}_female_{idx}.obj")
            # wandb.log({f"fake_female_epoch_{epoch}": fake_female})
            wandb.log({f"fake_female_epoch_{epoch}": wandb.Object3D(fake_female_vertices) })

            fake_male_vertices = fake_male[0].detach().cpu().numpy()
            fake_male = trimesh.Trimesh(vertices=fake_male_vertices)
            fake_male.export(f"{folder_name}/epoch_{epoch}_male_{idx}.obj")
            wandb.log({f"fake_male_epoch_{epoch}": wandb.Object3D(fake_male_vertices) })

        # save idx, D_loss, G_loss, mse, L1 in csv file
        with open(f'output/loss_{currentTime}.csv', 'a') as f: 
           f.write(f'{idx},{D_loss},{loss_G_M},{loss_G_F},{cycle_loss},{G_loss},{epoch}\n')
        wandb.log({
    "idx": idx,
    "D_loss": D_loss,
    "loss_G_M": loss_G_M,
    "loss_G_F": loss_G_F,
    "cycle_loss": cycle_loss,
    "G_loss": G_loss,
    "epoch": epoch
})
        
        # update progress bar
        loop.set_postfix(M_real=M_reals / (idx + 1), M_fake=M_fakes / (idx + 1), epoch=epoch)


def main():
    # initialize Discriminators and Generators
    disc_F = Discriminator().to(config.DEVICE)
    disc_M = Discriminator().to(config.DEVICE)
    gen_F = Generator().to(config.DEVICE)
    gen_M = Generator().to(config.DEVICE)

    # define optimizers for Discriminators and Generators
    opt_disc = optim.Adam(
        list(disc_M.parameters()) + list(disc_F.parameters()),
        lr=(config.LEARNING_RATE*config.DISC_LR_FACTOR),
        betas=(0.5, 0.999),
    )
    opt_gen = optim.Adam(
        list(gen_F.parameters()) + list(gen_M.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999),
    )

    mse = nn.MSELoss() #Adverserial loss

    if config.SAVE_MODEL:
        GEN_M_filename = config.CHECKPOINT_GEN_M
        GEN_F_filename = config.CHECKPOINT_GEN_F
        DISC_M_filename = config.CHECKPOINT_DISC_M
        DISC_F_filename = config.CHECKPOINT_DISC_F
        PRE_GEN_M_filename = config.CHECKPOINT_FOLD_M
        PRE_GEN_F_filename = config.CHECKPOINT_FOLD_F

    if config.LOAD_FOLD_MODEL:
        load_checkpoint(
            config.SAVEDMODEL_GEN_M,
            gen_M,
            opt_gen,
            config.LEARNING_RATE,
        )
        load_checkpoint(
            config.SAVEDMODEL_GEN_F,
            gen_F,
            opt_gen,
            config.LEARNING_RATE,
        )

    # load previously trained model if LOAD_MODEL is True
    if config.LOAD_MODEL:
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
            config.CHECKPOINT_DISC_M,
            disc_M,
            opt_disc,
            config.LEARNING_RATE,
        )
        load_checkpoint(
            config.CHECKPOINT_DISC_F,
            disc_F,
            opt_disc,
            config.LEARNING_RATE,
        )
    transform = transforms.Lambda(lambda x: x / config.MAX_DISTANCE)
    # define train dataset
    dataset = ObjDataset(
        root_male=config.TRAIN_DIR + "/male", 
        root_female=config.TRAIN_DIR + "/female",
        transform=transform,
        n_points=config.N_POINTS
    )
    # define validation dataset
    val_dataset = ObjDataset(
        root_male=config.VAL_DIR + "/male",
        root_female=config.VAL_DIR + "/female",
        transform=transform,
        n_points=config.N_POINTS
    )

    # define dataloader for train and validation dataset
    loader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader( val_dataset, batch_size=config.BATCH_SIZE, shuffle=True, pin_memory=True)
    
    # scalers to run in float 16. Handles overflow and underflow
    g_scaler = torch.cuda.amp.GradScaler() 
    d_scaler = torch.cuda.amp.GradScaler()

    #create csv file to store losses
    currentDateAndTime = datetime.now()
    currentTime = currentDateAndTime.strftime("%m.%d.%H.%M.%S")

    # save loss in csv file
    with open(f'output/loss_{currentTime}.csv', 'w') as f: 
        f.write(f"meta:{config.TRAIN_DIR=},{config.BATCH_SIZE=},{config.GAN_NUM_EPOCHS=},{config.LEARNING_RATE=},{config.LAMBDA_CYCLE},{config.GAN_NUM_EPOCHS},{config.LOAD_MODEL=},{config.N_POINTS=}\n")
        f.write('idx,D_loss,G_M_loss,G_F_loss,cycle_loss,G_loss,epoch\n')

    # create folder to save generated point clouds in
    folder_name = f"pre_saved_models/pcds/{currentTime}"
    os.makedirs(folder_name)

    if config.TRAIN_FOLD:
        for epoch in range(config.FOLD_NUM_EPOCH):
            train_fold(
                gen_M,
                gen_F,
                loader,
                opt_gen,
                g_scaler,
                epoch,
                folder_name
            )

            if config.SAVE_MODEL and epoch % 50 == 0 and epoch !=0:
                    PRE_GEN_M_filename = PRE_GEN_M_filename[:-8] + f"_E{epoch}.pth.tar"
                    PRE_GEN_F_filename = PRE_GEN_F_filename[:-8] + f"_E{epoch}.pth.tar"
                    save_checkpoint(gen_M, opt_gen, filename=PRE_GEN_M_filename)
                    save_checkpoint(gen_F, opt_gen, filename=PRE_GEN_F_filename)

    # create folder to save generated point clouds in
    folder_name = f"saved_pcds/{currentTime}"
    os.makedirs(folder_name)

    if config.TRAIN_GAN:
    # train the model
        for epoch in range(config.GAN_NUM_EPOCHS):
            train_fn(
                disc_M,
                disc_F,
                gen_F,
                gen_M,
                loader,
                opt_disc,
                opt_gen,
                mse,
                d_scaler,
                g_scaler,
                epoch,
                currentTime,
                folder_name
            )

        # save model for every epoch 
        if config.SAVE_MODEL:
            save_checkpoint(gen_M, opt_gen, filename=GEN_M_filename)
            save_checkpoint(gen_F, opt_gen, filename=GEN_F_filename)
            save_checkpoint(disc_M, opt_disc, filename=DISC_M_filename)
            save_checkpoint(disc_F, opt_disc, filename=DISC_F_filename)
    
    wandb.finish()

if __name__ == "__main__":
    main()
