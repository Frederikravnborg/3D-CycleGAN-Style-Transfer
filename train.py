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
from foldingnet_model import Generator, ChamferLoss
import trimesh
from torchvision import transforms
import wandb
import numpy as np

# start a new wandb run to track this script
wandb.init(
    # set the wandb project where this run will be logged
    project="Fagprojekt",
    
    # track hyperparameters and run metadata
    config={
    "BATCH_SIZE": config.BATCH_SIZE,
    "GAN_NUM_EPOCHS": config.GAN_NUM_EPOCHS,
    "LAMBDA_CYCLE": config.LAMBDA_CYCLE,
    "LEARNING_RATE": config.LEARNING_RATE,
    "DISC_LR_FACTOR": config.DISC_LR_FACTOR,
    "DISC_WIDTH_REDUCER": config.DISC_WIDTH_REDUCER,
    "FOLD_SHAPE": config.FOLD_SHAPE,
    "TRAIN_GAN": config.TRAIN_GAN,
    "LOAD_MODEL": config.LOAD_MODEL,
    "SAVE_MODEL": config.SAVE_MODEL,
    "TRAIN_FOLD": config.TRAIN_FOLD,
    "FOLD_NUM_EPOCH": config.FOLD_NUM_EPOCH,
    "LOAD_FOLD_MODEL": config.LOAD_FOLD_MODEL,
    },
    mode = "online" if config.USE_WANDB else "disabled"
)
wandb.define_metric("epoch_step")
wandb.define_metric("fake_female", step_metric = "epoch_step")
wandb.define_metric("fake_male", step_metric = "epoch_step")
wandb.define_metric("OG_male", step_metric = "epoch_step")
wandb.define_metric("OG_female", step_metric = "epoch_step")
wandb.define_metric("cycle_male", step_metric = "epoch_step")
wandb.define_metric("cycle_female", step_metric = "epoch_step")

def pretrain_FoldingNet(gen_M, gen_F, loader, opt_gen, epoch, folder_name):
    loop = tqdm(loader, leave=True) #Progress bar

    # loop through the data loader
    for idx, (female, male) in enumerate(loop):
        # define data to fit either on cpu or gpu (DEVICE parameter)
        female = female.to(config.DEVICE).float()
        male = male.to(config.DEVICE).float()

        """ TRAIN GENERATORS TO RECONSTRUCT INPUT """
        # Generate reconstructions
        fake_female, _, female_reconstruction_loss = gen_F(female)
        fake_male, _, male_reconstruction_loss = gen_M(male)

        # compute generator losses
        reconstruction_loss = (female_reconstruction_loss + male_reconstruction_loss) * config.LAMBDA_CYCLE
        
        # update of weights
        opt_gen.zero_grad()  #compute zero gradients
        reconstruction_loss.backward()
        opt_gen.step()

        # save point clouds every SAVE_RATE iterations
        if config.FOLD_SAVE_OBJ and ((epoch+1) % config.SAVE_RATE == 0 or epoch==0) and idx == 0:

            female_vertices = fake_female[0].detach().cpu().numpy().transpose(1,0)
            fake_female = trimesh.Trimesh(vertices=female_vertices)
            fake_female.export(f"{folder_name}/epoch_{epoch}_female_{idx}.obj")
            wandb.log({f"FOLD_female": wandb.Object3D(female_vertices) })
            
            male_vertices = fake_male[0].detach().cpu().numpy().transpose(1,0)
            fake_male = trimesh.Trimesh(vertices=male_vertices)
            fake_male.export(f"{folder_name}/epoch_{epoch}_male_{idx}.obj")
            wandb.log({f"FOLD_male": wandb.Object3D(male_vertices) }, commit=False)
        # update progress bar
        loop.set_postfix(epoch=epoch, reconstruction_loss=reconstruction_loss.item())

# define the training function
def train(
    disc_M, disc_F, gen_F, gen_M, loader, opt_disc, opt_gen, mse, epoch, currentTime, folder_name, chamfer_loss
):

    loop = tqdm(loader, leave=True) #Progress bar

    # loop through the data loader
    for idx, (female, male) in enumerate(loop):
        # define data to fit either on cpu or gpu (DEVICE parameter)
        female = female.to(config.DEVICE).float()
        male = male.to(config.DEVICE).float()

        ####################  TRAIN DISCRIMINATORS  ####################
        """  FEMALE -> MALE  """
        fake_male, _, _ = gen_M(female) #Creating fake input
        D_M_real = disc_M(torch.transpose(male,1,2).long())[0] #Giving discriminator real input
        D_M_fake = disc_M(fake_male.detach())[0] #Giving discriminator fake input
        # error between discriminator output and expected output
        D_M_real_loss = mse(D_M_real, torch.ones_like(D_M_real)) #MSE of D_M_real, expect 1
        D_M_fake_loss = mse(D_M_fake, torch.zeros_like(D_M_fake)) #MSE of D_M_fake, expect 0
        D_M_loss = D_M_real_loss + D_M_fake_loss #Sum of loss

        """  MALE -> FEMALE  """
        fake_female, _, _ = gen_F(male)
        D_F_real = disc_F(torch.transpose(female,1,2))[0]
        D_F_fake = disc_F(fake_female.detach())[0]
        # error between discriminator output and expected output
        D_F_real_loss = mse(D_F_real, torch.ones_like(D_F_real)) #MSE of D_F_real, expect 1
        D_F_fake_loss = mse(D_F_fake, torch.zeros_like(D_F_fake)) #MSE of D_F_fake, expect 0
        D_F_loss = D_F_real_loss + D_F_fake_loss #Sum of loss

        # define total discriminator loss as the average of the two
        D_loss = (D_M_loss + D_F_loss) / 2 

        opt_disc.zero_grad()  # compute zero gradients
        D_loss.backward()     # compute loss gradients wrt. to discriminator parameters
        opt_disc.step()       # update discriminator parameters

        ####################  TRAIN GENERATORS  ####################
        # adversarial loss for both generators
        D_M_fake = disc_M(fake_male)[0] #fake_male generated by gen_M
        D_F_fake = disc_F(fake_female)[0] #fake_female generated by gen_F
        #adversarial loss for male
        loss_G_M = mse(D_M_fake, torch.ones_like(D_M_fake)) #Real = 1, trick discriminator
        #adversarial loss for female
        loss_G_F = mse(D_F_fake, torch.ones_like(D_F_fake)) #Real = 1, trick discriminator

        # generate cycle-female and cycle-male
        cycle_female, _, _ = gen_F(fake_male.transpose(2,1))
        cycle_male, _, _ = gen_M(fake_female.transpose(2,1))

        # compute chamfer loss between real and cycle point clouds
        cycle_female_loss = chamfer_loss(cycle_female.transpose(2,1), female)
        cycle_male_loss = chamfer_loss(cycle_male.transpose(2,1), male)

        # compute cycle-consistency loss
        cycle_loss = (cycle_female_loss + cycle_male_loss) * config.LAMBDA_CYCLE

        # add all generator losses together to obtain full generator loss
        G_loss = (loss_G_F + loss_G_M + cycle_loss)

        opt_gen.zero_grad()  # compute zero gradients
        G_loss.backward()    # compute loss gradients wrt. to generator parameters
        opt_gen.step()       # update generator parameters

        # save point clouds every SAVE_RATE iterations
        if config.SAVE_OBJ and ((epoch+1) % config.SAVE_RATE == 0 or epoch==0) and idx == 0:
            wandb.log({"D_M_real": D_M_real}, commit=False)
            wandb.log({"D_M_fake": D_M_fake}, commit=False)
            wandb.log({"D_F_real": D_F_real}, commit=False)
            wandb.log({"D_F_fake": D_F_fake}, commit=False)
            
            fake_female_vertices = fake_female.transpose(2,1)[0].detach().cpu().numpy()
            fake_female = trimesh.Trimesh(vertices=fake_female_vertices)
            fake_female.export(f"{folder_name}/epoch_{epoch}_female_{idx}.obj")
            wandb.log({"fake_female": wandb.Object3D(fake_female_vertices), "epoch_step": epoch})
            wandb.log({"OG_male": wandb.Object3D(male[0].detach().cpu().numpy()), "epoch_step": epoch}, commit=False)
            wandb.log({"cycle_male": wandb.Object3D(cycle_male.transpose(2,1)[0].detach().cpu().numpy()), "epoch_step": epoch}, commit=False)

            fake_male_vertices = fake_male.transpose(2,1)[0].detach().cpu().numpy()
            fake_male = trimesh.Trimesh(vertices=fake_male_vertices)
            fake_male.export(f"{folder_name}/epoch_{epoch}_male_{idx}.obj")
            wandb.log({"fake_male": wandb.Object3D(fake_male_vertices), "epoch_step": epoch}, commit=False)
            wandb.log({"OG_female": wandb.Object3D(female[0].detach().cpu().numpy()), "epoch_step": epoch}, commit=False)
            wandb.log({"cycle_female": wandb.Object3D(cycle_female.transpose(2,1)[0].detach().cpu().numpy()), "epoch_step": epoch}, commit=False)

        # save idx, D_loss, G_loss, mse, L1 in csv file
        with open(f'output/loss_{currentTime}.csv', 'a') as f: 
           f.write(f'{D_loss},{loss_G_M},{loss_G_F},{cycle_loss},{G_loss},{epoch}\n')
        wandb.log({"D_loss": D_loss, "loss_G_M": loss_G_M, "loss_G_F": loss_G_F,
                   "cycle_loss": cycle_loss, "G_loss": G_loss, "epoch_step": epoch}, commit=False)
        
        # update progress bar
        loop.set_postfix(D_loss = D_loss.item(), G_loss = G_loss.item(), epoch=epoch)


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
    
    chamfer_loss = ChamferLoss() #Chamfer loss used for cycle loss
    mse = nn.MSELoss() # mse loss used for adverserial loss

    # save models optionally
    if config.SAVE_MODEL:
        GEN_M_filename = config.CHECKPOINT_GEN_M
        GEN_F_filename = config.CHECKPOINT_GEN_F
        PRE_GEN_M_filename = config.CHECKPOINT_FOLD_M
        PRE_GEN_F_filename = config.CHECKPOINT_FOLD_F

    # load previously pretrained FoldingNet models optionally
    if config.LOAD_FOLD_MODEL:
        load_checkpoint(config.SAVEDMODEL_GEN_M, gen_M, opt_gen, config.LEARNING_RATE,)
        load_checkpoint(config.SAVEDMODEL_GEN_F, gen_F, opt_gen, config.LEARNING_RATE,)

    # load previously trained models optionally
    if config.LOAD_MODEL:
        load_checkpoint(config.CHECKPOINT_GEN_M, gen_M, opt_gen, config.LEARNING_RATE,)
        load_checkpoint(config.CHECKPOINT_GEN_F, gen_F, opt_gen, config.LEARNING_RATE,)
        load_checkpoint(config.CHECKPOINT_DISC_M, disc_M, opt_disc, config.LEARNING_RATE,)
        load_checkpoint(config.CHECKPOINT_DISC_F, disc_F, opt_disc, config.LEARNING_RATE,)
    
    # define train dataset
    dataset = ObjDataset(
        root_male=config.TRAIN_DIR + "/male", 
        root_female=config.TRAIN_DIR + "/female",
        transform=None,
        n_points=config.N_POINTS
    )

    # define dataloader for train dataset
    loader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS, pin_memory=True)

    #create csv file to store losses
    currentDateAndTime = datetime.now()
    currentTime = currentDateAndTime.strftime("%m.%d.%H.%M.%S")

    # save loss in csv file
    with open(f'output/loss_{currentTime}.csv', 'w') as f: 
        f.write(f"meta:{config.TRAIN_DIR=},{config.BATCH_SIZE=},{config.GAN_NUM_EPOCHS=},{config.LEARNING_RATE=},{config.LAMBDA_CYCLE},{config.GAN_NUM_EPOCHS},{config.LOAD_MODEL=},{config.N_POINTS=}\n")
        f.write('D_loss,G_M_loss,G_F_loss,cycle_loss,G_loss,epoch\n')

    # create folder to save generated point clouds in
    folder_name = f"pre_saved_models/pcds/{currentTime}"
    os.makedirs(folder_name)

    # pretrain FoldingNet generators
    if config.TRAIN_FOLD:
        for epoch in range(config.FOLD_NUM_EPOCH):
            pretrain_FoldingNet(
                gen_M = gen_M,
                gen_F = gen_F,
                loader = loader,
                opt_gen = opt_gen,
                epoch = epoch,
                folder_name = folder_name
            )
            # save models optionally
            if config.SAVE_MODEL and epoch % 50 == 0 and epoch !=0:
                    PRE_GEN_M_filename = PRE_GEN_M_filename[:-8] + f"_E{epoch}.pth.tar"
                    PRE_GEN_F_filename = PRE_GEN_F_filename[:-8] + f"_E{epoch}.pth.tar"
                    save_checkpoint(gen_M, opt_gen, filename=PRE_GEN_M_filename)
                    save_checkpoint(gen_F, opt_gen, filename=PRE_GEN_F_filename)

    # create folder to save generated point clouds in
    folder_name = f"saved_pcds/{currentTime}"
    os.makedirs(folder_name)

    # train the CycleGAN model
    if config.TRAIN_GAN:
        for epoch in range(config.GAN_NUM_EPOCHS):
            train(
                disc_M = disc_M,
                disc_F = disc_F,
                gen_F = gen_F,
                gen_M = gen_M,
                loader = loader,
                opt_disc = opt_disc,
                opt_gen = opt_gen,
                mse = mse,
                epoch = epoch,
                currentTime = currentTime,
                folder_name = folder_name,
                chamfer_loss = chamfer_loss
            )

        # save model for specific epochs
        if config.SAVE_MODEL and (epoch == 299 or epoch==599 or epoch==899 or epoch==1199):
            save_checkpoint(gen_M, opt_gen, filename=GEN_M_filename+f"E{epoch+1}")
            save_checkpoint(gen_F, opt_gen, filename=GEN_F_filename+f"E{epoch+1}")
    
    wandb.finish()

if __name__ == "__main__":
    main()