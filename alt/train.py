
#import of relevant modules and scripts
import torch
from dataloader_dataset import PointCloudDataset
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
import os
import config
import torch.optim as optim
from utils import save_checkpoint, load_checkpoint, ChamferLoss
from Generator import ReconstructionNet as Generator_Fold
# from Discriminator import get_model as Discriminator_Point
from pointnet_model import Discriminator as Discriminator_Point
import wandb
import trimesh



# start a new wandb run to track this script
wandb.init(
    # set the wandb project where this run will be logged
    project="Fagprojekt",
    
    # track hyperparameters and run metadata
    config={
    "BATCH_SIZE": config.BATCH_SIZE,
    "LEARNING_RATE": config.LEARNING_RATE,
    "GAN_NUM_EPOCHS": config.NUM_EPOCHS,
    },
    mode = 'online' if config.USE_WANDB else 'disabled'
)



def train_one_epoch(disc_M, disc_FM, gen_M, gen_FM, loader, opt_disc, opt_gen, mse, chamferloss, return_loss, save_pcl=False):
    train_loop = tqdm(loader, leave=True)
    
    for idx, data in enumerate(train_loop):
        female = data['pc_female'].to(config.DEVICE)
        male = data['pc_male'].to(config.DEVICE)
        fem_ids = data['f_id']
        male_ids = data['m_id']
    

        '''''''''''
        DISCRIMINATORS
        '''''''''''

        # Male discriminator
        fake_male, _ = gen_M(female)
        D_M_real = disc_M(male)[0]
        D_M_fake = disc_M(fake_male.detach())[0]
        
        #if D_M_real.detach()[:,0] > 0.5:
            #D_correct += 1

        #Calculating MSE loss for male
        D_M_real_loss = mse(D_M_real, torch.ones_like(D_M_real))          
        D_M_fake_loss = mse(D_M_fake, torch.zeros_like(D_M_fake))         
        D_M_loss = D_M_real_loss + D_M_fake_loss
        
        #Female discriminator
        fake_female, _ = gen_FM(male)                      # Generating a female from a male pointcloud
        D_FM_real = disc_FM(female)[0]                    # Putting a true female from data through the discriminator
        D_FM_fake = disc_FM(fake_female.detach())[0]       # Putting a generated female through the discriminator

        #Calculate MSE loss for female
        D_FM_real_loss = mse(D_FM_real, torch.ones_like(D_FM_real))           
        D_FM_fake_loss = mse(D_FM_fake, torch.zeros_like(D_FM_fake))          
        D_FM_loss = D_FM_real_loss + D_FM_fake_loss

        #Total discriminator loss
        D_loss = (D_M_loss + D_FM_loss) / 2

        #Update the optimizer for the discriminator
        opt_disc.zero_grad()
        D_loss.backward()
        opt_disc.step()        

        
        '''''''''
        GENERATORS
        '''''''''
        
        #Adviserial loss for both generators
        D_M_fake = disc_M(fake_male)[0]
        D_FM_fake = disc_FM(fake_female)[0]
        loss_G_M = mse(D_M_fake, torch.ones_like(D_M_fake))                          
        loss_G_FM = mse(D_FM_fake, torch.ones_like(D_FM_fake))             

        #Cycle loss
        cycle_female, _ = gen_FM(fake_male)
        cycle_male, _ = gen_M(fake_female)

       
        # Set chamfer loss ind
        cycle_female_loss = chamferloss(cycle_female.transpose(2,1), female.transpose(2,1))
        cycle_male_loss = chamferloss(cycle_male.transpose(2,1), male.transpose(2,1))


        #Adding all generative losses together:
        G_loss = (
            loss_G_FM
            + loss_G_M
            + cycle_female_loss * config.LAMBDA_CYCLE
            + cycle_male_loss * config.LAMBDA_CYCLE
        )

        #Update the optimizer for the generator
        opt_gen.zero_grad()
        G_loss.backward()
        opt_gen.step()
       

        #Save pointclouds for a chosen index:
        if save_pcl:
            if idx == 0:
                original_man = male[0]
                female_male = fake_female[0]
                cycle_man = cycle_male[0]

                print(original_man.shape)
                print(female_male.shape)
                print(cycle_man.shape)

                print("Saving pointclouds")
                wandb.log({'male_OG': wandb.Object3D(original_man.transpose(-2,1).cpu().numpy()),
                        'female_fake': wandb.Object3D(female_male.detach().transpose(-2,1).cpu().numpy()),
                        'male_cyc': wandb.Object3D(cycle_man.detach().transpose(-2,1).cpu().numpy())})
                
                root = os.listdir("./Saved_pointclouds/")
                m = len([i for i in root if f'male_{config.START_SHAPE}' in i]) // 3

                fake_female_vertices = fake_female[0].detach().cpu().numpy().transpose(1,0)
                fake_female = trimesh.Trimesh(vertices=fake_female_vertices)
                fake_female.export(f"./Saved_pointclouds/fake_female_{config.START_SHAPE}_{m*config.save_pointclouds}.obj")

                # torch.save(original_man, f=f"./Saved_pointclouds/male_{config.START_SHAPE}_original_{m*config.save_pointclouds}.pt")
                # torch.save(female_male, f=f"./Saved_pointclouds/male_{config.START_SHAPE}_female_{m*config.save_pointclouds}.pt")
                # torch.save(cycle_man, f=f"./Saved_pointclouds/male_{config.START_SHAPE}_cycle_{m*config.save_pointclouds}.pt")
                

            # if 'SPRING1084.obj' in fem_ids:
            #     idx_female = fem_ids.index('SPRING1084.obj')
                original_woman = female[0]
                male_female = fake_male[0]
                cycle_woman = cycle_female[0]


                wandb.log({'female_OG': wandb.Object3D(original_woman.transpose(-2,1).cpu().numpy()),
                        'male_fake':wandb.Object3D(male_female.detach().transpose(-2,1).cpu().numpy()),
                        'female_cyc':wandb.Object3D(cycle_woman.detach().transpose(-2,1).cpu().numpy())})
                
                root = os.listdir("./Saved_pointclouds/")
                w = len([i for i in root if f'woman_{config.START_SHAPE}' in i]) // 3

                fake_male_vertices = fake_male[0].detach().cpu().numpy().transpose(1,0)
                fake_male = trimesh.Trimesh(vertices=fake_male_vertices)
                fake_male.export(f"./Saved_pointclouds/fake_male_{config.START_SHAPE}_{m*config.save_pointclouds}.obj")

                # torch.save(original_woman, f=f"./Saved_pointclouds/woman_{config.START_SHAPE}_original{w*config.save_pointclouds}.pt")
                # torch.save(male_female, f=f"./Saved_pointclouds/woman_{config.START_SHAPE}_man{w*config.save_pointclouds}.pt")
                # torch.save(cycle_woman, f=f"./Saved_pointclouds/woman_{config.START_SHAPE}_cycle{w*config.save_pointclouds}.pt")
                

    if return_loss:
        return D_loss, G_loss, cycle_female_loss + cycle_male_loss, loss_G_FM + loss_G_M
    



def main():
    args_gen = config.get_parser_gen()

    disc_M = Discriminator_Point(k=2).to(config.DEVICE)
    disc_FM = Discriminator_Point(k=2).to(config.DEVICE)
    gen_M = Generator_Fold(args_gen).to(config.DEVICE)
    gen_FM = Generator_Fold(args_gen).to(config.DEVICE)
    
    return_loss = config.RETURN_LOSS

    opt_disc = optim.Adam(
        list(disc_FM.parameters()) + list(disc_M.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999),
    )

    opt_gen = optim.Adam(
        list(gen_FM.parameters()) + list(gen_M.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999),
    )

    mse = nn.MSELoss()
    chamferloss = ChamferLoss()

    if args_gen.shape == 'feature_shape':
        chamferloss = lambda x, y : torch.sqrt((x - y)**2).sum()
    
    
    #load pretrained wheights from checkpoints
    if config.LOAD_MODEL:
        load_checkpoint(
            config.CHECKPOINT_ALL,
            models=[disc_FM, disc_M, gen_FM, gen_M],
            optimizers=[opt_disc, opt_gen],
            lr=config.LEARNING_RATE,
        )
    
    
    #load training dataset
    if args_gen.dataset == 'dataset':
        dataset = PointCloudDataset(
            root_female=config.TRAIN_DIR + "/female",
            root_male=config.TRAIN_DIR + "/male",
            transform=config.transform
        )    

    loader = DataLoader(dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=True,
            num_workers=config.NUM_WORKERS,
            pin_memory=True,
            collate_fn=config.collate_fn
            )

    for epoch in range(config.NUM_EPOCHS):
        if config.save_pointclouds:
            save_pcl = True if epoch % config.save_pointclouds == 0 else False

        if return_loss:
            D, G, cycle, adv = train_one_epoch(disc_M, disc_FM, gen_M, gen_FM, loader, opt_disc, opt_gen, mse, chamferloss, return_loss, save_pcl)
            wandb.log({"LossD": D, "LossG": G,"Adviserial_loss": adv, "Cycle_loss": cycle, "epoch": epoch+1})
        else: train_one_epoch(disc_M, disc_FM, gen_M, gen_FM, loader, opt_disc, opt_gen, mse, chamferloss, return_loss)
        models, opts = [disc_FM, disc_M, gen_FM, gen_M], [opt_disc, opt_gen]
        if config.SAVE_MODEL and return_loss and epoch+1 == config.NUM_EPOCHS:
            losses = [D, G] 
            save_checkpoint(epoch, models, opts, losses, filename=f"MODEL_OPTS_LOSSES_{config.START_SHAPE}_{epoch+1}.pth.tar")
        #elif config.SAVE_MODEL: save_checkpoint(epoch, models, opts, losses=None, filename=f"MODEL_OPTS_LOSSES_{epoch+1}.pth.tar")
        print(f'The best Discriminator loss for epoch {epoch+1} is {D} and the Generator loss is {G}')
    wandb.finish()

if __name__ == "__main__":
    main()
    