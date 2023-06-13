import pandas as pd
import matplotlib.pyplot as plt
from load_data import ObjDataset
from torch.utils.data import DataLoader
import config
from foldingnet_model import ChamferLoss
from tqdm import tqdm
import trimesh
import torch
import numpy as np
import glob
import os
import config

def visualize_loss():
    filename = "output/loss_06.07.15.18.59.csv"

    # read data from csv file using pandas
    meta = pd.read_csv(filename, nrows=1)
    data = pd.read_csv(filename ,skiprows=1)
    print(meta)

    # extract data from dataframe
    epoch = data['epoch']
    G_loss = []
    cycle_loss = []
    D_loss = []

    indexes = []
    step = 0
    for i in range (len(epoch)):
        indexes = data.index.get_indexer(data.query(f'epoch == {step}').index)
        G_loss.append(data['G_loss'][indexes].mean())
        cycle_loss.append(data['cycle_loss'][indexes].mean())
        D_loss.append(data['D_loss'][indexes].mean())
        step += 1

    # generate 2 plots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.plot(G_loss, label='G_loss')
    ax1.plot(cycle_loss, label='cycle_loss')
    ax1.legend()
    # add labels
    ax1.set(xlabel='Epoch', ylabel='Loss', title='Losses')
    ax2.plot(D_loss, label='D_loss')
    ax2.legend()
    ax2.set(xlabel='Epoch', ylabel='Loss', title='discriminator loss')
    plt.show()

# Function that takes Chamfer Loss for all combinations of generated models, and real models
def visualize_chamfer_loss(val_loader, gen_loader):
    meshes = []
    gen_meshes = []

    loop = tqdm(val_loader, leave=True) #Progress bar
    gen_loop = tqdm(gen_loader, leave=True) #Progress bar

    for _, (female, male) in enumerate(loop):
        female = female.to(config.DEVICE)
        male = male.to(config.DEVICE)

        meshes.append(female)

    for _, (gen_female, gen_male) in enumerate(gen_loop):
        gen_female = gen_female.to(config.DEVICE)
        gen_male = gen_male.to(config.DEVICE)

        gen_meshes.append(gen_female)


    

    # loop through the data loader
    
    
    return meshes, gen_meshes



if __name__ == '__main__':

    # define validation dataset
    val_dataset = ObjDataset(
        root_male= config.VAL_DIR + "/male",
        root_female= config.VAL_DIR + "/female",
        transform=None,
        n_points=config.N_POINTS
    )
    gen_dataset = ObjDataset(
        root_male= config.VAL_DIR + "/generated_male",
        root_female= config.VAL_DIR + "/generated_female",
        transform=None,
        n_points=config.N_POINTS
    )

    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, pin_memory=True)

    meshes = []
    gen_meshes = []

    real_pcds = glob.glob(os.path.join("data/val/female", '*.obj'))

    loop = tqdm(val_loader, leave=True) #Progress bar

    #for _, (female, male) in enumerate(loop):
    #    female = female.float()
    #    temp = female.detach().cpu().numpy()
    #    meshes.append(temp[0])

    for i in range(306):
        gen_female = trimesh.load(f"data/val/generated_female/female_{i}.obj")

        gen_temp = gen_female.vertices
        #gen_temp = np.asarray(gen_temp)

        gen_meshes.append(gen_temp)
    
    print("hi")

    loss = ChamferLoss()
    gen_loss = np.zeros((len(gen_meshes), len(gen_meshes)))

    
    gen_loss[i, x] = loss(torch.Tensor(gen_meshes), torch.Tensor(gen_meshes))
    
    print("hi")
