#%%
import pandas as pd
import matplotlib.pyplot as plt
from load_data import ObjDataset
from torch.utils.data import DataLoader
import config
from tqdm import tqdm
import trimesh
import numpy as np
import config
from sklearn.neighbors import NearestNeighbors


def chamfer_distance(x, y, metric='l2', direction='bi'):
    """Chamfer distance between two point clouds
    Parameters
    ----------
    x: numpy array [n_points_x, n_dims]
        first point cloud
    y: numpy array [n_points_y, n_dims]
        second point cloud
    metric: string or callable, default ‘l2’
        metric to use for distance computation. Any metric from scikit-learn or scipy.spatial.distance can be used.
    direction: str
        direction of Chamfer distance.
            'y_to_x':  computes average minimal distance from every point in y to x
            'x_to_y':  computes average minimal distance from every point in x to y
            'bi': compute both
    Returns
    -------
    chamfer_dist: float
        computed bidirectional Chamfer distance:
            sum_{x_i \in x}{\min_{y_j \in y}{||x_i-y_j||**2}} + sum_{y_j \in y}{\min_{x_i \in x}{||x_i-y_j||**2}}
    """
    
    if direction=='y_to_x':
        x_nn = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm='kd_tree', metric=metric).fit(x)
        min_y_to_x = x_nn.kneighbors(y)[0]
        chamfer_dist = np.mean(min_y_to_x)
    elif direction=='x_to_y':
        y_nn = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm='kd_tree', metric=metric).fit(y)
        min_x_to_y = y_nn.kneighbors(x)[0]
        chamfer_dist = np.mean(min_x_to_y)
    elif direction=='bi':
        x_nn = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm='kd_tree', metric=metric).fit(x)
        min_y_to_x = x_nn.kneighbors(y)[0]
        y_nn = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm='kd_tree', metric=metric).fit(y)
        min_x_to_y = y_nn.kneighbors(x)[0]
        chamfer_dist = np.mean(min_y_to_x) + np.mean(min_x_to_y)
    else:
        raise ValueError("Invalid direction type. Supported types: \'y_x\', \'x_y\', \'bi\'")
        
    return chamfer_dist

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


#%%
if __name__ == '__main__':

    # define validation dataset
    val_dataset = ObjDataset(
        root_male= config.VAL_DIR + "/male",
        root_female= config.VAL_DIR + "/female",
        transform=None,
        n_points=config.N_POINTS
    )

    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True, pin_memory=True)


    meshes = []
    gen_meshes = []

    loop = tqdm(val_loader, leave=True) #Progress bar
    count = 0

    
    

    for _, (female, male) in enumerate(loop):
        female = female.float()
        temp = np.asarray(female[0])

        meshes.append(temp)
    
    losses = np.zeros((len(meshes), len(meshes)))
    
    # calcuate chamfer loss between each array within gen_meshes:
    for i in range(0, len(meshes)):
        for j in range(0, len(meshes)):
            if i != j:
                chamfer_loss = chamfer_distance(meshes[i], meshes[j], metric='euclidean', direction='bi')
                losses[i][j] = chamfer_loss
    # save losses to npy
    np.save("losses.npy", losses)

    for x in range(0, 306):
        gen_mesh = trimesh.load(f"data/val/generated_female/female_{x}" + ".obj")
        temp = np.asarray(gen_mesh.vertices)

        gen_meshes.append(temp)
    
    gen_losses = np.zeros((len(gen_meshes), len(gen_meshes)))
    
    # calcuate chamfer loss between each array within gen_meshes:
    for i in range(0, len(gen_meshes)):
        for j in range(0, len(gen_meshes)):
            if i != j:
                chamfer_loss = chamfer_distance(gen_meshes[i], gen_meshes[j], metric='euclidean', direction='bi')
                gen_losses[i][j] = chamfer_loss
    # save losses to npy
    np.save("gen_losses.npy", gen_losses)
    print("done")

    

    


#%%

# load meshes data
losses = np.load("losses.npy")

# visualize loss in histogram
plt.hist(np.mean(losses, axis=1), bins=100)
plt.xlabel('Chamfer distance')
plt.ylabel('Frequency')
plt.title('Chamfer distance between generated meshes')
plt.show()

#%%
losses = np.load("gen_losses.npy")

# visualize loss in histogram
plt.hist(np.mean(losses, axis=1), bins=100)
plt.xlabel('Chamfer distance')
plt.ylabel('Frequency')
plt.title('Chamfer distance between generated meshes')
plt.show()
# %%
