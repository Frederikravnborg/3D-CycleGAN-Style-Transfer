# Import libraries
import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import open3d as o3d
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# Set seed
torch.manual_seed(0)

max_dist_female = torch.tensor(1.0407)
max_dist_male = torch.tensor(1.0428)

# Define a custom dataset class that inherits from Dataset
class ObjDataset(Dataset):
    # Initialize the dataset with the folder path and transform
    def __init__(self, folder_path, n_points, transform=None):
        # Get the list of obj file names in the folder
        self.file_names = [f for f in os.listdir(folder_path) if f.endswith(".obj")]
        # Store the folder path and transform
        self.folder_path = folder_path
        self.n_points = n_points
        self.transform = transform
    
    # Return the length of the dataset
    def __len__(self):
        return len(self.file_names)
    
    # Return the item at the given index
    def __getitem__(self, index):
        # Get the file name at the index
        file_name = self.file_names[index]

        # Load the .obj file
        mesh = o3d.io.read_triangle_mesh(os.path.join(self.folder_path, file_name))
        #for mesh in meshes:
            # Down sample the mesh to n points
        sampled_mesh = mesh.sample_points_uniformly(number_of_points=self.n_points)

        # Convert vertices and faces to PyTorch tensors
        np_mesh = np.asarray(sampled_mesh.points)
        verts = torch.from_numpy(np_mesh).float()
        #verts = torch.tensor(sampled_mesh.vertices).float()

        # Apply the transform if given
        if self.transform:
            verts = self.transform(verts)
        # Return the vertices and faces as a tuple
        return verts

def load(path, n_points=2048):
    # Create an instance of the dataset with a given folder path and no transform
    dataset = ObjDataset(path, n_points)

    ### Compute the maximum distance of any vertex from the origin in the dataset ###
    # max_dist = 0 # Initialize max_dist with zero
    # with torch.no_grad(): # No need to track gradients for this computation
    #     for verts in dataset: # Iterate over the dataset
    #         dists = torch.norm(verts, dim=1) # Compute the Euclidean distances of vertices from origin
    #         max_dist = max(max_dist, torch.max(dists)) # Update max_dist with the maximum distance in this item

    # Create a normalize transform using the computed max_dist
    if "female" in path:
        max_dist = max_dist_female
    else:
        max_dist = max_dist_male

    normalize = transforms.Lambda(lambda x: x / max_dist)

    # Create a new instance of the dataset with the same folder path and normalize transform
    dataset = ObjDataset(path, n_points, transform=normalize)
    
    return dataset

def data_split(n_points):

    # Define female and male data separately
    female_train = load("data/female_train", n_points)
    male_train = load("data/male_train", n_points)
    female_test = load("data/female_test", n_points)
    male_test = load("data/male_test", n_points)

    # Create a data loader with a given batch size and shuffle option
    batch_size = 32

    female_data_loader_train = DataLoader(female_train, batch_size=batch_size, shuffle=True)
    male_data_loader_train = DataLoader(male_train, batch_size=batch_size, shuffle=True)

<<<<<<< Updated upstream
    female_data_loader_test = DataLoader(female_test, batch_size=batch_size, shuffle=True)
    male_data_loader_test = DataLoader(male_test, batch_size=batch_size, shuffle=True)

    # Iterate over the data loader and print the shapes of the batches
    # for batch in female_data_loader:
    #     print(batch.shape)

    return female_data_loader_train, female_data_loader_test, male_data_loader_train, male_data_loader_test

print("Done")
=======
# Iterate over the data loader and print the shapes of the batches
# for batch in female_data_loader:
#     print(batch.shape)


# Create a 3d tensor of shape (4, 4, 4) with random values
tensor = female_train[0]

# Convert the tensor to a numpy array
array = tensor.numpy()

# Plot the array as a 3d surface
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
x, y = np.meshgrid(np.arange(4), np.arange(4))
ax.plot_surface(x, y, array[0], cmap='viridis')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.show()
>>>>>>> Stashed changes
