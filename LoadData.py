# Import libraries
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import open3d as o3d
import warnings
warnings.filterwarnings("ignore")

# Define a custom dataset class that inherits from Dataset
class ObjDataset(Dataset):
    # Initialize the dataset with the folder path and transform
    def __init__(self, folder_path, transform=None, target=None):
        # Get the list of obj file names in the folder
        self.file_names = [f for f in os.listdir(folder_path) if f.endswith(".obj")]
        # Store the folder path and transform
        self.folder_path = folder_path
        self.transform = transform
        self.target = target
    
    # Return the length of the dataset
    def __len__(self):
        return len(self.file_names)
    
    # Return the item at the given index
    def __getitem__(self, index):
        # Get the file name at the index
        file_name = self.file_names[index]
        # Load the .obj file
        mesh = o3d.io.read_triangle_mesh(os.path.join(self.folder_path, file_name))
        # Convert vertices and faces to PyTorch tensors
        pcd = torch.tensor(mesh.vertices).float()
        # Apply the transform if given
        if self.transform:
            pcd = self.transform(pcd)
        # Return the point cloud and its index
        return (pcd, self.target), index

def load(path, target):
    # Create an instance of the dataset with a given folder path and no transform
    dataset = ObjDataset(path,target)
    ### Compute the maximum distance of any vertex from the origin in the dataset ###
    # max_dist = 0 # Initialize max_dist with zero
    # with torch.no_grad(): # No need to track gradients for this computation
    #     for verts in dataset: # Iterate over the dataset
    #         dists = torch.norm(verts, dim=1) # Compute the Euclidean distances of vertices from origin
    #         max_dist = max(max_dist, torch.max(dists)) # Update max_dist with the maximum distance in this item
    # Define the maximum distance as a constant
    max_dist = torch.tensor(1.0428)
    # Define the transformation as a lambda function that takes a point cloud and normalizes it
    normalize = transforms.Lambda(lambda x: x / max_dist)
    # Create a new instance of the dataset with the same folder path and normalize transform
    dataset = ObjDataset(path, transform=None)
    return dataset


# Define female and male data separately
female_train = load("data/female_train", target=[1,0])
male_train = load("data/male_train", target=[0,1])
female_test = load("data/female_test", target=[1,0])
male_test = load("data/male_test", target=[0,1])

# Create a data loader with a given batch size and shuffle option
batch_size = 32
female_loader_train = DataLoader(   female_train,  batch_size=batch_size, shuffle=True)
male_loader_train = DataLoader(     male_train,    batch_size=batch_size, shuffle=True)
female_loader_test = DataLoader(    female_test,   batch_size=batch_size, shuffle=True)
male_loader_test = DataLoader(      male_test,     batch_size=batch_size, shuffle=True)


