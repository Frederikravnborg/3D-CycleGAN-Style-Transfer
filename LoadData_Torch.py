# Import libraries
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from pytorch3d.io import load_obj
import warnings
warnings.filterwarnings("ignore")

# Define a custom dataset class that inherits from Dataset
class ObjDataset(Dataset):
    # Initialize the dataset with the folder path and transform
    def __init__(self, folder_path, transform=None):
        # Get the list of obj file names in the folder
        self.file_names = [f for f in os.listdir(folder_path) if f.endswith(".obj")]
        # Store the folder path and transform
        self.folder_path = folder_path
        self.transform = transform
    
    # Return the length of the dataset
    def __len__(self):
        return len(self.file_names)
    
    # Return the item at the given index
    def __getitem__(self, index):
        # Get the file name at the index
        file_name = self.file_names[index]
        # Load the obj file using pytorch3d's load_obj function
        verts, faces, aux = load_obj(os.path.join(self.folder_path, file_name))
        # Convert the vertices and faces to tensors
        verts = torch.tensor(verts, dtype=torch.float32)
        faces = torch.tensor(faces.verts_idx, dtype=torch.int64)
        # Apply the transform if given
        if self.transform:
            verts = self.transform(verts)
        # Return the vertices and faces as a tuple
        return verts

def load(path):
    # Create an instance of the dataset with a given folder path and no transform
    dataset = ObjDataset(path)

    # Compute the maximum distance of any vertex from the origin in the dataset
    max_dist = 0 # Initialize max_dist with zero
    with torch.no_grad(): # No need to track gradients for this computation
        for verts in dataset: # Iterate over the dataset
            dists = torch.norm(verts, dim=1) # Compute the Euclidean distances of vertices from origin
            max_dist = max(max_dist, torch.max(dists)) # Update max_dist with the maximum distance in this item

    # Create a normalize transform using the computed max_dist
    print("Max distance: ", max_dist)
    normalize = transforms.Lambda(lambda x: x / max_dist)

    # Create a new instance of the dataset with the same folder path and normalize transform
    dataset = ObjDataset(path, transform=normalize)
    
    return dataset

# Define female and male data separately
female_data = load("data/mesh_female")
# male_data = load("data/mesh_male")

# Create a data loader with a given batch size and shuffle option
# female_data_loader = DataLoader(female_data, batch_size=32, shuffle=True)

# Iterate over the data loader and print the shapes of the batches
# for batch in female_data_loader:
#     print(batch.shape)