# Import libraries
import os
from torch.utils.data import Dataset
import warnings
import trimesh
import glob
import numpy as np
import config
warnings.filterwarnings("ignore")

# Define a custom dataset class that inherits from Dataset
class ObjDataset(Dataset):
    # Initialize the dataset with the folder path and transform
    def __init__(self, root_female, root_male, transform=None, n_points=2048):
        self.root_female = root_female
        self.root_male = root_male

        self.n_points = n_points
        self.female_pcds = glob.glob(os.path.join(root_female, '*.obj'))
        self.male_pcds = glob.glob(os.path.join(root_male, '*.obj'))

        self.length_dataset = max(len(self.female_pcds), len(self.male_pcds))
        self.female_len = len(self.female_pcds)
        self.male_len = len(self.male_pcds)
            
    def __len__(self):
        return self.length_dataset 
    
    # Return the item at the given index
    def __getitem__(self, index):
        female_pcd = self.female_pcds[index % self.female_len]             # get female pcds by index
        male_pcd = self.male_pcds[index % self.male_len]                   # get male pcds by index

        # load via trimesh
        female_pcd = trimesh.load(female_pcd)                                           
        male_pcd = trimesh.load(male_pcd)

        female_pcd = female_pcd.sample(self.n_points)
        male_pcd = male_pcd.sample(self.n_points)

        # normalize the point cloud data
        female_pcd = (female_pcd - np.mean(female_pcd, axis=0)) / config.MAX_DISTANCE
        male_pcd = (male_pcd - np.mean(male_pcd, axis=0)) / config.MAX_DISTANCE

        return female_pcd, male_pcd


