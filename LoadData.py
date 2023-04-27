# Import libraries
import os
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import open3d as o3d
import warnings
import numpy as np
warnings.filterwarnings("ignore")

# Define a custom dataset class that inherits from Dataset
class ObjDataset(Dataset):
    # Initialize the dataset with the folder path and transform
    def __init__(self, root_female, root_male, transform=None, n_points=2048):
        self.root_female = root_female
        self.root_male = root_male
        self.transform = transform
        self.n_points = n_points

        self.female_pcds = os.listdir(root_female)
        self.male_pcds = os.listdir(root_male)
        self.length_dataset = max(len(self.female_pcds), len(self.male_pcds))
        self.female_len = len(self.female_pcds)
        self.male_len = len(self.male_pcds)
            
    def __len__(self):
        return self.length_dataset
 
    # Return the item at the given index
    def __getitem__(self, index):
        female_pcd = self.female_pcds[index % self.female_len]
        male_pcd = self.male_pcds[index % self.male_len]

        #female_path = f"{self.root_female}/{female_pcd}"
        #male_path = f"{self.root_male}/{male_pcd}"

        female_path = os.path.join(self.root_female, female_pcd)
        male_path = os.path.join(self.root_male, male_pcd)

        female_pcd = o3d.io.read_triangle_mesh(female_path)
        male_pcd = o3d.io.read_triangle_mesh(male_path)

        female_pcd = female_pcd.sample_points_uniformly(number_of_points=self.n_points)
        male_pcd = male_pcd.sample_points_uniformly(number_of_points=self.n_points)
       
        female_pcd = np.asarray(female_pcd.points)
        male_pcd = np.asarray(male_pcd.points)

        # if self.transform:
        #     female_pcd = self.transform(female_pcd)
        #     male_pcd = self.transform(male_pcd)

        # print(type(female_pcd), type(male_pcd))

        return female_pcd, male_pcd