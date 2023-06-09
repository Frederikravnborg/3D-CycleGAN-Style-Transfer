
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset
import config as c
import os
import trimesh as tr

class PointCloudDataset(Dataset):

    def __init__(self, root_female="./data/train/female", root_male="./data/train/male", transform = None):
        self.root_female = root_female
        self.root_male = root_male
        self.transform = transform
        self.sample_points = c.SAMPLE_POINTS

        self.object_female = os.listdir(root_female)
        self.object_male = os.listdir(root_male)
        self.length_dataset = max(len(self.object_female), len(self.object_male))
        self.male_len = len(self.object_male)
        self.female_len = len(self.object_female)
        #self.object_all = np.concatenate((self.object_female, self.object_male), axis=0)

        self.furthest_distance = c.FURTHEST_DISTANCE #calculated in furthest_distance.py 


    def __len__(self):
        return self.length_dataset

    #given index, output a pointcloud as a tensor from both domains, have been normalized
    def __getitem__(self, idx):
        #get index for both domains
        male_obj = self.object_male[idx % self.male_len]
        female_obj = self.object_female[idx % self.female_len]

        #create path to the indexed objects
        male_path = os.path.join(self.root_male, male_obj)
        female_path = os.path.join(self.root_female, female_obj)
       
       # load meshes 
       
        male_file = tr.load(male_path)
        female_file = tr.load(female_path)
       
        #sample points from meshes
        
        pcl_male,_ = tr.sample.sample_surface_even(male_file, self.sample_points)
        pcl_female,_ = tr.sample.sample_surface_even(female_file, self.sample_points)
        if len(pcl_male[0])<self.sample_points:
            pcl_male,_ = tr.sample.sample_surface_even(male_file, self.sample_points+50)
            pcl_male = pcl_male[:self.sample_points,:]
            
        if len(pcl_female[0])<self.sample_points:
            pcl_female,_ = tr.sample.sample_surface_even(female_file, self.sample_points+50)
            pcl_female = pcl_female[:self.sample_points,:]
            
       
        male_array, female_array = np.asarray(pcl_male), np.asarray(pcl_female)
        
        

        #convert from .obj to torch tensor
        #pcl_male = o3d.io.read_triangle_mesh(male_path).sample_points_uniformly(number_of_points=self.sample_points) 
        #pcl_female = o3d.io.read_triangle_mesh(female_path).sample_points_uniformly(number_of_points=self.sample_points)
        
        #male_array, female_array = np.asarray(pcl_male.points), np.asarray(pcl_female.points)
        male_array, female_array = male_array.astype(np.float32), female_array.astype(np.float32)
        #perform transformation / augmentation if turned on
        if self.transform:
            female_array, male_array = self.transform(female=female_array, male = male_array)
        
        #normalize with regard to furtherst point in the whole dataset
        for pcl in (male_array, female_array):
            centroid = np.mean(pcl, axis=0)
            pcl -= centroid
            pcl /= self.furthest_distance
        #return a tensor pointcloud for each domain
        
        male_pointcloud, female_pointcloud = torch.from_numpy(male_array), torch.from_numpy(female_array)
        
        #return female_pointcloud, male_pointcloud
        return {"f_pcs": female_pointcloud,"m_pcs" : male_pointcloud, "id_female":female_obj, "id_male":male_obj }

    #Make a function that returns the normalvector for the points in a pointcloud
    

if __name__ == "__main__":
    data = PointCloudDataset()
    print(data[424])
    data
# data = PointCloudDataset()
# female, male = data[4]
# breakpoint()
# print(female)


