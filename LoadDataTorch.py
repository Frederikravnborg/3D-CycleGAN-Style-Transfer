import json
import os
import random

import numpy as np
import torch
import torch.utils.data as data
import glob
import open3d as o3d
import matplotlib.pyplot as plt


class Human_dataset(data.Dataset):
    def __init__(self, npoints=1024, split='train', normalize=False, data_augmentation=False):
        self.npoints = npoints
        self.split = split
        self.normalize = normalize
        self.data_augmentation = data_augmentation
        self.cat2id = {}

        """ # load classname and class id
        with open('misc/modelnet40category2id.txt', 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.cat2id[ls[0]] = int(ls[1])
        self.id2cat = {v: k for k, v in self.cat2id.items()}

        # find all .h5 files
        with open(os.path.join(self.root, '{}_files.txt'.format(split)), 'r') as f:
            data_files = [os.path.join(
                self.root, line.strip().split('/')[-1]) for line in f] """

    # Define a function that loads data from obj files
    def load_data(self):
        # Specify the path of the folder containing obj files
        path = "data/mesh_female/"
        # List all the files with .obj extension in the path
        files = glob.glob(path + "*.obj")
        print(files)
        # Create an empty list to store meshes
        female_meshes = []
        # Loop over each file name

        for file in files[:4]:
            # Read a cloud from an obj file
            mesh = o3d.io.read_triangle_mesh(file)
            # Append the cloud to the list of point clouds
            female_meshes.append(mesh)

        path = "data/mesh_male/"
        # List all the files with .obj extension in the path
        files = glob.glob(path + "*.obj")
        # Create an empty list to store meshes
        male_meshes = []
        # Loop over each file name

        for file in files[:4]:
            # Read a cloud from an obj file
            mesh = o3d.io.read_triangle_mesh(file)
            # Append the cloud to the list of clouds
            male_meshes.append(mesh)

        female_pcds = []
        male_pcds = []

        for mesh in female_meshes:
            # Create a point cloud object from the mesh
            pcd = mesh.sample_points_uniformly(number_of_points=2048)
            # Save it as a pcd file
            # pdc = o3d.io.write_point_cloud("example.pcd", pcd)
            female_pcds.append(pcd)

        for mesh in male_meshes:
            # Create a point cloud object from the mesh
            pcd = mesh.sample_points_uniformly(number_of_points=2048)
            # Save it as a pcd file
            # pdc = o3d.io.write_point_cloud("example.pcd", pcd)
            male_pcds.append(pcd)

        self.male_pcds = male_pcds
        self.female_pcds = female_pcds
        return self.female_pcds, self.male_pcds

    def __getitem__(self, index):
        females = np.asarray(self.female_pcds[index])
        males = np.asarray(self.male_pcds[index])

        #point_cloud = np.asarray(point_cloud.points)

        # select self.npoints from the original point cloud randomly
        #choice = np.random.choice(len(point_cloud), self.npoints, replace=True)
        #point_cloud = point_cloud[choice, :]
        # Define a function that normalizes a list of point clouds

        # Convert point clouds to numpy arrays
        points_list = [np.asarray(pcd.points) for pcd in self.female_pcds + self.male_pcds]
        
        # Compute the global centroid and the maximum distance from the centroid
        global_centroid = np.mean(np.concatenate(points_list, axis=0), axis=0)
        max_dist = np.max([np.max(np.linalg.norm(points - global_centroid, axis=1)) for points in points_list])
        
        # Translate and scale the points to fit within a unit circle
        female_points = [np.asarray(pcd.points) for pcd in self.female_pcds]
        male_points = [np.asarray(pcd.points) for pcd in self.male_pcds]
        normalized_points_female = [(points - global_centroid) / max_dist for points in female_points]
        normalized_points_male = [(points - global_centroid) / max_dist for points in male_points]

        # Create new point clouds with the normalized points
        normalized_pcds_female = [o3d.geometry.PointCloud() for _ in range(len(self.female_pcds))]
        for i in range(len(self.female_pcds)):
            normalized_pcds_female[i].points = o3d.utility.Vector3dVector(normalized_points_female[i])

        normalized_pcds_male = [o3d.geometry.PointCloud() for _ in range(len(self.male_pcds))]
        for i in range(len(self.male_pcds)):
            normalized_pcds_male[i].points = o3d.utility.Vector3dVector(normalized_points_male[i])
        # Return the normalized point clouds
            # normalize into a sphere whose radius is 1


        # data augmentation - random rotation and random jitter
        # if self.data_augmentation:
        #     #random jitter
        #     females += np.random.normal(0, 0.02, size=females.shape)
        female_pcds = [torch.from_numpy(np.asarray(female)) for female in normalized_pcds_female]
        return female_pcds


if __name__ == '__main__':
    females = Human_dataset(
        split='train', data_augmentation=False, npoints=2048)
    males = Human_dataset(
        split='test', data_augmentation=False, npoints=2048)

    females.load_data()
    point_cloud = females[random.randint(
        0, 0)]
    print(point_cloud.shape)

    # Visualize the point cloud using plotly

    # Extract the x, y, z coordinates from the tensor
    x = point_cloud[:, 0].numpy()
    y = point_cloud[:, 1].numpy()
    z = point_cloud[:, 2].numpy()

    # Create a figure and a 3D axis
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the points with blue markers
    ax.scatter(x, y, z, c='b', marker='.')

    # Set the axis labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Point Cloud Visualization')
    ax.set_box_aspect((1,1,1)) # Constrain the axes
    ax.set_proj_type('ortho') # Use orthographic projection
    ax.set_xlim(-1,1) # Set x-axis range
    ax.set_ylim(-1,1) # Set y-axis range
    ax.set_zlim(-1,1) # Set z-axis range

    # Show the figure
    plt.show()