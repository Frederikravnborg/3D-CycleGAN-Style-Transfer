import matplotlib.pyplot as plt
#import pyntcloud as PyntCloud
import pandas as pd
import numpy as np
import torch
import matplotlib.colors as colors
import trimesh
import os
from utilities.foldingnet_model_utils import create_sphere


os.environ['KMP_DUPLICATE_LIB_OK']='True'

#female = trimesh.load("/data/train/female/SPRING0014.obj")

def visualize_point_cloud(pcd):
    my_cmap = plt.get_cmap('hsv')
    # Convert it to a numpy array
    #points = np.asarray(pcd)
    #points = pcd.detach().numpy()
    points = pcd

    # Plot it using matplotlib with tiny points and constrained axes
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(points[:,0], points[:,1], points[:,2], s=0.1, cmap=my_cmap)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    #ax.set_box_aspect((1,1,1)) # Constrain the axes
    ax.set_proj_type('ortho') # Use orthographic projection
    #ax.set_xlim(-1,1) # Set x-axis range
    #ax.set_ylim(-1,1) # Set y-axis range
    #ax.set_zlim(-1,1) # Set z-axis range
    plt.show()

#visualize sphere
sphere = create_sphere(2048)
print(sphere.shape)
visualize_point_cloud(sphere)


quit()



def save_cloud_rgb(cloud, filename):
    green = torch.Tensor([int(x*255) for x in colors.to_rgb('forestgreen')]).unsqueeze(1).to("cuda")
    yellow = torch.Tensor([int(x*255) for x in colors.to_rgb('gold')]).unsqueeze(1).to("cuda")
    blue = torch.Tensor([int(x*255) for x in colors.to_rgb('blue')]).unsqueeze(1).to("cuda")
    red = torch.Tensor([255, 0, 0]).unsqueeze(1).to("cuda")
    cloud = cloud.cpu()
    d = {'x': cloud[0],
         'y': cloud[1],
         'z': cloud[2],
         'red': red,
         'green': green,
         'blue': blue}
    cloud_pd = pd.DataFrame(data=d)
    cloud_pd[['red', 'green', 'blue']] = cloud_pd[['red', 'green', 'blue']].astype(np.uint8) 
    cloud = PyntCloud(cloud_pd)
    cloud.to_file(filename)

def color_pc(point_cloud):
    green = torch.Tensor([int(x*255) for x in colors.to_rgb('forestgreen')]).unsqueeze(1).to("cuda")
    yellow = torch.Tensor([int(x*255) for x in colors.to_rgb('gold')]).unsqueeze(1).to("cuda")
    red = torch.Tensor([255, 0, 0]).unsqueeze(1).to("cuda")
    green_to_yellow = yellow - green
    yellow_to_red = red - yellow
    # normalize x-axis of point cloud to be between 0 and 1
    normalized_x = (point_cloud[:, 0, :] - point_cloud[:, 0, :].min()) / (
                point_cloud[:, 0, :].max() - point_cloud[:, 0, :].min())
    normalized_x = 1 -normalized_x.squeeze()
    color_per_point = []
    for x in normalized_x:
        if x < 0.5: # green to yellow
            new_scale_value = x.item() / 0.5
            new_color = green + green_to_yellow*new_scale_value
            color_per_point.append(new_color.int().squeeze().tolist())
        else: # yellow to red
            new_scale_value = (x.item() - 0.5) * 2
            new_color = yellow + yellow_to_red * new_scale_value
            color_per_point.append(new_color.int().squeeze().tolist())
    color_per_point = np.array(color_per_point)
    return color_per_point

save_cloud_rgb(torch.rand(3, 1000).to("cuda"), "test.ply")