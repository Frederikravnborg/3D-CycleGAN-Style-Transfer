
import torch
import numpy as np
import trimesh
import matplotlib.pyplot as plt
from matplotlib import colors
import glob
import os
import json

# function for converting from json to obj
def convert_to_obj(path):
    # Load the .pts.json file
    with open(path, 'r') as f:
        data = json.load(f)

    # Extract the points from the data
    points = np.array(data)

    # Create a Trimesh object from the points
    mesh = trimesh.Trimesh(vertices=points)

    # Save the mesh as an .obj file in the same directory
    mesh.export(path[:-9] + '.obj')

# function for coloring point clouds
def color_pcd(pcd):
    if pcd.requires_grad:
        pcd = pcd.detach()

    # Normalize x and y coordinates to be between 0 and 1
    x_min, x_max = pcd[:, 2].min(), pcd[:, 2].max()
    y_min, y_max = pcd[:, 0].min(), pcd[:, 0].max()
    normalized_x = (pcd[:, 2] - x_min) / (x_max - x_min)
    normalized_y = (pcd[:, 0] - y_min) / (y_max - y_min)

    # Define colors
    red = torch.Tensor([int(x*255) for x in colors.to_rgb('green')]).unsqueeze(1)
    yellow = torch.Tensor([int(x*255) for x in colors.to_rgb('yellow')]).unsqueeze(1)
    cyan = torch.Tensor([int(x*255) for x in colors.to_rgb('cyan')]).unsqueeze(1)
    green = torch.Tensor([int(x*255) for x in colors.to_rgb('red')]).unsqueeze(1)
    blue = torch.Tensor([int(x*255) for x in colors.to_rgb('blue')]).unsqueeze(1)

    blue_to_yellow = yellow - blue
    yellow_to_cyan = cyan - yellow
    green_to_yellow = yellow - green
    yellow_to_red = red - yellow

    # Compute color for each point
    color_per_point = []
    for x, y in zip(normalized_x, normalized_y):
        if x > 0.5:  # yellow to red
            if y > 0.5:
                # Compute color between yellow and cyan
                new_scale_value_x = (x - 0.5) * 2
                new_scale_value_y = (y - 0.5) * 2
                new_color = ((yellow + yellow_to_red * new_scale_value_x) +
                             (yellow + yellow_to_cyan * new_scale_value_y)) / 2
                color_per_point.append(new_color.int().squeeze().tolist())
            else:
                # Compute color between yellow and blue
                new_scale_value_x = (x - 0.5) * 2
                new_scale_value_y = y / 0.5
                new_color = ((yellow + yellow_to_red * new_scale_value_x) +
                             (blue + blue_to_yellow * new_scale_value_y)) / 2
                color_per_point.append(new_color.int().squeeze().tolist())
        else:  # green to yellow
            if y > 0.5:
                # Compute color between green and yellow
                new_scale_value_x = x / 0.5
                new_scale_value_y = (y - 0.5) * 2
                new_color = ((green + green_to_yellow * new_scale_value_x) +
                             (yellow + yellow_to_cyan * new_scale_value_y)) / 2
                color_per_point.append(new_color.int().squeeze().tolist())
            else:
                # Compute color between green and blue
                new_scale_value_x = x / 0.5
                new_scale_value_y = y / 0.5
                new_color = ((green + green_to_yellow * new_scale_value_x) +
                             (blue + blue_to_yellow * new_scale_value_y)) / 2
                color_per_point.append(new_color.int().squeeze().tolist())

    return np.array(color_per_point)

# function for visualizing point clouds
def visual_pcd(path, axislim=0.6, dotsize=20, border=0.5):
    pcd = trimesh.load(path)
    pcd = torch.from_numpy(pcd.vertices).float()
    color_per_point = color_pcd(pcd)
    pcd = pcd.squeeze().cpu()
    if pcd.requires_grad:
        pcd = pcd.detach()
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.scatter(pcd[:, 0], pcd[:, 1], pcd[:, 2], c=color_per_point/255.0, s=dotsize, edgecolors='black', linewidths=border)
    ax.set_xlim3d(-axislim, axislim)
    ax.set_ylim3d(-axislim, axislim)
    ax.set_zlim3d(-axislim, axislim)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=20, azim=-170, roll=0)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    plt.axis('off')
    plt.show()

# function for visualizing point clouds with angles
def visual_pcd_angles(path, axislim=0.6, dotsize=20, border=0.5, name = "noname", save_file = False):
    pcd = trimesh.load(path)
    pcd = torch.from_numpy(pcd.vertices).float()
    color_per_point = color_pcd(pcd)
    pcd = pcd.squeeze().cpu()
    if pcd.requires_grad:
        pcd = pcd.detach()
    fig, axs = plt.subplots(1, 2, figsize=(15, 5), subplot_kw={'projection': '3d'})
    fig.subplots_adjust(hspace=0, wspace=0.05)
    rotations = [120, -145]
    for i, ax in enumerate(axs):
        ax.scatter(pcd[:, 0], pcd[:, 1], pcd[:, 2], c=color_per_point/255.0, s=dotsize, edgecolors='black', linewidths=border)
        ax.set_xlim3d(-axislim, axislim)
        ax.set_ylim3d(-axislim, axislim)
        ax.set_zlim3d(-axislim, axislim)
        ax.set_box_aspect((1, 1, 1))
        ax.view_init(elev=20, azim=rotations[i], roll=0)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.set_axis_off()
    fig.suptitle(name, fontsize=16)
    if save_file:
        plt.savefig(f"./images/{name}", dpi=300)
    else:
        plt.show()

# function for visualizing multiple point clouds through epochs
def visual_pcd_onetype(path, gender):
    axislim=0.6
    dotsize=20
    border=0.2
    step_size = 5
    num_per_row = 5
    num_row = 2
    num_pcds = num_per_row * num_row 
    epoch_list = [0,1,4,9,19,49,99,199,499,1199]
    fig, axs = plt.subplots(num_row, num_per_row, figsize=(20, 8), subplot_kw={'projection': '3d'})
    fig.subplots_adjust(hspace=0.1, wspace=0.1)
    
    for epoch in range(num_pcds):
        # current_path = f"{path}/epoch_{epoch*step_size**2}_{'female' if gender == 0 else 'male'}_0.obj"
        current_path = f"{path}/epoch_{epoch_list[epoch]}_{'female' if gender == 0 else 'male'}_0.obj"


        pcd = trimesh.load(current_path)
        pcd = torch.from_numpy(pcd.vertices).float()
        color_per_point = color_pcd(pcd)
        pcd = pcd.squeeze().cpu()
        if pcd.requires_grad:
            pcd = pcd.detach()
        
        if num_row > 1:
            ax = axs[epoch // num_per_row, epoch % num_per_row]
        else:
            ax = axs[epoch % num_per_row]
        ax.scatter(pcd[:, 0], pcd[:, 1], pcd[:, 2], c=color_per_point/255.0, s=dotsize, edgecolors='black', linewidths=border)
        ax.set_xlim3d(-axislim, axislim)
        ax.set_ylim3d(-axislim, axislim)
        ax.set_zlim3d(-axislim, axislim)
        ax.set_box_aspect((1, 1, 1))
        ax.view_init(elev=20, azim=-170, roll=0)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        # ax.set_title(f'Epoch {(epoch*step_size**2)+1}')
        ax.set_title(f'Epoch {epoch_list[epoch]+1}')
        ax.axis('off')
    
    plt.show()

# function for visualizing generated validation point clouds
def visual_generated_pcd(path, save_file = False):
    axislim=0.6
    dotsize=20
    border=0.2
    num_per_row = 4
    num_row = 1
    num_pcds = num_per_row * num_row 
    epoch_list = [0, 99, 199, 299]
    fig, axs = plt.subplots(num_row, num_per_row, figsize=(20, 8), subplot_kw={'projection': '3d'})
    fig.subplots_adjust(hspace=0.1, wspace=0.1)
    
    for epoch in range(num_pcds):
        current_path = path
        pcd = trimesh.load(current_path)
        pcd = torch.from_numpy(pcd.vertices).float()
        color_per_point = color_pcd(pcd)
        pcd = pcd.squeeze().cpu()
        if pcd.requires_grad:
            pcd = pcd.detach()
        
        if num_row > 1:
            ax = axs[epoch // num_per_row, epoch % num_per_row]
        else:
            ax = axs[epoch % num_per_row]
        ax.scatter(pcd[:, 0], pcd[:, 1], pcd[:, 2], c=color_per_point/255.0, s=dotsize, edgecolors='black', linewidths=border)
        ax.set_xlim3d(-axislim, axislim)
        ax.set_ylim3d(-axislim, axislim)
        ax.set_zlim3d(-axislim, axislim)
        ax.set_box_aspect((1, 1, 1))
        ax.view_init(elev=20, azim=-170, roll=0)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        # ax.set_title(f'Epoch {(epoch*step_size**2)+1}')
        ax.set_title(f'Fake Female {epoch_list[epoch]+1}')
        ax.axis('off')
    if save_file:
        plt.savefig(f"./images/mode_collapse", dpi=300)
    else:
        plt.show()

# function for visualizing pcds
def visual_custom_pcds(paths, titles, name=None, save_file=False):
    axislim=0.6
    dotsize=20
    border=0.2
    num_per_row = len(paths)
    num_row = 1
    num_pcds = num_per_row * num_row
    fig, axs = plt.subplots(num_row, num_per_row, figsize=(20, 8), subplot_kw={'projection': '3d'})
    fig.subplots_adjust(hspace=0.1, wspace=0.1)
    
    for i in range(num_pcds):
        current_path = paths[i]
        pcd = trimesh.load(current_path)
        pcd = torch.from_numpy(pcd.vertices).float()
        color_per_point = color_pcd(pcd)
        pcd = pcd.squeeze().cpu()
        if pcd.requires_grad:
            pcd = pcd.detach()
        
        if num_row > 1:
            ax = axs[i // num_per_row, i % num_per_row]
        else:
            ax = axs[i % num_per_row]
        ax.scatter(pcd[:, 0], pcd[:, 1], pcd[:, 2], c=color_per_point/255.0, s=dotsize, edgecolors='black', linewidths=border)
        ax.set_xlim3d(-axislim, axislim)
        ax.set_ylim3d(-axislim, axislim)
        ax.set_zlim3d(-axislim, axislim)
        ax.set_box_aspect((1, 1, 1))
        ax.view_init(elev=20, azim=-170, roll=0)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.set_title(titles[i])
        ax.axis('off')
    if save_file:
        plt.savefig(f"./images/{name}", dpi=300)
    else:
        plt.show()

# function for visualizing pretrained foldingnet point clouds
def visual_pcd_pretrain(path, gender):
    axislim = 0.6
    dotsize = 5
    border = 0.2
    step_size = 5
    epoch_list = [0, 2, 9, 49, 199]
    num_per_row = len(epoch_list)
    num_row = len(path)
    num_pcds = num_per_row * num_row 
    fig, axs = plt.subplots(num_row, num_per_row, figsize=(20, 8), subplot_kw={'projection': '3d'})
    fig.subplots_adjust(hspace=0, wspace=0)
    
    for i in range(num_row):
        current_path = path[i]
        for epoch in range(num_per_row):
            file_path = f"{current_path}epoch_{epoch_list[epoch]}_{'female' if gender == 0 else 'male'}_0.obj"

            pcd = trimesh.load(file_obj=file_path, header=None)
            pcd = torch.from_numpy(pcd.vertices).float()
            color_per_point = color_pcd(pcd)
            pcd = pcd.squeeze().cpu()
            if pcd.requires_grad:
                pcd = pcd.detach()
            
            if num_row > 1:
                ax = axs[epoch // num_per_row + i, epoch % num_per_row]
            else:
                ax = axs[epoch % num_per_row]
                
            ax.scatter(pcd[:, 0], pcd[:, 1], pcd[:, 2], c=color_per_point / 255.0, s=dotsize, edgecolors='black', linewidths=border)
            ax.set_xlim3d(-axislim, axislim)
            ax.set_ylim3d(-axislim, axislim)
            ax.set_zlim3d(-axislim, axislim)
            ax.set_box_aspect((1, 1, 1))
            ax.view_init(elev=20, azim=-170, roll=0)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_zticks([])
            if i == 0:            
                ax.set_title(f'Epoch {epoch_list[epoch] + 1}')
            ax.axis('off')
    plt.savefig('./images/pretrain_females.png', dpi=400)
    plt.show()

if __name__ == "__main__":
    '''Pretrain'''
    # path = ["./results_pcd/1200_E25/",
    #         "./results_pcd/1200_E50/",
    #         "./results_pcd/1200_E75/",
    #         "./results_pcd/1200_E100/"]

    # gender = 0
    # visual_pcd_pretrain(path, gender)


    '''One type'''
    # path = "./results_pcd/baseline"
    # gender = 0
    # visual_pcd_onetype(path, gender)



    '''mode collapse'''
    # path = "./data/val/generated_female/female_0.obj"
    # visual_generated_pcd(path, save_file=True)

    # '''custom pcds'''
    paths = [
             "/Users/frederikravnborg/Documents/DTU-FredMac/Fagprojekt/Fagprojekt_Git/results_pcd/OG/OG_male_461.obj",
             "/Users/frederikravnborg/Documents/DTU-FredMac/Fagprojekt/Fagprojekt_Git/results_pcd/baseline/epoch_460_female_0.obj",
             "/Users/frederikravnborg/Documents/DTU-FredMac/Fagprojekt/Fagprojekt_Git/results_pcd/OG/cycle_male_460.obj",
             ]
    paths = [
            "/Users/frederikravnborg/Documents/DTU-FredMac/Fagprojekt/Fagprojekt_Git/results_pcd/OG/female_0.obj",
            "/Users/frederikravnborg/Documents/DTU-FredMac/Fagprojekt/Fagprojekt_Git/results_pcd/baseline/epoch_600_male_0.obj",
            "/Users/frederikravnborg/Documents/DTU-FredMac/Fagprojekt/Fagprojekt_Git/results_pcd/baseline/epoch_454_female_0.obj" 
    ]
    
    titles = ["OG female", "fake male", "cycle female"]
    visual_custom_pcds(paths, titles, name="illustration_FMF", save_file=True)


    '''Angles'''
    # path = paths[2]
    # epoch = 460
    # gender = 1
    # gender_path = ["female" if gender == 0 else "male"][0]
    # root = f"epoch_{epoch}_{gender_path}_0.obj"
    # visual_pcd_angles(path, name = "cycle_male_461", save_file=True)


    # convert_to_obj("./results_pcd/OG/OG_male_461.pts.json")



    # # plot row titles alone
    # texts = ["Pretrained 100 epochs", "Pretrained 75 epochs", "Pretrained 50 epochs", "Pretrained 25 epochs"]
    # fig, ax = plt.subplots()
    # for i, text in enumerate(texts):
    #     ax.text(0, i*2, text, fontsize=8)
    # ax.set_xlim([-1, 1])
    # ax.set_ylim([-0.2, len(texts)*2-0.2])
    # ax.set_xlabel('X Label')
    # ax.set_ylabel('Y Label')
    # ax.axis('off')
    # plt.savefig('./images/xxx', dpi=300)
    