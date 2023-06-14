
import torch
import numpy as np
import trimesh
import matplotlib.pyplot as plt
from matplotlib import colors
import glob
import os


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



# def visual_pcd(path, axislim=0.6, dotsize=20, border=0.5):
#     pcd = trimesh.load(path)
#     pcd = torch.from_numpy(pcd.vertices).float()
#     color_per_point = color_pcd(pcd)
#     pcd = pcd.squeeze().cpu()
#     if pcd.requires_grad:
#         pcd = pcd.detach()
#     fig = plt.figure()
#     ax = fig.add_subplot(projection="3d")
#     ax.scatter(pcd[:, 0], pcd[:, 1], pcd[:, 2], c=color_per_point/255.0, s=dotsize, edgecolors='black', linewidths=border)
#     ax.set_xlim3d(-axislim, axislim)
#     ax.set_ylim3d(-axislim, axislim)
#     ax.set_zlim3d(-axislim, axislim)
#     ax.set_box_aspect((1, 1, 1))
#     ax.view_init(elev=20, azim=-170, roll=0)
#     ax.set_xticks([])
#     ax.set_yticks([])
#     ax.set_zticks([])
#     plt.axis('off')
#     plt.show()

    

# if __name__ == "__main__":
#     path = "./results_pcd/Pretrain_Foldingnet/"
#     epoch = 10
#     gender = 0
#     gender_path = ["female" if gender == 0 else "male"][0]
#     root = f"epoch_{epoch}_{gender_path}_0.obj"


def visual_pcd(path, gender):
    axislim=0.6
    dotsize=20
    border=0.2
    step_size = 5
    num_per_row = 5
    num_row = 2
    num_pcds = num_per_row * num_row 
    epoch_list = [0,4,9,19,49,99,199,499,799,1199]
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

if __name__ == "__main__":
    path = "./results_pcd/baseline/"
    gender = 1
    visual_pcd(path, gender)