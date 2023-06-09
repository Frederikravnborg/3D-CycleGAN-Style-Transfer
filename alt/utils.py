import random, torch, os, numpy as np
import torch.nn as nn
import torch.nn.parallel
import config
import copy
import torch.nn.functional as F
import matplotlib.pyplot as plt
from dataloader_dataset import PointCloudDataset

def save_checkpoint(epoch, models : list, optimizers, losses, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    checkpoint = {}
    checkpoint['epoch']= epoch
    
    for m in range(len(models)):
        checkpoint[("state_dict_"+str(m))] = models[m].state_dict()
    for opt in range(len(optimizers)):
        checkpoint[("optimizer_"+str(opt))] = optimizers[opt].state_dict()
    #"optimizer": optimizers.state_dict()
    if losses is not None:
        checkpoint["losses"] = losses
    torch.save(checkpoint, filename)


def load_checkpoint(checkpoint_file, models, optimizers, lr):
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location=config.DEVICE)
    for m in range(len(models)):
        
        models[m].load_state_dict(checkpoint["state_dict_"+str(m)],strict = False)
    for opt in range(len(optimizers)):
        optimizers[opt].load_state_dict(checkpoint['optimizer_'+str(opt)])
        for param_group in optimizers[opt].param_groups:
            param_group["lr"] = lr
    epoch = checkpoint["epoch"]
    losses = checkpoint["losses"]
    # If we don't do this then it will just have learning rate of old checkpoint
    # and it will lead to many hours of debugging \:
    

class ChamferLoss(nn.Module):
    def __init__(self):
        super(ChamferLoss, self).__init__()
        self.use_cuda = torch.cuda.is_available()
    
    def batch_pairwise_dist(self, x, y):
        bs, num_points_x, points_dim = x.size()
        _, num_points_y, _ = y.size()
        xx = x.pow(2).sum(dim=-1)
        yy = y.pow(2).sum(dim=-1)
        zz = torch.bmm(x, y.transpose(2,1))
        rx = xx.unsqueeze(1).expand_as(zz.transpose(2, 1))
        ry = yy.unsqueeze(1).expand_as(zz)
        P = (rx.transpose(2, 1) + ry - 2 * zz)
        return P

    def forward(self, preds, gts):
        P = self.batch_pairwise_dist(gts, preds)
        mins, _ = torch.min(P, 1)
        loss_1 = torch.mean(mins)
        mins, _ = torch.min(P, 2)
        loss_2 = torch.mean(mins)
        return loss_1 + loss_2

class CrossEntropyLoss(nn.Module):
    def __init__(self, smoothing=True):
        super(CrossEntropyLoss, self).__init__()
        self.smoothing = smoothing
    
    def forward(self, preds, gts):
        gts = gts.contiguous().view(-1)

        if self.smoothing:
            eps = 0.2
            n_class = preds.size(1)

            one_hot = torch.zeros_like(preds).scatter(1, gts.view(-1, 1), 1)
            one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
            log_prb = F.log_softmax(preds, dim=1)

            loss = -(one_hot * log_prb).sum(dim=1).mean()
        else:
            loss = F.cross_entropy(preds, gts, reduction='mean')

        return loss


def visualize(cloud,id,gender):
    name = f"Visualization of {gender} pointcloud {id}"
    points = cloud.numpy() #takes tensor makes it into np.array [x,y,z] koordinates
    # Creating figure
    fig = plt.figure()
    ax = plt.axes(projection ="3d")

    x,y,z = points[:,0], points[:,1], points[:,2]
    # Add x, y gridlines
    ax.grid(b = True, color ='grey',
            linestyle ='-.', linewidth = 0.3,
            alpha = 0.8)


    # Creating color map
    my_cmap = plt.get_cmap('cool')#plt.get_cmap('hsv')

    # Creating plot
    sctt = ax.scatter3D(x, y, z,
                        alpha = 0.5,
                        c = (x + y + z),
                        cmap = my_cmap,
                        marker = 'o',
                        s=0.5)

    plt.title(str(name))
    ax.set_xlabel('X-axis', fontweight ='bold')
    ax.set_ylabel('Y-axis', fontweight ='bold')
    ax.set_zlabel('Z-axis', fontweight ='bold')
    ax.view_init(elev=20,azim=-160,roll=0)
    ax.set_box_aspect((1,1,1)) # Constrain the axes
    ax.set_proj_type('ortho') # Use orthographic projection
    ax.set_xlim(-0.75,0.75) # Set x-axis range
    ax.set_ylim(-0.75,0.75) # Set y-axis range
    ax.set_zlim(-0.75,0.75) # Set z-axis range
    #fig.colorbar(sctt, ax = ax, shrink = 0.5, aspect = 5)

    # show plot
    plt.show()


def isqrt(n):
    x = n
    y = (x + 1) // 2
    while y < x:
        x = y
        y = (x + n // x) // 2
    return int(x ** 2)

if __name__ == "__main__":
    #pcl = torch.load("./Saved_pointclouds/male_cycle2.pt", map_location=torch.device('cpu'))
    #any(val in x  for x in lst)
    print(['yes','no'].index('yes'))
    # print(isqrt(2048))
    # data = PointCloudDataset()
    # pcl = data[3]['m_pcs']
    # visualize(pcl, id = '1', gender = 'male')