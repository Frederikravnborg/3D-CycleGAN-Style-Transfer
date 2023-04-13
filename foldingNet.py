#%%
from mpl_toolkits import mplot3d
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from LoadData_Torch import data_split
#from utils import index_points, knn

#%%

def index_points(point_clouds, index):
    """
    Given a batch of tensor and index, select sub-tensor.
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, N, k]
    Return:
        new_points:, indexed points data, [B, N, k, C]
    """
    device = point_clouds.device
    batch_size = point_clouds.shape[0]
    view_shape = list(index.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(index.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(batch_size, dtype=torch.long, device=device).view(view_shape).repeat(repeat_shape)
    new_points = point_clouds[batch_indices, index, :]
    return new_points

def knn(x, k):
    """
    K nearest neighborhood.
    Parameters
    ----------
        x: a tensor with size of (B, C, N)
        k: the number of nearest neighborhoods
    
    Returns
    -------
        idx: indices of the k nearest neighborhoods with size of (B, N, k)
    """
    inner = -2 * torch.matmul(x.transpose(2, 1), x)  # (B, N, N)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)  # (B, 1, N)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)  # (B, 1, N), (B, N, N), (B, N, 1) -> (B, N, N)
    
    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (B, N, k)
    
    return idx


# Define a function that visualizes a point cloud
def visualize_point_cloud(pcd):
    my_cmap = plt.get_cmap('hsv')
    # Convert it to a numpy array
    #points = np.asarray(pcd)
    points = pcd.detach().numpy()
    print(pcd)
    print(pcd.shape)
    print(points)
    print(points.shape)

    # Plot it using matplotlib with tiny points and constrained axes
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(points[0,:], points[1,:], points[2,:], s=0.1, cmap=my_cmap)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    #ax.set_box_aspect((1,1,1)) # Constrain the axes
    ax.set_proj_type('ortho') # Use orthographic projection
    #ax.set_xlim(-1,1) # Set x-axis range
    #ax.set_ylim(-1,1) # Set y-axis range
    #ax.set_zlim(-1,1) # Set z-axis range
    plt.show()
#%%

class GraphLayer(nn.Module):
    """
    Graph layer.

    in_channel: it depends on the input of this network.
    out_channel: given by ourselves.
    """
    def __init__(self, in_channel, out_channel, k=16):
        super(GraphLayer, self).__init__()
        self.k = k
        self.conv = nn.Conv1d(in_channel, out_channel, 1)
        self.bn = nn.BatchNorm1d(out_channel)

    def forward(self, x):
        """
        Parameters
        ----------
            x: tensor with size of (B, C, N)
        """
        # KNN
        knn_idx = knn(x, k=self.k)  # (B, N, k)
        knn_x = index_points(x.permute(0, 2, 1), knn_idx)  # (B, N, k, C)

        # Local Max Pooling
        x = torch.max(knn_x, dim=2)[0].permute(0, 2, 1)  # (B, N, C)
        
        # Feature Map
        x = F.relu(self.bn(self.conv(x)))
        return x


class Encoder(nn.Module):
    """
    Graph based encoder.
    """
    def __init__(self):
        super(Encoder, self).__init__()

        self.conv1 = nn.Conv1d(12, 64, 1)
        self.conv2 = nn.Conv1d(64, 64, 1)
        self.conv3 = nn.Conv1d(64, 64, 1)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)

        self.graph_layer1 = GraphLayer(in_channel=64, out_channel=128, k=16)
        self.graph_layer2 = GraphLayer(in_channel=128, out_channel=1024, k=16)

        self.conv4 = nn.Conv1d(1024, 512, 1)
        self.bn4 = nn.BatchNorm1d(512)

    def forward(self, x):
        b, c, n = x.size()

        # get the covariances, reshape and concatenate with x
        knn_idx = knn(x, k=16)
        knn_x = index_points(x.permute(0, 2, 1), knn_idx)  # (B, N, 16, 3)
        mean = torch.mean(knn_x, dim=2, keepdim=True)
        knn_x = knn_x - mean
        covariances = torch.matmul(knn_x.transpose(2, 3), knn_x).view(b, n, -1).permute(0, 2, 1)
        x = torch.cat([x, covariances], dim=1)  # (B, 12, N)

        # three layer MLP
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))


        # two consecutive graph layers
        x = self.graph_layer1(x)
        x = self.graph_layer2(x)

        x = self.bn4(self.conv4(x))

        x = torch.max(x, dim=-1)[0]
        return x


class FoldingLayer(nn.Module):
    """
    The folding operation of FoldingNet
    """

    def __init__(self, in_channel: int, out_channels: list):
        super(FoldingLayer, self).__init__()

        layers = []
        for oc in out_channels[:-1]:
            conv = nn.Conv1d(in_channel, oc, 1)
            bn = nn.BatchNorm1d(oc)
            active = nn.ReLU(inplace=True)
            layers.extend([conv, bn, active])
            in_channel = oc
        out_layer = nn.Conv1d(in_channel, out_channels[-1], 1)
        layers.append(out_layer)
        
        self.layers = nn.Sequential(*layers)

    def forward(self, grids, codewords):
        """
        Parameters
        ----------
            grids: reshaped 2D grids or intermediam reconstructed point clouds
        """
        # concatenate
        x = torch.cat([grids, codewords], dim=1)
        # shared mlp
        x = self.layers(x)
        
        return x


class Decoder(nn.Module):
    """
    Decoder Module of FoldingNet
    """

    def __init__(self, in_channel=512):
        super(Decoder, self).__init__()

        # Sample the grids in 2D space
        xx = np.linspace(-0.3, 0.3, 45, dtype=np.float32)
        yy = np.linspace(-0.3, 0.3, 45, dtype=np.float32)
        self.grid = np.meshgrid(xx, yy)   # (2, 45, 45)

        # reshape
        self.grid = torch.Tensor(self.grid).view(2, -1)  # (2, 45, 45) -> (2, 45 * 45)
        
        self.m = self.grid.shape[1]

        self.fold1 = FoldingLayer(in_channel + 2, [512, 512, 3])
        self.fold2 = FoldingLayer(in_channel + 3, [512, 512, 3])
        self.fold3 = FoldingLayer(in_channel + 3, [512, 512, 3])
        self.fold4 = FoldingLayer(in_channel + 3, [512, 512, 3])
        self.fold5 = FoldingLayer(in_channel + 3, [512, 512, 3])
        self.fold6 = FoldingLayer(in_channel + 3, [512, 512, 3])

    def forward(self, x):
        """
        x: (B, C)
        """
        batch_size = x.shape[0]

        # repeat grid for batch operation
        grid = self.grid.to(x.device)                      # (2, 45 * 45)
        grid = grid.unsqueeze(0).repeat(batch_size, 1, 1)  # (B, 2, 45 * 45)
        
        # repeat codewords
        x = x.unsqueeze(2).repeat(1, 1, self.m)            # (B, 512, 45 * 45)
        
        # two folding operations
        recon1 = self.fold1(grid, x)
        recon2 = self.fold2(recon1, x)
        recon3 = self.fold3(recon2, x)
        #recon4 = self.fold4(recon3, x)
        #recon5 = self.fold5(recon4, x)
        #recon6 = self.fold6(recon5, x)
        
        return recon3


class AutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

#%%

if __name__ == '__main__':
    n: int = 1024*2

    female_data_loader_train, _, _, _ = data_split(n_points=n)
    for batch in female_data_loader_train:
        #visualize_point_cloud(batch[0, :, :])
        pcs = batch.reshape(32, 3, n)
        #print(pcs.size())
        break

    encoder = Encoder()
    codewords = encoder(pcs)
    print(codewords.size())

    decoder = Decoder(codewords.size(1))
    recons = decoder(codewords)
    print(recons.size())

    ae = AutoEncoder()
    y = ae(pcs)
    print(y.size())



#%%
x = 20
print(pcs)
visualize_point_cloud(pcs[0, :, :])
#visualize_point_cloud(codewords)
#visualize_point_cloud(recons[:, :2, :])
#visualize_point_cloud(y[:, :2, :])

# %%
print(pcs[0, :, :].size())
print(recons[0, :, :])
#print(y.size())
# %%
