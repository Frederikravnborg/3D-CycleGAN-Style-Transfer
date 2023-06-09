
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import numpy as np
import itertools
from utils import ChamferLoss, CrossEntropyLoss
import config
from dataloader_dataset import PointCloudDataset



def knn(x, k):
    batch_size = x.size(0)
    num_points = x.size(2)

    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
 
    idx = pairwise_distance.topk(k=k, dim=-1)[1]            # (batch_size, num_points, k)

    if idx.get_device() == -1:
        idx_base = torch.arange(0, batch_size).view(-1, 1, 1)*num_points
    else:
        idx_base = torch.arange(0, batch_size, device=idx.get_device()).view(-1, 1, 1)*num_points
    idx = idx + idx_base
    idx = idx.view(-1)

    return idx

def local_maxpool(x, idx):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
 
    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()                      # (batch_size, num_points, num_dims)
    x = x.view(batch_size*num_points, -1)[idx, :]           # (batch_size*n, num_dims) -> (batch_size*n*k, num_dims)
    x = x.view(batch_size, num_points, -1, num_dims)        # (batch_size, num_points, k, num_dims)
    x, _ = torch.max(x, dim=2)                              # (batch_size, num_points, num_dims)

    return x

def local_cov(pts, idx):
    batch_size = pts.size(0)
    num_points = pts.size(2)
    pts = pts.view(batch_size, -1, num_points)              # (batch_size, 3, num_points)
 
    _, num_dims, _ = pts.size()

    x = pts.transpose(2, 1).contiguous()                    # (batch_size, num_points, 3)
    x = x.view(batch_size*num_points, -1)[idx, :]           # (batch_size*num_points*2, 3)
    x = x.view(batch_size, num_points, -1, num_dims)        # (batch_size, num_points, k, 3)

    #Uncomment the bottommost comment to use local covariance as proposed in the FoldingNet paper
    x = torch.matmul(x[:,:,0].unsqueeze(3), x[:,:,1].unsqueeze(2))  # (batch_size, num_points, 3, 1) * (batch_size, num_points, 1, 3) -> (batch_size, num_points, 3, 3)
    #x = torch.matmul(x[:,:,1:].transpose(3, 2), x[:,:,1:])

    x = x.view(batch_size, num_points, 9).transpose(2, 1)   # (batch_size, 9, num_points)

    x = torch.cat((pts, x), dim=1)                          # (batch_size, 12, num_points)

    return x

class FoldNet_Encoder(nn.Module):
    def __init__(self, args):
        super(FoldNet_Encoder, self).__init__()
        if args.k == None:
            self.k = 16
        else:
            self.k = args.k
        self.n = 2048   # input point cloud size
        self.mlp1 = nn.Sequential(
            nn.Conv1d(12, 64, 1),
            nn.ReLU(),
            nn.Conv1d(64, 64, 1),
            nn.ReLU(),
            nn.Conv1d(64, 64, 1),
            nn.ReLU(),
        )
        self.linear1 = nn.Linear(64, 64)
        self.conv1 = nn.Conv1d(64, 128, 1)
        self.linear2 = nn.Linear(128, 128)
        self.conv2 = nn.Conv1d(128, 1024, 1)
        self.mlp2 = nn.Sequential(
            nn.Conv1d(1024, args.feat_dims, 1),
            nn.ReLU(),
            nn.Conv1d(args.feat_dims, args.feat_dims, 1),
        )
        

    def graph_layer(self, x, idx):           
        x = local_maxpool(x, idx)    
        x = self.linear1(x)  
        x = x.transpose(2, 1)                                     
        x = F.relu(self.conv1(x))                            
        x = local_maxpool(x, idx)  
        x = self.linear2(x) 
        x = x.transpose(2, 1)                                   
        x = self.conv2(x)                       
        return x

    def forward(self, pts):
        #pts = pts.transpose(2, 1)               # (batch_size, 3, num_points)
        idx = knn(pts, k=self.k)
        x = local_cov(pts, idx)                 # (batch_size, 3, num_points) -> (batch_size, 12, num_points])            
        x = self.mlp1(x)                        # (batch_size, 12, num_points) -> (batch_size, 64, num_points])
        global feature_shape
        feature_shape = x 
        x = self.graph_layer(x, idx)            # (batch_size, 64, num_points) -> (batch_size, 1024, num_points)
        x = torch.max(x, 2, keepdim=True)[0]    # (batch_size, 1024, num_points) -> (batch_size, 1024, 1)
        x = self.mlp2(x)                        # (batch_size, 1024, 1) -> (batch_size, feat_dims, 1)
        feat = x.transpose(2,1)                 # (batch_size, feat_dims, 1) -> (batch_size, 1, feat_dims)
        return feat                             # (batch_size, 1, feat_dims)

class FoldNet_Decoder(nn.Module):
    def __init__(self, args):
        super(FoldNet_Decoder, self).__init__()
        self.m = 2025  # 45 * 45.
        self.shape = args.shape
        if self.shape == 'feature_shape': self.m = 2048
        self.meshgrid = [[-1, 1, int(np.sqrt(self.m))], [-1, 1, int(np.sqrt(self.m))]]
        self.sphere = np.load("sphere.npy")
        self.gaussian = np.load("gaussian.npy")
        if self.shape == 'plane':
            self.folding1 = nn.Sequential(
                nn.Conv1d(args.feat_dims+2, args.feat_dims, 1),
                nn.ReLU(),
                nn.Conv1d(args.feat_dims, args.feat_dims, 1),
                nn.ReLU(),
                nn.Conv1d(args.feat_dims, 3, 1),
            )
        else:
            self.folding1 = nn.Sequential(
                nn.Conv1d(args.feat_dims+3, args.feat_dims, 1),
                nn.ReLU(),
                nn.Conv1d(args.feat_dims, args.feat_dims, 1),
                nn.ReLU(),
                nn.Conv1d(args.feat_dims, 3, 1),
            )  
        self.folding2 = nn.Sequential(
            nn.Conv1d(args.feat_dims+3, args.feat_dims, 1),
            nn.ReLU(),
            nn.Conv1d(args.feat_dims, args.feat_dims, 1),
            nn.ReLU(),
            nn.Conv1d(args.feat_dims, 3, 1),
        )
        # MLP for our experiment with a new shape, which is created based on the given input
        if self.shape == 'feature_shape':
            self.mlp3 = nn.Sequential(
                nn.Conv1d(64, 256, 1),
                nn.ReLU(),
                nn.Conv1d(256, 256, 1),
                nn.ReLU(),
                nn.Conv1d(256, 3, 1),
                nn.Tanh(),
            )

    def build_grid(self, batch_size):
        if self.shape == 'plane':
            x = np.linspace(*self.meshgrid[0])
            y = np.linspace(*self.meshgrid[1])
            points = np.array(list(itertools.product(x, y)))
        elif self.shape == 'sphere':
            points = self.sphere
        elif self.shape == 'gaussian':
            points = self.gaussian
        elif self.shape == 'feature_shape':
            points = self.mlp3(feature_shape)
            return points.transpose(2,1).float()
        points = np.repeat(points[np.newaxis, ...], repeats=batch_size, axis=0)
        points = torch.tensor(points)
        return points.float()

    def forward(self, x):
        x = x.transpose(1, 2).repeat(1, 1, self.m)      # (batch_size, feat_dims, num_points)
        points = self.build_grid(x.shape[0]).transpose(1, 2)  # (batch_size, 2, num_points) or (batch_size, 3, num_points)
        if x.get_device() != -1:
            points = points.cuda(x.get_device())
        cat1 = torch.cat((x, points), dim=1)            # (batch_size, feat_dims+2, num_points) or (batch_size, feat_dims+3, num_points)
        folding_result1 = self.folding1(cat1)           # (batch_size, 3, num_points)
        cat2 = torch.cat((x, folding_result1), dim=1)   # (batch_size, 515, num_points)
        folding_result2 = self.folding2(cat2)           # (batch_size, 3, num_points)
        return folding_result2.transpose(1, 2)          # (batch_size, num_points ,3)

class ReconstructionNet(nn.Module):
    def __init__(self, args):
        super(ReconstructionNet, self).__init__()
        #if args.encoder == 'foldnet':
        self.encoder = FoldNet_Encoder(args)
        self.decoder = FoldNet_Decoder(args)
        self.loss = ChamferLoss()

    def forward(self, input):
        feature = self.encoder(input)
        output = self.decoder(feature)
        return output.transpose(2,1), feature

    def get_parameter(self):
        return list(self.encoder.parameters()) + list(self.decoder.parameters())

    def get_loss(self, input, output):
        # input shape  (batch_size, 2048, 3)
        # output shape (batch_size, 2025, 3)
        return self.loss(input, output)

if __name__ == '__main__':
    
    data = PointCloudDataset()
    female = data[0]['f_pcs']
    
    args = config.get_parser_gen()
    Gen = ReconstructionNet(args)
    Gen(female.unsqueeze(0).transpose(2,1))
    female
