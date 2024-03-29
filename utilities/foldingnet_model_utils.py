import numpy as np
import torch

# create a sphere in 3D space with n_points points
def create_sphere(n_points):
    # create points on a sphere
    points = []
    phi = np.pi * (3. - np.sqrt(5.))            # golden angle in radians

    for i in range(n_points):
        y = 1 - (i / float(n_points - 1)) * 2
        radius = np.sqrt(1 - y * y)
        theta = phi * i

        x = np.cos(theta) * radius
        z = np.sin(theta) * radius

        points.append([x, y, z])

    return np.array(points)

# create gaussian noise in 3D space space with n_points points
def create_gaussian(n_points):
    return np.random.normal(size=(n_points, 3))

# define knn function
def knn(x, k):
    batch_size = x.size(0)
    num_points = x.size(2)

    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
 
    idx = pairwise_distance.topk(k=k, dim=-1)[1]                    # (batch_size, num_points, k)

    if idx.get_device() == -1:
        idx_base = torch.arange(0, batch_size).view(-1, 1, 1)*num_points
    else:
        idx_base = torch.arange(0, batch_size, device=idx.get_device()).view(-1, 1, 1)*num_points
    idx = idx + idx_base
    idx = idx.view(-1)

    return idx

# define local covariance function
def local_cov(pts, idx):
    batch_size = pts.size(0)
    num_points = pts.size(2)
    pts = pts.view(batch_size, -1, num_points)                      # (batch_size, 3, num_points)
 
    _, num_dims, _ = pts.size()

    x = pts.transpose(2, 1).contiguous()                            # (batch_size, num_points, 3)
    x = x.view(batch_size*num_points, -1)[idx, :]                   # (batch_size*num_points*2, 3)
    x = x.view(batch_size, num_points, -1, num_dims)                # (batch_size, num_points, k, 3)

    x = torch.matmul(x[:,:,0].unsqueeze(3), x[:,:,1].unsqueeze(2))  # (batch_size, num_points, 3, 1) * (batch_size, num_points, 1, 3) -> (batch_size, num_points, 3, 3)
    x = x.view(batch_size, num_points, 9).transpose(2, 1)           # (batch_size, 9, num_points)

    x = torch.cat((pts, x), dim=1)                                  # (batch_size, 12, num_points)

    return x

# define local maxpool function
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

# define graph feature function
def get_graph_feature(x, k=3, idx=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)                              # (batch_size, num_dims, num_points)
    if idx is None:
        idx = knn(x, k=k)                                               # (batch_size, num_points, k)
 
    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()                                  # (batch_size, num_points, num_dims)
    feature = x.view(batch_size*num_points, -1)[idx, :]                 # (batch_size*n, num_dims) -> (batch_size*n*k, num_dims)
    feature = feature.view(batch_size, num_points, k, num_dims)         # (batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)  # (batch_size, num_points, k, num_dims)
    
    feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2)      # (batch_size, num_points, k, 2*num_dims) -> (batch_size, 2*num_dims, num_points, k)
  
    return feature                                                      # (batch_size, 2*num_dims, num_points, k)
