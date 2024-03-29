import torch.nn as nn
import torch.nn.functional as F
from utilities.pointnet_utils import PointNetEncoder
import config

class Discriminator(nn.Module):
    def __init__(self, k=2):
        super(Discriminator, self).__init__()
        channel = 3
        r = config.DISC_WIDTH_REDUCER

        # pointnet encoder
        self.feat = PointNetEncoder(global_feat=True, 
                                    feature_transform=True, 
                                    channel=channel)
        
        # linear block
        self.fc1 = nn.Linear(1024, int(512/r))
        self.fc2 = nn.Linear(int(512/r), int(256/r))
        self.fc3 = nn.Linear(int(256/r), k)

        # dropout layer
        self.dropout = nn.Dropout(p=0.4)     

        # batch normalization layers                                  
        self.bn1 = nn.BatchNorm1d(int(512/r))
        self.bn2 = nn.BatchNorm1d(int(256/r))

        # activation function
        self.relu = nn.ReLU()

    def forward(self, x):
        x, _, trans_feat = self.feat(x)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        x = self.fc3(x)

        x = F.softmax(x, dim=1)
        return x, trans_feat

# class get_loss(torch.nn.Module):
#     def __init__(self, mat_diff_loss_scale=0.001):
#         super(get_loss, self).__init__()
#         self.mat_diff_loss_scale = mat_diff_loss_scale

#     def forward(self, pred, target, trans_feat):
#         loss = F.nll_loss(pred, target)
#         mat_diff_loss = feature_transform_regularizer(trans_feat)

#         total_loss = loss + mat_diff_loss * self.mat_diff_loss_scale
#         return total_loss
