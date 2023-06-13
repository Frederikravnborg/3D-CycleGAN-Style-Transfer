import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
from utilities.pointnet_utils import PointNetEncoder
import config
from utilities.pointnet_utils import feature_transform_regularizer
import numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class get_model(nn.Module):
    def __init__(self, k=2):
        super(get_model, self).__init__()
        channel = 3
        r = config.DISC_WIDTH_REDUCER
        self.feat = PointNetEncoder(global_feat=True, feature_transform=True, channel=channel)
        self.fc1 = nn.Linear(1024, int(512/r))
        self.fc2 = nn.Linear(int(512/r), int(256/r))
        self.fc3 = nn.Linear(int(256/r), k)
        self.dropout = nn.Dropout(p=0.4)
        self.bn1 = nn.BatchNorm1d(int(512/r))
        self.bn2 = nn.BatchNorm1d(int(256/r))
        self.relu = nn.ReLU()

    def forward(self, x):
        #x = x.transpose(2, 1)
        #print(x.shape)
        x = torch.from_numpy(x.cpu().numpy()).float()
        x, trans, trans_feat = self.feat(x.to(device))
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        x = self.fc3(x)
        # return torch.sigmoid(x)

        x = F.log_softmax(x, dim=1)
        # x = F.softmax(x, dim=1)
        return x, trans_feat

class get_loss(torch.nn.Module):
    def __init__(self, mat_diff_loss_scale=0.001):
        super(get_loss, self).__init__()
        self.mat_diff_loss_scale = mat_diff_loss_scale

    def forward(self, pred, target, trans_feat):
        loss = F.nll_loss(pred.squeeze(), target.squeeze().long())
        mat_diff_loss = feature_transform_regularizer(trans_feat)

        total_loss = loss + mat_diff_loss * self.mat_diff_loss_scale
        return total_loss
