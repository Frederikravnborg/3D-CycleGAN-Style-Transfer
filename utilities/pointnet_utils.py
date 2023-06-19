import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import config

r = config.DISC_WIDTH_REDUCER

# pointnet first T-net transfomer
class STN3d(nn.Module):
    def __init__(self, channel):
        super(STN3d, self).__init__()
        self.conv1 = torch.nn.Conv1d(channel, int(64/r), 1)
        self.conv2 = torch.nn.Conv1d(int(64/r), int(128/r), 1)
        self.conv3 = torch.nn.Conv1d(int(128/r), int(1024/r), 1)
        self.fc1 = nn.Linear(int(1024/r), int(512/r))
        self.fc2 = nn.Linear(int(512/r), int(256/r))
        self.fc3 = nn.Linear(int(256/r), 9)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(int(64/r))
        self.bn2 = nn.BatchNorm1d(int(128/r))
        self.bn3 = nn.BatchNorm1d(int(1024/r))
        self.bn4 = nn.BatchNorm1d(int(512/r))
        self.bn5 = nn.BatchNorm1d(int(256/r))

    def forward(self, x):
        batchsize = x.size()[0]                                                 # get the batch size from the input tensor
        x = F.relu(self.bn1(self.conv1(x)))                                     # apply the first convolutional layer, batch normalization and relu activation
        x = F.relu(self.bn2(self.conv2(x)))                                     # apply the second convolutional layer, batch normalization and relu activation
        x = F.relu(self.bn3(self.conv3(x)))                                     # apply the third convolutional layer, batch normalization and relu activation
        x = torch.max(x, 2, keepdim=True)[0]                                    # apply max pooling along the second dimension (feature dimension)
        x = x.view(-1, int(1024/r))                                             # reshape the tensor to have a size of (batchsize, 1024)

        x = F.relu(self.bn4(self.fc1(x)))                                       # apply the first fully connected layer, batch normalization and relu activation
        x = F.relu(self.bn5(self.fc2(x)))                                       # apply the second fully connected layer, batch normalization and relu activation
        x = self.fc3(x)                                                         # apply the third fully connected layer

        iden = Variable(torch.from_numpy(
                        np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]
                        ).astype(np.float32))).view(1, 9).repeat(batchsize, 1)  # create an identity matrix of size (batchsize, 9)
        if x.is_cuda:
            iden = iden.cuda()                                                  # move the identity matrix to cuda if the input tensor is on cuda
        x = x + iden                                                            # add the identity matrix to the output of the third fully connected layer
        x = x.view(-1, 3, 3)                                                    # reshape the tensor to have a size of (batchsize, 3, 3)
        return x                                                                # return the output tensor

# pointnet second T-net transfomer
class STNkd(nn.Module):
    def __init__(self, k=int(64/r)):
        super(STNkd, self).__init__()
        self.conv1 = torch.nn.Conv1d(k, int(64/r), 1)
        self.conv2 = torch.nn.Conv1d(int(64/r), int(128/r), 1)
        self.conv3 = torch.nn.Conv1d(int(128/r), int(1024/r), 1)
        self.fc1 = nn.Linear(int(1024/r), int(512/r))
        self.fc2 = nn.Linear(int(512/r), int(256/r))
        self.fc3 = nn.Linear(int(256/r), k * k)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(int(64/r))
        self.bn2 = nn.BatchNorm1d(int(128/r))
        self.bn3 = nn.BatchNorm1d(int(1024/r))
        self.bn4 = nn.BatchNorm1d(int(512/r))
        self.bn5 = nn.BatchNorm1d(int(256/r))

        self.k = k

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, int(1024/r))

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.eye(self.k).flatten().astype(np.float32))).view(1, self.k * self.k).repeat(
            batchsize, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x

# pointnet feature transform network
class PointNetEncoder(nn.Module):
    def __init__(self, global_feat=True, feature_transform=False, channel=3):
        super(PointNetEncoder, self).__init__()
        self.stn = STN3d(channel)
        self.conv1 = torch.nn.Conv1d(channel, int(64/r), 1)
        self.conv2 = torch.nn.Conv1d(int(64/r), int(128/r), 1)
        self.conv3 = torch.nn.Conv1d(int(128/r), int(1024/r), 1)
        self.bn1 = nn.BatchNorm1d(int(64/r))
        self.bn2 = nn.BatchNorm1d(int(128/r))
        self.bn3 = nn.BatchNorm1d(int(1024/r))
        self.global_feat = global_feat
        self.feature_transform = feature_transform
        if self.feature_transform:
            self.fstn = STNkd(k=int(64/r))

    def forward(self, x):
        B, D, N = x.size()
        trans = self.stn(x)
        x = x.transpose(2, 1)
        if D > 3:
            feature = x[:, :, 3:]
            x = x[:, :, :3]
        x = torch.bmm(x, trans)
        if D > 3:
            x = torch.cat([x, feature], dim=2)
        x = x.transpose(2, 1)
        x = F.relu(self.bn1(self.conv1(x)))

        if self.feature_transform:
            trans_feat = self.fstn(x)
            x = x.transpose(2, 1)
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2, 1)
        else:
            trans_feat = None

        pointfeat = x
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, int(1024/r))
        if self.global_feat:
            return x, trans, trans_feat
        else:
            x = x.view(-1, int(1024/r), 1).repeat(1, 1, N)
            return torch.cat([x, pointfeat], 1), trans, trans_feat


def feature_transform_regularizer(trans):
    d = trans.size()[1]
    I = torch.eye(d)[None, :, :]
    if trans.is_cuda:
        I = I.cuda()
    loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2, 1)) - I, dim=(1, 2)))
    return loss
