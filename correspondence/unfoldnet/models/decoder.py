import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable
import os

class ImplicitFun(nn.Module):
    def __init__(self, z_dim=256, num_branches=12):
        super(ImplicitFun, self).__init__()
        input_dim = z_dim+3

        self.unfold1 = mlpAdj(nlatent=input_dim)
        self.unfold2 = mlpAdj(nlatent=input_dim)

    def forward(self, z, points):

        num_pts = points.shape[1]
        z = z.unsqueeze(1).repeat(1, num_pts, 1)
        pointz = torch.cat((points, z), dim=2).float()

        x1 = self.unfold1(pointz)
        x2 = torch.cat((x1, z), dim=2)
        x3 = self.unfold2(x2)

        return x3


class mlpAdj(nn.Module):
    def __init__(self, nlatent = 1024):
        """Atlas decoder"""

        super(mlpAdj, self).__init__()
        self.nlatent = nlatent
        self.conv1 = torch.nn.Conv1d(self.nlatent, self.nlatent, 1)
        self.conv2 = torch.nn.Conv1d(self.nlatent, self.nlatent//2, 1)
        self.conv3 = torch.nn.Conv1d(self.nlatent//2, self.nlatent//4, 1)
        self.conv4 = torch.nn.Conv1d(self.nlatent//4, 3, 1)

        self.th = nn.Tanh()
        self.bn1 = torch.nn.BatchNorm1d(self.nlatent)
        self.bn2 = torch.nn.BatchNorm1d(self.nlatent//2)
        self.bn3 = torch.nn.BatchNorm1d(self.nlatent//4)

    def forward(self, x):
        batchsize = x.size()[0]
        x = x.transpose(2,1).contiguous()
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.th(self.conv4(x))
        return x.transpose(2,1)
