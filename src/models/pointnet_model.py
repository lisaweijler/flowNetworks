#######
## Code from here https://github.com/yanx27/Pointnet_Pointnet2_pytorch/blob/master/models/pointnet_utils.py

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

from .base_model import BaseModel


class STN3d(nn.Module):
    def __init__(self, channel):
        super(STN3d, self).__init__()
        self.channel = channel
        self.conv1 = torch.nn.Conv1d(channel, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, channel * channel)
        self.relu = nn.GELU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.gelu(self.bn1(self.conv1(x)))
        x = F.gelu(self.bn2(self.conv2(x)))
        x = F.gelu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.gelu(self.bn4(self.fc1(x)))
        x = F.gelu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        # iden = Variable(torch.from_numpy(np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]).astype(np.float32))).view(1, 9).repeat(
        #    batchsize, 1)
        iden = (
            Variable(torch.eye(self.channel, dtype=torch.float32))
            .view(1, self.channel * self.channel)
            .repeat(batchsize, 1)
        )

        iden = iden.to(x.device)
        x = x + iden
        x = x.view(-1, self.channel, self.channel)
        return x


class STNkd(nn.Module):
    def __init__(self, k=64):
        super(STNkd, self).__init__()
        self.conv1 = torch.nn.Conv1d(k, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)
        self.relu = nn.GELU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.k = k

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.gelu(self.bn1(self.conv1(x)))
        x = F.gelu(self.bn2(self.conv2(x)))
        x = F.gelu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.gelu(self.bn4(self.fc1(x)))
        x = F.gelu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = (
            Variable(torch.from_numpy(np.eye(self.k).flatten().astype(np.float32)))
            .view(1, self.k * self.k)
            .repeat(batchsize, 1)
        )

        iden = iden.to(x.device)
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x


class PointNetEncoder(nn.Module):
    def __init__(
        self,
        use_global_feat=True,
        return_global_feat=False,
        feature_transform=False,
        point_transform=True,
        channel=3,
        adapted=False,
        concat=False,
    ):
        super(PointNetEncoder, self).__init__()
        self.point_transform = point_transform
        self.use_global_feat = use_global_feat
        self.concat = concat
        self.adapted = adapted
        if self.point_transform:
            self.stn = STN3d(channel)
        self.conv1 = torch.nn.Conv1d(channel, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.return_global_feat = return_global_feat
        self.feature_transform = feature_transform
        if self.feature_transform:
            self.fstn = STNkd(k=64)

    def forward(self, x):

        B, D, N = x.size()
        if self.point_transform:
            trans = self.stn(x)
            x = x.transpose(2, 1)
            # if D > 3:
            #    feature = x[:, :, 3:]
            #    x = x[:, :, :3]
            x = torch.bmm(x, trans)
            # if D > 3:
            #    x = torch.cat([x, feature], dim=2)
            x = x.transpose(2, 1)
        else:
            trans = None
        x = F.gelu(self.bn1(self.conv1(x)))

        if self.feature_transform:
            trans_feat = self.fstn(x)
            x = x.transpose(2, 1)
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2, 1)
        else:
            trans_feat = None

        pointfeat = x
        x = F.gelu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        if self.use_global_feat:

            x_global = torch.max(x, 2, keepdim=True)[0]
            x_global = x_global.view(-1, 1024)

            if self.return_global_feat:
                return x_global, trans, trans_feat
            else:

                x_global = x_global.view(-1, 1024, 1).repeat(1, 1, N)
                if not self.adapted:
                    return torch.cat([x, pointfeat], 1), trans, trans_feat
                else:
                    if self.concat:
                        return torch.cat([x, x_global], 1), trans, trans_feat
                    else:
                        return x + x_global, trans, trans_feat
        else:
            return x, trans, trans_feat


def feature_transform_reguliarzer(trans):
    d = trans.size()[1]
    I = torch.eye(d)[None, :, :]
    I = I.to(trans.device)
    loss = torch.mean(
        torch.norm(torch.bmm(trans, trans.transpose(2, 1)) - I, dim=(1, 2))
    )
    return loss


class PointNetSemSeg(BaseModel):
    def __init__(
        self,
        num_class: int,
        use_global_feat: bool,
        return_global_feat: bool,
        feature_transform: bool,
        point_transform: bool,
        channel: int,
        adapted: bool = False,
        concat: bool = False,
    ):
        super(PointNetSemSeg, self).__init__()
        self.k = num_class
        self.feat = PointNetEncoder(
            use_global_feat=use_global_feat,
            return_global_feat=return_global_feat,
            feature_transform=feature_transform,
            point_transform=point_transform,
            channel=channel,
            adapted=adapted,
            concat=concat,
        )
        if use_global_feat and not adapted:
            self.conv1 = torch.nn.Conv1d(1088, 512, 1)
        elif use_global_feat and concat:
            self.conv1 = torch.nn.Conv1d(2048, 512, 1)
        else:
            self.conv1 = torch.nn.Conv1d(1024, 512, 1)
        self.conv2 = torch.nn.Conv1d(512, 256, 1)
        self.conv3 = torch.nn.Conv1d(256, 128, 1)
        self.conv4 = torch.nn.Conv1d(128, self.k, 1)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)

    def forward(self, x):
        # B,N, D -> trasnpose it so it fits in this model
        x = x.transpose(2, 1).contiguous()
        batchsize = x.size()[0]
        n_pts = x.size()[2]
        x, trans, trans_feat = self.feat(x)
        x = F.gelu(self.bn1(self.conv1(x)))
        x = F.gelu(self.bn2(self.conv2(x)))
        x = F.gelu(self.bn3(self.conv3(x)))
        x = self.conv4(x)
        x = x.transpose(2, 1).contiguous()
        # x = F.log_softmax(x.view(-1,self.k), dim=-1)
        x = x.view(batchsize, n_pts, self.k)
        return x, trans_feat


class PointNetLoss(torch.nn.Module):
    def __init__(self, mat_diff_loss_scale=0.001):
        super(PointNetLoss, self).__init__()
        self.mat_diff_loss_scale = mat_diff_loss_scale

    def forward(self, pred, target, trans_feat, weight=None):

        loss = F.binary_cross_entropy_with_logits(pred, target)
        if trans_feat is not None:
            mat_diff_loss = feature_transform_reguliarzer(trans_feat)
            total_loss = loss + mat_diff_loss * self.mat_diff_loss_scale
            return total_loss
        else:
            return loss
