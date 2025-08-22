import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
from mmdet3d.models.layers.pointnet2_utils import knn_point

class STN3d(nn.Module):
    def __init__(self, channel):
        super(STN3d, self).__init__()
        self.conv1 = torch.nn.Conv1d(channel, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x, use_hybrid=False):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        if use_hybrid:
            return x

        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]).astype(np.float32))).view(1, 9).repeat(
            batchsize, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, 3, 3)
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
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.k = k

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

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

class PointNetEncoder(nn.Module):
    def __init__(self, global_feat=True, feature_transform=False, channel=3):
        super(PointNetEncoder, self).__init__()
        self.stn = STN3d(channel)
        self.conv1 = torch.nn.Conv1d(channel, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.global_feat = global_feat
        self.feature_transform = feature_transform
        if self.feature_transform:
            self.fstn = STNkd(k=64)

    def forward(self, xyz, use_hybrid=False):
        B, D, N = xyz.size()
        trans = self.stn(xyz, use_hybrid)

        if use_hybrid:
            return xyz, trans
            
        x = xyz.transpose(2, 1)
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
        return xyz, x

def feature_transform_reguliarzer(trans):
    d = trans.size()[1]
    I = torch.eye(d)[None, :, :]
    if trans.is_cuda:
        I = I.cuda()
    loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2, 1)) - I, dim=(1, 2)))
    return loss

class PointNet(nn.Module):
    def __init__(self, k=40, normal_channel=True, use_hybrid=False):
        super(PointNet, self).__init__()
        print("\033[91mPointNet Created\033[0m")

        if normal_channel:
            channel = 6
        else:
            channel = 3
        self.feat = PointNetEncoder(global_feat=True, feature_transform=True, channel=channel)
        self.use_hybrid = use_hybrid

    def forward(self, x, backbone_list):
        # print("\033[91mPointNet input\033[0m")

        xyz, x = self.feat(x, self.use_hybrid)

        # print("\033[91mPointNet forward return xyz, x\033[0m")
        # print("\033[91xyz (input):\033[0m ", xyz.shape)
        # print("\033[91x (feature):\033[0m ", x.shape)
        return xyz, x

class PointNet_6C(nn.Module):
    def __init__(self, k=40, normal_channel=True, use_hybrid=False, ED_nsample=10, use_precomputed_eigen=False):
        super(PointNet_6C, self).__init__()
        print("\033[91mPointNet_6C Created\033[0m")

        channel = 6
        self.ED_nsample = ED_nsample
        self.use_precomputed_eigen = use_precomputed_eigen
        self.feat = PointNetEncoder(global_feat=True, feature_transform=True, channel=channel)
        self.use_hybrid = use_hybrid

    def forward(self, x, backbone_list):
        if self.use_precomputed_eigen:
            # Use pre-computed 6-channel data (x, y, z, eig1, eig2, eig3)
            # Input x already has shape [B, 6, N] with eigenvalues included
            xyz, x = self.feat(x, self.use_hybrid)
            
            # Return only 3D coordinates for attention layers
            xyz_3d = xyz[:, :3, :]
            return xyz_3d, x
        else:
            # Original on-the-fly eigenvalue computation
            xyz = x.permute(0,2,1)
            
            # Eigenvalue computation (same as ED_PointNet)
            group_idx = knn_point(nsample=self.ED_nsample, xyz=xyz, new_xyz=xyz)
            batch_indices = torch.arange(xyz.shape[0]).view(-1, 1, 1).expand(-1, xyz.shape[1], self.ED_nsample)
            neighborhood_points = xyz[batch_indices, group_idx]  # (B, N, k, 3)
            centered_points = neighborhood_points - neighborhood_points.mean(dim=2, keepdim=True)
            cov_matrices = centered_points.transpose(-2, -1).matmul(centered_points) / self.ED_nsample  # (B, N, 3, 3)
            eigenvalues = torch.linalg.eigvalsh(cov_matrices)  # (B, N, 3)

            x_6c = torch.cat((xyz, eigenvalues), dim=2)  # [B, N, 6]
            x_6c = x_6c.permute(0, 2, 1)
            
            xyz, x = self.feat(x_6c, self.use_hybrid)
            
            # Return only 3D coordinates for attention layers, but use 6-channel features internally
            xyz_3d = xyz[:, :3, :]  # Extract only x, y, z coordinates

            return xyz_3d, x
    
class ED_PointNet(nn.Module):
    def __init__(self, k=40, normal_channel=True, use_hybrid=False, ED_nsample=10, ED_conv_out=4, use_precomputed_eigen=False):
        super(ED_PointNet, self).__init__()
        print("\033[91mED_PointNet Created\033[0m")

        self.use_precomputed_eigen = use_precomputed_eigen

        if normal_channel:
            channel = 6
        else:
            channel = 3
        self.feat = PointNetEncoder(global_feat=True, feature_transform=True, channel=channel)
        self.use_hybrid = use_hybrid

        # Eigen ###############################################################################################################
        self.ED_nsample = ED_nsample
        self.ED_conv_out = ED_conv_out
        self.sub3_ED = nn.Sequential(
                            nn.Linear(3, ED_conv_out),
                            nn.ReLU(),
                            nn.Linear(ED_conv_out, ED_conv_out))
        
        # Final ###############################################################################################################
        self.conv_final = nn.Conv1d(1024 + ED_conv_out, 1024, 1)
        self.bn_final = nn.BatchNorm1d(1024)

    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = pc[..., 3:].contiguous()
        return xyz, features

    def forward(self, x, backbone_list):
        x = x.permute(0,2,1)

        xyz, eigenvalues = self._break_up_pc(x)
        xyz = xyz.permute(0,2,1)

        out, feat = self.feat(xyz, self.use_hybrid)

        if not self.use_precomputed_eigen:
            # Eigen ###############################################################################################################
            group_idx = knn_point(nsample=self.ED_nsample, xyz=xyz, new_xyz=xyz)
            batch_indices = torch.arange(xyz.shape[0]).view(-1, 1, 1).expand(-1, xyz.shape[1], self.ED_nsample)
            neighborhood_points = xyz[batch_indices, group_idx]  # (B, N, k, 3)
            centered_points = neighborhood_points - neighborhood_points.mean(dim=2, keepdim=True)
            cov_matrices = centered_points.transpose(-2, -1).matmul(centered_points) / self.ED_nsample  # (B, N, 3, 3)
            eigenvalues = torch.linalg.eigvalsh(cov_matrices)  # (B, N, 3)
        
        eigen_feature = self.sub3_ED(eigenvalues)

        # Final ###############################################################################################################
        z = torch.cat((feat, eigen_feature.permute(0,2,1)), dim=1)
        z = F.relu(self.bn_final(self.conv_final(z)))

        return out, z