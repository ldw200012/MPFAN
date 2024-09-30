import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.optimize import linear_sum_assignment
from .pointnet2_utils import fps_subset_split

from .backbone_net import Pointnet_Backbone
from .pointnet import PointNet
from .attention import corss_attention

class ConcatCloneNeXt(nn.Module):
    def __init__(self, input_channels, use_xyz, SA_conv_out, conv_out, k, normal_channel, nhead, attention, nsample):
        super(ConcatCloneNeXt, self).__init__()

        self.sub1_SA = Pointnet_Backbone(input_channels=input_channels, use_xyz=use_xyz, conv_out=SA_conv_out, nsample=nsample)
        self.sub2_PN = PointNet(k, normal_channel)

        # 1024 to 32
        self.PN_conv1 = nn.Conv1d(1024, 256, 1)
        self.PN_conv2 = nn.Conv1d(256, 64, 1)
        self.PN_conv3 = nn.Conv1d(64, int(SA_conv_out/4), 1)

        self.PN_bn1 = nn.BatchNorm1d(256)
        self.PN_bn2 = nn.BatchNorm1d(64)
        self.PN_bn3 = nn.BatchNorm1d(int(SA_conv_out/4))

        self.conv1 = nn.Conv1d(SA_conv_out + int(SA_conv_out/4), 512, 1)
        self.conv2 = nn.Conv1d(512, 256, 1)
        self.conv3 = nn.Conv1d(256, conv_out, 1)

        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(conv_out)

    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = pc[..., 3:].transpose(1, 2).contiguous() if pc.size(-1) > 3 else None
        return xyz, features

    def verify_concatenation(self, original, subset1, subset2):
        concatenated = torch.cat((subset1, subset2), dim=1)
        original_sorted = original.sort(dim=1)[0]
        concatenated_sorted = concatenated.sort(dim=1)[0]
        
        return torch.allclose(original_sorted, concatenated_sorted)
    
    def closest_points(self, subset_1, subset_2):
        # Compute the distance matrix between each pair of points in subset_1 and subset_2
        subset_1 = np.asarray(subset_1.cpu())
        subset_2 = np.asarray(subset_2.cpu())

        N = subset_1.shape[0]
        distance_matrix = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                distance_matrix[i, j] = np.linalg.norm(subset_1[i] - subset_2[j])

        # Apply the Hungarian algorithm to find the optimal one-to-one pairing with minimal distance
        row_ind, col_ind = linear_sum_assignment(distance_matrix)

        # Reorder subset_2 according to the optimal pairing
        ordered_subset_2 = subset_2[col_ind]

        return torch.tensor(subset_1).cuda(), torch.tensor(ordered_subset_2).cuda()

    def forward(self, pointcloud, numpoints):
        xyz, features = self._break_up_pc(pointcloud)  # xyz: (B, N, C)

        # Clone 1 through Self-Attention
        out1, h1 = self.sub1_SA(xyz, numpoints)     # h1 shape = [B, conv_out, N]
        
        # Clone 2 through PointNet
        out2, h2 = self.sub2_PN(xyz.permute(0,2,1), backbone_list=numpoints)    # h2 shape = [B, conv_out, N]
        h2_ =  F.relu(self.PN_bn1(self.PN_conv1(h2)))
        h2_ =  F.relu(self.PN_bn2(self.PN_conv2(h2_)))
        h2_ =  F.relu(self.PN_bn3(self.PN_conv3(h2_)))

        print("DualReID ({}) | h1:{}, h2:{}".format(self.fe_module, h1.shape, h2_.shape))
        z = torch.cat((h1, h2_), dim=1)

        z_ = F.relu(self.bn1(self.conv1(z)))
        z_ = F.relu(self.bn2(self.conv2(z_)))
        z_ = F.relu(self.bn3(self.conv3(z_)))
        
        return xyz, z_ # [B, N/2, 3], [B, conv_out=64, N/2]


