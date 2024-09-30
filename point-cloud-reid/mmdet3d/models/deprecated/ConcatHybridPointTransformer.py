import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.optimize import linear_sum_assignment
from .pointnet2_utils import fps_subset_split

from .backbone_net import Pointnet_Backbone
from .pointnet import PointNet
from .attention import corss_attention

class ConcatHybridPointTransformer(nn.Module):
    def __init__(self, input_channels, use_xyz, SA_conv_out, conv_out, k, normal_channel, nhead, attention):
        super(ConcatHybridPointTransformer, self).__init__()

        self.sub1_SA = Pointnet_Backbone(input_channels=input_channels, use_xyz=use_xyz, conv_out=conv_out)
        self.sub2_PN = PointNet(k, normal_channel)

        self.conv1 = nn.Conv1d(1024+conv_out, 512, 1)
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
        subset1, subset2 = fps_subset_split(xyz=xyz, npoint=int(xyz.shape[1]/2))

        # ordered_subset_1, ordered_subset_2 = self.closest_points(subset1, subset2)

        out1, h1 = self.sub1_SA(subset1, numpoints)     # h1 shape = [B, conv_out, N]
        out2, h2 = self.sub2_PN(subset2.permute(0,2,1), backbone_list=numpoints)    # h2 shape = [B, conv_out, N]
        # out2 = out2.permute(0,2,1)

        z = torch.cat((h1, h2), dim=1)

        z1 = F.relu(self.bn1(self.conv1(z)))
        z2 = F.relu(self.bn2(self.conv2(z1)))
        z3 = F.relu(self.bn3(self.conv3(z2)))
        
        return subset2, z3 # [B, N/2, 3], [B, conv_out=64, N/2]


