import torch
import torch.nn as nn
import torch.nn.functional as F
from .pointnet2_utils import fps_subset_split

from .backbone_net import Pointnet_Backbone
from .pointnet import PointNet
from .attention import corss_attention

class HybridPointTransformer(nn.Module):
    def __init__(self, input_channels, use_xyz, SA_conv_out, conv_out, k, normal_channel, nhead, attention):
        super(HybridPointTransformer, self).__init__()

        self.sub1_SA = Pointnet_Backbone(input_channels=input_channels, use_xyz=use_xyz, conv_out=SA_conv_out)
        self.sub2_PN = PointNet(k, normal_channel)
        self.corss_attention = corss_attention(d_model=SA_conv_out, nhead=nhead, attention=attention)

        self.fc1 = nn.Conv1d(SA_conv_out, 512, 1)
        self.fc2 = nn.Conv1d(512, 256, 1)
        self.fc3 = nn.Conv1d(256, conv_out, 1)

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

    def forward(self, pointcloud, numpoints):
        xyz, features = self._break_up_pc(pointcloud)  # xyz: (B, N, C)
        subset1, subset2 = fps_subset_split(xyz=xyz, npoint=int(xyz.shape[1]/2))

        out1, h1 = self.sub1_SA(subset1, numpoints)     # h1 shape = [B, conv_out, N/2]
        out2, h2 = self.sub2_PN(subset2.permute(0,2,1), backbone_list=numpoints)    # h2 shape = [B, conv_out, N/2]
        out2 = out2.permute(0,2,1)

        z = self.corss_attention(search_feat=h2, search_xyz=out2, template_feat=h1, template_xyz=out1, mask=None)
        z1 = F.relu(self.bn1(self.fc1(z)))
        z2 = F.relu(self.bn2(self.fc2(z1)))
        z3 = F.relu(self.bn3(self.fc3(z2)))
        
        return subset2, z3 # [B, N/2, 3], [B, conv_out=64, N/2]


