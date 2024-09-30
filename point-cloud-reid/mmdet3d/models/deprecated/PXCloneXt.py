import torch
import torch.nn as nn
import torch.nn.functional as F 

from .backbone_net import Pointnet_Backbone
from .PointNeXt import PointNeXt

class PXCloneXt(nn.Module):
    def __init__(self, input_channels, use_xyz, SA_conv_out, conv_out, nsample):
        super(PXCloneXt, self).__init__()

        self.sub1_SA = Pointnet_Backbone(input_channels=input_channels, use_xyz=use_xyz, conv_out=SA_conv_out, nsample=nsample)
        self.sub2_PX = PointNeXt() # output = 64

        # 64 to 32 for PXCNN
        self.PX_conv1 = nn.Conv1d(64, int(SA_conv_out/4), 1)
        self.PX_bn1 = nn.BatchNorm1d(int(SA_conv_out/4))

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

    def forward(self, pointcloud, numpoints):
        xyz, features = self._break_up_pc(pointcloud)  # xyz: (B, N, C)

        # Clone 1 through Self-Attention
        out1, h1 = self.sub1_SA(xyz, numpoints)     # h1 shape = [B, conv_out, N]

        # print("\033[91{}\033[0m".format())
        # print("\033[91{}: {}\033[0m".format("xyz shape", xyz.shape))
        
        # Clone 2 through PX
        out2, h2 = self.sub2_PX(xyz, numpoints)
        # print("\033[91{}: {}\033[0m".format("h1 shape", h1.shape))
        # print("\033[91{}: {}\033[0m".format("h2 shape", h2.shape))

        h2_ =  F.relu(self.PX_bn1(self.PX_conv1(h2)))
        # print("\033[91{}: {}\033[0m".format("h2_ shape", h2_.shape))

        # out2 = out2.permute(0,2,1)
        z = torch.cat((h1, h2_), dim=1)
        # print("\033[91{}: {}\033[0m".format("z shape", z.shape))

        z_ = F.relu(self.bn1(self.conv1(z)))
        z_ = F.relu(self.bn2(self.conv2(z_)))
        z_ = F.relu(self.bn3(self.conv3(z_)))
        
        return xyz, z_ # [B, N/2, 3], [B, conv_out=64, N/2]