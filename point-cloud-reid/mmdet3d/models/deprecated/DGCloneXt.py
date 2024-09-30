import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone_net import Pointnet_Backbone, EDPointTransformer
from .dgcnn_orig import DGCNN

class DGCloneXt(nn.Module):
    def __init__(self, input_channels, use_xyz, SA_conv_out, conv_out, nsample):
        super(DGCloneXt, self).__init__()

        self.sub1_SA = Pointnet_Backbone(input_channels=input_channels, use_xyz=use_xyz, conv_out=SA_conv_out, nsample=nsample)
        self.sub2_DG = DGCNN(dropout=0.5,emb_dims=1024, k=20, output_channels=40) # output = emb_dims = 1024

         # 1024 to 32 for DGCNN
        self.DG_conv1 = nn.Conv1d(1024, 256, 1)
        self.DG_conv2 = nn.Conv1d(256, 64, 1)
        self.DG_conv3 = nn.Conv1d(64, int(SA_conv_out/4), 1)

        self.DG_bn1 = nn.BatchNorm1d(256)
        self.DG_bn2 = nn.BatchNorm1d(64)
        self.DG_bn3 = nn.BatchNorm1d(int(SA_conv_out/4))

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
        
        # Clone 2 through DGCNN
        out2, h2 = self.sub2_DG(xyz.permute(0,2,1), numpoints)
        h2_ =  F.relu(self.DG_bn1(self.DG_conv1(h2)))
        h2_ =  F.relu(self.DG_bn2(self.DG_conv2(h2_)))
        h2_ =  F.relu(self.DG_bn3(self.DG_conv3(h2_)))

        # out2 = out2.permute(0,2,1)
        z = torch.cat((h1, h2_), dim=1)

        z_ = F.relu(self.bn1(self.conv1(z)))
        z_ = F.relu(self.bn2(self.conv2(z_)))
        z_ = F.relu(self.bn3(self.conv3(z_)))
        
        return xyz, z_ # [B, N/2, 3], [B, conv_out=64, N/2]
    
class ED_Ptr_DG(nn.Module):
    def __init__(self, input_channels, use_xyz, SA_conv_out, conv_out, nsample, ED_nsample=10, ED_conv_out=4):
        super(ED_Ptr_DG, self).__init__()
        print("\033[91mED_Ptr_DG Network Created\033[0m")

        self.sub1_SA = EDPointTransformer(input_channels=input_channels, use_xyz=use_xyz, conv_out=SA_conv_out, nsample=nsample, ED_nsample=ED_nsample, ED_conv_out=ED_conv_out)
        self.sub2_DG = DGCNN(dropout=0.5,emb_dims=1024, k=20, output_channels=40) # output = emb_dims = 1024

         # 1024 to 32 for DGCNN
        self.DG_conv1 = nn.Conv1d(1024, 256, 1)
        self.DG_conv2 = nn.Conv1d(256, 64, 1)
        self.DG_conv3 = nn.Conv1d(64, int(SA_conv_out/4), 1)

        self.DG_bn1 = nn.BatchNorm1d(256)
        self.DG_bn2 = nn.BatchNorm1d(64)
        self.DG_bn3 = nn.BatchNorm1d(int(SA_conv_out/4))

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
        
        # Clone 2 through DGCNN
        out2, h2 = self.sub2_DG(xyz.permute(0,2,1), numpoints)
        h2_ =  F.relu(self.DG_bn1(self.DG_conv1(h2)))
        h2_ =  F.relu(self.DG_bn2(self.DG_conv2(h2_)))
        h2_ =  F.relu(self.DG_bn3(self.DG_conv3(h2_)))

        # out2 = out2.permute(0,2,1)
        z = torch.cat((h1, h2_), dim=1)

        z_ = F.relu(self.bn1(self.conv1(z)))
        z_ = F.relu(self.bn2(self.conv2(z_)))
        z_ = F.relu(self.bn3(self.conv3(z_)))
        
        return xyz, z_ # [B, N/2, 3], [B, conv_out=64, N/2]