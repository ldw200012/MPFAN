import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet3d.models.layers.pointnet2_utils import knn_point

from mmdet3d.models.backbone.pointnet import PointNet, PointNet_6C, ED_PointNet
from mmdet3d.models.backbone.pointnext import PointNeXt, ED_PointNeXt
from mmdet3d.models.backbone.dgcnn_orig import DGCNN, DGCNN_6C, ED_DGCNN
from mmdet3d.models.backbone.deepgcn import DeepGCN, DeepGCN_6C, ED_DeepGCN
from mmdet3d.models.backbone.pointtransformer_backbone import PointTransformerBackbone, PointTransformerBackbone_6C, ED_PointTransformerBackbone
from mmdet3d.models.backbone.spotr import SPoTr, SPoTr_6C, ED_SPoTr

######################################################
# DualReID(DGCNN) ==> DGCloneXt
######################################################

class DualReID(nn.Module):
    def __init__(self, SA_conv_out=128, conv_out=128, nsample=[16,16,16]):
        super(DualReID, self).__init__()
        # torch.cuda.synchronize()

        self.sub1_SA = PointTransformerBackbone(input_channels=0, use_xyz=True, conv_out=SA_conv_out, nsample=nsample)
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

class DualReID_6C(nn.Module):
    def __init__(self, SA_conv_out=128, conv_out=128, nsample=[16,16,16]):
        super(DualReID_6C, self).__init__()
        # torch.cuda.synchronize()

        self.sub1_SA = PointTransformerBackbone_6C(input_channels=0, use_xyz=True, conv_out=SA_conv_out, nsample=nsample)
        self.sub2_DG = DGCNN_6C(dropout=0.5,emb_dims=1024, k=20, output_channels=40) # output = emb_dims = 1024

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
        features = None
        return xyz, features

    def forward(self, pointcloud, numpoints):
        # Clone 1 through Self-Attention
        out1, h1 = self.sub1_SA(pointcloud, numpoints)     # h1 shape = [B, conv_out, N]
        
        # Clone 2 through DGCNN
        out2, h2 = self.sub2_DG(pointcloud.permute(0,2,1), numpoints)
        h2_ =  F.relu(self.DG_bn1(self.DG_conv1(h2)))
        h2_ =  F.relu(self.DG_bn2(self.DG_conv2(h2_)))
        h2_ =  F.relu(self.DG_bn3(self.DG_conv3(h2_)))

        # out2 = out2.permute(0,2,1)
        z = torch.cat((h1, h2_), dim=1)

        z_ = F.relu(self.bn1(self.conv1(z)))
        z_ = F.relu(self.bn2(self.conv2(z_)))
        z_ = F.relu(self.bn3(self.conv3(z_)))
        
        xyz, features = self._break_up_pc(pointcloud)  # xyz: (B, N, C)

        return xyz, z_ # [B, N/2, 3], [B, conv_out=64, N/2]

######################################################
# DualReID(DGCNN)-Eigen
######################################################

class ED_DualReID(nn.Module):
    def __init__(self, SA_conv_out=128, conv_out=128, nsample=[16,16,16], ED_conv_out=4):
        super(ED_DualReID, self).__init__()
        # torch.cuda.synchronize()

        print("\033[91mED_DualReID Created\033[0m")

        self.sub1_SA = PointTransformerBackbone(input_channels=0, use_xyz=True, conv_out=SA_conv_out, nsample=nsample)
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

        # Eigen ###############################################################################################################
        self.ED_conv_out = ED_conv_out
        self.sub3_ED = nn.Sequential(
                            nn.Linear(3, ED_conv_out),
                            nn.ReLU(),
                            nn.Linear(ED_conv_out, ED_conv_out))
        
        # Final ###############################################################################################################
        self.conv_final = nn.Conv1d(128 + ED_conv_out, 128, 1)
        self.bn_final = nn.BatchNorm1d(128)

    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = pc[..., 3:].contiguous()
        return xyz, features

    def forward(self, pointcloud, numpoints):
        xyz, eigenvalues = self._break_up_pc(pointcloud)  # xyz: (B, N, C)

        # Clone 1 through Self-Attention
        out1, h1 = self.sub1_SA(xyz, numpoints)     # h1 shape = [B, conv_out, N]
        
        # Clone 2 through DGCNN
        out2, h2 = self.sub2_DG(xyz.permute(0,2,1), numpoints)
        h2_ =  F.relu(self.DG_bn1(self.DG_conv1(h2)))
        h2_ =  F.relu(self.DG_bn2(self.DG_conv2(h2_)))
        h2_ =  F.relu(self.DG_bn3(self.DG_conv3(h2_)))

        # out2 = out2.permute(0,2,1)
        f = torch.cat((h1, h2_), dim=1)

        f_ = F.relu(self.bn1(self.conv1(f)))
        f_ = F.relu(self.bn2(self.conv2(f_)))
        f_ = F.relu(self.bn3(self.conv3(f_)))

        eigen_feature = self.sub3_ED(eigenvalues)

        # print("eigen_feature: {}".format(eigen_feature.shape))

        # Final ###############################################################################################################
        z = torch.cat((f_, eigen_feature.permute(0,2,1)), dim=1)
        z_ = F.relu(self.bn_final(self.conv_final(z)))
        
        # return xyz, z_, h1, h2_, eigen_feature.permute(0,2,1) # [B, N/2, 3], [B, 128, N/2]
        
        return xyz, z_ # [B, N/2, 3], [B, conv_out=64, N/2]