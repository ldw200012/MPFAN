import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone_net import Pointnet_Backbone
from .dgcnn_orig import DGCNN
from .pointnet2_utils import knn_point
from mmdet3d.utils.utils_ed_plus import Batched_ED_Plus

class EDDG(nn.Module):
    def __init__(self, input_channels, use_xyz, SA_nsample, ED_nsample,
                 conv_out=128, SA_conv_out=128, DG_conv_out=1024, ED_conv_out=4):
        super(EDDG, self).__init__()

        # PointTransformer ####################################################################################################
        self.sub1_SA = Pointnet_Backbone(input_channels=input_channels, use_xyz=use_xyz, conv_out=SA_conv_out, nsample=SA_nsample)

        # DGCNN ###############################################################################################################
        self.sub2_DG = DGCNN(dropout=0.5,emb_dims=DG_conv_out, k=20, output_channels=40)
        self.DG_conv1 = nn.Conv1d(DG_conv_out, 256, 1)
        self.DG_conv2 = nn.Conv1d(256, 64, 1)
        self.DG_conv3 = nn.Conv1d(64, int(SA_conv_out/4), 1)
        self.DG_bn1 = nn.BatchNorm1d(256)
        self.DG_bn2 = nn.BatchNorm1d(64)
        self.DG_bn3 = nn.BatchNorm1d(int(SA_conv_out/4))

        # Eigen ###############################################################################################################
        self.ED_nsample = ED_nsample
        self.sub3_ED = nn.Sequential(
                            nn.Linear(3, ED_conv_out),
                            nn.ReLU(),
                            nn.Linear(ED_conv_out, ED_conv_out)
                        )

        # Final ###############################################################################################################
        self.conv1 = nn.Conv1d(SA_conv_out + int(SA_conv_out/4) + ED_conv_out, 512, 1)
        self.conv2 = nn.Conv1d(512, 256, 1)
        self.conv3 = nn.Conv1d(256, conv_out, 1)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(conv_out)

    def forward(self, pointcloud, numpoints):
        xyz = pointcloud[..., 0:3].contiguous() # xyz: (B, N, C)
        
        # PointTransformer
        _, h1 = self.sub1_SA(xyz, numpoints)     # h1 shape = [B, conv_out, N]
        
        # DGCNN
        _, h2 = self.sub2_DG(xyz.permute(0,2,1), numpoints)
        h2 =  F.relu(self.DG_bn1(self.DG_conv1(h2)))
        h2 =  F.relu(self.DG_bn2(self.DG_conv2(h2)))
        h2 =  F.relu(self.DG_bn3(self.DG_conv3(h2)))

        group_idx = knn_point(nsample=self.ED_nsample, xyz=xyz, new_xyz=xyz)
        batch_indices = torch.arange(xyz.shape[0]).view(-1, 1, 1).expand(-1, xyz.shape[1], self.ED_nsample)
        neighborhood_points = xyz[batch_indices, group_idx]  # (B, N, k, 3)
        centered_points = neighborhood_points - neighborhood_points.mean(dim=2, keepdim=True)
        cov_matrices = centered_points.transpose(-2, -1).matmul(centered_points) / self.ED_nsample  # (B, N, 3, 3)

        # option = 1
        # eigenvalues = None
        # if option == 1:
        #     # ED
        #     eigenvalues = torch.linalg.eigvalsh(cov_matrices)  # (B, N, 3)
        # else:
        #     # ED Plus
        #     batched_ed_plus = Batched_ED_Plus.apply
        #     _, eigenvalues = batched_ed_plus(cov_matrices)

        eigenvalues = torch.linalg.eigvalsh(cov_matrices)  # (B, N, 3)
        h3 = self.sub3_ED(eigenvalues)

        # Final
        z = torch.cat((torch.cat((h1, h2), dim=1), h3.permute(0,2,1)), dim=1)
        z = F.relu(self.bn1(self.conv1(z)))
        z = F.relu(self.bn2(self.conv2(z)))
        z = F.relu(self.bn3(self.conv3(z)))
        
        return xyz, z # [B, N, 3], [B, conv_out=128, N]
    
class Adaptive_EDDG(nn.Module):
    def __init__(self, input_channels, use_xyz, SA_nsample,
                 conv_out=128, SA_conv_out=128, DG_conv_out=1024, ED_conv_out=4, ED_radius_factor=0.1):
        super(Adaptive_EDDG, self).__init__()

        # PointTransformer ####################################################################################################
        self.sub1_SA = Pointnet_Backbone(input_channels=input_channels, use_xyz=use_xyz, conv_out=SA_conv_out, nsample=SA_nsample)

        # DGCNN ###############################################################################################################
        self.sub2_DG = DGCNN(dropout=0.5,emb_dims=DG_conv_out, k=20, output_channels=40)
        self.DG_conv1 = nn.Conv1d(DG_conv_out, 256, 1)
        self.DG_conv2 = nn.Conv1d(256, 64, 1)
        self.DG_conv3 = nn.Conv1d(64, int(SA_conv_out/4), 1)
        self.DG_bn1 = nn.BatchNorm1d(256)
        self.DG_bn2 = nn.BatchNorm1d(64)
        self.DG_bn3 = nn.BatchNorm1d(int(SA_conv_out/4))

        # Eigen ###############################################################################################################
        self.ED_radius_factor = ED_radius_factor
        self.sub3_ED = nn.Sequential(
                            nn.Linear(3, ED_conv_out),
                            nn.ReLU(),
                            nn.Linear(ED_conv_out, ED_conv_out)
                        )

        # Final ###############################################################################################################
        self.conv1 = nn.Conv1d(SA_conv_out + int(SA_conv_out/4) + ED_conv_out, 512, 1)
        self.conv2 = nn.Conv1d(512, 256, 1)
        self.conv3 = nn.Conv1d(256, conv_out, 1)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(conv_out)

    def calculate_distances(self, point_cloud):
        if point_cloud.size(0) < 2:
            return None, None  # Not enough points to calculate distances

        # Expand point cloud to calculate pairwise differences
        point_cloud_expanded = point_cloud.unsqueeze(1)  # Shape: [N, 1, 3]
        differences = point_cloud_expanded - point_cloud  # Broadcasting to get pairwise differences
        distances = torch.sqrt((differences ** 2).sum(dim=2))  # Euclidean distances

        # Mask the diagonal of the distance matrix to exclude zero distances (distance to itself)
        mask = torch.eye(distances.size(0), dtype=torch.bool, device=point_cloud.device)
        distances = distances.masked_fill(mask, float('inf'))

        # Flatten the upper triangle of the distance matrix to get all unique pairs
        distances = distances.triu(1)
        distances_list = distances[distances != float('inf')].tolist()  # Convert to list excluding 'inf'
        distances_list = distances[distances > float(0.0)].tolist()  # Convert to list excluding '0.0'

        min_distance = min(distances_list)
        max_distance = max(distances_list)

        return distances_list, min_distance, max_distance
    
    def radius_neighbors(self, point_cloud, radius):
        distances = torch.cdist(point_cloud, point_cloud)
        distances = distances.fill_diagonal_(float('inf'))
        neighbors_mask = distances < radius
        return neighbors_mask # (N, k)

    def forward(self, pointcloud, numpoints):

        xyz = pointcloud[..., 0:3].contiguous() # xyz: (B, N, C)
        device = xyz.get_device()
        B, N, C = xyz.shape
        # print("BNC: ", B, N, C) # 256. 128. 3
        
        # PointTransformer
        _, h1 = self.sub1_SA(xyz, numpoints)     # h1 shape = [B, conv_out, N]
        
        # DGCNN
        _, h2 = self.sub2_DG(xyz.permute(0,2,1), numpoints)
        h2 =  F.relu(self.DG_bn1(self.DG_conv1(h2)))
        h2 =  F.relu(self.DG_bn2(self.DG_conv2(h2)))
        h2 =  F.relu(self.DG_bn3(self.DG_conv3(h2)))

        eigenvalues_B = torch.empty((0, N, 3), dtype=torch.float32) # (B, N, 3)
        for bn in range(B):
            _, min_d, max_d = self.calculate_distances(xyz[bn])
            adaptive_radius = max_d*self.ED_radius_factor
            neighbors_mask = self.radius_neighbors(xyz[bn], adaptive_radius)

            eigenvalues_N = torch.empty((0, 3), dtype=torch.float32) # (N, 3)
            for i in range(N):
                neighborhoods_k = xyz[bn][neighbors_mask[i]] # (k, 3)
                centered_points = neighborhoods_k - neighborhoods_k.mean(dim=0, keepdim=True)

                k = neighbors_mask[i].shape[0]
                if k > 3:
                    cov_matrices = centered_points.transpose(-2, -1).matmul(centered_points) / k  # (3, 3)
                    eigenvalues = torch.linalg.eigvalsh(cov_matrices).unsqueeze(0) # (1, 3)
                else:
                    eigenvalues = torch.zeros(1, 3)
                    
                # print("k: ", eigenvalues.shape)
                eigenvalues_N = torch.cat((eigenvalues_N.to(device), eigenvalues.to(device)), 0)

            # print("N: ", eigenvalues_N.shape)
            eigenvalues_B = torch.cat((eigenvalues_B.to(device), eigenvalues_N.unsqueeze(0).to(device)), 0)
        
        h3 = self.sub3_ED(eigenvalues_B)

        # Final
        z = torch.cat((torch.cat((h1, h2), dim=1), h3.permute(0,2,1)), dim=1)
        z = F.relu(self.bn1(self.conv1(z)))
        z = F.relu(self.bn2(self.conv2(z)))
        z = F.relu(self.bn3(self.conv3(z)))
        
        return xyz, z # [B, N, 3], [B, conv_out=128, N]