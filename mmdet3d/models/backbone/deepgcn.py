import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet3d.models.layers.pointnet2_utils import knn_point
from openpoints.models.backbone import DeepGCNEncoder

class DeepGCN(nn.Module):
    def __init__(self, emb_dims=1024):
        super(DeepGCN, self).__init__()
        print("\033[91mDeepGCN Created\033[0m")

        torch.cuda.synchronize()

        in_channels = 3
        self.encoder = DeepGCNEncoder(in_channels=in_channels, channels=64, emb_dims=emb_dims, n_blocks=14, # n_blocks=14
                                      conv='edge', block='no', k=16, epsilon=0.2, #block='res'
                                      use_stochastic=True, use_dilation=True,
                                      norm_args={'norm': 'bn'}, act_args={'act': 'relu'}, conv_args={'order': 'conv-norm-act'},
                                      is_seg=False)
                                      
    def forward(self, data, numpoints):
        _, f = self.encoder.forward_seg_feat(data)
        return data, f

class DeepGCN_6C(nn.Module):
    def __init__(self, emb_dims=1024, use_precomputed_eigen=False):
        super(DeepGCN_6C, self).__init__()
        print("\033[91mDeepGCN_6C Created\033[0m")

        torch.cuda.synchronize()

        in_channels = 6 if use_precomputed_eigen else 3
        self.use_precomputed_eigen = use_precomputed_eigen
        
        self.encoder = DeepGCNEncoder(in_channels=in_channels, channels=64, emb_dims=emb_dims, n_blocks=14, # n_blocks=14
                                      conv='edge', block='no', k=16, epsilon=0.2, #block='res'
                                      use_stochastic=True, use_dilation=True,
                                      norm_args={'norm': 'bn'}, act_args={'act': 'relu'}, conv_args={'order': 'conv-norm-act'},
                                      is_seg=False)
                                      
    def forward(self, data, numpoints):
        if self.use_precomputed_eigen:
            # Use pre-computed 6-channel data (x, y, z, eig1, eig2, eig3)
            # Input data already has shape [B, 6, N] with eigenvalues included
            _, f = self.encoder.forward_seg_feat(data)
            
            # Return only 3D coordinates for attention layers
            data_3d = data[:, :3, :]  # Extract only x, y, z coordinates
            return data_3d, f
        else:
            # Original 3-channel behavior
            _, f = self.encoder.forward_seg_feat(data)
            return data, f

class ED_DeepGCN(nn.Module):
    def __init__(self, emb_dims=1024, ED_nsample=10, ED_conv_out=4, use_precomputed_eigen=False):
        super(ED_DeepGCN, self).__init__()
        print("\033[91mED_DeepGCN Created\033[0m")
        
        torch.cuda.synchronize()

        in_channels = 3
        self.use_precomputed_eigen = use_precomputed_eigen
        self.encoder = DeepGCNEncoder(in_channels=in_channels, channels=64, emb_dims=emb_dims, n_blocks=14, # n_blocks=14
                                      conv='edge', block='no', k=16, epsilon=0.2, #block='res'
                                      use_stochastic=True, use_dilation=True,
                                      norm_args={'norm': 'bn'}, act_args={'act': 'relu'}, conv_args={'order': 'conv-norm-act'},
                                      is_seg=False)
        
        # Eigen ###############################################################################################################
        self.ED_nsample = ED_nsample
        self.ED_conv_out = ED_conv_out
        self.sub3_ED = nn.Sequential(
                            nn.Linear(3, ED_conv_out),
                            nn.ReLU(),
                            nn.Linear(ED_conv_out, ED_conv_out))
        
        # Final ###############################################################################################################
        self.conv_final = nn.Conv1d(emb_dims + ED_conv_out, emb_dims, 1)
        self.bn_final = nn.BatchNorm1d(emb_dims)

    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = pc[..., 3:].contiguous()
        return xyz, features
                                      
    def forward(self, data, numpoints):
        print("DATA SHAPE: ", data.shape) # [B, N, C]

        xyz, eigenvalues = self._break_up_pc(data)

        _, f = self.encoder.forward_seg_feat(xyz)

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
        z = torch.cat((f, eigen_feature.permute(0,2,1)), dim=1)
        z = F.relu(self.bn_final(self.conv_final(z)))
        
        return xyz, z
