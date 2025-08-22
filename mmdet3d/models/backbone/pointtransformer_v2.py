import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet3d.models.layers.pointnet2_utils import knn_point
from openpoints.models.backbone import PointTransformerV2Encoder, PointTransformerV2Decoder

class PointTransformerV2(nn.Module):
    def __init__(self):
        super(PointTransformerV2, self).__init__()
        print("\033[91mPointTransformerV2 Created\033[0m")

        torch.cuda.synchronize()

        in_channels = 3
        self.encoder = PointTransformerV2Encoder(
            blocks=[1, 4, 7, 4, 4], 
            strides=[1, 3, 3, 3, 3],
            sa_layers=1, 
            sa_use_res=False,
            width=64, 
            in_channels=in_channels, 
            expansion=4, 
            radius=0.1, 
            nsample=32,
            aggr_args={'feature_type':'dp_fj', 'reduction':'max'}, 
            group_args={'NAME':'ballquery', 'normalize_dp':True}, 
            conv_args={'order':'conv-norm-act'},
            act_args={'act':'relu'}, 
            norm_arg={'norm':'bn'}
        )
        
        self.decoder = PointTransformerV2Decoder(
            encoder_channel_list=self.encoder.channel_list if hasattr(self.encoder,'channel_list') else None,
            decoder_layers=2, 
            decoder_stages=4, 
            in_channels=in_channels
        )

    def forward(self, data, numpoints):
        p, f = self.encoder.forward_seg_feat(data)

        if self.decoder is not None:
            f = self.decoder(p, f).squeeze(-1)

        return data, f

class PointTransformerV2_6C(nn.Module):
    def __init__(self, use_precomputed_eigen=False):
        super(PointTransformerV2_6C, self).__init__()
        print("\033[91mPointTransformerV2_6C Created\033[0m")

        torch.cuda.synchronize()

        in_channels = 6 if use_precomputed_eigen else 3
        self.use_precomputed_eigen = use_precomputed_eigen
        
        # Create encoder with 6-channel input capability
        self.encoder = PointTransformerV2Encoder(
            blocks=[1, 4, 7, 4, 4], 
            strides=[1, 3, 3, 3, 3],
            sa_layers=1, 
            sa_use_res=False,
            width=64, 
            in_channels=in_channels, 
            expansion=4, 
            radius=0.1, 
            nsample=32,
            aggr_args={'feature_type':'dp_fj', 'reduction':'max'}, 
            group_args={'NAME':'ballquery', 'normalize_dp':True}, 
            conv_args={'order':'conv-norm-act'},
            act_args={'act':'relu'}, 
            norm_arg={'norm':'bn'}
        )
        
        self.decoder = PointTransformerV2Decoder(
            encoder_channel_list=self.encoder.channel_list if hasattr(self.encoder,'channel_list') else None,
            decoder_layers=2, 
            decoder_stages=4, 
            in_channels=in_channels
        )

    def forward(self, data, numpoints):
        if self.use_precomputed_eigen:
            # Use pre-computed 6-channel data (x, y, z, eig1, eig2, eig3)
            # Input data has shape [B, 6, N] with eigenvalues included
            
            # PointTransformerV2 expects: p0 (position) in [B, N, C] format
            # Transpose and make contiguous: [B, 6, N] -> [B, N, 6]
            data_transposed = data.transpose(1, 2).contiguous()
            
            p, f = self.encoder.forward_seg_feat(data_transposed)
            
            if self.decoder is not None:
                f = self.decoder(p, f).squeeze(-1)
            
            # Return only 3D coordinates for attention layers
            # Extract xyz from the first position data
            p_3d = p[0][:, :3].transpose(1, 2).contiguous()  # [B, N, 3] -> [B, 3, N]
            return p_3d, f
        else:
            # Original 3-channel behavior
            p, f = self.encoder.forward_seg_feat(data)

            if self.decoder is not None:
                f = self.decoder(p, f).squeeze(-1)

            return data, f

class ED_PointTransformerV2(nn.Module):
    def __init__(self, ED_nsample=10, ED_conv_out=4, use_precomputed_eigen=False):
        super(ED_PointTransformerV2, self).__init__()
        print("\033[91mED_PointTransformerV2 Created\033[0m")
        
        self.use_precomputed_eigen = use_precomputed_eigen
        
        torch.cuda.synchronize()

        in_channels = 6 if use_precomputed_eigen else 3
        self.encoder = PointTransformerV2Encoder(
            blocks=[1, 4, 7, 4, 4], 
            strides=[1, 3, 3, 3, 3],
            sa_layers=1, 
            sa_use_res=False,
            width=64, 
            in_channels=in_channels, 
            expansion=4, 
            radius=0.1, 
            nsample=32,
            aggr_args={'feature_type':'dp_fj', 'reduction':'max'}, 
            group_args={'NAME':'ballquery', 'normalize_dp':True}, 
            conv_args={'order':'conv-norm-act'},
            act_args={'act':'relu'}, 
            norm_arg={'norm':'bn'}
        )
        
        self.decoder = PointTransformerV2Decoder(
            encoder_channel_list=self.encoder.channel_list if hasattr(self.encoder,'channel_list') else None,
            decoder_layers=2, 
            decoder_stages=4, 
            in_channels=in_channels
        )
        
        # Eigen ###############################################################################################################
        self.ED_nsample = ED_nsample
        self.ED_conv_out = ED_conv_out
        self.sub3_ED = nn.Sequential(
                            nn.Linear(3, ED_conv_out),
                            nn.ReLU(),
                            nn.Linear(ED_conv_out, ED_conv_out))
        
        # Final ###############################################################################################################
        self.conv_final = nn.Conv1d(64 + ED_conv_out, 64, 1)
        self.bn_final = nn.BatchNorm1d(64)

    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = pc[..., 3:].contiguous()
        return xyz, features

    def forward(self, data, numpoints):
        if self.use_precomputed_eigen:
            # Use pre-computed 6-channel data
            data_transposed = data.transpose(1, 2).contiguous()  # [B, 6, N] -> [B, N, 6]
            xyz, eigenvalues = self._break_up_pc(data_transposed)
            
            p, f = self.encoder.forward_seg_feat(data_transposed)
            f = self.decoder(p, f).squeeze(-1)

            # Use pre-computed eigenvalues
            eigen_feature = self.sub3_ED(eigenvalues)

            # Final ###############################################################################################################
            z = torch.cat((f, eigen_feature.permute(0,2,1)), dim=1)
            z = F.relu(self.bn_final(self.conv_final(z)))
            
            # Return 3D coordinates for attention layers
            p_3d = p[0][:, :3].transpose(1, 2).contiguous()  # [B, N, 3] -> [B, 3, N]
            return p_3d, z
        else:
            # Original 3-channel behavior with on-the-fly eigenvalue computation
            p, f = self.encoder.forward_seg_feat(data)
            f = self.decoder(p, f).squeeze(-1)

            # Eigen ###############################################################################################################
            group_idx = knn_point(nsample=self.ED_nsample, xyz=data, new_xyz=data)
            batch_indices = torch.arange(data.shape[0]).view(-1, 1, 1).expand(-1, data.shape[1], self.ED_nsample)
            neighborhood_points = data[batch_indices, group_idx]  # (B, N, k, 3)
            centered_points = neighborhood_points - neighborhood_points.mean(dim=2, keepdim=True)
            cov_matrices = centered_points.transpose(-2, -1).matmul(centered_points) / self.ED_nsample  # (B, N, 3, 3)
            eigenvalues = torch.linalg.eigvalsh(cov_matrices)  # (B, N, 3)
            eigen_feature = self.sub3_ED(eigenvalues)

            # Final ###############################################################################################################
            z = torch.cat((f, eigen_feature.permute(0,2,1)), dim=1)
            z = F.relu(self.bn_final(self.conv_final(z)))

            return data, z
