import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet3d.models.layers.pointnet2_utils import knn_point
from openpoints.models.backbone import SPoTrEncoder, SPoTrDecoder

class SPoTr(nn.Module):
    def __init__(self):
        super(SPoTr, self).__init__()
        print("\033[91mSPoTr Created\033[0m")
        
        in_channels = 3
        self.encoder = SPoTrEncoder(blocks=[1,5,5,5,5], strides=[1,3,3,3,3],
                                    width=64, in_channels=in_channels, expansion=4, radius=0.1, nsample=32, gamma=16, num_gp=16, tau_delta=0.5,
                                    aggr_args={'feature_type':'dp_df', 'reduction':'max'}, group_args={'NAME':'ballquery', 'normalize_dp':True}, conv_args={'order':'conv-norm-act'},
                                    act_args={'act':'relu'}, norm_arg={'norm':'bn'})
        
        self.decoder = SPoTrDecoder(encoder_channel_list=self.encoder.channel_list if hasattr(self.encoder,'channel_list') else None,
                                    decoder_layers=2, decoder_stages=4, in_channels=in_channels)

    def forward(self, data, numpoints):
        p, f = self.encoder.forward_seg_feat(data)
        f = self.decoder(p, f).squeeze(-1)
        
        return data, f

class SPoTr_6C(nn.Module):
    def __init__(self):
        super(SPoTr_6C, self).__init__()
        print("\033[91mSPoTr_6C Created\033[0m")
        
        in_channels = 6
        
        self.encoder = SPoTrEncoder(blocks=[1,5,5,5,5], strides=[1,3,3,3,3],
                                    width=64, in_channels=in_channels, expansion=4, radius=0.1, nsample=32, gamma=16, num_gp=16, tau_delta=0.5,
                                    aggr_args={'feature_type':'dp_df', 'reduction':'max'}, group_args={'NAME':'ballquery', 'normalize_dp':True}, conv_args={'order':'conv-norm-act'},
                                    act_args={'act':'relu'}, norm_arg={'norm':'bn'})
        
        self.decoder = SPoTrDecoder(encoder_channel_list=self.encoder.channel_list if hasattr(self.encoder,'channel_list') else None,
                                    decoder_layers=2, decoder_stages=4, in_channels=in_channels)

    def forward(self, data, numpoints):
        p, f = self.encoder.forward_seg_feat(data)
        f = self.decoder(p, f).squeeze(-1)
        
        # Return only 3D coordinates for attention layers
        p_3d = p[:, :3, :]  # Extract only x, y, z coordinates
        return p_3d, f
    
class ED_SPoTr(nn.Module):
    def __init__(self, ED_conv_out=4):
        super(ED_SPoTr, self).__init__()
        print("\033[91mED_SPoTr Created\033[0m")
        
        in_channels = 3
        self.encoder = SPoTrEncoder(blocks=[1,5,5,5,5], strides=[1,3,3,3,3],
                                    width=64, in_channels=in_channels, expansion=4, radius=0.1, nsample=32, gamma=16, num_gp=16, tau_delta=0.5,
                                    aggr_args={'feature_type':'dp_df', 'reduction':'max'}, group_args={'NAME':'ballquery', 'normalize_dp':True}, conv_args={'order':'conv-norm-act'},
                                    act_args={'act':'relu'}, norm_arg={'norm':'bn'})
        
        self.decoder = SPoTrDecoder(encoder_channel_list=self.encoder.channel_list if hasattr(self.encoder,'channel_list') else None,
                                    decoder_layers=2, decoder_stages=4, in_channels=in_channels)
        
        # Eigen ###############################################################################################################
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
        print("DATA SHAPE: ", data.shape) # [B, N, C]

        xyz, eigenvalues = self._break_up_pc(data)

        p, f = self.encoder.forward_seg_feat(xyz)
        f = self.decoder(p, f).squeeze(-1)

        eigen_feature = self.sub3_ED(eigenvalues)

        # Final ###############################################################################################################
        z = torch.cat((f, eigen_feature.permute(0,2,1)), dim=1)
        z = F.relu(self.bn_final(self.conv_final(z)))
        
        return xyz, z