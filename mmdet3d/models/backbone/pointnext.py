import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet3d.models.layers.pointnet2_utils import knn_point
from openpoints.models.backbone import PointNextEncoder, PointNextDecoder

class PointNeXt(nn.Module):
    def __init__(self):
        super(PointNeXt, self).__init__()

        print("\033[91mPointNext Created\033[0m")

        torch.cuda.synchronize()

        in_channels = 3
        self.encoder = PointNextEncoder(blocks=[1, 4, 7, 4, 4], strides=[1, 3, 3, 3, 3],
                                        sa_layers=1, sa_use_res=False,
                                        width=64, in_channels=in_channels, expansion=4, radius=0.1, nsample=32,
                                        aggr_args={'feature_type':'dp_fj', 'reduction':'max'}, group_args={'NAME':'ballquery', 'normalize_dp':True}, conv_args={'order':'conv-norm-act'},
                                        act_args={'act':'relu'}, norm_arg={'norm':'bn'})
        
        self.decoder = PointNextDecoder(encoder_channel_list=self.encoder.channel_list if hasattr(self.encoder,'channel_list') else None,
                                    decoder_layers=2, decoder_stages=4, in_channels=in_channels)

    def forward(self, data, numpoints):
        p, f = self.encoder.forward_seg_feat(data)

        if self.decoder is not None:
            f = self.decoder(p, f).squeeze(-1)

        return data, f

# data: [B, N, C]
class PointNeXt_6C(nn.Module):
    def __init__(self):
        super(PointNeXt_6C, self).__init__()
        
        print("\033[91mPointNeXt_6C Created\033[0m")
        
        torch.cuda.synchronize()
        
        in_channels = 3
        self.encoder = PointNextEncoder(blocks=[1, 4, 7, 4, 4], strides=[1, 3, 3, 3, 3],
                                        sa_layers=1, sa_use_res=False,
                                        width=64, in_channels=in_channels, expansion=4, radius=0.1, nsample=32,
                                        aggr_args={'feature_type':'dp_fj', 'reduction':'max'}, group_args={'NAME':'ballquery', 'normalize_dp':True}, conv_args={'order':'conv-norm-act'},
                                        act_args={'act':'relu'}, norm_arg={'norm':'bn'})
        
        self.decoder = PointNextDecoder(encoder_channel_list=self.encoder.channel_list if hasattr(self.encoder,'channel_list') else None,
                                    decoder_layers=2, decoder_stages=4, in_channels=in_channels)
        
    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        eigenvalues = pc[..., 3:].contiguous()
        return xyz, eigenvalues

    def forward(self, data, numpoints):
        xyz, eigenvalues = self._break_up_pc(data)

        p, f = self.encoder.forward_seg_feat(xyz, f0=eigenvalues.permute(0,2,1))
        
        if self.decoder is not None:
            f = self.decoder(p, f).squeeze(-1)

        return xyz, f

# data: [B, N, C]
class ED_PointNeXt(nn.Module):
    def __init__(self, ED_conv_out=4):
        super(ED_PointNeXt, self).__init__()

        print("\033[91mED_PointNext Created\033[0m")

        torch.cuda.synchronize()

        in_channels = 3
        self.encoder = PointNextEncoder(blocks=[1, 4, 7, 4, 4], strides=[1, 3, 3, 3, 3],
                                        sa_layers=1, sa_use_res=False,
                                        width=64, in_channels=in_channels, expansion=4, radius=0.1, nsample=32,
                                        aggr_args={'feature_type':'dp_fj', 'reduction':'max'}, group_args={'NAME':'ballquery', 'normalize_dp':True}, conv_args={'order':'conv-norm-act'},
                                        act_args={'act':'relu'}, norm_arg={'norm':'bn'})
        
        self.decoder = PointNextDecoder(encoder_channel_list=self.encoder.channel_list if hasattr(self.encoder,'channel_list') else None,
                                    decoder_layers=2, decoder_stages=4, in_channels=in_channels)
        
        self.ED_conv_out = ED_conv_out
        self.sub3_ED = nn.Sequential(
                            nn.Linear(3, ED_conv_out),
                            nn.ReLU(),
                            nn.Linear(ED_conv_out, ED_conv_out)
                            )
        
        self.conv_final = nn.Conv1d(64 + ED_conv_out, 64, 1)
        self.bn_final = nn.BatchNorm1d(64)

    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = pc[..., 3:].contiguous()
        return xyz, features

    def forward(self, data, numpoints):

        xyz, eigenvalues = self._break_up_pc(data)

        p, f = self.encoder.forward_seg_feat(xyz)
        f = self.decoder(p, f).squeeze(-1)

        eigen_feature = self.sub3_ED(eigenvalues)

        z = torch.cat((f, eigen_feature.permute(0,2,1)), dim=1)
        z = F.relu(self.bn_final(self.conv_final(z)))

        return xyz, z