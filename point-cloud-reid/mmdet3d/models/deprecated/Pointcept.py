import torch
import torch.nn as nn
from pointcept.models.point_transformer_v3 import PointTransformerV3
from pointcept.models.utils.structure import Point

class Pointcept(nn.Module):
    def __init__(self):
        super().__init__()

        print("\033[91mPointcept Created\033[0m")
        self.backbone = PointTransformerV3( in_channels=3,
                                            # order=("z", "z-trans", "hilbert", "hilbert-trans"),
                                            order=("z", "z-trans"),
                                            stride=(2, 2, 2, 2),
                                            enc_depths=(2, 2, 2, 6, 2),
                                            enc_channels=(32, 64, 128, 256, 512),
                                            enc_num_head=(2, 4, 8, 16, 32),
                                            enc_patch_size=(1024, 1024, 1024, 1024, 1024),
                                            dec_depths=(2, 2, 2, 2),
                                            dec_channels=(64, 64, 128, 256),
                                            dec_num_head=(4, 4, 8, 16),
                                            dec_patch_size=(1024, 1024, 1024, 1024),
                                            mlp_ratio=4,
                                            qkv_bias=True,
                                            qk_scale=None,
                                            attn_drop=0.0,
                                            proj_drop=0.0,
                                            drop_path=0.3,
                                            pre_norm=True,
                                            shuffle_orders=True,
                                            enable_rpe=False,
                                            enable_flash=False,
                                            upcast_attention=False,
                                            upcast_softmax=False,
                                            cls_mode=False,
                                            pdnorm_bn=False,
                                            pdnorm_ln=False,
                                            pdnorm_decouple=True,
                                            pdnorm_adaptive=False,
                                            pdnorm_affine=True,
                                            pdnorm_conditions=("nuScenes", "SemanticKITTI", "Waymo"),)
        
    def forward(self, data, numpoints):

        B, N, C = data.shape
        data = data.reshape(B*N,C)
        print("Data Shape View: ", data.shape)

        data_dict = dict(
            feat = data,
            coord = data,
            grid_coord = data,
            offset = torch.arange(1, B + 1, device=data.device) * N
        )

        print("Before PTV3")
        point = self.backbone(data_dict)
        print("After PTV3")

        print(point.shape)        
        return point

        # data = Point(data_dict)
        # data.serialization(order=self.order, shuffle_orders=self.shuffle_orders)
        # data.sparsify()

        # data = data.permute(0,2,1)
        # print("\033[91mPointcept Input: \033[0m", data.shape)
        # points = self.backbone(data)
        # print("\033[91mPointcept Output: \033[0m", type(points))
        # print("\033[91mPointcept Output: \033[0m", points.shape)

        # return data, points