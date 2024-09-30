import sys
import torch
import torch.nn as nn
import spconv.pytorch as spconv
from collections import OrderedDict
from pointcept.models.utils.structure import Point


class PointModule(nn.Module):
    r"""PointModule
    placeholder, all module subclass from this will take Point in PointSequential.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class PointSequential(PointModule):
    r"""A sequential container.
    Modules will be added to it in the order they are passed in the constructor.
    Alternatively, an ordered dict of modules can also be passed in.
    """

    def __init__(self, *args, **kwargs):
        super().__init__()
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for key, module in args[0].items():
                self.add_module(key, module)
        else:
            for idx, module in enumerate(args):
                self.add_module(str(idx), module)
        for name, module in kwargs.items():
            if sys.version_info < (3, 6):
                raise ValueError("kwargs only supported in py36+")
            if name in self._modules:
                raise ValueError("name exists.")
            self.add_module(name, module)

    def __getitem__(self, idx):
        if not (-len(self) <= idx < len(self)):
            raise IndexError("index {} is out of range".format(idx))
        if idx < 0:
            idx += len(self)
        it = iter(self._modules.values())
        for i in range(idx):
            next(it)
        return next(it)

    def __len__(self):
        return len(self._modules)

    def add(self, module, name=None):
        if name is None:
            name = str(len(self._modules))
            if name in self._modules:
                raise KeyError("name exists")
        self.add_module(name, module)

    # Example of setting up a DataLoader that also calculates spatial bounds
    def get_spatial_shape(self, data_loader):
        spatial_min = []
        spatial_max = []
        for data in data_loader:
            spatial_min.append(torch.min(data, dim=0).values)
            spatial_max.append(torch.max(data, dim=0).values)
        overall_min = torch.stack(spatial_min).min(dim=0).values
        overall_max = torch.stack(spatial_max).max(dim=0).values
        return torch.ceil(overall_max - overall_min).int().tolist()

    def voxelize(self, points, grid_size=(10, 10, 10), batch_size=256):
        # Points: Tensor of shape [batch_size, point_num, 3] (assuming last dimension contains XYZ coords)
        # Create an empty grid with a single channel where we will accumulate features
        voxel_grid = torch.zeros((batch_size, 1, *grid_size))
        
        # Normalization factor to scale point coordinates to voxel indices
        scaling_factors = torch.tensor(grid_size) - 1
        
        # Iterate over each item in the batch
        for b in range(batch_size):
            # Scale and convert point coordinates to grid indices
            grid_indices = (points[b, :, :3].to('cuda') * scaling_factors.to('cuda')).long()
            
            # Clamp indices to lie within the grid size
            grid_indices = torch.clamp(grid_indices.to('cuda'), torch.tensor(0).to('cuda'), torch.tensor(grid_size).to('cuda') - 1)

            print("Grid Indices: ", grid_indices)
            
            # Use features as values to accumulate in the voxel grid
            features = points[b, :, 3:]  # Assuming additional features beyond XYZ
            for i in range(points.shape[1]):
                print(voxel_grid[b, 0, grid_indices[i, 0], grid_indices[i, 1], grid_indices[i, 2]])
                print(features[i])
                voxel_grid[b, 0, grid_indices[i, 0], grid_indices[i, 1], grid_indices[i, 2]] += features[i]

        return voxel_grid
    
    def forward(self, input):
        print("\033[91mInput Type:\033[0m", type(input))
        print("\033[91mInput Shape:\033[0m", input.shape)

        # input = self.voxelize(input, grid_size=(10, 10, 10), batch_size=input.shape[0])
        print("\033[91mInput Shape:\033[0m", input.shape)

        for k, module in self._modules.items():
            # Point module
            if isinstance(module, PointModule):
                input = module(input)

            # Spconv module
            elif spconv.modules.is_spconv_module(module):
                if isinstance(input, Point):
                    input.sparse_conv_feat = module(input.sparse_conv_feat)
                    input.feat = input.sparse_conv_feat.features
                else:
                    input = module(input)

            # # Spconv module
            # elif spconv.modules.is_spconv_module(module):
            #     if isinstance(input, Point):
            #         if hasattr(input, 'sparse_conv_feat') and isinstance(input.sparse_conv_feat, spconv.SparseConvTensor):
            #             input.sparse_conv_feat = module(input.sparse_conv_feat)
            #             input.feat = input.sparse_conv_feat.features
            #         else:
            #             raise TypeError("Expected input to have 'sparse_conv_feat' of type SparseConvTensor")
            #     elif isinstance(input, spconv.SparseConvTensor):
            #         input = module(input)
            #     else:
            #         raise TypeError("Input type not supported for spconv module")

            # PyTorch module
            else:
                print(8.3)
                if isinstance(input, Point):
                    input.feat = module(input.feat)
                    if "sparse_conv_feat" in input.keys():
                        input.sparse_conv_feat = input.sparse_conv_feat.replace_feature(
                            input.feat
                        )
                elif isinstance(input, spconv.SparseConvTensor):
                    if input.indices.shape[0] != 0:
                        input = input.replace_feature(module(input.features))
                else:
                    print(9.6)
                    input = module(input)
                    print(9.7)
        print(100)
        return input

    # def forward(self, input):
    #     for k, module in self._modules.items():
    #         # PyTorch module
    #         if isinstance(input, Point):
    #             input.feat = module(input.feat)
    #             if "sparse_conv_feat" in input.keys():
    #                 input.sparse_conv_feat = input.sparse_conv_feat.replace_feature(
    #                     input.feat
    #                 )
    #         elif isinstance(input, spconv.SparseConvTensor):
    #             if input.indices.shape[0] != 0:
    #                 input = input.replace_feature(module(input.features))
    #         else:
    #             input = module(input)
    #     return input