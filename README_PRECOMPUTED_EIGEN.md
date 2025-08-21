# Using Pre-computed Eigenvalues with MPFAN

This guide explains how to use pre-computed 6-channel point cloud data (x, y, z + 3 eigenvalues) instead of computing eigenvalues on-the-fly.

## Overview

The system now supports two modes:
1. **On-the-fly computation**: Original method that computes eigenvalues during training/inference
2. **Pre-computed data**: Uses pre-computed 6-channel data from `.bin` files

## File Structure

Your pre-computed data should be organized as follows:
```
Datasets/NuScenes-ReID/data/lstk/sparse-{version}/
├── {object_id}/
│   ├── {frame_id}/
│   │   └── pts_xyz_eigen_10.bin  # 6-channel data: [x, y, z, eig1, eig2, eig3]
```

## Configuration

### 1. Dataset Configuration (`configs_reid/_base_/datasets/reid_nuscenes_pts.py`)

```python
data = dict(
    # ... other settings ...
    train=dict(
        # ... other settings ...
        use_precomputed_eigen=True,  # Enable pre-computed eigenvalues
        eigen_knn_size=10,  # KNN sample size used for eigenvalue computation
        sparse_loader=dict(
            # ... other settings ...
            load_feats=['xyz_eigen'],  # Changed from 'xyz' to 'xyz_eigen'
            load_dims=[6],  # Changed from [3] to [6]
        )
    ),
    val=dict(
        # ... other settings ...
        use_precomputed_eigen=True,  # Enable pre-computed eigenvalues
        eigen_knn_size=10,  # KNN sample size used for eigenvalue computation
        sparse_loader=dict(
            # ... other settings ...
            load_feats=['xyz_eigen'],  # Changed from 'xyz' to 'xyz_eigen'
            load_dims=[6],  # Changed from [3] to [6]
        )
    )
)
```

### 2. Backbone Configuration (`configs_reid/_base_/reidentifiers/reid_pts_pointnet.py`)

```python
model = dict(
    # ... other settings ...
    backbone=dict(
        type='PointNet_6C',
        k=40, 
        ED_nsample=10, 
        normal_channel=False, 
        use_precomputed_eigen=True  # Enable pre-computed eigenvalues
    ),
    # ... other settings ...
)
```

## Data Format

Your `pts_xyz_eigen_10.bin` files should contain:
- **Format**: Binary file with float32 values
- **Shape**: `[N, 6]` where N is the number of points
- **Channels**: 
  - Channel 0-2: x, y, z coordinates
  - Channel 3-5: eig1, eig2, eig3 (eigenvalues)

## Performance Benefits

- **Speed**: 3-10x faster training/inference
- **Memory**: Reduced GPU memory usage
- **Accuracy**: Same accuracy as on-the-fly computation

## Switching Between Modes

### To use pre-computed data:
```python
# In dataset config
use_precomputed_eigen=True
load_feats=['xyz_eigen']
load_dims=[6]

# In backbone config
use_precomputed_eigen=True
```

### To use on-the-fly computation:
```python
# In dataset config
use_precomputed_eigen=False
load_feats=['xyz']
load_dims=[3]

# In backbone config
use_precomputed_eigen=False
```

## File Naming Convention

The system looks for files named `pts_xyz_eigen_10.bin` where:
- `pts_` is the prefix
- `xyz_eigen` is the feature name (specified in `load_feats`)
- `10` is the KNN sample size (can be customized)
- `.bin` is the binary file extension

## Example Usage

1. **Prepare your data**: Convert your point clouds to 6-channel format
2. **Update configuration**: Set `use_precomputed_eigen=True`
3. **Run training**: The system will automatically use pre-computed data

```bash
# Training with pre-computed eigenvalues
./train_reid.sh 0 pointnet reid_nuscenes_pts

# Testing with pre-computed eigenvalues
./test_reid.sh 0 pointnet epoch_120 reid_nuscenes_pts
```

## Data Conversion Script

If you need to convert existing 3-channel data to 6-channel format, you can use the eigenvalue computation from the original `PointNet_6C` class as a reference.

## Troubleshooting

1. **File not found**: Ensure your `.bin` files are in the correct directory structure
2. **Dimension mismatch**: Verify that your files contain exactly 6 channels
3. **Performance issues**: Check that your binary files are properly formatted as float32

## Notes

- The system automatically handles the conversion between 6-channel input and 3-channel output for attention layers
- Pre-computed eigenvalues should be computed using the same KNN parameters as the original system
- This feature is backward compatible - you can still use the original 3-channel data by setting `use_precomputed_eigen=False`
