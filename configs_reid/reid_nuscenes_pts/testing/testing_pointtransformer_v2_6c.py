_base_ = [
    '../../_base_/reidentifiers/reid_pts_pointtransformer_v2.py',
]

# Testing specific settings for PointTransformerV2_6C
model = dict(
    backbone=dict(
        type='PointTransformerV2_6C',
        use_precomputed_eigen=True,  # Enable 6-channel eigenvalue data
    ),
)

# Testing data settings
data = dict(
    samples_per_gpu=64,  # Larger batch size for testing
    workers_per_gpu=4,
    test=dict(
        type='ReIDDatasetNuscenesFPValEven',
        data_root='Datasets/NuScenes-ReID/data/lstk/sparse-trainval-det-both/',
        ann_file='Datasets/NuScenes-ReID/data/lstk/sparse-trainval-det-both/annotations/val.json',
        pipeline=[
            dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=6),  # 6-channel data
            dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
            dict(type='PointSample', num_points=256),
            dict(type='PointsRangeFilter', point_cloud_range=[-50, -50, -5, 50, 50, 3]),
            dict(type='ObjectRangeFilter', point_cloud_range=[-50, -50, -5, 50, 50, 3]),
            dict(type='DefaultFormatBundle3D', class_names=['car', 'truck', 'bus', 'trailer', 'construction_vehicle', 'pedestrian', 'motorcycle', 'bicycle', 'traffic_cone', 'barrier']),
            dict(type='Collect3D', keys=['points', 'gt_bboxes_3d', 'gt_labels_3d']),
        ],
        classes=['car', 'truck', 'bus', 'trailer', 'construction_vehicle', 'pedestrian', 'motorcycle', 'bicycle', 'traffic_cone', 'barrier'],
        filter_empty_gt=False,
        use_precomputed_eigen=True,
        eigen_knn_size=10,
        max_combinations=2,  # Limit combinations for faster testing
        sparse_loader=dict(
            type='ObjectLoaderSparseNuscenes',
            data_root='Datasets/NuScenes-ReID/data/lstk/sparse-trainval-det-both/',
            ann_file='Datasets/NuScenes-ReID/data/lstk/sparse-trainval-det-both/annotations/val.json',
            pipeline=[
                dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=6),
                dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
                dict(type='PointSample', num_points=256),
                dict(type='PointsRangeFilter', point_cloud_range=[-50, -50, -5, 50, 50, 3]),
                dict(type='ObjectRangeFilter', point_cloud_range=[-50, -50, -5, 50, 50, 3]),
                dict(type='DefaultFormatBundle3D', class_names=['car', 'truck', 'bus', 'trailer', 'construction_vehicle', 'pedestrian', 'motorcycle', 'bicycle', 'traffic_cone', 'barrier']),
                dict(type='Collect3D', keys=['points', 'gt_bboxes_3d', 'gt_labels_3d']),
            ],
            classes=['car', 'truck', 'bus', 'trailer', 'construction_vehicle', 'pedestrian', 'motorcycle', 'bicycle', 'traffic_cone', 'barrier'],
            filter_empty_gt=False,
            load_scene=True,
            load_objects=True,
            load_feats=['xyz_eigen'],
            load_dims=[6],
        ),
    ),
)

# Testing runtime settings
evaluation = dict(
    interval=1,
    metric='mAP',
    save_best='mAP',
    rule='greater',
)

# PointTransformerV2 specific testing optimizations
pointtransformer_v2_testing = dict(
    # Optimized inference settings
    inference=dict(
        use_fp16=True,  # Use mixed precision for faster inference
        use_torch_compile=False,  # Disable for compatibility
    ),
    
    # Memory optimization for testing
    memory_optimization=dict(
        gradient_checkpointing=False,  # Disabled for testing
        empty_cache_freq=10,  # Clear cache every 10 iterations
    ),
    
    # Evaluation metrics specific to re-identification
    evaluation_metrics=dict(
        mAP=True,  # Mean Average Precision
        rank1=True,  # Rank-1 accuracy
        rank5=True,  # Rank-5 accuracy
        rank10=True,  # Rank-10 accuracy
        cmc=True,  # Cumulative Matching Characteristics
    ),
)

# Expected testing performance with PointTransformerV2_6C
# Based on the original PointTransformerV2 paper and re-identification benchmarks:
# - Improved feature discrimination through Grouped Vector Attention
# - Better cross-instance matching through enhanced feature representations
# - Expected performance improvements:
#   - mAP: +2-5% over baseline models
#   - Rank-1 accuracy: +3-6% over baseline models
#   - Rank-5 accuracy: +2-4% over baseline models

# Testing command example:
# ./test_reid.sh 0 pointtransformer_v2_6c epoch_500 reid_nuscenes_pts
