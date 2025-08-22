_base_ = [
    '../datasets/reid_nuscenes_pts.py',
    '../default_runtime.py',
    '../schedules/cyclic_500e_lr1e-5.py',
]

# model settings
model = dict(
    type='ReIDNet',
    backbone=dict(
        type='PointTransformerV2_6C',
        use_precomputed_eigen=True,  # Enable 6-channel eigenvalue data
    ),
    # backbone=dict(type='PointTransformerV2',),  # Original 3-channel version
    # backbone=dict(type='ED_PointTransformerV2', ED_nsample=10, ED_conv_out=4, use_precomputed_eigen=True),  # ED version with 6-channel
    numpoints=256,
    use_dgcnn=False,
    use_attention=True,
    attention_layers=['cross_attention', 'local_self_attention'],
    attention_params=dict(
        cross_attention=dict(
            type='cross_attention',
            in_channels=64,
            out_channels=64,
            num_heads=8,
            dropout=0.1,
        ),
        local_self_attention=dict(
            type='local_self_attention',
            in_channels=64,
            out_channels=64,
            num_heads=8,
            dropout=0.1,
            k=16,  # Number of neighbors for local attention
        ),
    ),
    # ReID specific settings
    reid_head=dict(
        type='ReIDHead',
        in_channels=64,
        hidden_channels=128,
        out_channels=256,  # Feature dimension for re-identification
        dropout=0.5,
        num_classes=1000,  # Number of vehicle classes
    ),
    # Loss settings
    loss=dict(
        type='ReIDLoss',
        triplet_margin=0.3,
        classification_weight=1.0,
        triplet_weight=1.0,
    ),
)

# optimizer settings
optimizer = dict(type='AdamW', lr=1e-4, weight_decay=1e-4)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))

# learning rate scheduler
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 10,
    step=[400, 450]
)

# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=500)
checkpoint_config = dict(interval=10)
evaluation = dict(interval=10, metric='mAP', save_best='mAP')

# data settings
data = dict(
    samples_per_gpu=32,  # Reduced batch size for PointTransformerV2
    workers_per_gpu=4,
    train=dict(
        use_precomputed_eigen=True,  # Enable 6-channel data
        eigen_knn_size=10,
        sparse_loader=dict(
            load_feats=['xyz_eigen'],
            load_dims=[6],
        )
    ),
    val=dict(
        use_precomputed_eigen=True,  # Enable 6-channel data
        eigen_knn_size=10,
        sparse_loader=dict(
            load_feats=['xyz_eigen'],
            load_dims=[6],
        )
    ),
)

# logging settings
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook'),
    ]
)

# model checkpoint and evaluation
evaluation = dict(
    interval=10,
    metric='mAP',
    save_best='mAP',
    rule='greater',
)

# PointTransformerV2 specific settings
# Based on the original PointTransformerV2 paper and implementation
pointtransformer_v2_settings = dict(
    # Architecture settings from PointTransformerV2 paper
    encoder_blocks=[1, 4, 7, 4, 4],  # Number of blocks in each stage
    encoder_strides=[1, 3, 3, 3, 3],  # Downsampling ratios
    width=64,  # Base width for feature channels
    expansion=4,  # Expansion factor for feature channels
    radius=0.1,  # Ball query radius
    nsample=32,  # Number of neighbors for ball query
    # Grouped Vector Attention settings
    grouped_vector_attention=True,
    num_groups=8,  # Number of groups for grouped attention
    # Partition-based Pooling settings
    partition_based_pooling=True,
    partition_size=0.1,  # Partition size for pooling
)

# Performance expectations based on PointTransformerV2 paper
# PointTransformerV2 achieved state-of-the-art performance on:
# - ScanNet: 75.4% mIoU
# - S3DIS: 70.4% mIoU  
# - ScanNet200: 29.3% mIoU
# For re-identification tasks, we expect similar improvements in feature learning
# and cross-instance discrimination capabilities.

# Citation for PointTransformerV2
# @inproceedings{wu2022point,
#   title     = {Point transformer V2: Grouped Vector Attention and Partition-based Pooling},
#   author    = {Wu, Xiaoyang and Lao, Yixing and Jiang, Li and Liu, Xihui and Zhao, Hengshuang},
#   booktitle = {NeurIPS},
#   year      = {2022}
# }
