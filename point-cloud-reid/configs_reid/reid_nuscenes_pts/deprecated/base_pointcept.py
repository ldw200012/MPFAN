_base_ = [
    "../_base_/datasets/reid_nuscenes_pts.py",
    "../_base_/reidentifiers/reid_pts_pointcept_point-cat.py",
]

model = dict(
# 0) Siamese MoE
    type='ReIDNet',

# 1) Choose Feature Backbone Model
    backbone=dict(type='Pointcept'),

# 2) Loss Functions to use
    losses_to_use=dict(
        match=True,
        kl=False,
        triplet=False,
        fp=False,  #BCE
        cls=False, #CE
        ),
)

# 3) Batchsize
_bs = 128

# -) min_points
_min_points = 128

# 4) subsample mode
_subsample_mode = "random"
# _subsample_mode = "fps"
# _subsample_mode = "rand_crop"

# _val_subsample_mode = "random"
_val_subsample_mode = "fps"

# 5) subsample size
_subsample_sparse = 128

data = dict(
    samples_per_gpu = _bs,
    val_samples_per_gpu = _bs*2,
    workers_per_gpu = 4,
    train = dict(
        subsample_sparse = _subsample_sparse,
        subsample_mode = _subsample_mode,
        val_subsample_mode=_val_subsample_mode,
        sparse_loader = dict(
            min_points = _min_points
        ),
    ),
    val = dict(
        subsample_sparse = _subsample_sparse,
        subsample_mode = _subsample_mode,
        val_subsample_mode=_val_subsample_mode,
        sparse_loader = dict(
            min_points = _min_points
        ),
    ),
)

# (val_)samples_per_gpu < subsample_sparse <= min_points

resume_from = None