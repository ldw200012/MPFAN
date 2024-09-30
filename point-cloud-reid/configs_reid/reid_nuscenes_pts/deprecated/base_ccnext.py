_base_ = [
    "../_base_/datasets/reid_nuscenes_pts.py",
    "../_base_/reidentifiers/reid_pts_ccnext_point-cat.py",
]

model = dict(
    type='ReIDNet',

    backbone=dict(type='ConcatCloneNeXt', nsample=[32,48,48]),

    losses_to_use=dict(
        match=True,
        kl=False,
        triplet=True,
        fp=False,  #BCE
        cls=False, #CE
        ),
)

_bs = 128
_min_points = 128
_subsample_mode = "random" # random | fps | rand_crop
_val_subsample_mode = "fps" # random | fps | rand_crop
_subsample_sparse = 128

data = dict(
    samples_per_gpu = _bs,
    val_samples_per_gpu = _bs*2, # (val_)samples_per_gpu < subsample_sparse <= min_points
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

resume_from = None