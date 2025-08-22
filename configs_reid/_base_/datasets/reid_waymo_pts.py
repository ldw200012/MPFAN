CLASSES = ['car',
           'truck',
           'bus',
           'motorcycle',
           'bicycle',
           'pedestrian']

tracking_classes = {
    'bicycle':'bicycle',
    'truck':'truck',
    'car':'car',
    'trailer':'trailer',
    'bus':'bus',
    'motorcycle':'motorcycle',
    'pedestrian':'pedestrian'
}

tracking_classes_fp = tracking_classes

cls_to_idx = {
    'none_key':-1,
    'car':0,
    'truck':1,
    'bus':2,
    'motorcycle':3,
    'bicycle':4,
    'pedestrian':5
}

cls_to_idx_fp = {
    'none_key':-1,
    'car':0,
    'truck':1,
    'bus':2,
    'motorcycle':3,
    'bicycle':4,
    'pedestrian':5,
    'FP_car':6,
    'FP_truck':7,
    'FP_bus':8,
    'FP_motorcycle':9,
    'FP_bicycle':10,
    'FP_pedestrian':11,
}

train_metadata_version = 'waymo-det-both-train'
val_metadata_version = 'waymo-det-both-val'

resume_from = None

#################### CUSTOMIZE HERE ####################
load_feats = ['xyz_eigen']  # 'xyz' / 'xyz_eigen'
load_dims = [6]             # [3] / [6]
eigen_knn_size = 16
use_precomputed_eigen = True
#################### CUSTOMIZE HERE ####################

data = dict(
    samples_per_gpu=64,
    val_samples_per_gpu=128,
    workers_per_gpu=4,
    train=dict(type='ReIDDatasetWaymoFP',
               train=True,
               cls_to_idx=cls_to_idx,
               cls_to_idx_fp=cls_to_idx_fp,
               tracking_classes=tracking_classes,
               tracking_classes_fp=tracking_classes_fp,
               subsample_sparse=128,
               subsample_mode="random",
               val_subsample_mode="fps",
               CLASSES=CLASSES,
               return_mode='dict',
               verbose=False,
               validation_seed=0,
               use_precomputed_eigen=use_precomputed_eigen,  # New option to use pre-computed eigenvalues
               eigen_knn_size=eigen_knn_size,  # KNN sample size used for eigenvalue computation
               sparse_loader=dict(type='ObjectLoaderSparseWaymo',
                                metadata_path='Datasets/Waymo-ReID/data/lstk/updated_sparse-{}/metadata'.format(train_metadata_version),
                                data_root='Datasets/Waymo-ReID/data/lstk/updated_sparse-{}'.format(train_metadata_version),
                                min_points=128,
                                tracking_classes=tracking_classes,
                                load_scene=True,
                                load_objects=True,
                                load_feats=load_feats,
                                load_dims=load_dims,),
            ),
    val=dict(type='ReIDDatasetWaymoFPValEven',
               train=False,
               cls_to_idx=cls_to_idx,
               cls_to_idx_fp=cls_to_idx_fp,
               tracking_classes=tracking_classes,
               tracking_classes_fp=tracking_classes_fp,
               subsample_sparse=128,
               subsample_mode="random",
               val_subsample_mode="fps",
               CLASSES=CLASSES,
               return_mode='dict',
               verbose=False,
               validation_seed=0,
               max_combinations=10,
               use_precomputed_eigen=use_precomputed_eigen,  # New option to use pre-computed eigenvalues
               eigen_knn_size=eigen_knn_size,  # KNN sample size used for eigenvalue computation
               sparse_loader=dict(type='ObjectLoaderSparseWaymo',
                                metadata_path='Datasets/Waymo-ReID/data/lstk/sparse-{}/metadata'.format(val_metadata_version),
                                data_root='Datasets/Waymo-ReID/data/lstk/sparse-{}'.format(val_metadata_version),
                                min_points=128,
                                tracking_classes=tracking_classes,
                                use_metdata_fix=True,
                                load_scene=True,
                                load_objects=True,
                                load_feats=load_feats,
                                load_dims=load_dims,),
            ),
)