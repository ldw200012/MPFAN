# <p align="center">CALM-Net</p>

<b>C</b>urvature-<b>A</b>ware <b>L</b>iDAR point cloud-based <b>M</b>ulti-branch <b>N</b>etwork <b>(CALM-Net)</b> is a LiDAR point cloud-based 3D object re-identification model, powered by the aggregation of multiple feature vectors extracted using various sub-network structures.

## BEFORE START
📌 You need to SignUp to NeptuneAI, and input your Project & Token as below to two files.
- CALM-Net/configs_reid/\_base\_/reidentification_runtime.py
- CALM-Net/configs_reid/\_base\_/reidentification_runtime_testing.py
```
init_kwargs={
'project':"dongwooklee1201/mpfan-eval",
'api_token':"eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJiNmY3MjBiMi01Mjg2LTQwOTYtODIzYy00Mjk4MGIwMTQ4ZjcifQ==",
...
},
```

## BUILD REPOSITORY
1. Clone the CALM-Net git repository (Main Workspace)
```
git clone https://github.com/ldw200012/CALM-Net.git
```
2. Clone LAMTK git repository (Lamtk Library)
```
cd CALM-Net/
git clone https://github.com/c7huang/lamtk
```
## ENVIRONMENT SETUP (Docker)
1. Pull docker image (the image is with CUDA-11.3)
```
docker pull daldidan/mpfan:latest
```
2. Run docker container. You need to fix the parameters in run_docker.py
```
python3 tools/run_docker.py
```
3. (In docker container) Setup the dependencies
```
cd /mpfan
python setup.py develop --user

cd /mpfan/lamtk
pip install -e .
```

## GET PRETRAINED WEIGHTS
We provide you the pretrained weights for the following models: PointNet, PointNeXt, DGCNN, DeepGCN, Point Transformer, SPoTr, CALM-Net.

| Model          | Trained Epoch | # Params | Download  |
| -------------- | ------------- | -------- | --------- |
| PointNeXt      | 500           | -        | [LINK](#) |
| DGCNN          | 500           | -        | [LINK](#) |
| DeepGCN        | 500           | -        | [LINK](#) |
| Point Transformer | 500        | -        | [LINK](#) |
| SPoTr          | 500           | -        | [LINK](#) |
| CALM-Net          | 500           | -        | [LINK](https://drive.usercontent.google.com/download?id=1pGCarCGP6N-qt4nYr8WU7YqgYSuvEJUT) |

## <img src="https://cdn-icons-png.freepik.com/512/4834/4834296.png" width=15/> TRAIN
```
CUDA_VISIBLE_DEVICES={GPU-ID} MASTER_ADDR=localhost torchpack dist-run -v -np 1 python tools/train.py configs_reid/reid_{dataset_name}_pts/training/training_{model_name}.py --seed 66  --run-dir runs/
```

## <img src="https://cdn-icons-png.flaticon.com/512/5671/5671391.png" width=15/> TEST
```
CUDA_VISIBLE_DEVICES={GPU-ID} MASTER_ADDR=localhost torchpack dist-run -v -np 1 python tools/train.py configs_reid/reid_{dataset_name}_pts/testing/testing_{model_name}.py --checkpoint weights/{checkpoint_name}.pth
```

## Complexity Analysis Tools (Inference)
The repository provides tools for performing comprehensive complexity analysis on the ReIDNet models for testing/inference.

- A. Parameter Count: Model size in millions of parameters
- B. FLOPs/MACs: Estimated computational complexity for a fixed input (1024 points)
- C. Inference Latency: Runtime performance in milliseconds

### Run All Analyses via Script
Use the convenience script to run all analyses (parameter count, FLOPs/MACs estimation, and latency):
```
./tools/run_complexity_analysis.sh configs_reid/reid_nuscenes_pts/base_mpfan.py
```
Replace the config path with any supported configuration.

### Run Individually
- Parameter Count
```
python tools/count_params.py --config configs_reid/reid_nuscenes_pts/base_mpfan.py
```
- FLOPs/MACs (estimated)
```
python tools/flops_thop.py --config configs_reid/reid_nuscenes_pts/base_mpfan.py
```
- Inference Latency
```
python tools/latency.py --config configs_reid/reid_nuscenes_pts/base_mpfan.py
```

### Notes
- The FLOPs/MACs are estimated based on model architecture and parameters for models with complex input structures.
- Latency is measured with warmup iterations followed by measurement iterations. Use `--warmup` and `--iters` to control measurement fidelity.
- Scripts assume CUDA is available but will fall back to CPU if not.

## ACKNOWLEDGEMENTS
Out repository is based on <a href="https://github.com/bentherien/point-cloud-reid.git">point-cloud-reid</a>, <a href="https://github.com/open-mmlab/mmdetection3d.git">mmdetection3d</a>, and <a href="https://github.com/guochengqian/openpoints.git">openpoints</a>.

<!-- ## CITE OUR WORK -->