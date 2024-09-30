# MPFAN: Multi-Perspective Feature Aggregation Network
MPFAN is a model aimed to discover the complementarity of the existing backbone networks (or feature extracting mechanisms).

## 1. How to run
1. Clone the repository
```
git clone https://github.com/ldw200012/MPFAN.git
```

2. Pull the Docker image (if docker is not installed, please install. <font style="color:red;">This Docker image is linked with CUDA-11.3</font>) and check if it is correctly done.
```
docker image pull daldidan/dev:bentherien_modified

docker images # (this should output like below)
# REPOSITORY     TAG                   IMAGE ID       CREATED         SIZE
# daldidan/dev   bentherien_modified   21438bf47948   3 months ago    34.2GB
```

3. Run the docker image with below command.
```
cd MPFAN/point-cloud-reid/tools
python3 run_docker.py
```

4. (Only once forever) Setup the libraries
```
cd MPFAN/point-cloud-reid
python setup.py develop --user
```

## 2. Try the model
1. Download the pre-trained weight and locate it in ./weights folder.
<a href="https://drive.usercontent.google.com/download?id=1pGCarCGP6N-qt4nYr8WU7YqgYSuvEJUT">https://drive.usercontent.google.com/download?id=1pGCarCGP6N-qt4nYr8WU7YqgYSuvEJUT</a>
3. Run the code below to run the MPFAN model with a pre-trained weight
```
CUDA_VISIBLE_DEVICES=0 MASTER_ADDR=localhost torchpack dist-run -v -np 1 python tools/train.py configs_reid/reid_nuscenes_pts/testing/testing_dualreid.py --checkpoint weights/MPFAN_epoch_500.pth
```
