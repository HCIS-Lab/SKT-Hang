# SKT-Hang

- This GitHub repo is the implementation of the paper "SKT-Hang: Hanging Everyday Objects via Object-Agnostic Semantic Keypoint Trajectory Generation"

![SKT-Hang](images/skt-animation.png)

## System Requirements
- Linux (Teseted on Ubuntu 20.04)
- Python 3 (Tested on Python 3.7)
- Torch (Tested on Torch 1.13.1)
- Cuda (Tested on Cuda 11.8)
- GPU (Tested on Nvidia RTX3090, RTX4090)
- CPU (Tested on Intel COre i7-12700, Intel Xeon Silver 4210R)

## Setup
- Clone This Repo
```
$ git clone https://github.com/HCIS-Lab/SKT-Hang.git
```
- Create Conda Environment
```
$ cd SKT-Hang
$ conda create -n skt-hang python=3.7
$ conda activate skt-hang
$ pip install -r requirements.txt
```
- Install [PointNet++](https://github.com/erikwijmans/Pointnet2_PyTorch)
```
$ git clone https://github.com/erikwijmans/Pointnet2_PyTorch
$ cd Pointnet2_PyTorch
$ pip install -r requirements.txt
$ pip install -e .
```

## File Structures
```
skt-hang/
    ├── config/
    │   ├── affordance/ # config yamls for training affordance prediction module
    │   ├── sctdn/ # config yamls for training SCTDN
    │   ├── vatmart/ # config yamls for vatmart
    │   └── modified_vatmart/ # config yamls for modified vatmart
    │
    ├── dataset/ # put all the dataset folders here
    │
    ├── shapes/ # put all the 3D shapes here
    │   ├── hook_all_new/ # all the supporting items
    │   ├── inference_objs_5/ # 5 objects for validation
    │   ├── inference_objs_50/ # 50 objects for testing
    │   └── wall/ # environment
    │
    └── src/ # all the source code
        ├── checkpoints/ # put all the checkpoints here
        ├── dataset/ # all the dataset modules
        ├── inference/ # for the inference results (.gifs, .pngs)
        └── models/ # all the network architectures (SCTDN, VAT-Mart, Modified Vat-Mart)
        ├── pybullet_robot_envs/ # robot manipulation framework
        ├── utils/ # useful tools and scripts
        ├── run_sctdn.py # for SCTDN training and inference
        ├── run_sctdn.sh # scripts for SCTDN training and inference
        ├── run_vatmart.py # for VAT-Mart training and inference
        ├── run_vatmart.sh # scripts for VAT-Mart training and inference
        ├── run_modified_vatmart.py # for Modified VAT-Mart training and inference
        └── run_modified_vatmart.sh # scripts for Modified VAT-Mart training and inference

```

## Citation
```
@article{skthang2023,
  title={SKT-Hang: Hanging Everyday Objects via Object-Agnostic Semantic Keypoint Trajectory Generation},
  author={Chia-Liang Kuo, Yu-Wei Chao, Yi-Ting Chen},
  year={2023},
  booktitle={arXiv},
}
```

