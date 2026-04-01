# F-DSNet: a frustum-guided lightweight 3D object detection method in complex scenes

This repository is the code for our  paper [[IEEEXplore]](https://iopscience.iop.org/article/10.1088/2631-8695/ae5462).

## Citation

If you find this work useful in your research, please consider citing.

```BibTeX
@article{zhang2026fdsnet,
    author = {Zhang, X. and Chen, M. and Song, C. and others},
    title = {F-DSNet: A Frustum-guided Lightweight 3D Object Detection Method in Complex Scenes},
    journal = {Engineering Research Express},
    year = {2026}
}
```

## Installation

### Requirements

* PyTorch 1.8+
* Python 3.8+

We test our code under Ubuntu-18.04 with CUDA-11.3

### Clone the repository and install dependencies

```shell
git clone https://github.com/Asov000/F-DSNet.git
```

You may need to install extra packages, like pybind11, opencv, yaml, tensorflow(optional).

If you want to use tensorboard to visualize the training status, you should install tensorflow (CPU version is enough).
Otherwise, you should set the config 'USE_TFBOARD: False' in cfgs/\*.yaml.

### Compile extension

```shell
cd ops
bash clean.sh
bash make.sh
```

## Download data

Download the SUNRGB-D 3D object detection dataset from [here](https://rgbd.cs.princeton.edu/) and organize them as follows.


## Training and evaluation

### First stage

Run following command to prepare pickle files for car training. We use the 2D detection results from F-PointNets.
The pickle files will be saved in `sunrgbd/data/pickle_data`.

```shell
python sunrgbd/prepare_data.py --gen_train --gen_val --gen_val_rgb_detection
```

Run following commands to train and evaluate the final model. You can use `export CUDA_VISIBLE_DEVICES=?` to specify which GPU to use.
And you can modify the setting after `OUTPUT_DIR` to set a directory to save the log, model files and evaluation results.  All the config settings are under the configs/config.py.

```shell
python train/train_net_det.py --cfg cfgs/det_sample_sunrgbd.yaml OUTPUT_DIR output/sunrgbd
python train/test_net_det_sunrgbd.py --cfg cfgs/det_sample_sunrgbd.yaml OUTPUT_DIR output/sunrgbd TEST.WEIGHTS output/sunrgbd/model_final.pth SAVE_SUB_DIR test_gt2D FROM_RGB_DET False
```

We also provide the shell script, so you can also run `bash scripts/sunrgbd_train.sh` instead.


## Pretrained models
We provide the pretrained models for car category, you can download from [here](https://pan.baidu.com/s/1z0KJ9a3rw_ZAJbQrUk0_jw).
Extraction code AAAA


## Acknowledgements

Part of the code was adapted from [F-ConvNet](https://github.com/Gorilla-Lab-SCUT/frustum-convnet.git).


