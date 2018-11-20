# CubeNet: Equivariance to 3D Rotation and Translation

# Overview
This repo contains the basic code you need to reimplement [our ECCV18 paper](https://arxiv.org/abs/1804.04458). 

## Requirements
You will need
- Python 3.6
- Tensorflow 1.8
- Standard libraries: Numpy, sympy, scikit-image etc..

# ModelNet10
To see the Modelnet 10 experiment go into `./modelnet`. You need to download the data and then you can run the `train.py` and `test.py` scripts.

## The data
I've already gone through the hassle of downloading the data and reformatting it. Thanks to [Daniel Maturana's](https://github.com/dimatura/voxnet) Voxnet code and [this handy code](http://vision.princeton.edu/projects/2014/3DShapeNets/3DShapeNetsCode.zip) from the [ShapeNet guys](http://vision.princeton.edu/projects/2014/3DShapeNets/) for doing most of the leg work. In the original `3DShapeNets` folder is a subdirectory named `volumetric_data`. In there you will find all the numpy files.

Due to lack of time and some annoying idiosyncrasies of Tensorflow, I have gone for a rather strange, but hopefully understandable data reprentation. We have decompressed all the model files into `.png`s, where I have reshaped [32,32] -> [32,32x32]. These means we can use the TF dataset classes with minimal hassle... I should really change this at some point...

## What you have to do
Place the train and test data in folders named
`<root>/data/modelnet<num>_train`
`<root>/data/modelnet<num>_test`
where `<num>=10` or `<num>=40` depending on the dataset. Note that you will first need to untar the files with a command like `tar -xvf modelnet10_train.tar`

## Training
To train a model you have to specific two things: 
1) use the `--architecture` flag, you have options `GVGG, GResnet`. There refer to a group-CNN version of a VGG network and Resnet.
2) use the `--group` flag to specify the specific rotation subgroup with options `V,T4,S4` corresponding to 4 rotations, 12 rotations, and 24 rotations, respectively.
A typical call is then
```
python train.py --architecture GVGG --group V
```

# About the code
The `cubenet` folder contains the core code. In here you will find 4 files
- `layers.py`
- `V_group.py`, `T4_group.py`, `S4_group.py`

`layers.py` contains a `Layer` class, with key operations: `conv` `Gconv`, `conv_block`, `Gconv_block`, `Gres_block`. The most important for us are `Gconv` and `Gconv_block`. 
- `Gconv()` constructs a group convolution
- `Gconv_block()` constructs a group convolution with group-equivariant batch norm and pointwise nonlinearity

To construct a `Layer`, we pass a
