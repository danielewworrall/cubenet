# CubeNet: Equivariance to 3D Rotation and Translation

# Overview
This repo contains the basic code you need to reimplement [our ECCV18 paper](https://arxiv.org/abs/1804.04458)

## Requirements
You will need
- Python 3.6
- Tensorflow 1.4
- Sympy, scipy, numpy, scikit-image

## Code highlights
## Models
## Training and testing

# ModelNet10

## The data
I've already gone through the hassle of downloading the data and reformatting it. Thanks to [Daniel Maturana's](https://github.com/dimatura/voxnet) Voxnet code and [this handy code](http://vision.princeton.edu/projects/2014/3DShapeNets/3DShapeNetsCode.zip) from the [ShapeNet guys](http://vision.princeton.edu/projects/2014/3DShapeNets/) for doing most of the leg work. In the original `3DShapeNets` folder is a subdirectory named `volumetric_data`. In there you will find all the numpy files.

Due to lack of time and some annoying idiosyncrasies of Tensorflow, I have gone for a rather strange, but hopefully understandable data reprentation. We have decompressed all the model files into `.png`s, where I have reshaped [32,32] -> [32,32x32]. These means we can use the TF dataset classes with minimal hassle... I should really change this at some point...

### What you have to do
Place the train and test data in folders named
`<root>/data/modelnet<num>_train`
`<root>/data/modelnet<num>_test`
where `<num>=10` or `<num>=40` depending on the dataset. Note that you will first need to untar the files with a command like `tar -xvf modelnet10_train.tar`
