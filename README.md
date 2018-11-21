# CubeNet: Equivariance to 3D Rotation and Translation

This code contains a Tensorflow implementation of a discrete 3D roto-translation convolution and some example of models using them on the ModelNet10 benchmark. More details can be found in [our ECCV18 paper](https://arxiv.org/abs/1804.04458). 

If you find code useful, please cite us as
```
@inproceedings{Worrall18,
  title     = {CubeNet: Equivariance to 3D Rotation and Translation},
  author    = {Daniel E. Worrall and Gabriel J. Brostow},
  booktitle = {Computer Vision - {ECCV} 2018 - 15th European Conference, Munich,
               Germany, September 8-14, 2018, Proceedings, Part {V}},
  pages     = {585--602},
  year      = {2018},
  doi       = {10.1007/978-3-030-01228-1\_35},
}
```

## Requirements
You will need
- Python 3.6
- Tensorflow 1.8
- Standard libraries: numpy, sympy, scikit-image etc..

## ModelNet100
To see the Modelnet 10 experiment go into `./modelnet`. You need to download the data and then you can run the `train.py` and `test.py` scripts.

### The data
I've already gone through the hassle of downloading the data and reformatting it. Thanks to [Daniel Maturana's](https://github.com/dimatura/voxnet) Voxnet code and [this handy code](http://vision.princeton.edu/projects/2014/3DShapeNets/3DShapeNetsCode.zip) from the [ShapeNet guys](http://vision.princeton.edu/projects/2014/3DShapeNets/) for doing most of the leg work. In the original `3DShapeNets` folder is a subdirectory named `volumetric_data`. In there you will find all the numpy files.

Due to lack of time and some annoying idiosyncrasies of Tensorflow, I have gone for a rather strange, but hopefully understandable data reprentation. We have decompressed all the model files into `.png`s, where I have reshaped [32,32] -> [32,32x32]. These means we can use the TF dataset classes with minimal hassle... I should really change this at some point...

### What you have to do
Place the train and test data in folders named
`<root>/data/modelnet<num>_train`
`<root>/data/modelnet<num>_test`
where `<num>=10` or `<num>=40` depending on the dataset. Note that you will first need to untar the files with a command like `tar -xvf modelnet10_train.tar`

### Training
To train a model you have to specific two things: 
1) use the `--architecture` flag, you have options `GVGG, GResnet`. There refer to a group-CNN version of a VGG network and Resnet.
2) use the `--group` flag to specify the specific rotation subgroup with options `V,T4,S4` corresponding to 4 rotations, 12 rotations, and 24 rotations, respectively.
A typical call is then
```
python train.py --architecture GVGG --group V
```

# About the code
The `cubenet` folder contains the core code. In here you will find 4 files `layers.py`, `V_group.py`, `T4_group.py`, and `S4_group.py`

`layers.py` contains a `Layer` class, with key operations: `conv` `Gconv`, `conv_block`, `Gconv_block`, `Gres_block`. The most important for us are `Gconv` and `Gconv_block`. 
- `Gconv()` constructs a group convolution
- `Gconv_block()` constructs a group convolution with group-equivariant batch norm and pointwise nonlinearity

## Creating a group CNN
There are just two things you need to bear mind mind when designing a group CNN. 

1) The activation tensors are 6D arrays. Therefore inputs to any Gconv modules must be 6D! This can be achieved with
```
x = tf.expand_dims(x, -1)
```
We use the convention `[batch_size, depth, height, width, channels, group_dim]`. Notice the extra axis `group_dim`, this corresponds to the 'rotation dimension', it stores the activations at each discrete rotation of the kernel. (In hindsight, we should have placed `group_dim` before `channels` for aesthetic reason, but hey ho!)

2) Unless you study and understand the code inside out, stick to using just one group throughout the entire network. For this we advise to create a single `Layer` object, which you will use to construct all group convolutions. 

## But how do I construct a layer?
To construct a `Layer`, you need to first choose a group, your choices are from the strings `"V","T4","S4"`. For instance to create a four-group layer we write
```
myLayer = Layer("V")
```
After that you can construct multiple four-group convolutional layers using your `mygroup` object. For instance, to create three layers of group convolution with the same number of channels and kernel size we would write
```
activations1 = myLayer.Gconv(input, kernel_size, channels_out, is_training)
activations2 = myLayer.Gconv(activations1, kernel_size, channels_out, is_training)
activations3 = myLayer.Gconv(activations2, kernel_size, channels_out, is_training)
```
If we want to include batch norm and ReLUs, then we should have instead used a `Gconv_block`, so
```
activations1 = myLayer.Gconv_block(input, kernel_size, channels_out, is_training, use_bn=True, fnc=tf.nn.relu)
activations2 = myLayer.Gconv_block(activations1, kernel_size, channels_out, is_training, use_bn=True, fnc=tf.nn.relu)
activations3 = myLayer.Gconv_block(activations2, kernel_size, channels_out, is_training, use_bn=True, fnc=tf.nn.relu)
```
