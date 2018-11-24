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

# About the code
The `cubenet` folder contains the core code. In here you will find 4 files `layers.py`, `V_group.py`, `T4_group.py`, and `S4_group.py`

`layers.py` contains a `Layer` class, with key operations: `conv`, `Gconv`, `conv_block`, `Gconv_block`, `Gres_block`. The most important for us are `Gconv` and `Gconv_block`. 
- `Gconv()` constructs a group convolution
- `Gconv_block()` constructs a group convolution with group-equivariant batch norm and pointwise nonlinearity

## Creating a group CNN
Group CNNs are little more intricate than standard CNNs (technically called Z-CNNs). We have tried to make them as easy as possible to use in our code. You just need to pay attention at the _input_ and the _output_.

#### At the input
The activation tensors are 6D arrays, therefore __inputs to any Gconv modules must be 6D!__ We use the convention `[batch_size, depth, height, width, channels, group_dim]`. Notice the extra axis `group_dim`, this corresponds to the 'rotation dimension', it stores the activations at each discrete rotation of the kernel. For instance, for the four-group `group_dim=4`. (In hindsight, we should have placed `group_dim` before `channels` for aesthetic reasons, but hey ho!). 

At the input to a collection of group convolutional layers, you will have a `[batch_size, depth, height, width, channels]` input. To feed this into our code all you have to do is
```
x = tf.expand_dims(x, -1)
```
then feed `x` into a `Gconv` or `Gconv_block` layer. Our code will detect that the `group_dim` axis has only 1 channel and apply the appropriate form of convolution. The output will have `group_dim=4,12,24`.

#### Constructing layers
To construct a `Layer`, you need to first choose a group, your choices are from the strings `"V","T4","S4"`. For instance to create a four-group layer we write
```
myLayer = Layer("V")
```
After that you can construct multiple four-group convolutional layers using your `myLayer` object. For instance, to create three layers of group convolution with the same number of channels and kernel size we would write
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
Unless you study and understand the code inside out, stick to using just one particular choice of group throughout the entire network. For this we advise to create a single `Layer` object, which you will use to construct all group convolutions. 

#### At the output
If you are looking for activations, which are rotation invariant (in the sense of the particular group you have chosen), then you must coset pool (see [Section 6.3 of Cohen and Welling](https://arxiv.org/abs/1602.07576)). We found the easiest and most effective thing to do is to average pool. This is just averaging over the last dimension of your 6D tensor, so
```
x = tf.reduce_mean(x, -1)
```
You can then treat this 5D tensor just like a standard CNN tensor in a 3D translation-equivariant CNN.


# Example: ModelNet10
To see the Modelnet10 experiment go into `./modelnet`. You need to download the data (link below) and then you can run the `train.py` scripts.

### The data
I've already gone through the hassle of downloading the data and reformatting it. Thanks to [Daniel Maturana's](https://github.com/dimatura/voxnet) Voxnet code and [this handy code](http://vision.princeton.edu/projects/2014/3DShapeNets/3DShapeNetsCode.zip) from the [ShapeNet guys](http://vision.princeton.edu/projects/2014/3DShapeNets/) for doing most of the leg work. 

Due to lack of time and some annoying idiosyncrasies of Tensorflow, I have gone for a rather strange, but hopefully understandable data reprentation. We have decompressed all the model files into `.png`s, where I have reshaped [32,32,32] -> [32,32x32], i.e. each file is a 2D image containing a collection of cross-sections through the 3D model. This means we can use the TF dataset classes with minimal hassle (I should really change this at some point). When we load the data, we just read in a 2D `.png` and reshape into a 3D binary volumetric tensor.

### What you have to do
Download the [data](https://drive.google.com/file/d/1aO48z-Qzsctd29zWpeuOOqvoKF3hXfbU/view?usp=sharing) and [addresses](https://drive.google.com/file/d/1XsXEI0U9t6jdWrHp_NyW3ua-PWsUMDbT/view?usp=sharing). Place both `.zip` files in the `modelnet` folder and run
```
unzip addresses.zip 
rm addresses.zip
unzip data.zip 
rm data.zip
```

### Training
The basic call to train is `python train.py`. On its own it will do nothing, because you have to specify two things: 
1) use the `--architecture` flag, you have options `GVGG, GResnet`. They refer to a group-CNN version of a VGG network and Resnet.
2) use the `--group` flag to specify the specific rotation subgroup with options `V,T4,S4` corresponding to 4 rotations, 12 rotations, and 24 rotations, respectively.
A typical call is then
```
python train.py --architecture GVGG --group V
```
This will create a `models/` folder with the default first being `models/model_0`. Rerunning the code will ask you to overwrite this model. If you do not want that use the `--path_increment` flag to automatically increment this to `models/model_1`, otherwise you are free to change the naming conventions via tha `--save_dir` and `--log_dir` flags. Just note that the model name should be of the form `<myModelName>_0`, and `myModelName` may not contain any underscores.

### Testing
Just run 
```
  python test.py
```
Note that _this only works for the default GVGG model at the moment_. If you have used a different path for the model file, then you need to add the flag `--save_dir <path_to_folder_containing_checkpoint>`. Do note that `test.py` is very bittle (my bad) and you should avoid changing things like `batch_size` or the `shuffle` option, because rotation averaging will break. 

You should have results in the region of if you train long enough
```
Test accuracy: 0.9420
Test accuracy rot avg: 0.9460
 ```
