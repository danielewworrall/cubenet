"""Models for the RFNN experiments"""
import argparse
import os
import sys
import time

import numpy as np
import tensorflow as tf

#from ops_layer import get_steerable_basis
# ------- Spherical harmonics -------- #
from scipy.special import sph_harm
from sympy.physics.quantum.spin import Rotation



class DiscreteModel(object):
    def __init__(self, images, n_classes, args, is_training):
        # Constants
        self.batch_size = images.get_shape().as_list()[0]
        self.color_chn = images.get_shape().as_list()[3]
        print("...Computing Cayley table")
        self.cayley = self.get_cayleytable()

        # Inputs
        self.images = images
        self.n_classes = n_classes
        self.fnc = lambda x: tf.nn.relu(x) - 0.2*tf.nn.relu(-x)
        self.ks = args.kernel_size
        self.first_ks = args.first_kernel_size
        self.nc = args.n_channels_out
        self.basis = self.get_steerable_filters(self.ks, 4, 4)
        self.batch_size = tf.shape(images)[0]

        # model predictions
        print("...Constructing network")
        self.pred_logits = self.get_pred(self.images, is_training)

    def get_pred(self, x, is_training, reuse=False):
        with tf.variable_scope('prediction', reuse=reuse) as scope:
            init = tf.contrib.layers.variance_scaling_initializer()
            use_bn = True
            group_dim = 4
            nc = int(self.nc/group_dim)

            x = tf.expand_dims(x, -1)
            x = self.Vconv_block(x, self.first_ks, nc, is_training, use_bn=use_bn, name="S4_1a", basis=self.basis)
            x = self.Vconv_block(x, self.ks, nc, is_training, use_bn=use_bn, name="S4_1b", basis=self.basis)
            print(x)
            x = self.Vconv_block(x, self.ks, 2*nc, is_training, use_bn=use_bn, strides=2, name="S4_2a", basis=self.basis)
            x = self.Vconv_block(x, self.ks, 2*nc, is_training, use_bn=use_bn, strides=1, name="S4_2b", basis=self.basis)
            print(x)
            x = self.Vconv_block(x, self.ks, 4*nc, is_training, use_bn=use_bn, strides=2, name="S4_3a", basis=self.basis)
            x = self.Vconv_block(x, self.ks, 4*nc, is_training, use_bn=use_bn, strides=1, name="S4_3b", basis=self.basis)
            print(x)
            x = self.Vconv_block(x, self.ks, 8*nc, is_training, use_bn=use_bn, strides=2, name="S4_4a", basis=self.basis)
            x = self.Vconv_block(x, self.ks, 8*nc, is_training, use_bn=use_bn, strides=1, name="S4_4b", basis=self.basis)
            x = self.Vconv_block(x, self.ks, 8*nc, is_training, use_bn=use_bn, strides=1, name="S4_4c", basis=self.basis)
            print(x)
            x = self.Vconv_block(x, self.ks, 16*nc, is_training, use_bn=use_bn, strides=2, name="S4_5a", basis=self.basis)
            x = self.Vconv_block(x, self.ks, 16*nc, is_training, use_bn=use_bn, strides=1, name="S4_5b", basis=self.basis)
            x = self.Vconv_block(x, self.ks, 16*nc, is_training, use_bn=use_bn, strides=1, name="S4_5c", basis=self.basis)
            print(x)


            keep_prob = 1. - 0.5*tf.to_float(is_training)
            # Cyclic pool (mean)
            x = tf.reduce_mean(x, [1,2,3,5])
            x = tf.reshape(x, [self.batch_size,1,1,1,x.get_shape().as_list()[-1]])

            # Fully connected layers
            x = tf.nn.dropout(x, keep_prob)
            x = self.conv_block(x, 1, 512, is_training, use_bn=False, name="fc1")

            x = tf.nn.dropout(x, keep_prob)
            with tf.variable_scope("logits"):
                pred_logits = self.conv_block(x, 1, self.n_classes, is_training, use_bn=False, fnc=tf.identity)
                return tf.squeeze(pred_logits)


    ##### Helpful functions #####
    # Here we provide two flavors of convolution, standard and structured receptive field
    ## Conv
    def get_kernel(self, name, shape, factor=2.0, trainable=True):
        init = tf.contrib.layers.variance_scaling_initializer(factor=factor)
        return tf.get_variable(name, shape, initializer=init, trainable=trainable)


    def conv(self, x, kernel_size, n_out, strides=1, padding="SAME"):
        """A basic 3D convolution"""
        with tf.variable_scope("conv"):
            n_in = x.get_shape().as_list()[-1]
            W = self.get_kernel('W', [kernel_size,kernel_size,kernel_size,n_in,n_out])
            return tf.nn.conv3d(x, W, (1,strides,strides,strides,1), padding)


    def conv_block(self, x, kernel_size, n_out, is_training, use_bn=True, strides=1,
                   padding="SAME", fnc=tf.nn.relu, name="conv_block"):
        """Convolution with batch normalization/bias and nonlinearity"""
        with tf.variable_scope(name):
            y = self.conv(x, kernel_size, n_out, strides=strides, padding=padding)
            beta_init = tf.constant_initializer(0.01)
            if use_bn:
                return fnc(tf.layers.batch_normalization(y, training=is_training,
                           beta_initializer=beta_init))
            else:
                bias = tf.get_variable("bias", [n_out], initializer=beta_init)
                return fnc(tf.nn.bias_add(y, bias))


    def Vconv_block(self, x, kernel_size, n_out, is_training, use_bn=True, strides=1,
                   padding="SAME", fnc=tf.nn.relu, name="p4conv_block", basis=[]):
        """Convolution with batch normalization/bias and nonlinearity"""
        with tf.variable_scope(name):
            y = self.Vconv(x, kernel_size, n_out, strides=strides, padding=padding, basis=basis, is_training=is_training)
            beta_init = tf.constant_initializer(0.01)
            y = tf.transpose(y, perm=[0,1,2,3,5,4])
            ysh = y.get_shape().as_list()
            if use_bn:
                y = tf.layers.batch_normalization(y, training=is_training, beta_initializer=beta_init)
            else:
                bias = tf.get_variable("bias", [n_out], initializer=beta_init)
                y = tf.nn.bias_add(y, bias)
            return tf.transpose(fnc(y), perm=[0,1,2,3,5,4])


    def Vconv(self, x, kernel_size, n_out, strides=1, padding="SAME", basis=[], is_training=False):
        """Perform a discretized convolution on SO(3)

        Args:
            x: [batch_size, height, width, n_in, group_dim/1]
            kernel_size: int for the spatial size of the kernel
            n_out: int for number of output channels
            strides: int for spatial stride length
            padding: "valid" or "same" padding
        Returns:
            [batch_size, new_height, new_width, new_depth, n_out, group_dim] tensor in p4
        """
        group_dim = 4
        with tf.variable_scope('pNconv'):
            xsh = x.get_shape().as_list()
            init = tf.variance_scaling_initializer()
            # W is the base filter. We rotate it 4 times for a p4 convolution over
            # R^2. For a p4 convolution over p4, we rotate it, and then shift up by
            # one dimension in the channels.
            #W = get_kernel("W", [kernel_size, kernel_size, kernel_size, xsh[4]*xsh[5]*n_out])
            weights = self.get_kernel("weights", [basis.shape[-1], xsh[4]*xsh[5]*n_out])
            basis = np.reshape(basis, [kernel_size*kernel_size*kernel_size,-1])
            basis = tf.constant(basis.astype(np.float32))

            W = basis @ weights
            W = tf.reshape(W, [kernel_size, kernel_size, kernel_size, -1])
            WN = self.get_Vrotations(W)
            WN = tf.stack(WN, -1)
            # Reshape and rotate the io filters 4 times. Each input-output pair is
            # rotated and stacked into a much bigger kernel
            print(x)
            xN = tf.reshape(x, [self.batch_size, xsh[1], xsh[2], xsh[3], xsh[4]*xsh[5]])
            if xsh[-1] == 1:
                # A convolution on R^2 is just standard convolution with 3 extra 
                # output channels for eacn rotation of the filters
                WN = tf.reshape(WN, [kernel_size, kernel_size, kernel_size, xsh[4], -1])
            elif xsh[-1] == group_dim:
                # A convolution on p4 is different to convolution on R^2. For each
                # dimension of the group output, we need to both rotate the filters
                # and circularly shift them in the input-group dimension. In a
                # sense, we have to spiral the filters
                WN = tf.reshape(WN, [kernel_size, kernel_size, kernel_size, xsh[4], group_dim, n_out, group_dim])
                print("WN: {}".format(WN))
                # [kernel_size, kernel_size, kernel_size, n_in, 4, n_out, 4]
                # Shift over axis 4
                WN_shifted = self.V_permutation(WN)
                WN = tf.stack(WN_shifted, -1)
                # Shift over axis 6
                # Stack the shifted tensors and reshape to 4D kernel
                WN = tf.reshape(WN, [kernel_size, kernel_size, kernel_size, xsh[4]*group_dim, n_out*group_dim])

            # Convolve
            # Gaussian dropout on the weights
            WN *= (1 + 0.1*tf.to_float(is_training)*tf.random_normal(WN.get_shape()))

            yN = tf.nn.conv3d(xN, WN, (1,strides,strides,strides,1), padding)
            ysh = yN.get_shape().as_list()
            y = tf.reshape(yN, [self.batch_size, ysh[1], ysh[2], ysh[3], n_out, group_dim])
        return y


    def get_Vrotations(self, x):
        """Rotate the tensor x with all 4 Klein Vierergruppe rotations

        Args:
            x: [h,w,d,n_channels]
        Returns:
            list of 4 rotations of x [[h,w,d,n_channels],....]
        """
        xsh = x.get_shape().as_list()
        angles = [0.,np.pi]
        rx = []
        for i in range(2):
            # 2x 180. rotations about the z axis
            perm = [1,0,2,3]
            y = tf.transpose(x, perm=perm)
            y = tf.contrib.image.rotate(y, angles[i])
            y = tf.transpose(y, perm=perm)

            # 2x 180. rotations about another axis
            for j in range(2):
                perm = [2,1,0,3]
                z = tf.transpose(y, perm=perm)
                z = tf.contrib.image.rotate(z, angles[j])
                z = tf.transpose(z, perm=perm)
                rx.append(z)
        return rx


    def get_cayleytable(self):
        cayley = np.asarray([[0,1,2,3],
                             [1,0,3,2],
                             [2,3,0,1],
                             [3,2,1,0]])
        return cayley


    def V_permutation(self, W):
        """Permute the outputs of the group convolution"""
        Wsh = W.get_shape().as_list()
        cayley = self.get_cayleytable()
        U = []
        for i in range(4):
            perm_mat = self.get_permutation_matrix(cayley, i)
            w = W[:,:,:,:,:,:,i]
            w = tf.transpose(w, [0,1,2,3,5,4])
            w = tf.reshape(w, [-1, 4])
            w = w @ perm_mat
            w = tf.reshape(w, Wsh[:4]+[-1,4])
            U.append(tf.transpose(w, [0,1,2,3,5,4]))
        return U


    def get_permutation_matrix(self, perm, dim):
        ndim = perm.shape[0]
        mat = np.zeros((ndim, ndim))
        for j in range(ndim):
            mat[j,perm[j,dim]] = 1
        return mat



    def get_sph_harm(self, kernel_size, max_degree):
        """Return the spherical harmonics on a Cartesian grid of edge length 
        kernel_size and max degree.

        These harmonics use the quantum mechanical Condon-Shortley phase

        Args:
            kernel_size: int
            max_degree: int
        Returns:
            dict of 4D-tensors of harmonics, radii
        """
        # Compute the radius of the kernel
        r = kernel_size // 2 + 1

        # Compute the spherical polar basis
        lin = np.arange(-r+1,r)
        Y,X,Z = np.meshgrid(lin,lin,lin)
        
        ## Radius of each point
        R = np.sqrt(X**2 + Y**2 + Z**2)
        ## theta: polar angle in [0,2*pi)
        theta = np.arctan2(Y,X)
        ## phi: azimuthal angle in [0,\pi]
        zero_mask = (R==0)
        phi = np.arccos(Z/(R+zero_mask*1e-6))

        # Construct a dict of spherical harmonics. Each entry is structured as
        # {degree: [kernel_size, kernel_size, kernel_size, 2*degree+1]}
        # where degree in Z_{>=0}
        # the 5-tensor is an X,Y,Z Cartesian volume with -1 axis for harmonic
        # order ranging from degree to -degree top to bottom
        harmonics = {}
        for l in range(1,max_degree+1):
            degree_l = []
            for m in range(l,-l-1,-1):
                degree_l.append(sph_harm(m,l,theta,phi))
            degree_l = np.stack(degree_l, -1)
            harmonics[l] = degree_l
        return harmonics, R


    def get_WignerD_matrices(self, alpha, beta, gamma, max_degree):
        """Return the transposed Wigner-D matrices of all degrees up to max_degree.

        Args:
            alpha:
            beta:
            gamma:
            max_degree
        Returns:
            dict of matrices
        """
        WignerD = {}
        for j in range(1,max_degree+1):
            D = np.zeros((2*j+1,2*j+1), dtype=np.complex_)
            for m in range(j, -j-1, -1):
                for n in range(j, -j-1, -1):
                    D[m,n] = Rotation.D(j,m,n,alpha,beta,gamma).doit()
            WignerD[j] = D
        return WignerD


    def get_steerable_filters(self, kernel_size, max_degree, n_radii):
        """Return a set of learnable 3D steerable filters"""
        # Get harmonics and rotation matrices (Wigner-D matrices)
        harmonics, R = self.get_sph_harm(kernel_size, max_degree)
        #WignerD = get_WignerD_matrices(0,0,0,max_degree) 
        
        sigma = 1./np.pi
        radius = int(kernel_size / 2)
        # Mask the center voxel of the non-constant harmonics
        mask = np.ones((kernel_size,kernel_size,kernel_size,1)).astype(np.float32)
        mask[radius,radius,radius,0] = 0

        # For each ring of the patch produce a radial weighting function 
        ## Width of each Gaussian ring is 1./pi
        radials = []
        for ring in np.linspace(kernel_size/(2.*n_radii), kernel_size/2, num=n_radii):
            radials.append(np.exp(-0.5 * (R - ring) * (R - ring) / (sigma * sigma))[...,np.newaxis])

        # The radially-weighted degree >0 harmonics are stored in AC as 
        # [k,k,k,n_radii,2*degree+1,2] tensors
        AC = []
        for k in range(1, max_degree+1):
            block_harmonic = []
            for radial in radials:
                block_harmonic.append(mask*radial*harmonics[k])
            AC.append(np.stack(block_harmonic, -2))

        transformed = np.concatenate(AC, -1)

        # Combine
        basis = np.reshape(transformed, [kernel_size,kernel_size,kernel_size,-1])
        real = np.real(basis)
        imag = np.imag(basis)
        radials = np.concatenate(radials, -1)
        AC = np.concatenate([real, imag], -1)
        return np.concatenate([radials, AC], -1)
