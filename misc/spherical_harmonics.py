import os
import sys
import time

import numpy as np

import scipy as sp

from scipy.special import sph_harm
from sympy.physics.quantum.spin import Rotation

import importlib
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class HarmonicExpansion(object):
    def __init__(self, kernel_size, max_degree):
        self.kernel_size = kernel_size
        self.max_degree = max_degree
        self.radius = kernel_size // 2
        """
        self.euler_angles = [(0.,       0.,     0.),
                             (np.pi,    0.,     0.),
                             (0.,       np.pi,  0.),
                             (np.pi,    np.pi,  0.)]
        """
        self.euler_angles = []
        n = 12
        for i in range(n):
            self.euler_angles.append((2.*np.pi*i / n, 0., 0.))
        self.n_rot = len(self.euler_angles)
        

    def get_sph_harm(self):
        """Return the spherical harmonics on a Cartesian grid of edge length 
        kernel_size and max degree.

        These harmonics use the quantum mechanical Condon-Shortley phase

        Args:
            kernel_size: int
            max_degree: int
        Returns:
            dict of 4D-tensors of harmonics, radii
        """
        # Compute the spherical polar basis
        azi = np.linspace(0., np.pi, num=self.kernel_size)
        pol = np.linspace(0., 2.*np.pi, num=2*self.kernel_size)
        theta, phi = np.meshgrid(pol, azi)
        
        ## theta: polar angle in [0,2*pi)
        ## phi: azimuthal angle in [0,\pi]

        # Construct a dict of spherical harmonics. Each entry is structured as
        # {degree: [kernel_size, kernel_size, kernel_size, 2*degree+1]}
        # where degree in Z_{>=0}
        # the 5-tensor is an X,Y,Z Cartesian volume with -1 axis for harmonic
        # order ranging from degree to -degree top to bottom
        harmonics = {}
        for l in range(1,self.max_degree+1):
            degree_l = []
            for m in range(l,-l-1,-1):
                degree_l.append((sph_harm(m,l,theta,phi)))
            degree_l = np.stack(degree_l, -1)
            harmonics[l] = degree_l
        return harmonics


    def get_WignerD_matrices(self, alpha, beta, gamma, max_degree):
        """Return the transposed Wigner-D matrices of all degrees up to max_degree.

        The convention is z-y-z Euler angles, passive

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
                    m_idx = j - m
                    n_idx = j - n
                    print(m_idx, n_idx)
                    D[m_idx,n_idx] = Rotation.D(j,m,n,alpha,beta,gamma).doit()
            WignerD[j] = D.T.conj()
        return WignerD


    def get_rotation_matrices(self):
        """Return the WignerD matrices for each rotation"""
        filename = "./WignerD.npy"
        if os.path.exists(filename):
            print("Loading pre-computed Wigner-D matrices from {}".format(filename))
            WignerD = np.load(filename)
        else:
            WignerD = []
            for angles in self.euler_angles:
                print("Computing WignerD matrix for: {}".format(angles))
                WignerD.append(self.get_WignerD_matrices(*angles, self.max_degree))
            np.save(filename, WignerD)
            print("Saving pre-computed Wigner-D matrices to {}".format(filename))
        return WignerD


    def get_radials(self, R):
        """Return radially-weighted profiles"""
        sigma = 1./np.pi
        # TODO: remove
        sigma *= 5
        radius = int(self.kernel_size / 2)
        # Mask the center voxel of the non-constant harmonics
        mask = np.ones(list(R.shape) +[1,]).astype(np.float32)
        mask[radius,radius,radius,0] = 0

        # For each ring of the patch produce a radial weighting function 
        ## Width of each Gaussian ring is 1./pi
        radials = []
        for ring in np.linspace(radius/self.n_radii, radius, num=self.n_radii):
            radials.append(np.exp(-0.5*(R-ring)*(R-ring) / (sigma*sigma))[...,np.newaxis])
        return radials, mask


    def get_steerable_filters(self):
        """Return a set of learnable 3D steerable filters"""
        # Get harmonics and rotation matrices (Wigner-D matrices)
        harmonics = self.get_sph_harm()

        """
        plt.figure()
        plt.ion()
        plt.show()

        for l in range(1,self.max_degree+1):
            for m in range(2*l+1):
                harm = harmonics[l][...,m]
                plt.imshow(np.real(harm), interpolation='nearest')
                plt.draw()
                input()
        """


        # The radially-weighted degree >0 harmonics are stored in AC as 
        # [theta,phi,2*degree+1] tensors with dimensions: height, width,
        # depth, radius, order, complex
        AC = harmonics
        print(AC[1])

        # Rotate the basis
        WignerD = self.get_rotation_matrices()
        print(len(WignerD))
        for WD in WignerD:
            print(np.angle(WD[1]))

        # Get rotations of basis
        basis = []
        for rot in WignerD:
            mat = []
            for j in range(1, self.max_degree+1):
                ac = np.reshape(AC[j], [-1,2*j+1])
                ac = np.reshape(ac @ rot[j], AC[j].shape)
                mat.append(ac)
            basis.append(np.concatenate(mat, -1))
        #basis = np.stack([np.real(basis), np.imag(basis)], -1)
        """
        basis = []
        for l in range(1,self.max_degree+1):
            basis.append(AC[l])
        basis = np.concatenate(basis, -1)
        """

        # Combine
        # TODO: combine the DC component
        weights = np.random.randn(basis[0].shape[-1],1) + np.random.randn(basis[0].shape[-1],1)*1j
        #weights = np.concatenate([np.zeros((4,1)), np.ones((1,1)), np.zeros((1,1))], 0)
        #weights = np.asarray([1,0,1,0,1,0])[:,np.newaxis]
        #weights = np.ones((3,1)) + np.zeros((3,1))*1j
        steerables = []
        for base in basis:
            steerable = np.real(base @ weights)
            steerables.append(steerable[:,:,0])
        
        amin = np.amin(steerables)
        amax = np.amax(steerables)

        # Display
        """
        plt.figure(figsize=(6,12))
        for i, steerable in enumerate(steerables):
            plt.subplot(len(steerables),1,i+1)
            plt.imshow(steerable, interpolation='nearest')
        plt.show()
        """
        
        azi = np.linspace(0., np.pi, num=self.kernel_size)[:,np.newaxis]
        pol = np.linspace(0., 2.*np.pi, num=2*self.kernel_size)[:,np.newaxis]

        x = np.cos(pol)*np.sin(azi).T
        y = np.sin(pol)*np.sin(azi).T
        z = np.ones((2*self.kernel_size,1))*np.cos(azi).T

        fig = plt.figure()
        for i, steerable in enumerate(steerables):
            plt.subplot(1,len(steerables),i+1, projection='3d') 

            rgba = steerable.T
            rgba -= np.amin(rgba)
            rgba /= np.amax(rgba)
            print(rgba.shape)
            print(x.shape)

            ax = fig.gca(projection='3d')
            ax.plot_surface(x, y, z, facecolors=plt.cm.jet(rgba), rstride=1, cstride=1, linewidth=0)
            plt.axis("square")
        plt.show()


"""
- Need the spherical harmonics and the wigner d matrices to steer them
- Need the radial gaussian function
- Need the real R^3 projection to combine these filters
"""


if __name__ == "__main__":
    sf = HarmonicExpansion(21, 1)
    sf.get_steerable_filters()

















































