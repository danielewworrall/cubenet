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
    def __init__(self, kernel_size, max_degree, n_radii):
        self.kernel_size = kernel_size
        self.max_degree = max_degree
        self.radius = kernel_size // 2
        self.n_radii = n_radii
        self.euler_angles = [(-np.pi/2,    -np.pi/2,    np.pi/2.),
                             (0.,          -np.pi/2.,   0.),
                             (-np.pi/2.,   0.,          0.)]
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
        lin = np.arange(-self.radius,self.radius+1)
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
        for l in range(self.max_degree+1):
            degree_l = []
            for m in range(l,-l-1,-1):
                degree_l.append(sph_harm(m,l,theta,phi))
            degree_l = np.stack(degree_l, -1)
            harmonics[l] = degree_l
        return harmonics, R


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
        for j in range(max_degree+1):
            D = np.zeros((2*j+1,2*j+1), dtype=np.complex_)
            for m in range(j, -j-1, -1):
                for n in range(j, -j-1, -1):
                    m_idx = j - m
                    n_idx = j - n
                    D[m_idx,n_idx] = Rotation.D(j,m,n,alpha,beta,gamma).doit()
            WignerD[j] = D.conj()
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
        sigma *= 3
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


    def get_S4(self, WignerD):
        """Return all rotations of the basis"""
        # 1) Build all WignerD matrices
        new_WD = [{} for __ in range(24)]
        for j in range(self.max_degree+1):
            Xrot = WignerD[0][j]
            Yrot = WignerD[1][j]
            Zrot = WignerD[2][j]
            counter = 0
            # Z4 rotations about the Y axis
            for i in range(4):
                W = np.eye(2*j+1)
                for __ in range(i):
                    W = Yrot @ W
                # Rotations in the quotient space (sphere S^2)
                # i) Z4 rotations about the Z axis
                for k in range(4):
                    U = W.copy()
                    for __ in range(k):
                        U = Zrot @ U
                    new_WD[counter][j] = U
                    counter += 1
                # ii) X pole rotations
                new_WD[counter][j] = Xrot @ W
                counter += 1
                new_WD[counter][j] = Xrot.T.conj() @ W
                counter += 1

        return new_WD


    def get_T4(self, WignerD):
        """Return all rotations of the basis"""
        def rotZ3(rot, W)
            Us = []
            for i in range(3):
                U = W.copy()
                for __ in range(i):
                    U = rot @ U
                Us.append(U)
            return Us


        # 1) Build all WignerD matrices
        new_WD = [{} for __ in range(12)]
        for j in range(self.max_degree+1):
            Xrot = WignerD[0][j]
            Yrot = WignerD[1][j]
            Zrot = WignerD[2][j]
            counter = 0

            # a) do nothing
            W = np.eye(2*j+1)
            mats = rotZ3(Zrot.T.conj() @ Xrot.T.conj(), W)
            for mat in mats:
                new_WD[counter][j] = mat
                counter += 1
            # b) swap 1
            W = np.eye(2*j+1)
            

        return new_WD


    def apply_rotation_to_basis(self, WignerD, AC):
        """Returns [k,k,k,n_rot,n_radii,sph_harm]"""
        # 2) Apply each WignerD matrix to the spherical harmonic basis
        basis = []
        for rot in WignerD:
            mat = []
            for j in range(self.max_degree+1):
                mat.append(AC[j] @ rot[j])
            basis.append(np.concatenate(mat, -1))
        basis = np.stack(basis, -3)
        basis = np.stack([np.real(basis), np.imag(basis)], -1)
        return np.reshape(basis, list(basis.shape[:4])+[-1,])
        

    def get_steerable_filters(self):
        """Return a set of learnable 3D steerable filters"""
        # Get harmonics and rotation matrices (Wigner-D matrices)
        harmonics, R = self.get_sph_harm()
        radial_profiles, mask = self.get_radials(R)

        # The radially-weighted degree >0 harmonics are stored in AC as 
        # [k,k,k,n_radii,2*degree+1,2] tensors with dimensions: height, width,
        # depth, radius, order, complex
        AC = {}
        for j in range(self.max_degree+1):
            block_harmonic = []
            for radial in radial_profiles:
                block_harmonic.append(mask*radial*harmonics[j])
            AC[j] = np.stack(block_harmonic, -2)

        # Rotate the basis
        WignerD = self.get_rotation_matrices()
        WignerD = self.get_T4(WignerD)
        basis = self.apply_rotation_to_basis(WignerD, AC)
        #return basis
        
        # Combine
        weights = np.random.randn(basis.shape[-1],1)
        
        # Note that we use a right-multiplication convention!
        steerables = np.real(basis @ weights)

        
        # Display
        fig = plt.figure(1)

        for i in range(12):
            plt.subplot(4,3,i+1,projection='3d')
            voxels = steerables[:,:,:,i,0]
            voxels -= np.amin(voxels)
            voxels /= np.amax(voxels)    

            filled = R < self.radius
            alpha = 0.99*filled
            colors = np.stack([voxels, voxels, voxels], -1) #, alpha], -1)
            
            #plt.cla()
            ax = fig.gca(projection='3d')
            ax.voxels(filled, facecolors=colors) #, edgecolor='k')
            
            plt.axis("square")
        plt.show()


"""
- Need the spherical harmonics and the wigner d matrices to steer them
- Need the radial gaussian function
- Need the real R^3 projection to combine these filters
"""


if __name__ == "__main__":
    sf = HarmonicExpansion(5, 2, 1)
    sf.get_steerable_filters()

















































