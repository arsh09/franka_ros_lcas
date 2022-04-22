from __future__ import division
import numpy as np
import numpy.matlib as mat
import tensorflow as tf
# from skimage.io import imread
# from skimage.transform import resize
import matplotlib.pyplot as plt
from tensorflow.keras import backend as K
from natsort import natsorted
import os, json, csv, math
import pandas as pd
from glob import glob
import matplotlib.patches as mpatches

JOINT_NUM = 7
JOINT_WEIGHTS = 8
CARTESIAN_NUM = 3
CARTESIAN_WEIGHTS = 3
 

class ProMP:
    """
    ProMP class
        N = number of basis functions
        h = bandwidth of basis functions
        dt = time step
        covY = variance of original p(Y)
        Wm = mean of weights [Nx1]
        covW = variance of weights [NxN]
    internal:
        Phi = basis functions evaluated for every step [TxN]
    methods:
        printMP (plots an MP showing a standar deviation above and below)
            name = title of the plot
    """
    def __init__(self, n=8, h=0.07, f=1, dt=1/(150-1)):
        self.n = n
        self.h = h
        self.dt = dt
        self.f = f

    def basis_func_gauss_glb(self): #[Txn] this is the basis function used globally for loss function
        """
        Evaluates Gaussian basis functions in [0,1]
        n = number of basis functions
        h = bandwidth
        dt = time step
        f = modulation factor
        """
        # phi = ProMP().basis_func_gauss_glb()  Matrix of basis functions
        tf_ = 1/self.f;
        T = int(round(tf_/self.dt+1))
        F = np.zeros((T,self.n))
        for z in range(0,T):
            t = z*self.dt
            q = np.zeros((1, self.n))
            for k in range(1,self.n+1):
                c = (k-1)/(self.n-1)
                q[0,k-1] = np.exp(-(self.f*t - c)*(self.f*t - c)/(2*self.h))
            F[z,:self.n] = q[0, :self.n]
        F = tf.cast(F, tf.float32)

        # Normalize basis functions
        F = F/np.transpose(mat.repmat(np.sum(F,axis=1),self.n,1)); #[TxN]
        return F #[Txn]
 