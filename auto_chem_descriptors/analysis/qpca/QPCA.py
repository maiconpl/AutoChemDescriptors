#!/usr/bin/python3
'''
Created on January 22, 2026.

@author: maicon & clayton
Last modification by MPL: 24/01/2026: to try to understand the "precomputed" kernels.
'''

from sklearn.decomposition import KernelPCA
from qiskit_machine_learning.kernels import FidelityQuantumKernel

class QPCA:

      def __init__(self, n_components, feature_map):

          self.n_components = n_components
          self.feature_map = feature_map

      def get_qkernel(self, matrix_inp):

          kernel = FidelityQuantumKernel(feature_map=self.feature_map)
          qmatrix = kernel.evaluate(x_vec=matrix_inp)

          return qmatrix

      def transform(self, matrix_inp):

          # Get kernel matrix
          qkernel = self.get_qkernel(matrix_inp)

          print("qkernel:", qkernel)

          # call KernelPCA object
          #transformer = KernelPCA(n_components=self.n_components, kernel='precomputed')
          #kernel_PCA = KernelPCA(n_components=self.n_components, kernel='precomputed')

          kernel_PCA = KernelPCA(n_components=None, kernel='precomputed')

          # Fit and transform the matrix
          # Note: When using 'precomputed', the input to fit_transform is the kernel matrix

          kernel_PCA.fit(qkernel) # return the Object
          X_qpca = kernel_PCA.transform(qkernel) # return: X_new ndarray of shape (n_samples, n_components)

          #X_qpca = kernel_PCA.fit_transform(qkernel) # return: X_new ndarray of shape (n_samples, n_components)

          print("eigenvectors:", kernel_PCA.eigenvectors_)
          print("eigenvalues:", kernel_PCA.eigenvalues_, type(kernel_PCA.eigenvalues_))

          print("Projected Coordinates:")
          #print(data_transformed)
          print(X_qpca)

          #XXX = eigenvectors[:, valid_idx] * np.sqrt(eigenvalues[valid_idx])
          #import numpy as np
          #XXX = kernel_PCA.eigenvectors_ * np.sqrt( kernel_PCA.eigenvalues_)

          #print("XXX:", XXX)
          
          tmp01 = 0.0
          explained_variance_ratio = []
          eigenvectors = []
          eigenvectors = kernel_PCA.eigenvectors_

          #eigenvalues_list_sorted = sorted(transformer.eigenvalues_.tolist(), reverse=True)
          eigenvalues_list_sorted = sorted(kernel_PCA.eigenvalues_.tolist(), reverse=True)

          for i in range(self.n_components):
              tmp01 = eigenvalues_list_sorted[i]/sum(eigenvalues_list_sorted)
              explained_variance_ratio.append( tmp01 )
              tmp01 = 0.0

          print("explained_variance_ratio:", explained_variance_ratio)

          return X_qpca, explained_variance_ratio, eigenvectors
