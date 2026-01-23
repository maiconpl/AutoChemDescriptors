#!/usr/bin/python3
'''
Created on December 10, 2025.

@author: maicon & clayton
Last modification by MPL: 22/01/2026.
'''

from sklearn.decomposition import KernelPCA
from qiskit_machine_learning.kernels import FidelityQuantumKernel

class QPCA:

      def __init__(self, n_components, feature_map):

          self.n_components = n_components
          self.feature_map = feature_map

      def fit(self, matrix_inp):

          kernel = FidelityQuantumKernel(feature_map=self.feature_map)
          qmatrix = kernel.evaluate(x_vec=matrix_inp)

          return qmatrix

      def transform(self, matrix_inp):

          # Assuming 'matrix' is the 4x4 Qiskit matrix from your previous step
          # n_components=2 projects your data into a 2D space
          transformer = KernelPCA(n_components=self.n_components, kernel='precomputed')

          # Fit and transform the matrix
          # Note: When using 'precomputed', the input to fit_transform is the kernel matrix
          qkernel = self.fit(matrix_inp)

          print("qkernel:", qkernel)

          data_transformed = transformer.fit_transform(qkernel)

          print("eigenvectors:", transformer.eigenvectors_)
          print("eigenvalues:", transformer.eigenvalues_, type( transformer.eigenvalues_))

          print("Projected Coordinates:")
          print(data_transformed)
          
          tmp01 = 0.0
          explained_variance_ratio = []

          eigenvalues_list_sorted = sorted(transformer.eigenvalues_.tolist(), reverse=True)

          for i in range(self.n_components):
              tmp01 = eigenvalues_list_sorted[i]/sum(eigenvalues_list_sorted)
              explained_variance_ratio.append( tmp01 )
              tmp01 = 0.0

          print("explained_variance_ratio:", explained_variance_ratio)

          return data_transformed, explained_variance_ratio

          #import matplotlib.pyplot as plt
#
#          labels = ["r", "b", 'k']
#
#          plt.figure(figsize=(8, 6))
#          plt.scatter(data_transformed[:, 0], data_transformed[:, 1], cmap='viridis')
#          plt.title("Quantum Kernel PCA")
#          plt.xlabel("Principal Component 1")
#          plt.ylabel("Principal Component 2")
#          plt.grid(True)
#          plt.show()
