#!/usr/bin/env
"""
    Eigenvalues analysis

    (C) Jose Angel Velasco (joseangel.velasco@yahoo.es
    Nov. 2020
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing

np.random.seed(1)

A = np.random.normal(1,3,(1000,10))

# create a customised covariance matrix
S = np.random.uniform(0,1,(10,10))
for i in np.arange(0,9):
    for j in np.arange(0, 9):
        if j>i:
            S[j,i] = S[i,j]
        if j==i:
            S[i,j] = 1

# cholesky decomposition
L = np.tril(np.sqrt(S))

# correlated variables
y = np.transpose(np.dot(L,A.T))

# standarise the data
min_max_scaler = preprocessing.MinMaxScaler()
ys = min_max_scaler.fit_transform(y)

# covariance matrix
S2 = np.cov(y.T)

# obtain eigenvalues and eigenvectors of the correlation matrix
w,v = np.linalg.eig(S2)

# project the the new space
Y = np.dot(y,v)

# Plot first two components
plt.figure()
plt.plot(Y[:,0], Y[:,1],'.')
plt.show()