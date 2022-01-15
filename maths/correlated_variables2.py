#!/usr/bin/env
""" Example of generating correlated normally distributed random samples."

    source: https://scipy-cookbook.readthedocs.io/items/CorrelatedRandomSamples.html

    (C) Jose Angel Velasco (joseangel.velasco@yahoo.es
    Nov. 2020
"""


import numpy as np
from scipy.linalg import eigh, cholesky
from scipy.stats import norm
import pandas as pd
from pylab import plot, show, axis, subplot, xlabel, ylabel, grid
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix


# Choice of cholesky or eigenvector method.
#method = 'cholesky'
method = 'eigenvectors'

n = 1500 # samples

# The desired covariance matrix.
S = np.array([
        [  3.40, -2.75, -2.00, 3.00, 4.00],
        [ -2.75,  5.50,  1.50, 3.00, 4.00],
        [ -2.00,  1.50,  1.25, 3.00, 4.00],
        [  3.00,  3.00,  3.00, 3.00, 4.00],
        [  3.00,  3.00,  3.00, 3.00, 4.00],
    ])

m, m = S.shape # features
# Generate samples from three independent normally distributed random
# variables (with mean 0 and std. dev. 1).
x = np.random.normal(0,1,(n,m))
#x = norm.rvs(size=(n, m))

# We need a matrix `c` for which `L*L^T = S`.  We can use, for example,
# the Cholesky decomposition, or the we can construct `c` from the
# eigenvectors and eigenvalues.

if method == 'cholesky':
    # Compute the Cholesky decomposition.
    L = cholesky(S, lower=True)
else:
    # Compute the eigenvalues and eigenvectors.
    evals, evecs = eigh(S)
    # Construct c, so S = L*L^T
    L = np.dot(evecs, np.diag(np.sqrt(evals)))

# Convert the data to correlated random variables.
y = np.dot(L, x.T)



# Plot various projections of the samples.
if 0==1:
    subplot(2,2,1)
    plot(y[0], y[1], 'b.')
    ylabel('y[1]')
    axis('equal')
    grid(True)

    subplot(2,2,3)
    plot(y[0], y[2], 'b.')
    xlabel('y[0]')
    ylabel('y[df2]')
    axis('equal')
    grid(True)

    subplot(2,2,4)
    plot(y[1], y[2], 'b.')
    xlabel('y[1]')
    axis('equal')
    grid(True)

    show()



# multivariate Scatter plot (uncorrelated variables)
if 1==1:
    df_x = pd.DataFrame(x)
    fig = plt.figure(num=1, figsize=(8, 8), dpi=80, facecolor='w', edgecolor='k')
    scatter_matrix(df_x)
    plt.show()
    #plt.close()

# multivariate Scatter plot  (correlated variables)
if 1==1:
    df_y = pd.DataFrame(y.T)
    # multivariate Scatter plo
    fig = plt.figure(num=1, figsize=(8, 8), dpi=80, facecolor='w', edgecolor='k')
    scatter_matrix(df_y)
    plt.show()
    #plt.close()
