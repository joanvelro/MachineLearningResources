#!/usr/bin/env
"""
    Create correlated variables from non-correlated variables with
    Cholesky decomposition

    (C) Jose Angel Velasco (joseangel.velasco@yahoo.es
    Nov. 2020
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="ticks")
from scipy.linalg import cholesky
from scipy.stats import pearsonr
from matplotlib import rc

rc('font', **{'family': 'serif', 'serif': ['Times New Roman']})
rc('text', usetex=True)
np.random.seed(18031991)  # set random number generator

p = 4
n = 1000
z = np.random.normal(0, 1, (n, p))  # Phase C unbalance level
df = pd.DataFrame(z)
sns.pairplot(df)
plt.savefig('pairplot.png')
plt.close()

# Create desired covariance matrix
S = np.diag(np.ones(p))

S[0, 1] = np.random.uniform(0.5, 0.8)
S[0, 2] = np.random.uniform(0.5, 0.8)
S[0, 3] = np.random.uniform(0.5, 0.8)
S[1, 2] = np.random.uniform(0.5, 0.8)
S[1, 3] = np.random.uniform(0.5, 0.8)
S[2, 3] = np.random.uniform(0.5, 0.8)

S[1, 0] = S[0, 1]
S[2, 0] = S[0, 2]
S[3, 0] = S[0, 3]
S[2, 1] = S[1, 2]
S[3, 1] = S[1, 3]
S[2, 3] = S[3, 2]

# Cholesky Decomposition
L = np.zeros((S.shape[0], S.shape[0]))
for i in np.arange(0, S.shape[0]):
    for j in np.arange(0, S.shape[0]):
        if i == 0 and j == 0:
            L[i, j] = np.sqrt(S[i, j])
        if j == 0 and i > j:
            L[i, j] = S[i, j] / L[0, 0]
        if 0 < j < i:
            L[i, j] = (S[i, j] - sum(L[i, k] * L[j, k] for k in np.arange(0, j))) * (1 / L[j, j])
        if i == j and i > 0:
            L[i, j] = np.sqrt(S[i, j] - sum(L[i, k] ** 2 for k in np.arange(0, i)))

y = np.dot(L, z.T).T
df2 = pd.DataFrame(y)
sns.pairplot(df2)
plt.savefig('pairplot_correlated.png')
plt.close()

if 1 == 1:
    # Correlation matrix
    corr_mat = np.array([[1.0, 0.6, 0.4, 0.3],
                         [0.6, 1.0, 0.5, 0.5],
                         [0.4, 0.5, 1.0, 0.5],
                         [0.3, 0.5, 0.5, 1.0]])

    # Compute the (upper) Cholesky decomposition matrix
    Lu = cholesky(corr_mat)

    # Generate 3 series of normally distributed (Gaussian) numbers
    x = np.random.normal(0.0, 1.0, size=(n, p))

    # Finally, compute the inner product of upper_chol and rnd
    y = x @ Lu
    y2 = np.dot(x, Lu)
    df3 = pd.DataFrame(y2)
    sns.pairplot(df3)
    plt.savefig('pairplot_correlated2.png')
    plt.close()
