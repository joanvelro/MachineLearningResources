import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn import datasets, linear_model
from sklearn.ensemble import RandomForestRegressor

""" Make a regression problem
https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_regression.html#sklearn.datasets.make_regression
"""

X, y, coef = datasets.make_regression(n_samples=1000, n_features=4, n_targets=1,
                                      random_state=0,
                                      coef=True,
                                      noise=0.3,
                                      tail_strength=0.2,
                                      bias=0.1,
                                      n_informative=4)

df = pd.DataFrame({'x1': X[:, 0], 'x2': X[:, 1], 'x3': X[:, 2], 'x4': X[:, 3], 'y': y})

df.plot(subplots=True)

n_estimators_ = [int(x) for x in np.linspace(start=200, stop=2000, num=10)]
"""Number of features to consider at every split"""
max_features_ = ['auto', 'sqrt']
"""Maximum number of levels in tree"""
max_depth_ = [int(x) for x in np.linspace(10, 110, num=11)]
max_depth_.append(None)
"""Minimum number of samples required to split a node"""
min_samples_split_ = [2, 5, 10]
"""Minimum number of samples required at each leaf node"""
min_samples_leaf_ = [1, 2, 4]
""" Method of selecting samples for training each tree"""
bootstrap_ = [True, False]

for i in n_estimators_:
    rf = RandomForestRegressor(n_estimators=i,
                               max_depth=max_depth_[0],
                               max_features=max_features_[0],
                               min_samples_split=min_samples_split_[0],
                               min_samples_leaf=min_samples_leaf_[0],
                               bootstrap=bootstrap_[0],
                               random_state=0)

    score = cross_val_score(rf, X, y, cv=5).mean()
    print(score)
