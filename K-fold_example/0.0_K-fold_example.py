from sklearn.model_selection import KFold
import pandas as pd
import numpy as np


"""
5-fold cross-validation
"""
if 1 == 1:
    """simulate splitting a dataset of 25 observations into 5 folds"""
    kf = KFold(n_splits=5, shuffle=False).split(range(25))

    """print the contents of each training and testing set"""
    print('{} {:^61} {}'.format('Iteration', 'Training set observations', 'Testing set observations'))
    for iteration, data in enumerate(kf, start=1):
        print('{:^9} {} {:^25}'.format(iteration, data[0], str(data[1])))


"""
5-fold cross-validation with dummy data
"""
if 1 == 1:
    df = pd.DataFrame(data=np.random.random((100, 25)))
    X_ = df.values
    y_ = np.random.uniform(0, 2, 100).round().astype(int)
    """simulate splitting a dataset of 25 observations into 5 folds"""
    kf2 = KFold(n_splits=5, shuffle=False)

    for train_index, test_index in kf2.split(X_):
        print("TRAIN:", train_index, "TEST:", test_index)
        X_train_, X_test_ = X_[train_index], X_[test_index]
        y_train_, y_test_ = y_[train_index], y_[test_index]