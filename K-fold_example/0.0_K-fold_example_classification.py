"""https://medium.com/datadriveninvestor/k-fold-cross-validation-for-parameter-tuning-75b6cb3214f"""

"""https://github.com/justmarkham/scikit-learn-videos/blob/master/07_cross_validation.ipynb"""

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
import plotly.graph_objects as go
import plotly.io as pio
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

pio.renderers.default = "browser"

"""
Review of model evaluation procedures
Motivation: Need a way to choose between machine learning models
Goal is to estimate likely performance of a model on out-of-sample data
Initial idea: Train and test on the same data
But, maximizing training accuracy rewards overly complex models which overfit the training data
Alternative idea: Train/test split
Split the dataset into two pieces, so that the model can be trained and tested on different data
Testing accuracy is a better estimate than training accuracy of out-of-sample performance
But, it provides a high variance estimate since changing which observations happen to be in the testing set can 
significantly change testing accuracy
"""

if 1 == 1:
    """ read in the iris data"""
    iris = load_iris()

    """create X (features) and y (response)"""
    X = iris.data
    y = iris.target

    """use train/test split with different random_state values"""
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=4)

    """check classification accuracy of KNN with K=5"""
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    print(metrics.accuracy_score(y_test, y_pred))
"""
Question: What if we created a bunch of train/test splits, calculated the testing accuracy for each,
and averaged the results together?
Answer: That's the essense of cross-validation!
"""
"""
Steps for K-fold cross-validation
Split the dataset into K equal partitions (or "folds").
Use fold 1 as the testing set and the union of the other folds as the training set.
Calculate testing accuracy.
Repeat steps 2 and 3 K times, using a different fold as the testing set each time.
Use the average testing accuracy as the estimate of out-of-sample accuracy.
"""
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
Comparing cross-validation to train/test split
Advantages of cross-validation:
*More accurate estimate of out-of-sample accuracy
*More "efficient" use of data (every observation is used for both training and testing)
Advantages of train/test split:
*Runs K times faster than K-fold cross-validation
*Simpler to examine the detailed results of the testing process
"""
"""
Cross-validation recommendations
K can be any number, but K=10 is generally recommended
For classification problems, stratified sampling is recommended for creating the folds
Each response class should be represented with equal proportions in each of the K folds
scikit-learn cross_val_score function does this by default
"""
"""
Cross-validation example: parameter tuning

"""

"""10-fold cross-validation with K=5 for KNN (the n_neighbors parameter)
"""
if 1 == 1:
    knn = KNeighborsClassifier(n_neighbors=5)
    """Evaluate a 'score' by cross-validation"""
    scores = cross_val_score(knn, X, y, cv=10, scoring='accuracy')
    print(scores)

    """use average accuracy as an estimate of out-of-sample accuracy"""
    print(scores.mean())

    """search for an optimal value of K for KNN"""
    k_range = list(range(1, 31))
    k_scores = []
    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k)
        scores = cross_val_score(knn, X, y, cv=10, scoring='accuracy')
        k_scores.append(scores.mean())
    print(k_scores)

"""plot the value of K for KNN (x-axis) versus the cross-validated accuracy (y-axis)"""
if 1 == 1:
    fig = go.Figure(data=go.Scatter(x=k_range, y=k_scores))

    fig.update_layout(title='Average High and Low Temperatures in New York',
                      xaxis_title='Value of K for KNN',
                      yaxis_title='Cross-Validated Accuracy')
    fig.show()

"""Cross-validation example: model selection¶"""
if 1 == 1:
    """10-fold cross-validation with the best KNN model"""
    knn = KNeighborsClassifier(n_neighbors=20)
    print(cross_val_score(knn, X, y, cv=10, scoring='accuracy').mean())

    """10-fold cross-validation with logistic regression"""
    logreg = LogisticRegression()
    print(cross_val_score(logreg, X, y, cv=5, scoring='accuracy').mean())


""" Improvements to cross-validation

* Repeated cross-validation
** Repeat cross-validation multiple times (with different random splits of the data) and average the results
** More reliable estimate of out-of-sample performance by reducing the variance associated with a single trial of 
cross-validation

*Creating a hold-out set
** "Hold out" a portion of the data before beginning the model building process
** Locate the best model using cross-validation on the remaining data, and test it using the hold-out set
** More reliable estimate of out-of-sample performance since hold-out set is truly out-of-sample

* Feature engineering and selection within cross-validation iterations
** Normally, feature engineering and selection occurs before cross-validation
** Instead, perform all feature engineering and selection within each cross-validation iteration
** More reliable estimate of out-of-sample performance since it better mimics the application of the model to
out-of-sample data

"""
