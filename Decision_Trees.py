# Import Library
# Import other necessary libraries like pandas, numpy...
from sklearn import tree
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris



""" source = https://scikit-learn.org/stable/modules/tree.html
"""

"""CLASIFICATION"""
X = [[0, 0], [1, 1]]
Y = [0, 1]
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X, Y)
x_test = [[2., 2.]]
"""the model can then be used to predict the class of samples"""
print(clf.predict(x_test))
""""the probability of each class can be predicted, which is the fraction of training samples of the same class in a leaf:"""
print(clf.predict_proba(x_test))


clf = tree.DecisionTreeClassifier()
iris = load_iris()
"""iris data contains a matrix with fatures sepal length, sepal width, petal length, petal width """
"""iris.target contains the target classes: setosa (0), versicolor (1), virginica (2)"""
clf = clf.fit(iris.data, iris.target)
x_test = [[iris.data[:,0].mean(), iris.data[:,1].mean(), iris.data[:,2].mean(),iris.data[:,3].mean()],
          [iris.data[:,0].mean() + iris.data[:,0].std() , iris.data[:,1].mean() + iris.data[:,1].std(), iris.data[:,2].mean()  + iris.data[:,2].std(), iris.data[:,3].mean() + iris.data[:,3].std()],
          [iris.data[:,0].mean() - iris.data[:,0].std() , iris.data[:,1].mean() - iris.data[:,1].std(), iris.data[:,2].mean()  - iris.data[:,2].std(), iris.data[:,3].mean() - iris.data[:,3].std()],]

print('new plant:',clf.predict(x_test))


"""REGRESSION"""
X = [[0, 0], [2, 2]]
y = [0.5, 2.5]
clf = tree.DecisionTreeRegressor()
clf = clf.fit(X, y)
x_test = [[1, 1]]
print('new value:',clf.predict(x_test))



