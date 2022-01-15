import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import neural_network
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

# https://www.kaggle.com/hhllcks/neural-net-with-gridsearch
testset = pd.read_csv("test_nn_kaggle.csv")
trainset = pd.read_csv("train_nn_kaggle.csv")

sns.set()
sns.pairplot(trainset[["bone_length", "rotting_flesh", "hair_length", "has_soul", "type"]], hue="type")

trainset['hair_soul'] = trainset.apply(lambda row: row['hair_length']*row['has_soul'],axis=1)
testset['hair_soul'] = testset.apply(lambda row: row['hair_length']*row['has_soul'],axis=1)

x = trainset[["bone_length", "rotting_flesh", "hair_length", "has_soul"]]
x_hair_soul = trainset[["bone_length", "rotting_flesh", "hair_length", "has_soul", "hair_soul"]]

x_test = testset[["bone_length", "rotting_flesh", "hair_length", "has_soul"]]
x_test_hair_soul = testset[["bone_length", "rotting_flesh", "hair_length", "has_soul", "hair_soul"]]

y = trainset[["type"]]

if 1==1:
    parameters = {'solver': ['sgd'],
                  'max_iter': [1000],
                  'alpha': [0.0001, 0.001, 0.01, 0.1, 1] ,
                  'hidden_layer_sizes':[5, 6, 7, 8, 9],}
    clf_grid = GridSearchCV(neural_network.MLPClassifier(), cv = 4, param_grid=parameters  , n_jobs=-1)
    clf_grid.fit(x,y.values.ravel())

if 1==1:

    print("Best score: %0.4f" % clf_grid.best_score_)
    print("Using the following parameters:")
    print(clf_grid.best_params_)


    clf = neural_network.MLPClassifier(alpha=0.1, hidden_layer_sizes=(6), max_iter=500, random_state=3, solver='lbfgs')
    clf.fit(x, y.values.ravel())
    preds = clf.predict(x_test)