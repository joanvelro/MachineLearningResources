import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_multilabel_classification
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import CCA


def plot_hyperplane(clf, min_x, max_x, linestyle, label):
    # get the separating hyperplane
    w = clf.coef_[0]
    a = -w[0] / w[1]
    xx = np.linspace(min_x - 5, max_x + 5)  # make sure the line is long enough
    yy = a * xx - (clf.intercept_[0]) / w[1]
    plt.plot(xx, yy, linestyle, label=label)


def plot_subfigure(X, Y, subplot, title, transform):
    if transform == "pca":
        X = PCA(n_components=2).fit_transform(X)
    elif transform == "cca":
        X = CCA(n_components=2).fit(X, Y).transform(X)
    else:
        raise ValueError

    min_x = np.min(X[:, 0])
    max_x = np.max(X[:, 0])

    min_y = np.min(X[:, 1])
    max_y = np.max(X[:, 1])

    classif = OneVsRestClassifier(SVC(kernel='linear'))
    classif.fit(X, Y)

    plt.subplot(2, 2, subplot)
    plt.title(title)

    zero_class = np.where(Y[:, 0])
    one_class = np.where(Y[:, 1])
    two_class = np.where(Y[:, 1])
    plt.scatter(X[:, 0], X[:, 1], s=40, c='gray', edgecolors=(0, 0, 0))

    plt.scatter(X[zero_class, 0], X[zero_class, 1], s=160, edgecolors='blue',
                facecolors='none', linewidths=2, label='Class 1')

    plt.scatter(X[one_class, 0], X[one_class, 1], s=80, edgecolors='orange',
                facecolors='none', linewidths=2, label='Class 2')

    plt.scatter(X[one_class, 0], X[two_class, 1], s=80, edgecolors='red',
                facecolors='none', linewidths=2, label='Class 3')

    plot_hyperplane(classif.estimators_[0], min_x, max_x, 'k--',
                    'Boundary\nfor class 1')
    plot_hyperplane(classif.estimators_[1], min_x, max_x, 'k-.',
                    'Boundary\nfor class 2')
    plt.xticks(())
    plt.yticks(())

    plt.xlim(min_x - .5 * max_x, max_x + .5 * max_x)
    plt.ylim(min_y - .5 * max_y, max_y + .5 * max_y)
    if subplot == 2:
        plt.xlabel('First principal component')
        plt.ylabel('Second principal component')
        plt.legend(loc="upper left")


if 0==1:
    plt.figure(figsize=(8, 6))

    X1, Y1 = make_multilabel_classification(n_classes=2, n_labels=1,
                                          allow_unlabeled=True,
                                          random_state=1)

    plot_subfigure(X1, Y1, 1, "With unlabeled samples + CCA", "cca")
    plot_subfigure(X1, Y1, 2, "With unlabeled samples + PCA", "pca")

    X2, Y2 = make_multilabel_classification(n_classes=3, n_labels=1,
                                          allow_unlabeled=False,
                                          random_state=1)

    plot_subfigure(X2, Y2, 3, "Without unlabeled samples + CCA", "cca")
    plot_subfigure(X2, Y2, 4, "Without unlabeled samples + PCA", "pca")

    plt.subplots_adjust(.04, .02, .97, .94, .09, .2)
    plt.show()




# Generate a random multilabel classification problem.
X, Y3 = make_multilabel_classification(n_samples=100, n_classes=2, n_labels=1,
                                      allow_unlabeled=False,
                                   random_state=1)
for i in range(0, len(Y3)):
    if sum(Y3[i, :])>1:
        Y3[i,0]=1
        Y3[i,1]=0

X3 = PCA(n_components=2).fit_transform(X)

min_x = np.min(X3[:, 0])
max_x = np.max(X3[:, 0])

min_y = np.min(X3[:, 1])
max_y = np.max(X3[:, 1])

classif = OneVsRestClassifier(SVC(kernel='linear')) #clustering
classif.fit(X3, Y3)

class1 = np.where(Y3[:, 0]) # tuples
class2 = np.where(Y3[:, 1])

plt.figure(num=None, figsize=(8,6), dpi=120, facecolor='w', edgecolor='k')
plt.scatter(X3[:, 0], X3[:, 1], s=40, c='gray', edgecolors=(0, 0, 0))
plt.scatter(X3[class1, 0], X3[class1, 1], s=160, edgecolors='red',
            facecolors='none', linewidths=2, label='Class 1')
plt.scatter(X3[class2, 0], X3[class2, 1], s=80, edgecolors='black',
            facecolors='none', linewidths=2, label='Class 2')

plot_hyperplane(classif.estimators_[0], min_x, max_x, 'k--',
                    'Boundary\nfor class 1')
plot_hyperplane(classif.estimators_[1], min_x, max_x, 'k-.',
                    'Boundary\nfor class 2')

plt.tick_params(axis='both', labelsize=16)
#plt.xticks(())
#plt.yticks(())

plt.xlim(min_x - .5 * max_x, max_x + .5 * max_x)
plt.ylim(min_y - .5 * max_y, max_y + .5 * max_y)