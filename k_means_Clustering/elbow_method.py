"""https://stackoverflow.com/questions/19197715/scikit-learn-k-means-elbow-criterion"""

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

"""
K-Means clustering can be evaluated using either Elbow Criterion or Silhouette Coefficient.

Elbow Criterion Method:

The idea behind elbow method is to run k-means clustering on a given dataset for a range of values of
k (num_clusters, e.g k=1 to 10), and for each value of k, calculate Sum of Squared Errors (SSE).

After that, plot a line graph of the SSE for each value of k. If the line graph looks like an arm - a red
circle in below line graph (like angle), the "elbow" on the arm is the value of optimal k (number of
cluster). Here, we want to minimize SSE. SSE tends to decrease toward 0 as we increase k (and SSE is 0 when
is equal to the number of data points in the dataset, because then each data point is its own cluster, and
there is no error between it and the center of its cluster).

So the goal is to choose a small value of k that still has a low SSE, and the elbow usually represents where
we start to have diminishing returns by increasing k.

"""


iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris['feature_names'])
# print(X)
data = X[['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)']]

sse = {}
for k in range(1, 10):
    kmeans = KMeans(n_clusters=k, max_iter=1000).fit(data)
    data["clusters"] = kmeans.labels_
    # print(data["clusters"])
    sse[k] = kmeans.inertia_  # Inertia: Sum of distances of samples to their closest cluster center
plt.figure()
plt.plot(list(sse.keys()), list(sse.values()))
plt.xlabel("Number of cluster")
plt.ylabel("SSE")
plt.show()




"""
Silhouette Coefficient Method:

A higher Silhouette Coefficient score relates to a model with better-defined clusters. The Silhouette
Coefficient is defined for each sample and is composed of two scores: `

a: The mean distance between a sample and all other points in the same class.

b: The mean distance between a sample and all other points in the next nearest cluster.
The Silhouette Coefficient is for a single sample is then given as:

s=b-a/max(a,b)

Now, to find the optimal value of k for KMeans, loop through 1..n for n_clusters in KMeans and calculate
Silhouette Coefficient for each sample.

A higher Silhouette Coefficient indicates that the object is well matched to its own cluster and poorly
matched to neighboring clusters.
"""
from sklearn.metrics import silhouette_score

X = load_iris().data
y = load_iris().target

for n_cluster in range(2, 11):
    kmeans = KMeans(n_clusters=n_cluster).fit(X)
    label = kmeans.labels_
    sil_coeff = silhouette_score(X, label, metric='euclidean')
    print("For n_clusters={}, The Silhouette Coefficient is {}".format(n_cluster, sil_coeff))

"""
As we can see, n_clusters=2 has highest Silhouette Coefficient. This means that 2 should be the optimal
number of cluster, Right?

But here's the catch.

Iris dataset has 3 species of flower, which contradicts the 2 as an optimal number of cluster. So despite
n_clusters=2 having highest Silhouette Coefficient, We would consider n_clusters=3 as optimal number of
cluster due to -

Iris dataset has 3 species. (Most Important)
n_clusters=2 has a 2nd highest value of Silhouette Coefficient.
So choosing n_clusters=3 is the optimal no. of cluster for iris dataset.

Choosing optimal no. of the cluster will depend on the type of datasets and the problem we are trying to
solve. But most of the cases, taking highest Silhouette Coefficient will yield an optimal number of cluster.
"""
