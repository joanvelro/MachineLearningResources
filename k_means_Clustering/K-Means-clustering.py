

from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale

from time import time
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-paper')
plt.rcParams.update({'font.size': 12})
from matplotlib import rc
rc('font',**{'family':'serif','serif':['Times New Roman']})
rc('text', usetex=True)

np.random.seed(18031991) # set random number generator

digits = load_digits()
data = scale(digits.data)

n_samples, n_features = data.shape
n_digits = len(np.unique(digits.target))
labels = digits.target
sample_size = 300






if 1==1:
    print("n_digits: %d, \t n_samples %d, \t n_features %d"
          % (n_digits, n_samples, n_features))

    print(82 * '_')
    print('init\t\ttime\tinertia\thomo\tcompl\tv-meas\tARI\tAMI\tsilhouette')

    def bench_k_means(estimator, name, data):
        t0 = time()
        estimator.fit(data)
        print('%-9s\t%.2fs\t%i\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f'
              % (name, (time() - t0), estimator.inertia_,
                 metrics.homogeneity_score(labels, estimator.labels_),
                 metrics.completeness_score(labels, estimator.labels_),
                 metrics.v_measure_score(labels, estimator.labels_),
                 metrics.adjusted_rand_score(labels, estimator.labels_),
                 metrics.adjusted_mutual_info_score(labels, estimator.labels_, average_method='arithmetic'),
                 metrics.silhouette_score(data, estimator.labels_,
                                          metric='euclidean',
                                          sample_size=sample_size)))



    bench_k_means(KMeans(init='k-means++', n_clusters=n_digits, n_init=10),
                  name="k-means++", data=data)

    bench_k_means(KMeans(init='random', n_clusters=n_digits, n_init=10),
                  name="random", data=data)

    # in this case the seeding of the centers is deterministic, hence we run the
    # kmeans algorithm only once with n_init=1
    pca = PCA(n_components=n_digits).fit(data)
    bench_k_means(KMeans(init=pca.components_, n_clusters=n_digits, n_init=1),
                  name="PCA-based",
                  data=data)
    print(82 * '_')





if 1==1:

    # #############################################################################
    # Visualize the results on PCA-reduced data

    reduced_data = PCA(n_components=2).fit_transform(data)
    kmeans = KMeans(init='k-means++', n_clusters=n_digits, n_init=10)
    kmeans.fit(reduced_data)

    # Step size of the mesh. Decrease to increase the quality of the VQ.
    h = .02     # point in the mesh [x_min, x_max]x[y_min, y_max].

    # Plot the decision boundary. For that, we will assign a color to each
    x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
    y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Obtain labels for each point in mesh. Use last trained model.
    Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)

    plt.figure(1)
    plt.clf()
    plt.imshow(Z, interpolation='nearest',
               extent=(xx.min(), xx.max(), yy.min(), yy.max()),
               cmap=plt.cm.cmaps_listed['twilight'],
               aspect='auto', origin='lower')

    plt.plot(reduced_data[:, 0], reduced_data[:, 1], 'k.', markersize=2)
    # Plot the centroids as a white X
    centroids = kmeans.cluster_centers_
    plt.scatter(centroids[:, 0], centroids[:, 1],
                marker='x', s=169, linewidths=3,
                color='black', zorder=10)
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xticks(())
    plt.yticks(())
    plt.show()
    plt.savefig('clustering1.pdf')
    plt.close()


if 1==1:
    num_samples = 100
    num_classes = 4
    param1 = np.random.exponential(scale=0.2, size=num_samples) # Relative lateral branch length
    param2 = np.random.beta(a=1, b=0.4, size=num_samples) # Relative lateral branch length
    param3 = np.random.normal(0.3, 0.02, size=num_samples) # Weight of the lateral branch
    param4 = np.random.logistic(loc=0.3, scale=0.06, size=num_samples) # Weight of the lateral branch
    data =  np.vstack((param1, param2, param3, param4)).T
    Y = np.zeros((num_samples, num_classes))
    for i in range(0,len(Y)):
        f = np.random.randint(0,3)
        Y[i,f]=1


    # #############################################################################
    # Visualize the results on PCA-reduced data

    reduced_data = PCA(n_components=2).fit_transform(data)
    kmeans = KMeans(init='k-means++', n_clusters=num_classes, n_init=1)
    kmeans.fit(reduced_data)

    # Step size of the mesh. Decrease to increase the quality of the VQ.
    h = .02     # point in the mesh [x_min, x_max]x[y_min, y_max].

    # Plot the decision boundary. For that, we will assign a color to each
    offset = 0.3
    x_min, x_max = reduced_data[:, 0].min() - offset, reduced_data[:, 0].max() + offset
    y_min, y_max = reduced_data[:, 1].min() - offset, reduced_data[:, 1].max() + offset
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Obtain labels for each point in mesh. Use last trained model.
    Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)

    plt.figure(num=None, figsize=(8,6), dpi=120, facecolor='w', edgecolor='k')
    plt.clf()
    plt.plot(reduced_data[:, 0], reduced_data[:, 1], 'k.', markersize=6, marker='o')
    plt.legend(('Data',), fontsize=18)
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    #plt.xticks(())
    #plt.yticks(())
    plt.tick_params(axis='both', labelsize=18)
    plt.show()
    plt.ylabel('$Y_F$', fontsize=18)
    plt.xlabel('$X_F$', fontsize=18)
    plt.savefig('clustering2-only_data.pdf')
    plt.close()



    plt.figure(num=None, figsize=(8,6), dpi=120, facecolor='w', edgecolor='k')
    # Plot the centroids as a white X
    centroids = kmeans.cluster_centers_
    plt.scatter(centroids[:, 0], centroids[:, 1],
                marker='x', s=169, linewidths=3,
                color='black', zorder=10)
    plt.legend(('Centroids',), fontsize=18)
    plt.imshow(Z, interpolation='nearest',
               extent=(xx.min(), xx.max(), yy.min(), yy.max()),
               cmap=plt.cm.cmaps_listed['magma'] ,
               aspect='auto', origin='lower')
    #plt.colorbar()
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.tick_params(axis='both', labelsize=18)
    plt.show()
    plt.ylabel('$Y_F$', fontsize=18)
    plt.xlabel('$X_F$', fontsize=18)
    plt.savefig('clustering2-contour.pdf')
    plt.close()