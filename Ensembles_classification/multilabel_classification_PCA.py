import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_multilabel_classification
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import CCA
from sklearn.cluster import KMeans

plt.style.use('seaborn-paper')
plt.rcParams.update({'font.size': 12})
from matplotlib import rc
rc('font',**{'family':'serif','serif':['Times New Roman']})
rc('text', usetex=True)

num_samples = 500

np.random.seed(19680801)## Fixing random state for reproducibility


param1 = np.random.exponential(scale=0.18, size=num_samples) # Relative lateral branch length
param2 = np.random.uniform(0,1, size=num_samples) # Relative position branch length
param3 = np.random.normal(0.1, 0.02, size=num_samples) # Load Distribution
param4 = np.random.exponential(scale=0.12, size=num_samples) # DG level
param5 = np.random.normal(1/3, 0.1, size=num_samples) # unbalance load
param6 = np.random.uniform(0,1, size=num_samples) # PV Generation distribution

X =  np.vstack((param1, param2, param3, param4, param5, param6)).T

fig = plt.figure(num=None, figsize=(13,12), dpi=120, facecolor='w', edgecolor='k')

ax = fig.add_subplot(3, 2, 1)
ax.bar(np.arange(0,len(param1)), np.sort(param1), color='k')
ax.set_ylabel('Laterals Weight', fontsize=18)
ax.legend(('$LW$',),fontsize=18, loc='upper left')
plt.title('a)', fontsize=20)
ax.tick_params(axis='both', labelsize=18)

ax = fig.add_subplot(3, 2, 2)
ax.bar(np.arange(0,len(param2)), np.sort(param2), color='k')
ax.set_ylabel('Laterals Location', fontsize=18)
ax.legend(('$LL$',),fontsize=18, loc='upper left')
plt.title('b)', fontsize=20)
ax.tick_params(axis='both', labelsize=18)

ax = fig.add_subplot(3, 2, 3)
ax.bar(np.arange(0,len(param3)), np.sort(param3), color='k' )
ax.set_ylabel('Load Distribution', fontsize=18)
ax.legend(('$LD$',),fontsize=18, loc='upper left')
plt.title('c)', fontsize=20)
ax.tick_params(axis='both', labelsize=18)

ax = fig.add_subplot(3, 2, 4)
ax.bar(np.arange(0,len(param4)), np.sort(param4), color='k')
ax.set_ylabel('PV Generation Level', fontsize=18)
ax.legend(('PVL',),fontsize=18, loc='upper left')
plt.title('d)', fontsize=20)
ax.tick_params(axis='both', labelsize=18)

ax = fig.add_subplot(3, 2, 5)
ax.bar(np.arange(0,len(param5)), np.sort(param5), color='k')
ax.set_ylabel('Unbalance Load Level', fontsize=18)
ax.legend(('$ULL$',),fontsize=18, loc='upper left')
plt.title('e)', fontsize=20)
ax.tick_params(axis='both', labelsize=18)

ax = fig.add_subplot(3, 2, 6)
ax.bar(np.arange(0,len(param6)), np.sort(param6), color='k')
ax.set_ylabel('PV Generation distribution', fontsize=18)
ax.legend(('PVD',),fontsize=18, loc='upper left')
plt.title('f)', fontsize=20)
ax.tick_params(axis='both', labelsize=18)

plt.show()
plt.savefig('fig0.eps')




Xr = PCA(n_components=2).fit_transform(X)
x_min = np.min(Xr[:, 0])
x_max = np.max(Xr[:, 0])
y_min = np.min(Xr[:, 1])
y_max = np.max(Xr[:, 1])


plt.figure(num=None, figsize=(10,8), dpi=120, facecolor='w', edgecolor='k')
plt.clf()

#Plot data
plt.scatter(Xr[:, 0], Xr[:, 1], c='k', linewidths=12, label='Data')
plt.tick_params(axis='both', labelsize=16)
#plt.xticks(())
#plt.yticks(())
#plt.xlim(min_x - .5 * max_x, max_x + .5 * max_x)
#plt.ylim(min_y - .5 * max_y, max_y + .5 * max_y)
offset = 0.3
plt.xlim(x_min-offset, x_max+offset)
plt.ylim(y_min-offset, y_max+offset)
plt.tick_params(axis='both', labelsize=18)
plt.show()
plt.legend(fontsize=18, fancybox=True, edgecolor='black', shadow=True)
plt.ylabel('$Y_1$', fontsize=18)
plt.xlabel('$X_1$', fontsize=18)
plt.savefig('fig33.eps')



random_state = 170
num_clusters = 3

plt.figure(num=None, figsize=(10,8), dpi=120, facecolor='w', edgecolor='k')
plt.clf()
y_pred = KMeans(init='k-means++', n_clusters=num_clusters, random_state=random_state).fit_predict(Xr)
class1 = y_pred==0
class2 = y_pred==1
class3 = y_pred==2
class4 = y_pred==3
#plt.scatter(Xr[:, 0], Xr[:, 1], c=y_pred, marker = 'o', linewidths=12)
plt.scatter(Xr[class1, 0], Xr[class1, 1], c='b', marker = 'o', linewidths=12, label='Cluster 1')
plt.scatter(Xr[class2, 0], Xr[class2, 1], c='r', marker = 'o', linewidths=12, label='Cluster 2')
plt.scatter(Xr[class3, 0], Xr[class3, 1], c='g', marker = 'o', linewidths=12, label='Cluster 3')
if num_clusters==4:
    plt.scatter(Xr[class4, 0], Xr[class4, 1], c='k', marker = 'o', linewidths=12, label='Cluster 4')
#Plot the centroids as a white X
plt.scatter(np.mean(Xr[class1, 0]), np.mean(Xr[class1, 1]), marker='x', s=169, linewidths=3, color='k', zorder=10, label='Centroids')
plt.scatter(np.mean(Xr[class2, 0]), np.mean(Xr[class2, 1]), marker='x', s=169, linewidths=3, color='k', zorder=10)
plt.scatter(np.mean(Xr[class3, 0]), np.mean(Xr[class3, 1]), marker='x', s=169, linewidths=3, color='k', zorder=10)
#Plot data
plt.scatter(Xr[:, 0], Xr[:, 1], c='k')
plt.tick_params(axis='both', labelsize=16)
#plt.xticks(())
#plt.yticks(())
#plt.xlim(min_x - .5 * max_x, max_x + .5 * max_x)
#plt.ylim(min_y - .5 * max_y, max_y + .5 * max_y)
offset = 0.3
plt.xlim(x_min-offset, x_max+offset)
plt.ylim(y_min-offset, y_max+offset)
plt.tick_params(axis='both', labelsize=18)
plt.show()
plt.legend(fontsize=18, fancybox=True, edgecolor='black', shadow=True)
plt.ylabel('$Y_1$', fontsize=18)
plt.xlabel('$X_1$', fontsize=18)
plt.savefig('fig3.eps')


kmeans = KMeans(init='k-means++', n_clusters=num_clusters, n_init=1, random_state=random_state)
kmeans.fit(Xr)
# Step size of the mesh. Decrease to increase the quality of the VQ.
h = .02     # point in the mesh [x_min, x_max]x[y_min, y_max].

# Plot the decision boundary. For that, we will assign a color to each
offset = 0.3
x_min, x_max = Xr[:, 0].min() - offset, Xr[:, 0].max() + offset
y_min, y_max = Xr[:, 1].min() - offset, Xr[:, 1].max() + offset
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
# Obtain labels for each point in mesh. Use last trained model.
Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])
# Put the result into a color plot
Z = Z.reshape(xx.shape)


plt.figure(num=None, figsize=(10,8), dpi=120, facecolor='w', edgecolor='k')
# Plot the centroids as a white X
centroids = kmeans.cluster_centers_
plt.scatter(centroids[:, 0], centroids[:, 1], marker='x', s=169, linewidths=3, color='black', zorder=10)
plt.legend(('Centroids',), fontsize=18)
#cmap = plt.cm.get_cmap("winter")
cmap = plt.cm.get_cmap("jet")
plt.imshow(Z, interpolation='nearest',
           extent=(xx.min(), xx.max(), yy.min(), yy.max()),
           cmap = cmap,
           aspect='auto', origin='lower')
#plt.colorbar()
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.tick_params(axis='both', labelsize=18)
plt.show()
plt.ylabel('$Y_1$', fontsize=18)
plt.xlabel('$X_1$', fontsize=18)
plt.savefig('fig4.eps')