# birch clustering
from matplotlib import pyplot
from numpy import unique
from numpy import where
import numpy as np
import json
from sklearn.datasets import make_classification
from sklearn.cluster import Birch

# define dataset
# X, _ = make_classification(n_samples=10, n_features=2, n_informative=2, n_redundant=0, n_clusters_per_class=1,
#                            random_state=4)
X = np.array([[28.596596, 77.344098], [28.574783, 77.333393]])
X = np.append(X, [[28.596596, 77.344098], [28.574783, 77.333393], [28.582515, 77.246735],
                  [28.582915, 77.215735], [28.635639, 77.201197], [28.464873, 76.995451]],axis=0)
# define the model
model = Birch(threshold=0.01, n_clusters=3)
# fit the model
model.fit(X)
# assign a cluster to each example
yhat = model.predict(X)
# retrieve unique clusters
clusters = unique(yhat)
dic = {}
# create scatter plot for samples from each cluster
for cluster in clusters:
    # 	# get row indexes for samples with this cluster
    row_ix = where(yhat == cluster)
    # 	# create scatter of these samples
    dic[cluster] = row_ix[0]
    # print(clusters)
    # print('yhat:')
    # print(yhat)
    # print('cluster:')
    # print(cluster)
    # print('row_ix:')
    # print(row_ix)
    # print(X[row_ix, 0])
    # print(X[row_ix, 1])
    pyplot.scatter(X[row_ix, 0], X[row_ix, 1])
    # pyplot.plot(X[row_ix, 0], X[row_ix, 1], marker='o', linestyle='dashed',
    #             linewidth=2, markersize=12)
    # print('dict')

# # show the plot
print(dic)
pyplot.show()

