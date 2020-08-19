from matplotlib import pyplot
import numpy as np
from sklearn.cluster import Birch
from numpy import unique
from numpy import where


# Returns dictionary value from the function
def map_clusters(n_list, n_clusters):
    # x = np.array([[28.596596, 77.344098], [28.574783, 77.333393]])
    # x = np.append(x, [[28.596596, 77.344098], [28.574783, 77.333393], [28.582515, 77.246735],
    #                   [28.582915, 77.215735], [28.635639, 77.201197], [28.464873, 76.995451]], axis=0)
    x = np.array([[28.596596, 0], [28.574783, 0], [28.996596, 0], [28.674783, 0], [28.582515, 0],
                  [28.582915, 0], [28.635639, 0], [28.464873, 0]])
    # x = np.append(x, n_list, axis=0)
    # define the model
    model = Birch(threshold=0.01, n_clusters=3)
    # fit the model
    model.fit(x)
    # assign a cluster to each example
    yhat = model.predict(x)
    # retrieve unique clusters
    clusters = unique(yhat)
    dic = {}
    # create scatter plot for samples from each cluster
    for cluster in clusters:
        # 	# get row indexes for samples with this cluster
        row_ix = where(yhat == cluster)
        # 	# create scatter of these samples
        dic[cluster] = row_ix[0]
        pyplot.scatter(x[row_ix, 0], x[row_ix, 1])
    # print(dic)
    pyplot.show()
    return dic
