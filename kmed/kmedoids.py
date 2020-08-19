import numpy as np
from sklearn.datasets.samples_generator import make_blobs
from sklearn.metrics.pairwise import pairwise_distances
from kmed.constrained_kmedoids import KMedoids

# generate random points
X, _ = make_blobs(n_samples=40, centers=3)

# compute distance matrix
dist = pairwise_distances(X, metric='euclidean')

# k-medoids algorithm
km = KMedoids(distance_matrix=dist, n_clusters=5)
km.run(max_iterations=10, tolerance=0.001)

print(km.clusters)