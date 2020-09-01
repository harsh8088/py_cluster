import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_blobs
from sklearn.metrics.pairwise import pairwise_distances
from kmed.constrained_kmedoids import KMedoids
from time import process_time


def plot_graphs(data, clusters):
    colors = {0: 'b*', 1: 'g^', 2: 'ro', 3: 'c*', 4: 'm^', 5: 'yo', 6: 'ko', 7: 'w*', 8: 'b*', 9: 'g^', 10: 'ro'}
    index = 0
    for key in clusters.keys():
        temp_data = clusters[key]
        x = [data[i][0] for i in temp_data]
        y = [data[i][1] for i in temp_data]
        plt.plot(x, y, colors[index])
        index += 1
    plt.title('Cluster formations')
    plt.show()

    medoid_data_points = []
    for m in clusters.keys():
        medoid_data_points.append(data[m])
    x = [i[0] for i in data]
    y = [i[1] for i in data]
    x_ = [i[0] for i in medoid_data_points]
    y_ = [i[1] for i in medoid_data_points]
    plt.plot(x, y, 'yo')
    plt.plot(x_, y_, 'r*')
    plt.title('Mediods are highlighted in red')
    plt.show()


def main():
    # generate random points
    X, _ = make_blobs(n_samples=40)

    # compute distance matrix
    dist = pairwise_distances(X, metric='euclidean')

    # k-medoids algorithm
    km = KMedoids(distance_matrix=dist, n_clusters=4)
    km.run(max_iterations=10, tolerance=0.001)

    print(km.clusters)
    end_time = process_time()
    print(end_time - start_time, ":time")
    plot_graphs(X, km.clusters)


if __name__ == '__main__':
    start_time = process_time()
    main()
