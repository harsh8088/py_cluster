# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import cluster_map as cluster
import numpy as np


def print_value(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.
    n_list = [[28.596596, 77.344098], [28.574783, 77.333393], [28.582515, 77.246735],
              [28.582915, 77.215735], [28.635639, 77.201197], [28.464873, 76.995451]]
    print(cluster.map_clusters(np.array(n_list), 4))


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_value('Main')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
