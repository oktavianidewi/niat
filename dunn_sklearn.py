from operator import itemgetter
import pandas as pd
import numpy as np

def dunn(c, distances):
    """
    Dunn index for cluster validation (the bigger, the better)

    .. math:: D = \\min_{i = 1 \\ldots n_c; j = i + 1\ldots n_c} \\left\\lbrace \\frac{d \\left( c_i,c_j \\right)}{\\max_{k = 1 \\ldots n_c} \\left(diam \\left(c_k \\right) \\right)} \\right\\rbrace

    where :math:`d(c_i,c_j)` represents the distance between
    clusters :math:`c_i` and :math:`c_j`, given by the distances between its
    two closest data points, and :math:`diam(c_k)` is the diameter of cluster
    :math:`c_k`, given by the distance between its two farthest data points.

    The bigger the value of the resulting Dunn index, the better the clustering
    result is considered, since higher values indicate that clusters are
    compact (small :math:`diam(c_k)`) and far apart.

    .. [Kovacs2005] Kovacs, F., Legany, C., & Babos, A. (2005). Cluster validity measurement techniques. 6th International Symposium of Hungarian Researchers on Computational Intelligence.
    """

    unique_cluster_distances = np.unique(min_cluster_distances(c, distances))
    print 'unique cluster distances ', unique_cluster_distances
    max_diameter = max(diameter(c, distances))
    print 'max diameter ', max_diameter

    if np.size(unique_cluster_distances) > 1:
        return unique_cluster_distances[1] / max_diameter
        print 'unique cluster banyak ', unique_cluster_distances[1] / max_diameter
    else:
        return unique_cluster_distances[0] / max_diameter
        print 'unique cluster sedikit ', unique_cluster_distances[0] / max_diameter

def min_cluster_distances(c, distances):
    """Calculates the distances between the two nearest points of each cluster""" # nilai intracluster
    min_distances = np.zeros((max(c) + 1, max(c) + 1))
    print min_distances
    for i in np.arange(0, len(c)):
        if c[i] == -1: continue
        # for ii in np.arange(i + 1, len(c)):
        for ii in np.arange(0, len(c)):
            if c[ii] == -1: continue
            print i, ii
            print 'nilai c', c[i], c[ii]
            print 'distance ', distances[i, ii]
            print 'min distance ', min_distances[c[i], c[ii]]
            if c[i] != c[ii] and distances[i, ii] > min_distances[c[i], c[ii]]:
            # if distances[i, ii] > min_distances[c[i], c[ii]]:
                min_distances[c[i], c[ii]] = min_distances[c[ii], c[i]] = distances[i, ii]
                # min_distances[c[i], c[ii]] = min_distances[c[i], c[ii]] = distances[i, ii]
    print 'nilai min distance akhir ', min_distances
    return min_distances

def diameter(c, distances):
    """Calculates cluster diameters (the distance between the two farthest data points in a cluster)"""
    diameters = np.zeros(max(c) + 1)
    print 'diameters ', diameters
    for i in np.arange(0, len(c)):
        if c[i] == -1: continue
        for ii in np.arange(i + 1, len(c)):
            if c[ii] == -1: continue
            print 'diameters i, ii', i, ii
            if c[i] != -1 or c[ii] != -1 and c[i] == c[ii] and distances[i, ii] > diameters[c[i]]:
                diameters[c[i]] = distances[i, ii]
    print 'nilai diameters ', diameters
    return diameters

def max_intracluster(cluster_labels, f):
    distances = []
    # cluster_label = [ x for x in range(5) ]
    for cluster in cluster_labels:
        new_list = []
        new_list_particular = []
        # print cluster
        for x,y in f:
            # print cluster
            # cluster = 4
            if x == cluster:
                # print x
                # index = 0
                new_list.append(y)
                new_list_particular.append(y[cluster])

        index_min = new_list_particular.index(min(new_list_particular))
        index_max = new_list_particular.index(max(new_list_particular))
        distances.append(new_list[index_max])

        """
        print new_list
        print 'cluster : ', cluster
        print 'min intracluster ', new_list[index_min]
        print 'max intracluster ', new_list[index_max]
        """
    return distances

def dunn_(cluster_labels, labels_distance):
    # print c
    # print distances
    distances = max_intracluster(cluster_labels, labels_distance)
    # print distances

    division_list = []
    for cluster in cluster_labels:
        """
        print distances[cluster]
        print distances[cluster][cluster]
        print 'cluster ', cluster
        """
        division = distances[cluster]/distances[cluster][cluster]
        # print 'division ', division
        min_division = min(result for result in division if result > 1.0)
        # print 'min ', min_division
        division_list.append(min_division)
        # print cluster
    # print division_list
    # print 'min division list = dunn result ', min(division_list)
    return round(min(division_list), 4)

# distances = np.array(distances)
# print distances
# dunn_(cluster_label, distances)
