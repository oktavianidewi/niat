# level 1, to cluster from post-level to user-level features

import pandas as pd
import numpy as np
import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from operator import itemgetter
from dunn_sklearn import dunn_

# df.drop(['Unnamed: 0', 'UserId', 'PostTextLength'], axis=1, inplace=True)

# print df.dtypes
# print len(df)
# quit()

dunn_value = {}
def getKValue(minval, maxval, X):
    for k in range(minval, maxval):
        # print 'jumlah cluster : ', k
        # k = 4
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(X)
        # transform -> get the distance of each point within a cluster to the cluster center, the index of the lowest value indicates the label class
        distance = kmeans.transform(X)
        labels = kmeans.labels_
        labels_distance = zip(labels, distance)
        labels_distance.sort(key=lambda t: t[0])
        cluster_labels = [x for x in range(k)]
        dunn_value[k] = dunn_(cluster_labels, labels_distance)
    max_dunn = max(dunn_value.iteritems(), key=itemgetter(1))[0]
    return max_dunn

def print_posts_to_csv(df, labels, centroids, ClusterColumn):
    labels_centroid = []
    for label in labels:
        labels_centroid.append([label] + [centroids[label][x] for x in range(len(ClusterColumn))])
    # ClusterColumn
    for x in ['labels'] + ['Centroid' + x for x in ClusterColumn]:
        df[x] = 0
    df[['labels'] + ['Centroid' + x for x in ClusterColumn]] = labels_centroid
    # df.to_csv('posts_cluster.csv')
    return df

def post_userlevel_cluster(df):
    # X = df.values
    ClusterColumn = ['PostTextSubjectivity', 'PostTextPolarity']
    X = StandardScaler().fit_transform(df[ClusterColumn].values)
    ScaledColumn = [ 'Scaled'+x for x in ClusterColumn ]

    for x in ScaledColumn:
        df[ x ] = 0

    df[ScaledColumn] = X

    # find the highest dunn index
    k = getKValue(2, 4, X)

    # print 'jumlah recommended clusters : ', k
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(X)
    # transform -> get the distance of each point within a cluster to the cluster center, the index of the lowest value indicates the label class
    distance = kmeans.transform(X)
    centroids = kmeans.cluster_centers_
    labels = kmeans.labels_

    # df = print_posts_to_csv(df, labels, centroids, ClusterColumn)
    # print df.head()

    # print to csv -> untuk mengetahui centroid dan kecenderungan sentiment
    # print centroids
    # print labels


    # print labels
    # print distance

    # grouping based on userID, get the raw value
    labels_distance = zip(labels, distance)
    labels_distance.sort(key = lambda t: t[0])

    cluster_labels = [ x for x in range(k) ]

    df['cluster'] = labels

    cluster_dummies = pd.get_dummies(df['cluster'])
    df = df.join(cluster_dummies)
    dict_aggr = {x: np.count_nonzero for x in cluster_dummies}
    aggr_df = df.groupby(['UserId'], sort=True).agg(dict_aggr).reset_index()

    # print X
    # df.to_csv('allposts_sentimentscore_cluster.csv')
    return aggr_df

# df = pd.read_csv('allposts_sentimentscore.csv', sep=None, header=0)
# p = post_userlevel_cluster(df)
# print p