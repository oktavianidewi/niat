import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import csv
import pandas as pd
import json
import sys
import load_all_dataset as ld
from sklearn.metrics import silhouette_score
import math

# plt.scatter(x,y)
# plt.show()

k = 3
# data = np.mat(ownKmeans.loadDataSet('08.22.2016/271_all_post_processed_humanjudge7_subjpol - withoutNullValueSubjPol - onlyNumbers#2.txt'))

gs_df = pd.read_csv('data/userlevel_all_features_1007.csv', header=0)

df = ld.all_post()
# print df.dtypes
df['userId'] = df['postId'].str.split('_').str.get(0).astype(int)
df['postIds'] = df['postId'].str.split('_').str.get(1).astype(int)

df['interaction'] = df['LikesCount']+df['SharesCount']+df['CommentsCount']
df = df[['userId', 'postIds', 'postId', 'PostTextLength', 'interaction', 'PostTextSubjectivity', 'PostTextPolarity']].dropna()

userId = df['userId'].values
postId = df['postIds'].values
polarity = df['PostTextPolarity'].values
subjectivity = df['PostTextSubjectivity'].values

# data = df[['PostTextLength', 'interaction', 'PostTextSubjectivity', 'PostTextPolarity']].values
data = df[['PostTextSubjectivity', 'PostTextPolarity', 'PostTextLength']].values

kmeans = KMeans(n_clusters=k, max_iter=100)

userid_list = list( df.groupby(['userId'], sort=True).size() )
userid_list = [262, 35, 400, 68, 16]

# kalau semua
"""
print data
kmeans.fit(data)
centroids = kmeans.cluster_centers_
labels = kmeans.labels_
print list(centroids)
print list(labels)
"""
# kalau satu2
for user_id in userid_list:
    data = df[['PostTextSubjectivity', 'PostTextPolarity', 'PostTextLength']].loc[df['userId'] == user_id]
    print 'User ID ', user_id
    print 'Number of Posts ', len(data)
    if len(data) > 0:
        kmeans.fit(data)
        centroids = kmeans.cluster_centers_
        labels = kmeans.labels_
        print 'Cluster \t [PostTextSubjectivity, PostTextPolarity, PostTextLength]'
        for index, cluster_centroid in enumerate(centroids):
            print index, '\t', cluster_centroid
        print labels

"""
resultcluster_file = open("kmeans5_result_10112016.csv", "wb")
open_file_object = csv.writer(resultcluster_file)
open_file_object.writerow(["userId","postId","subjectivity","polarity","resultCluster"])
open_file_object.writerows(zip(userId, postId, subjectivity, polarity, labels))
resultcluster_file.close()
"""

"""
# untuk membuat cluster dalam kolom sendiri2
file_csv = 'kmeans_result_09172016.csv'
df = pd.read_csv(file_csv, header=0)
df['UserNum'] = df['postId'].str.split('_').str.get(0).astype(int)

# resultcluster level
resultCluster_dummies = pd.get_dummies(df['resultCluster'])
df = df.join(resultCluster_dummies)
df.rename(columns={0: 'Cluster_0', 1: 'Cluster_1', 2: 'Cluster_2', 3: 'Cluster_3'}, inplace=True)
print df.dtypes
df.to_csv('kmeans_result_09172016.csv')


# np.savetxt('08.22.2016/271_all_post_processed_humanjudge7_subjpol - withoutNullValueSubjPol - onlyNumbers_resultKmeansScikit.txt', labels, delimiter='\t')
# kalo bisa di pas2in warna green -> sharer, red -> monologue, turquoise -> sociable, blue -> critics
title = 'Scatter Plot Distribution'
opsiwarna = ['turquoise', 'red', 'green', 'blue', 'black']

for i in range(k):
    # select only data observations with cluster label == i
    # tentukan index mana aja yang di-print disini
    # ds = data[262:297]
    ds = data[np.where(labels==i)]
    # print 'ds awal ', ds
    # print 'ds ', ds[0:2]
    # plot the data observations
    # plt.plot(x,y,'o')
    # print 'ds ', ds
    plt.plot(ds[:,0],ds[:,1], 'o', color=opsiwarna[i])
    # plt.plot(ds[:,0],ds[:,1], 'o', color='white')
    # plot the centroids
    lines = plt.plot(centroids[i,0], centroids[i,1],'kx')
    # make the centroid x's bigger
    plt.setp(lines, ms=10.0)
    plt.setp(lines, mew=2.0)
plt.title(title)
plt.xlabel('PostTextSubjectivity')
plt.ylabel('PostTextPolarity')
plt.axis([-0.2, 1.2, -1.5, 1.5])
plt.show()
"""