import numpy as np

from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler


import load_all_dataset as ld
import pandas as pd

def dataset_allpost(type):
    if type == 'old':
        # Load the text dataset, drop null non-text, assign UserId
        all_df = ld.all_post()
        all_df = all_df[['postId', 'PostTextLength', 'PostTextPolarity', 'PostTextSubjectivity']].dropna()
        all_df['UserId'] = all_df['postId'].str.split('_').str.get(0).astype(int)
        all_df.drop('postId', axis=1, inplace=True)

    elif type == 'new':
        # Load the new dataset, drop null non-text, assign UserId
        ds_file_array = ['data/english_foodgroup_new.json', 'data/english_TEDtranslate_new.json',
                         'data/english_traveladdiction_new.json']
        photo_file = ['data/album_english_foodgroups.json', 'data/album_english_TEDtranslate.json',
                      'data/album_english_traveladdiction.json']
        friendsnum_file = ['data/english_foodgroups_friendsnum.json', 'data/english_TEDtranslate_friendsnum.json',
                           'data/english_traveladdiction_friendsnum.json']

        all_df = ld.new_dataset(ds_file_array)
        # print all_df.dtypes
        all_df = all_df[['UserID', 'PostTextLength', 'PostTextPolarity','PostTextSubjectivity']].dropna()
        all_df['UserId'] = all_df['UserID']
        all_df.drop('UserID', axis=1, inplace=True)

    return all_df

# *************************************** #
# """

# """
# merge between 2 datasets
all_post_old = dataset_allpost('old')
all_post_new = dataset_allpost('new')

# print all_post_old
# print all_post_new
all_posts = all_post_new.append(all_post_old[list(all_post_new.columns.values)], ignore_index=True)
# all_posts = all_post_old
# UserId = all_posts['UserId'].values
# all_posts.drop(['UserId'], axis=1, inplace=True)
all_posts.to_csv('allposts_sentimentscore.csv')
print all_posts.dtypes

quit()
# print all_posts.values

# task: masukkan ke clustering
# """

# access dataset
"""
centers = [[1, 1], [-1, -1], [1, -1]]
X, labels_true = make_blobs(n_samples=750, centers=centers, cluster_std=0.4, random_state=0)

print X
"""

# X = StandardScaler().fit_transform(X)
# X = StandardScaler().fit_transform(all_posts)
X = all_posts.as_matrix()
print len(X)

import datetime

a = datetime.datetime.now()
db = DBSCAN(eps=0.01, min_samples=1, algorithm="ball_tree", metric='euclidean').fit(X)
b = datetime.datetime.now()
delta = b - a
print 'clustering time : ', delta
cluster_labels = db.labels_
num_clusters = len(set(cluster_labels))
print 'jumlah clusters ', num_clusters
quit()

core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

"""
print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))
print("Adjusted Rand Index: %0.3f" % metrics.adjusted_rand_score(labels_true, labels))
"""

print('Estimated number of clusters: %d' % n_clusters_)
print("Adjusted Mutual Information: %0.3f" % metrics.adjusted_mutual_info_score(labels_true, labels))
print("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(X, labels))