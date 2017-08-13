import json
import pandas as pd
import numpy as np
import sys
sys.path.insert(0, '/githubrepository/AnalyzeFBCrawl/')
from post_identification_func import other, dailyActivitiesPost, sharedFeeling, uploadedPhoto, uploadedVideo, sharedNews

def all_post():
    # header
    header_file = open('header_271_all_post_processed_.txt', 'r')
    arr_header = header_file.read().splitlines()
    # arr_header = read_header.split('\n')

    # content
    presentFile = '271_all_post_processed_wpostId.json'
    data_file = open(presentFile)
    json_data = json.load(data_file)
    alluser = []

    # gabung header dan row
    # tempusers = ['Adam Dolhanyk', 'Dawn Deangelo', 'JaeLee Morel']
    for username in json_data:
        for row in json_data[username]:
            alluser.append(row)

    df = pd.DataFrame(alluser, columns=arr_header)
    df['UserID'] = df['postId'].str.split('_').str.get(0).astype(int)

    arr_convert_to_num = ['PostTextLength', 'LikesCount', 'SharesCount', 'CommentsCount', 'PostTextPolarity', 'PostTextSubjectivity']
    df[arr_convert_to_num] = df[arr_convert_to_num].convert_objects(convert_numeric=True)
    # df['LikesCount'] = df['LikesCount'].astype(int)

    """
    LikesCount                       65658 non-null float64
    SharesCount                      65669 non-null float64
    CommentsCount                    65671 non-null float64
    """

    # print all_df['LikesCount'].dropna().median()
    # print all_df['LikesCount'].median()
    # print all_df['LikesCount'].interpolate()
    # all_df['LikesCount'].fillna(all_df.groupby(['userId'])['LikesCount'].median(), inplace=True)
    # all_df['SharesCount'].fillna(all_df['SharesCount'].dropna().median(), inplace=True)
    # all_df['CommentsCount'].fillna(all_df['CommentsCount'].dropna().median(), inplace=True)

    # LikesCount, SharesCount, CommentsCount ada yang g lengkap, dilengkapi dulu dengan nilai interpolated
    df['LikesCount'] = df['LikesCount'].interpolate().astype(int)
    df['SharesCount'] = df['SharesCount'].interpolate().astype(int)
    df['CommentsCount'] = df['CommentsCount'].interpolate().astype(int)
    # print arr_header
    return df

def user_gender():
    # gender
    gender_file = 'data/user_gender.csv'
    gender_df = pd.read_csv(gender_file, header=0)
    gender_df.Sex[ gender_df.Sex.isnull() ] = gender_df.Sex.dropna().mode().values
    gender_df['Sex'] = gender_df['Sex'].map({'F':0, 'M':1}).astype(int)
    return gender_df

def post_cluster_result():
    # content
    # file = 'D:/githubrepository/text-analysis-master/09.01.2016/kmeans_post_result_09012016.csv'
    file = 'data/kmeans_result_all_features.csv'
    df = pd.read_csv(file, header=0)
    df['resultCluster'] = df['resultCluster'].astype('category', ordered = False)
    return df

def user_cluster_result():
    # content
    file = 'D:/githubrepository/text-analysis-master/09.01.2016/user_kmeans_result.csv'
    # file = 'data/kmeans_result_all_features.csv'
    df = pd.read_csv(file, header=0)
    df['resultCluster'] = df['resultCluster'].astype('category', ordered = False)
    # print df
    return df

def getUserId():
    userid_file = open('data/userid_only.txt', 'r')
    arr_userid = userid_file.read().splitlines()
    x = list(set(arr_userid))
    x = [ int(y) for y in x ]
    x = sorted(x)
    for i in x:
        print i
    pass

def averageSum_Feature_User():
    # content
    df = all_post()
    id_shared_df = df[['UserID', 'PostTitle']].values
    array_shared = []
    for row in id_shared_df:
        array_shared.append([row[0], sharedNews(row[1]), uploadedPhoto(row[1]), uploadedVideo(row[1]) ])

    shared_df = pd.DataFrame(array_shared, columns=['UserID', 'SharedNews', 'UploadPhoto', 'UploadVideo'])
    # concat_df = pd.concat([df, shared_df], axis=1)
    # concat_df = df.join(shared_df, on='UserID', how='left', lsuffix='_shared')
    concat_df = df.merge(shared_df, on='UserID', how='left')

    """
    print df.info()
    print shared_df.info()
    print concat_df.head()

    # untuk tanggal
    df['converted_date'] = pd.to_datetime(df['PostTime'])
    mask = ( (df['converted_date'] > '2015-1-1') & (df['converted_date'] < '2015-12-31') )
    df.loc[mask]
    """

    concat_df = concat_df.groupby(['UserID'], sort=False).agg({
        'PostTextLength':np.median,
        'LikesCount':np.sum,
        'SharesCount':np.sum,
        'CommentsCount':np.sum,
        'SharedNews':np.sum,
        'UploadPhoto':np.sum,
        'UploadVideo':np.sum
    }).reset_index()

    # gender_df = user_gender()
    # concat_gender_df = pd.merge(concat_df, gender_df, how='left', on='userId')
    # return concat_gender_df
    return concat_df

print averageSum_Feature_User()
quit()
def user_score_aggregation():
    file = 'data/271_all_post_aggregate_to_user_rerun.csv'
    df = pd.read_csv(file, header=0)
    # df['resultCluster'] = df['resultCluster'].astype('category', ordered = False)
    print df.dtypes
    return df

def all_user_features():
    # featureUser
    df_feature_user = averageSum_Feature_User()
    # print df_feature_user.dtypes

    # header
    header_file = open('data/header_271_user_other_label.txt', 'r')
    arr_header = header_file.read().splitlines()
    # print arr_header()
    # content
    file = 'data/271_user_other_label_.csv'
    df = pd.read_csv(file, header=0)
    df = df[arr_header]
    df['userId'] = df['UserNum']
    # df.drop('UserNum', axis=1, inplace=True)
    concat_df = df.join(df_feature_user, on='userId', how='left', lsuffix='_')
    return concat_df

# all_user_features()
# print averageSum_Feature_User()
# user_cluster_result()