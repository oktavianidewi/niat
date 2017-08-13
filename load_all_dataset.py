# -*- coding: utf-8 -*-
import json, csv
import pandas as pd
import numpy as np
import sys
sys.path.insert(0, '/githubrepository/AnalyzeFBCrawl/')
from post_identification_func import other, dailyActivitiesPost, sharedFeeling, uploadedPhoto, uploadedVideo, sharedNews
import warnings
warnings.filterwarnings('ignore')

def all_post():
    # header
    header_file = open('header_271_all_post_processed_.txt', 'r')
    arr_header = header_file.read().splitlines()

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

    df['userId'] = df['postId'].str.split('_').str.get(0).astype(int)

    arr_convert_to_num = ['PostTextLength', 'LikesCount', 'SharesCount', 'CommentsCount', 'PostTextPolarity', 'PostTextSubjectivity']
    df[arr_convert_to_num] = df[arr_convert_to_num].convert_objects(convert_numeric=True)
    # df['LikesCount'] = df['LikesCount'].astype(int)

    """
    LikesCount                       65658 non-null float64
    SharesCount                      65669 non-null float64
    CommentsCount                    65671 non-null float64
    """

    # print datetime.datetime.strptime(df['PostTime'], )
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

    return df

def func_week_part(df):
    weekend_dummies = pd.get_dummies(df['WeekendOrWeekday'])
    df = df.join(weekend_dummies)

    def most_frequent_week_part(x):
        if (x['Weekend'] > x['Weekday']):
            return 'Weekend'  # x['Weekend']
        else:
            return 'Weekday'  # x['Weekday']

    aggr_df = df.groupby(['userId'], sort=True).agg({'Weekday': np.count_nonzero, 'Weekend': np.count_nonzero}).reset_index()
    aggr_df['frequent_week_part'] = aggr_df.apply(most_frequent_week_part, axis=1)
    return aggr_df[['userId', 'frequent_day_part']]

def func_day_part(df):
    df['waktu'] = 'a'
    condition = (df['PostTime'].str.split(' at ').str.get(1).str[-2:] == 'pm')
    if_yes = df['PostTime'].str.split(' at ').str.get(1).str.split(':').str.get(0).fillna(0).astype(int) + 12
    if_no = df['PostTime'].str.split(' at ').str.get(1).str.split(':').str.get(0).fillna(0).astype(int)
    df['waktu'] = np.where(condition, if_yes, if_no)

    df['day_part'] = 'a'
    df['day_part'].loc[((df['waktu'] >= 1) & (df['waktu'] < 6))] = 'Early_Morning'
    df['day_part'].loc[((df['waktu'] >= 6) & (df['waktu'] < 12))] = 'Morning'
    df['day_part'].loc[((df['waktu'] >= 12) & (df['waktu'] < 18))] = 'Afternoon'
    df['day_part'].loc[((df['waktu'] >= 18) & (df['waktu'] <= 24))] = 'Evening'

    df.drop('waktu', axis=1, inplace=True)

    daypart_dummies = pd.get_dummies(df['day_part'])
    df = df.join(daypart_dummies)

    def most_frequent_day_part(x):
        if ( x['Morning'] > x['Afternoon'] and x['Morning'] > x['Evening'] and x['Morning'] > x['Early_Morning'] ):
            return 'Morning' # x['Morning']
        elif ( x['Afternoon'] > x['Morning'] and x['Afternoon'] > x['Evening'] and x['Afternoon'] > x['Early_Morning'] ):
            return 'Afternoon' # x['Afternoon']
        elif ( x['Evening'] > x['Morning'] and x['Evening'] > x['Afternoon'] and x['Evening'] > x['Early_Morning'] ):
            return 'Evening' # x['Evening']
        elif ( x['Early_Morning'] > x['Morning'] and x['Early_Morning'] > x['Afternoon'] and x['Early_Morning'] > x['Evening'] ):
            return 'Early_Morning' # x['Early Evening']
    aggr_df = df.groupby(['UserId'], sort=True).agg({'Morning':np.count_nonzero, 'Afternoon':np.count_nonzero, 'Evening':np.count_nonzero, 'Early_Morning':np.count_nonzero}).reset_index()
    aggr_df['frequent_day_part'] = aggr_df.apply( most_frequent_day_part , axis=1)
    return aggr_df[['UserId', 'frequent_day_part']]

def photo_dataset(filearr):
    y = []
    json_data = []
    for filename in filearr:
        # print filename
        # content
        data_file = open(filename)
        json_data = json.load(data_file)
        for username in json_data:
            if 'photos' in json_data[username]:
                z = json_data[username]["photos"]
                allcols = list( set([c for c in z]) - set(['Profile Pictures', 'Timeline Photos', 'Cover Photos', 'Mobile Uploads', 'Videos']) )
                # print allcols
                # to count all photos
                allphotos = 0
                up = 0
                for albums in allcols:
                    if z[albums] == -1:
                        numofphoto = 0
                    else:
                        numofphoto = int( z[albums].split(' ')[0].replace(',', '') )
                    allphotos += numofphoto

                if 'Profile Pictures' in z:
                    pp = z['Profile Pictures'].split(' ')[0]
                else:
                    pp = 0

                for selfuploaded in ['Timeline Photos', 'Mobile Uploads']:
                    if selfuploaded in z:
                        up += int( z[selfuploaded].split(' ')[0].replace(',', '') )

                if 'Cover Photos' in z:
                    cp = z['Cover Photos'].split(' ')[0]
                else:
                    cp = 0
            y.append([username, pp, cp, up, allphotos])
    # print y
    df = pd.DataFrame(y, columns=['UserID', 'NoProfilePhotos', 'NoCoverPhotos',  'UploadPhotoSum', 'NoPhotos'])
    arr_convert_to_num = ['NoProfilePhotos', 'NoCoverPhotos', 'UploadPhotoSum', 'NoPhotos']
    df[arr_convert_to_num] = df[arr_convert_to_num].convert_objects(convert_numeric=True)
    return df

# photo_file = ['data/album_english_foodgroups.json', 'data/album_english_TEDtranslate.json', 'data/album_english_traveladdiction.json']
# all_df = photo_dataset(photo_file)

def new_dataset(filearr):
    header_file = open('data/header_english_dataset.txt', 'r')
    arr_header = header_file.read().splitlines()+['ActiveInterests']

    # content
    all_arr = []
    json_data = []
    for filename in filearr:
        # print filename
        if 'food' in filename: ai = 'Food'
        elif 'travel' in filename: ai = 'Travelling'
        elif 'translate' in filename: ai = 'TEDTranslate'

        data_file = open(filename)
        json_data = json.load(data_file)
        if '.DS_Store' in json_data: del json_data['.DS_Store']
        # print len(read_file)
        for username in json_data:
            # print i, ai
            if 'timeline' in json_data[username]:
                for post in json_data[username]['timeline']:
                    post.insert(0, username)
                    post.append(ai)
                    all_arr.append(post[:len(arr_header)])

    df = pd.DataFrame(all_arr, columns=arr_header)
    # print df.dtypes
    # df['Interaction'] = df['SharesCount']+df['CommentsCount']+df['LikesCount']
    arr_convert_to_num = ['PostTextLength', 'LikesCount', 'SharesCount', 'CommentsCount', 'PostTextPolarity', 'PostTextSubjectivity']
    df[arr_convert_to_num] = df[arr_convert_to_num].convert_objects(convert_numeric=True)
    return df

def separate_postTextLength(df):
    # df = df[df.line_race != 0]
    df = df[df['PostTextLength'] != 0]
    df['PostTextLengthLevel'] = 'ShortPosts'
    df['PostTextLengthLevel'].loc[ df['PostTextLength'] > df['PostTextLength'].mean() ] = 'LongPosts'
    length_dummies = pd.get_dummies(df['PostTextLengthLevel'])
    df = df.join(length_dummies)
    df_cols = [col for col in df.columns if 'UserID' in col]
    # print df.dtypes
    if df_cols :
        userid = 'UserID'
        aggr_df = df.groupby(['UserID'], sort=True).agg({'ShortPosts': np.count_nonzero, 'LongPosts': np.count_nonzero}).reset_index()
    else:
        userid = 'UserId'
        aggr_df = df.groupby(['UserId'], sort=True).agg({'ShortPosts': np.count_nonzero, 'LongPosts': np.count_nonzero}).reset_index()
    return aggr_df[[userid, 'ShortPosts', 'LongPosts']]

# ds_file = ['data/english_foodgroup_new.json', 'data/english_TEDtranslate_new.json', 'data/english_traveladdiction_new.json']
# all_df = new_dataset(ds_file)
# all_df = all_post()
# print separate_postTextLength(all_df)
# quit()

def about_dataset(filearr):
    # content
    all_arr = []
    json_data = []

    for filename in filearr:
        # print filename
        data_file = open(filename)
        json_data = json.load(data_file)
        for username in json_data:
            if 'about' in json_data[username]:
                if 'Gender' in json_data[username]['about']:
                    genderinfo = json_data[username]['about']['Gender']
                else:
                    genderinfo = 'NA'
            else:
                genderinfo = 'NA'

            if 'timeline' in json_data[username]:
                userFB = json_data[username]['timeline'][0][0].split('com/')[1]

            all_arr.append([username, userFB, genderinfo])
    df = pd.DataFrame(all_arr, columns=['UserID', 'UserName', 'Gender'])
    df['GenderCode'] = df['Gender'].map({'Female':0, 'Male':1, 'NA':2})
    return df

def friendsnum_dataset(filearr):
    # content
    all_arr = []
    json_data = []

    for filename in filearr:
        data_file = open(filename)
        json_data = json.load(data_file)
        for username in json_data:
            if 'friendnum' in json_data[username]:

                # print type(json_data[username]['friendnum'])
                if type(json_data[username]['friendnum']) == unicode:
                    try:
                        friendnum = int(json_data[username]['friendnum'])
                    except Exception, e:
                        friendnum = 0
                else:
                    friendnum = 0
            else:
                friendnum = 0
            all_arr.append([username, friendnum])
    df = pd.DataFrame(all_arr, columns=['UserID', 'Friends'])
    return df

# friendsnum_file = ['data/english_foodgroups_friendsnum.json', 'data/english_TEDtranslate_friendsnum.json', 'data/english_traveladdiction_friendsnum.json']
# print friendsnum_dataset(friendsnum_file)

# ds_file = ['data/english_foodgroup_new.json', 'data/english_TEDtranslate_new.json', 'data/english_traveladdiction_new.json']
# print about_dataset(ds_file)

"""
# sometimes user id in csv file are truncated
def new_dataset(ds_file):
    # content
    ds_df = pd.read_csv(ds_file, delimiter=',', header=0, error_bad_lines=False)
    arr_convert_to_num = ['PostTextLength', 'LikesCount', 'SharesCount', 'CommentsCount', 'PostTextPolarity', 'PostTextSubjectivity']
    ds_df[arr_convert_to_num] = ds_df[arr_convert_to_num].convert_objects(convert_numeric=True)
    # LikesCount, SharesCount, CommentsCount ada yang g lengkap, dilengkapi dulu dengan nilai interpolated
    ds_df['LikesCount'] = ds_df['LikesCount'].interpolate().astype(int)
    ds_df['SharesCount'] = ds_df['SharesCount'].interpolate().astype(int)
    ds_df['CommentsCount'] = ds_df['CommentsCount'].interpolate().astype(int)
    return ds_df
"""
def user_gender():
    # gender
    gender_file = 'data/user_gender.csv'
    gender_df = pd.read_csv(gender_file, header=0)
    gender_df.Sex[ gender_df.Sex.isnull() ] = gender_df.Sex.dropna().mode().values
    gender_df['Sex'] = gender_df['Sex'].map({'F':0, 'M':1}).astype(int)
    return gender_df

def post_cluster_result(file):
    # content
    # file = 'D:/githubrepository/text-analysis-master/09.01.2016/kmeans_post_result_09012016.csv'
    # file = 'data/kmeans_result_all_features.csv'
    df = pd.read_csv(file, header=0)
    df['resultCluster'] = df['resultCluster'].astype('category', ordered = False)
    return df

def user_cluster_result(file):
    # content
    # file = 'D:/githubrepository/text-analysis-master/09.01.2016/user_kmeans_result.csv'
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

    id_shared_df = df[['userId', 'PostTitle']].values
    array_shared = []
    for row in id_shared_df:
        array_shared.append([row[0], sharedNews(row[1]), uploadedPhoto(row[1]), uploadedVideo(row[1]) ])

    shared_df = pd.DataFrame(array_shared, columns=['userIds', 'SharedNews', 'UploadPhoto', 'UploadVideo'])
    concat_df = pd.concat([df, shared_df], axis=1)

    """
    print df.info()
    print shared_df.info()

    # untuk tanggal
    df['converted_date'] = pd.to_datetime(df['PostTime'])
    mask = ( (df['converted_date'] > '2015-1-1') & (df['converted_date'] < '2015-12-31') )
    df.loc[mask]
    """

    concat_df = concat_df.groupby(['userId'], sort=False).agg({
        'userId':np.min,
        'PostTextLength':np.mean,
        'LikesCount':np.mean,
        'SharesCount':np.mean,
        'CommentsCount':np.mean,
        'SharedNews':np.sum,
        'UploadPhoto':np.sum,
        'UploadVideo':np.sum
    })

    gender_df = user_gender()
    concat_gender_df = pd.merge(concat_df, gender_df, how='left', on='userId')
    return concat_gender_df

def aggr_feature_user(filename, *args):
    # content
    df = new_dataset(filename)
    # df['userId'] = df['UserID']
    # print df.dtypes

    id_shared_df = df[['UserID', 'PostTitle']].values
    array_shared = []
    for row in id_shared_df:
        array_shared.append([row[0], sharedNews(row[1]), uploadedPhoto(row[1]), uploadedVideo(row[1]) ])
    # UserIDtemp is later deleted
    shared_df = pd.DataFrame(array_shared, columns=['UserIDtemp', 'SharedNews', 'UploadPhoto', 'UploadVideo'])
    # data di df dan shared_df juga masih urut
    concat_df = pd.concat([df, shared_df], axis=1)
    concat_df.drop('UserIDtemp', axis=1, inplace=True)

    concat_df = concat_df.groupby(['UserID'], sort=False).agg({
        'PostText':np.count_nonzero,
        'SharedNews':np.sum,
        'UploadVideo':np.sum,
        'PostTextLength':np.mean,
        'LikesCount':np.sum,
        'SharesCount':np.sum,
        'CommentsCount':np.sum
    }).reset_index()
    concat_df.rename(columns={'PostText':'NoPosts','SharedNews':'SharedNewsSum','UploadVideo':'UploadVideoSum','PostTextLength':'PostTextLengthMedian', 'LikesCount':'LikesCountSum','SharesCount':'SharesCountSum','CommentsCount':'CommentsCountSum'}, inplace=True)
    return concat_df

# print aggr_feature_user('data/english_traveladdiction_new.json').dtypes
# averageSum_Feature_User()
# print averageSum_Feature_User()
# user_cluster_result()

# print averageSum_Feature_User()