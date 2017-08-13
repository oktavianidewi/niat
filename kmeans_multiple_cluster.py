# cluster level 2
# plot masing2 user punya post apa aja

import load_all_dataset as ld
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split, cross_val_score, StratifiedShuffleSplit
from sklearn.metrics import mean_squared_error, accuracy_score, recall_score, precision_score, f1_score
import time
from sklearn.feature_selection import RFE
from sklearn.preprocessing import StandardScaler, RobustScaler
import warnings
warnings.filterwarnings('ignore')
from sklearn.cluster import KMeans

from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import statsmodels.formula.api as smf
import statsmodels.api as sm
from statsmodels.iolib.summary2 import summary_col

def normalize(df):
    # columns = ['po', 'ps', 'nto', 'nts', 'ngo', 'ngs']
    normal_df = (df[zone_columns] - df[zone_columns].min())/(df[zone_columns].max() - df[zone_columns].min())
    return normal_df

def scaler_modified(df):
    # columns = ['po', 'ps', 'nto', 'nts', 'ngo', 'ngs']
    standard_scaler = StandardScaler()
    scaled_df = pd.DataFrame(standard_scaler.fit_transform(df[zone_columns]), columns=[zone_columns])
    new_column = [x+'_scaled' for x in zone_columns]
    df[new_column] = scaled_df
    return df, new_column

def robust_modified(df):
    robust_scaler = RobustScaler()
    robust_df = pd.DataFrame(robust_scaler.fit_transform(df[zone_columns]), columns=[zone_columns])
    new_column = [x+'_robust' for x in zone_columns]
    df[new_column] = robust_df
    return df, new_column

def log_modified(df):
    # log function
    initial_column = df.columns.tolist()
    for column_name in initial_column:
        # print column_name
        df[column_name+'_log'] = np.log(df[column_name])
    new_column = [x+'_log' for x in initial_column]
    df = df.replace(-np.inf, 0)
    return df, new_column

def no_modified(df):
    new_column = [x for x in zone_columns]
    return df, new_column

def ratio_modified(df):
    total = 0
    for i in zone_columns:
        total += df[i]
    df['total_post'] = total
    for i in zone_columns:
        df[i+'_percent'] = df[i]/df['total_post']
    new_column = [x+'_percent' for x in zone_columns]
    return df, new_column
def smoothing():
    # piye carane
    pass

def entropy_features_(df, ent_):
    # entropy, kalo udah percentage
    df['entropy'] = 0
    if ent_ != 0:
        for x in ent_:
            df[x + '_log2'] = np.log2(df[x])
            df = df.replace(-np.inf, 0)
            df['entropy'] += -(df[x]) * df[x + '_log2']
    else:
        df['entropy'] = 0
    # entropy_column_name = [x+'_log2' for x in ent_polarity]+[y+'_log2' for y in ent_subjectivity]+['entropy_polarity', 'entropy_subjectivity']
    return df  # [entropy_column_name]


def entropy_features(df, ent_polarity, ent_subjectivity):
    # entropy, kalo udah percentage
    df['entropy_polarity'] = 0
    df['entropy_subjectivity'] = 0
    if ent_polarity != 0:
        for x in ent_polarity:
            df[x+'_log2'] = np.log2(df[x])
            df = df.replace(-np.inf, 0)
            df['entropy_polarity'] += -(df[x]) * df[x+'_log2']
    else:
        df['entropy_polarity'] = 0

    if ent_subjectivity != 0:
        for y in ent_subjectivity:
            df[y+'_log2'] = np.log2(df[y])
            df = df.replace(-np.inf, 0)
            df['entropy_subjectivity'] += -(df[y]) * df[y+'_log2']
    else:
        df['entropy_subjectivity'] = 0

    # entropy_column_name = [x+'_log2' for x in ent_polarity]+[y+'_log2' for y in ent_subjectivity]+['entropy_polarity', 'entropy_subjectivity']
    return df #[entropy_column_name]

def bestZones(all_df):
    # 3 polarity and 2 subjectivity
    subjectivity_zone = [0.0, 0.5, 1.0]
    # polarity_zone = [[-0.0, -0.448], [-0.448, -1.0], [0.0, 0.0], [0.0, 0.445], [0.445, 1.0]]
    polarity_zone = [[-0.0, -0.442], [-0.442, -1.0], [0.0, 0.0], [0.0, 0.4], [0.4, 1.0]]
    all_df['zone'] = 'a'
    all_df['entropy_all'] = 0
    for index_sub, subjectivity in enumerate(subjectivity_zone):
        # print subjectivity
        if index_sub < len(subjectivity_zone)-1:
            for index_pol, polarity in enumerate(polarity_zone):
                # print index_pol, polarity
                nama_zone = str(index_sub)+'_'+str(index_pol)
                # print subjectivity, polarity, nama_zone
                if polarity[1] < 0.0:
                    # jika polarity < 0
                    all_df['zone'].loc[ ((all_df['PostTextPolarity'] < polarity[0]) & (all_df['PostTextPolarity'] >= polarity[1])) &
                                        ((all_df['PostTextSubjectivity'] >= subjectivity_zone[index_sub] ) & (all_df['PostTextSubjectivity'] <= subjectivity_zone[index_sub+1])) ] = nama_zone

                    all_df['entropy_all'].loc[((all_df['PostTextPolarity'] < polarity[0]) & (all_df['PostTextPolarity'] >= polarity[1])) &
                                              ((all_df['PostTextSubjectivity'] >= subjectivity_zone[index_sub]) & (all_df['PostTextSubjectivity'] <= subjectivity_zone[index_sub + 1]))] = nama_zone+'_ratio'

                elif polarity[1] == 0.0:
                    # jika polarity = 0
                    all_df['zone'].loc[ ( all_df['PostTextPolarity'] == 0.0 ) &
                                        ((all_df['PostTextSubjectivity'] >= subjectivity_zone[index_sub] ) & (all_df['PostTextSubjectivity'] <= subjectivity_zone[index_sub+1])) ] = nama_zone

                    all_df['entropy_all'].loc[(all_df['PostTextPolarity'] == 0.0) &
                                              ((all_df['PostTextSubjectivity'] >= subjectivity_zone[index_sub]) & (all_df['PostTextSubjectivity'] <= subjectivity_zone[index_sub + 1]))] = nama_zone+'_ratio'


                else:
                    all_df['zone'].loc[ ((all_df['PostTextPolarity'] > polarity[0]) & (all_df['PostTextPolarity'] <= polarity[1])) &
                                        ((all_df['PostTextSubjectivity'] >= subjectivity_zone[index_sub] ) & (all_df['PostTextSubjectivity'] <= subjectivity_zone[index_sub+1])) ] = nama_zone

                    all_df['entropy_all'].loc[ ((all_df['PostTextPolarity'] > polarity[0]) & (all_df['PostTextPolarity'] <= polarity[1])) &
                                               ((all_df['PostTextSubjectivity'] >= subjectivity_zone[index_sub] ) & (all_df['PostTextSubjectivity'] <= subjectivity_zone[index_sub+1])) ] = nama_zone+'_ratio'

    zone_dummies = pd.get_dummies(all_df['zone'])
    column_zone_dummies = zone_dummies.columns.tolist()
    all_df = all_df.join(zone_dummies)
    zones = column_zone_dummies

    ent_ = [ x+'_ratio' for x in column_zone_dummies ]

    # zone_dummies = pd.get_dummies(all_df['zone'])
    # all_df = all_df.join(zone_dummies)

    ent_dummies = pd.get_dummies(all_df['entropy_all'])
    all_df = all_df.join(ent_dummies)

    dict_aggr = {x:np.sum for x in column_zone_dummies}
    dict_aggr.update({x:np.mean for x in ent_})
    aggr_df = all_df.groupby(['UserId'], sort=True).agg(dict_aggr).reset_index()
    # add entropy features
    aggr_df = entropy_features_(aggr_df, ent_)

    return aggr_df, zones

def threshold_zones(all_df, polarity_zone, subjectivity_zone):
    all_df['entropy_all'] = 0
    for index_sub, subjectivity in enumerate(subjectivity_zone):
        # print subjectivity
        if index_sub < len(subjectivity_zone) - 1:
            for index_pol, polarity in enumerate(polarity_zone):
                # print index_pol, polarity
                nama_zone = str(index_sub) + '_' + str(index_pol)
                if polarity[1] < 0.0:
                    # jika polarity < 0
                    all_df['zone'].loc[
                        ((all_df['PostTextPolarity'] < polarity[0]) & (all_df['PostTextPolarity'] >= polarity[1])) &
                        ((all_df['PostTextSubjectivity'] >= subjectivity_zone[index_sub]) & (
                            all_df['PostTextSubjectivity'] <= subjectivity_zone[index_sub + 1]))] = nama_zone

                    all_df['entropy_all'].loc[
                        ((all_df['PostTextPolarity'] < polarity[0]) & (all_df['PostTextPolarity'] >= polarity[1])) &
                        ((all_df['PostTextSubjectivity'] >= subjectivity_zone[index_sub]) & (
                            all_df['PostTextSubjectivity'] <= subjectivity_zone[index_sub + 1]))] = nama_zone + '_ratio'

                elif polarity[1] == 0.0:
                    # jika polarity = 0
                    all_df['zone'].loc[(all_df['PostTextPolarity'] == 0.0) &
                                       ((all_df['PostTextSubjectivity'] >= subjectivity_zone[index_sub]) & (
                                           all_df['PostTextSubjectivity'] <= subjectivity_zone[index_sub + 1]))] = nama_zone

                    all_df['entropy_all'].loc[(all_df['PostTextPolarity'] == 0.0) &
                                              ((all_df['PostTextSubjectivity'] >= subjectivity_zone[index_sub]) & (
                                                  all_df['PostTextSubjectivity'] <= subjectivity_zone[
                                                      index_sub + 1]))] = nama_zone + '_ratio'
                else:
                    all_df['zone'].loc[
                        ((all_df['PostTextPolarity'] > polarity[0]) & (all_df['PostTextPolarity'] <= polarity[1])) &
                        ((all_df['PostTextSubjectivity'] >= subjectivity_zone[index_sub]) & (
                            all_df['PostTextSubjectivity'] <= subjectivity_zone[index_sub + 1]))] = nama_zone

                    all_df['entropy_all'].loc[
                        ((all_df['PostTextPolarity'] > polarity[0]) & (all_df['PostTextPolarity'] <= polarity[1])) &
                        ((all_df['PostTextSubjectivity'] >= subjectivity_zone[index_sub]) & (
                            all_df['PostTextSubjectivity'] <= subjectivity_zone[index_sub + 1]))] = nama_zone + '_ratio'

    return all_df

def dataset(type, polarity_zone, subjectivity_zone):
    if type == 'old':
        # Load the text dataset, drop null non-text, assign UserId
        all_df = ld.all_post()
        all_df = all_df[['postId', 'LikesCount', 'SharesCount', 'CommentsCount', 'PostTextLength', 'PostTextPolarity', 'PostTextSubjectivity', 'PostTime']].dropna()
        all_df['UserId'] = all_df['postId'].str.split('_').str.get(0).astype(int)
        all_df['zone'] = 'a'

        all_df = threshold_zones(all_df, polarity_zone, subjectivity_zone)

        zone_dummies = pd.get_dummies(all_df['zone'])
        column_zone_dummies = zone_dummies.columns.tolist()

        ent_ = [x + '_ratio' for x in column_zone_dummies]
        ent_dummies = pd.get_dummies(all_df['entropy_all'])

        all_df = all_df.join(ent_dummies)
        all_df = all_df.join(zone_dummies)

        dict_aggr = {x:np.sum for x in column_zone_dummies}
        dict_aggr.update({x: np.mean for x in ent_})

        aggr_df = all_df.groupby(['UserId'], sort=True).agg(dict_aggr).reset_index()

        # add PostTextLengthLevel
        postTextLengthLevel_df = ld.separate_postTextLength(all_df)
        aggr_df = pd.merge(aggr_df, postTextLengthLevel_df, how='inner', left_on='UserId', right_on='UserId')

        # add day part
        day_part_df = ld.func_day_part(all_df)
        aggr_df = pd.merge(aggr_df, day_part_df, how='inner', left_on='UserId', right_on='UserId')

        # Load golden standard file
        gs_df = pd.read_csv('data/userlevel_all_features_1007.csv', header=0)

        # Merge zone file and golden standard
        aggr_df = pd.merge(aggr_df, gs_df, how='inner', left_on='UserId', right_on='UserNum')
        # aggr_df = aggr_df[['UserId', 'ActiveInterests']+column_zone_dummies+ent_]

        # harusnya digabung dulu, baru activeinterests dibikin dummies
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
        all_df = all_df[['UserID', 'LikesCount', 'SharesCount', 'CommentsCount', 'PostTextLength', 'PostTextPolarity','PostTextSubjectivity', 'ActiveInterests', 'PostTime']].dropna()
        all_df['UserId'] = all_df['UserID']

        all_df['zone'] = 'a'
        all_df = threshold_zones(all_df, polarity_zone, subjectivity_zone)

        zone_dummies = pd.get_dummies(all_df['zone'])
        column_zone_dummies = zone_dummies.columns.tolist()

        ent_ = [x + '_ratio' for x in column_zone_dummies]
        ent_dummies = pd.get_dummies(all_df['entropy_all'])

        all_df = all_df.join(ent_dummies)
        all_df = all_df.join(zone_dummies)

        dict_aggr = {x: np.sum for x in column_zone_dummies}
        dict_aggr.update({'ActiveInterests': np.min})
        dict_aggr.update({x: np.mean for x in ent_})

        aggr_df = all_df.groupby(['UserId'], sort=True).agg(dict_aggr).reset_index()

        # add PostTextLengthLevel
        postTextLengthLevel_df = ld.separate_postTextLength(all_df)
        aggr_df = pd.merge(aggr_df, postTextLengthLevel_df, how='inner', left_on='UserId', right_on='UserID')

        # obtain NoPosts, SharedNewsSum, UploadVideoSum
        df_1 = ld.aggr_feature_user(ds_file_array)
        aggr_df = pd.merge(aggr_df, df_1, how='inner', left_on='UserId', right_on='UserID')

        # obtain about
        df_2 = ld.about_dataset(ds_file_array)
        aggr_df = pd.merge(aggr_df, df_2, how='inner', left_on='UserId', right_on='UserID')

        # UserID, NoProfilePhotos, NoCoverPhotos, NoUploadedPhotos, NoPhotos
        df_3 = ld.photo_dataset(photo_file)
        aggr_df = pd.merge(aggr_df, df_3, how='inner', left_on='UserId', right_on='UserID')

        # NumOfFriends
        df_4 = ld.friendsnum_dataset(friendsnum_file)
        aggr_df = pd.merge(aggr_df, df_4, how='inner', left_on='UserId', right_on='UserID')

        # day_part
        df_5 = ld.func_day_part(all_df)
        aggr_df = pd.merge(aggr_df, df_5, how='inner', left_on='UserId', right_on='UserId')

        aggr_df.drop(['userId', 'UserID', 'UserID_y', 'UserID_x'], axis=1, inplace=True)
        # print 'data baru kolom', aggr_df.dtypes

    aggr_df['frequent_day_part'] = aggr_df['frequent_day_part'].map(
        {'Early_Morning': 0, 'Morning': 1, 'Afternoon': 2, 'Evening': 3})
    # aggr_df = all_df.groupby(['UserId'], sort=True).agg(dict_aggr).reset_index()
    # add entropy features
    aggr_df = entropy_features_(aggr_df, ent_)
    return aggr_df, column_zone_dummies #, target

# return in nilai shortposts dan longposts
# *************************************** #
# """
polarity_zone = [[-0.0, -0.442], [-0.442, -1.0], [0.0, 0.0], [0.0, 0.4], [0.4, 1.0]]
subjectivity_zone = [0.0, 0.5, 1.0]

# merge between 2 datasets
aggr_df_old, column_zone_dummies = dataset('old', polarity_zone, subjectivity_zone)
aggr_df_new, column_zone_dummies_new = dataset('new', polarity_zone, subjectivity_zone)
cols = list(aggr_df_new.columns.values)
aggr_df = aggr_df_new.append(aggr_df_old[cols], ignore_index=True)

# Separate golden standard column name
target_dummies = pd.get_dummies(aggr_df['ActiveInterests'])
# print 'target dummies ', target_dummies
aggr_df = aggr_df.join(target_dummies)
target = sorted( list( set( target_dummies.columns.tolist() ) - set( ['Random'] ) ) )

zone_columns = column_zone_dummies
# print cols# cluster level 2
# plot masing2 user punya post apa aja

import load_all_dataset as ld
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split, cross_val_score, StratifiedShuffleSplit
from sklearn.metrics import mean_squared_error, accuracy_score, recall_score, precision_score, f1_score
import time
from sklearn.feature_selection import RFE
from sklearn.preprocessing import StandardScaler, RobustScaler
import warnings
warnings.filterwarnings('ignore')
from sklearn.cluster import KMeans

from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import statsmodels.formula.api as smf
import statsmodels.api as sm
from statsmodels.iolib.summary2 import summary_col

def normalize(df):
    # columns = ['po', 'ps', 'nto', 'nts', 'ngo', 'ngs']
    normal_df = (df[zone_columns] - df[zone_columns].min())/(df[zone_columns].max() - df[zone_columns].min())
    return normal_df

def scaler_modified(df):
    # columns = ['po', 'ps', 'nto', 'nts', 'ngo', 'ngs']
    standard_scaler = StandardScaler()
    scaled_df = pd.DataFrame(standard_scaler.fit_transform(df[zone_columns]), columns=[zone_columns])
    new_column = [x+'_scaled' for x in zone_columns]
    df[new_column] = scaled_df
    return df, new_column

def robust_modified(df):
    robust_scaler = RobustScaler()
    robust_df = pd.DataFrame(robust_scaler.fit_transform(df[zone_columns]), columns=[zone_columns])
    new_column = [x+'_robust' for x in zone_columns]
    df[new_column] = robust_df
    return df, new_column

def log_modified(df):
    # log function
    initial_column = df.columns.tolist()
    for column_name in initial_column:
        # print column_name
        df[column_name+'_log'] = np.log(df[column_name])
    new_column = [x+'_log' for x in initial_column]
    df = df.replace(-np.inf, 0)
    return df, new_column

def no_modified(df):
    new_column = [x for x in zone_columns]
    return df, new_column

def ratio_modified(df):
    total = 0
    for i in zone_columns:
        total += df[i]
    df['total_post'] = total
    for i in zone_columns:
        df[i+'_percent'] = df[i]/df['total_post']
    new_column = [x+'_percent' for x in zone_columns]
    return df, new_column
def smoothing():
    # piye carane
    pass

def entropy_features_(df, ent_):
    # entropy, kalo udah percentage
    df['entropy'] = 0
    if ent_ != 0:
        for x in ent_:
            df[x + '_log2'] = np.log2(df[x])
            df = df.replace(-np.inf, 0)
            df['entropy'] += -(df[x]) * df[x + '_log2']
    else:
        df['entropy'] = 0
    # entropy_column_name = [x+'_log2' for x in ent_polarity]+[y+'_log2' for y in ent_subjectivity]+['entropy_polarity', 'entropy_subjectivity']
    return df  # [entropy_column_name]


def entropy_features(df, ent_polarity, ent_subjectivity):
    # entropy, kalo udah percentage
    df['entropy_polarity'] = 0
    df['entropy_subjectivity'] = 0
    if ent_polarity != 0:
        for x in ent_polarity:
            df[x+'_log2'] = np.log2(df[x])
            df = df.replace(-np.inf, 0)
            df['entropy_polarity'] += -(df[x]) * df[x+'_log2']
    else:
        df['entropy_polarity'] = 0

    if ent_subjectivity != 0:
        for y in ent_subjectivity:
            df[y+'_log2'] = np.log2(df[y])
            df = df.replace(-np.inf, 0)
            df['entropy_subjectivity'] += -(df[y]) * df[y+'_log2']
    else:
        df['entropy_subjectivity'] = 0

    # entropy_column_name = [x+'_log2' for x in ent_polarity]+[y+'_log2' for y in ent_subjectivity]+['entropy_polarity', 'entropy_subjectivity']
    return df #[entropy_column_name]

def dataset_raw(type):
    if type == 'old':
        all_df = ld.all_post()
        all_df = all_df[['postId', 'PostTextLength', 'PostTextPolarity','PostTextSubjectivity']].dropna()
        all_df['UserId'] = all_df['postId'].str.split('_').str.get(0).astype(int)

        # Load golden standard file
        gs_df = pd.read_csv('data/userlevel_all_features_1007.csv', header=0)
        gs_df['UserId'] = 0
        gs_df['UserId'] = gs_df['UserNum']
        gs_df = gs_df[['UserId', 'ActiveInterests']]

        # add PostTextLengthLevel
        postTextLengthLevel_df = ld.separate_postTextLength(all_df)
        print postTextLengthLevel_df
        aggr_df = pd.merge(gs_df, postTextLengthLevel_df, how='inner', left_on='UserId', right_on='UserId')

        """
        # add day part
        day_part_df = ld.func_day_part(all_df)
        aggr_df = pd.merge(aggr_df, day_part_df, how='inner', left_on='UserId', right_on='UserId')
        """
        # harusnya digabung dulu, baru activeinterests dibikin dummies
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
        # all_df = all_df[['UserID', 'LikesCount', 'SharesCount', 'CommentsCount', 'PostTextLength', 'PostTextPolarity','PostTextSubjectivity', 'ActiveInterests', 'PostTime']].dropna()
        all_df['UserId'] = all_df['UserID']
        all_df = all_df[['UserId', 'ActiveInterests', 'PostTextLength']].dropna()

        dict_aggr = {'ActiveInterests': np.min}
        aggr_df = all_df.groupby(['UserId'], sort=True).agg(dict_aggr).reset_index()

        # add PostTextLengthLevel
        postTextLengthLevel_df = ld.separate_postTextLength(all_df)
        aggr_df = pd.merge(aggr_df, postTextLengthLevel_df, how='inner', left_on='UserId', right_on='UserId')

        # aggr_df = pd.merge(aggr_df, postTextLengthLevel_df, how='inner', left_on='UserId', right_on='UserID')

        """
        # obtain NoPosts, SharedNewsSum, UploadVideoSum
        df_1 = ld.aggr_feature_user(ds_file_array)
        aggr_df = pd.merge(aggr_df, df_1, how='inner', left_on='UserId', right_on='UserID')

        # obtain about
        df_2 = ld.about_dataset(ds_file_array)
        aggr_df = pd.merge(aggr_df, df_2, how='inner', left_on='UserId', right_on='UserID')

        # UserID, NoProfilePhotos, NoCoverPhotos, NoUploadedPhotos, NoPhotos
        df_3 = ld.photo_dataset(photo_file)
        aggr_df = pd.merge(aggr_df, df_3, how='inner', left_on='UserId', right_on='UserID')

        # NumOfFriends
        df_4 = ld.friendsnum_dataset(friendsnum_file)
        aggr_df = pd.merge(aggr_df, df_4, how='inner', left_on='UserId', right_on='UserID')

        # day_part
        df_5 = ld.func_day_part(all_df)
        aggr_df = pd.merge(aggr_df, df_5, how='inner', left_on='UserId', right_on='UserId')

        aggr_df.drop(['userId', 'UserID', 'UserID_y', 'UserID_x'], axis=1, inplace=True)
        # print 'data baru kolom', aggr_df.dtypes

        aggr_df['frequent_day_part'] = aggr_df['frequent_day_part'].map(
            {'Early_Morning': 0, 'Morning': 1, 'Afternoon': 2, 'Evening': 3})
        # aggr_df = all_df.groupby(['UserId'], sort=True).agg(dict_aggr).reset_index()
        # add entropy features
        aggr_df = entropy_features_(aggr_df, ent_)
        """
    # return all_df
    return aggr_df

all_df_old = dataset_raw('old')
all_df_new = dataset_raw('new')

cols = list(all_df_old.columns.values)
all_df = all_df_new.append(all_df_old[cols], ignore_index=True)

print all_df

# print all_df_new

from tes_cluster import post_userlevel_cluster

df = pd.read_csv('allposts_sentimentscore.csv', sep=None, header=0)
p = post_userlevel_cluster(df)
print p

# gabung p dengan all_df
all_df = pd.merge(all_df, p, how='inner', left_on='UserId', right_on='UserId')
print 'merged all df : ', all_df

# buang kolom2 yang g perlu
quit()

# Separate golden standard column name
target_dummies = pd.get_dummies(aggr_df['ActiveInterests'])
# print 'target dummies ', target_dummies
aggr_df = aggr_df.join(target_dummies)
target = sorted( list( set( target_dummies.columns.tolist() ) - set( ['Random'] ) ) )

zone_columns = column_zone_dummies
# print cols

# List of Behavior Features
# behavior = ['Friends','NoPhotos','NoProfilePhotos','NoCoverPhotos','NoPosts','PostTextLengthMedian','SharedNewsSum','UploadVideoSum','UploadPhotoSum']
demographic = ['GenderCode']
categorical = ['frequent_day_part']
behavior = ['Friends','NoPhotos','NoProfilePhotos','NoCoverPhotos','NoPosts','CommentsCountSum','SharesCountSum','UploadVideoSum','UploadPhotoSum','SharedNewsSum','PostTextLengthMedian','LikesCountSum']

# Preprocessing Sentiment Features
x_df, zone_columns_pp = log_modified(aggr_df[zone_columns])
logbehavior_df, logbehavior = log_modified(aggr_df[behavior])

# x_df, zone_columns_pp = robust_modified(aggr_df[zone_columns])
# x_df, zone_columns_pp = ratio_modified(aggr_df[zone_columns])
aggr_df = aggr_df.join(x_df[zone_columns_pp])
aggr_df = aggr_df.join(logbehavior_df[logbehavior])
# x_df, zone_columns_pp = no_modified(aggr_df[zone_columns])

# tugas!
# task 1: panggil tes_cluster disini, gabungkan sebagai fitur
# task 2: hitung entropy sentiment
# task 3: hitung entropy post text length
# task 4: cluster lagi dengan k-means

# Feature collection
features = zone_columns + ['PostTextLengthMedian'] # + ['entropy']
aggr_df['frequent_day_part'].fillna(aggr_df['frequent_day_part'].mean(), inplace=True)
df_raw = aggr_df[features]
df_norm = (df_raw - df_raw.min())/(df_raw.max() - df_raw.min())
data = df_norm.values

# evaluation
"""
from dunn_sklearn import dunn_

for k in range(2, 15):
    # k = 4
    print k

    kmeans = KMeans(n_clusters=k)
    kmeans.fit(data)
    # transform -> get the distance of each point within a cluster to the cluster center, the index of the lowest value indicates the label class
    distance = kmeans.transform(data)
    centroids = kmeans.cluster_centers_
    labels = kmeans.labels_
    inertia = kmeans.inertia_

    print centroids
    # print labels
    # print len(distance)
    # print distance

    labels_distance = zip(labels, distance)
    labels_distance.sort(key = lambda t: t[0])

    # print labels_distance
    cluster_labels = [ x for x in range(k) ]

    dunn_value = dunn_(cluster_labels, labels_distance)
    print 'dunn value : ', dunn_value

quit()
"""
UserId_ = aggr_df['UserId'].values
k = 3
kmeans = KMeans(n_clusters=k)
kmeans.fit(data)
# transform -> get the distance of each point within a cluster to the cluster center, the index of the lowest value indicates the label class
distance = kmeans.transform(data)
centroids = kmeans.cluster_centers_
labels = kmeans.labels_
inertia = kmeans.inertia_
# cluster
# masukkan hasil ke dataframe lagi untuk diolah
df_labels = pd.DataFrame(zip(UserId_, labels), columns=['UserId', 'ClusterLabels'])
df_norm = df_norm.join(df_labels)
df_norm.to_csv('cluster_result_without_entropy.csv')

df_norm.columns.tolist()
column_name = list(set(df_norm.columns.tolist()) - set(['UserId', 'ClusterLabels']))
aggr_column_name = { x:np.mean for x in column_name }
aggr_column_name.update({'UserId':np.count_nonzero})
aggr_df_norm = df_norm.groupby(['ClusterLabels'], sort=False).agg(aggr_column_name).reset_index()

# print aggr_df_norm.head()

pos = [x for x in range(k)] # list(range(len(aggr_df_norm['0_0'])))
width = 0.05
legend_name = [ 'number of '+x for x in column_name ]
print len(legend_name)
color = ['maroon', 'gray', 'orange', 'black', 'green', 'yellow', 'pink', 'darkgoldenrod', 'magenta', 'red', 'white']

fig, ax = plt.subplots(figsize=(10,5))

# Plotting the bars

for i, v in enumerate(column_name):
    print aggr_df_norm[v]
    plt.bar([p + width*i for p in pos],
        #using df['pre_score'] data,
        aggr_df_norm[v],
        # of width
        width,
        # with alpha 0.5
        alpha=1.0,
        # with color
        color=color[i % len(color) ],
        # with label the first value in first_name
        # label='Cluster_'+aggr_df_norm['ClusterLabels'][i]+''
            )

# Set the y axis label
ax.set_ylabel('Percentage')
ax.set_xlabel('User Clusters')

# Set the chart's title
# ax.set_title('Users Clustered')

# Set the position of the x ticks
ax.set_xticks([p + 4 * width for p in pos])

# Set the labels for the x ticks
ax.set_xticklabels(aggr_df_norm['ClusterLabels'])
ax.set_position([0.1,0.1,0.5,0.8])
# Setting the x-axis and y-axis limits
# plt.xlim(min(pos)-width, max(pos)+width*4)
# plt.ylim([0, max(aggr_df_norm['pre_score'] + aggr_df_norm['mid_score'] + aggr_df_norm['post_score'])] )

# Adding the legend and showing the plot
plt.legend(legend_name, loc='center right', bbox_to_anchor=(1.35, 0.5))
plt.grid()
plt.show()

# List of Behavior Features
# behavior = ['Friends','NoPhotos','NoProfilePhotos','NoCoverPhotos','NoPosts','PostTextLengthMedian','SharedNewsSum','UploadVideoSum','UploadPhotoSum']
demographic = ['GenderCode']
categorical = ['frequent_day_part']
behavior = ['Friends','NoPhotos','NoProfilePhotos','NoCoverPhotos','NoPosts','CommentsCountSum','SharesCountSum','UploadVideoSum','UploadPhotoSum','SharedNewsSum','PostTextLengthMedian','LikesCountSum']

# Preprocessing Sentiment Features
x_df, zone_columns_pp = log_modified(aggr_df[zone_columns])
logbehavior_df, logbehavior = log_modified(aggr_df[behavior])

# x_df, zone_columns_pp = robust_modified(aggr_df[zone_columns])
# x_df, zone_columns_pp = ratio_modified(aggr_df[zone_columns])
aggr_df = aggr_df.join(x_df[zone_columns_pp])
aggr_df = aggr_df.join(logbehavior_df[logbehavior])
# x_df, zone_columns_pp = no_modified(aggr_df[zone_columns])

# tugas!
# task 1: panggil tes_cluster disini, gabungkan sebagai fitur
# task 2: hitung entropy sentiment
# task 3: hitung entropy post text length
# task 4: cluster lagi dengan k-means

# Feature collection
features = zone_columns + ['PostTextLengthMedian'] # + ['entropy']
aggr_df['frequent_day_part'].fillna(aggr_df['frequent_day_part'].mean(), inplace=True)
df_raw = aggr_df[features]
df_norm = (df_raw - df_raw.min())/(df_raw.max() - df_raw.min())
data = df_norm.values

# evaluation
"""
from dunn_sklearn import dunn_

for k in range(2, 15):
    # k = 4
    print k

    kmeans = KMeans(n_clusters=k)
    kmeans.fit(data)
    # transform -> get the distance of each point within a cluster to the cluster center, the index of the lowest value indicates the label class
    distance = kmeans.transform(data)
    centroids = kmeans.cluster_centers_
    labels = kmeans.labels_
    inertia = kmeans.inertia_

    print centroids
    # print labels
    # print len(distance)
    # print distance

    labels_distance = zip(labels, distance)
    labels_distance.sort(key = lambda t: t[0])

    # print labels_distance
    cluster_labels = [ x for x in range(k) ]

    dunn_value = dunn_(cluster_labels, labels_distance)
    print 'dunn value : ', dunn_value

quit()
"""
UserId_ = aggr_df['UserId'].values
k = 3
kmeans = KMeans(n_clusters=k)
kmeans.fit(data)
# transform -> get the distance of each point within a cluster to the cluster center, the index of the lowest value indicates the label class
distance = kmeans.transform(data)
centroids = kmeans.cluster_centers_
labels = kmeans.labels_
inertia = kmeans.inertia_
# cluster
# masukkan hasil ke dataframe lagi untuk diolah
df_labels = pd.DataFrame(zip(UserId_, labels), columns=['UserId', 'ClusterLabels'])
df_norm = df_norm.join(df_labels)
df_norm.to_csv('cluster_result_without_entropy.csv')

df_norm.columns.tolist()
column_name = list(set(df_norm.columns.tolist()) - set(['UserId', 'ClusterLabels']))
aggr_column_name = { x:np.mean for x in column_name }
aggr_column_name.update({'UserId':np.count_nonzero})
aggr_df_norm = df_norm.groupby(['ClusterLabels'], sort=False).agg(aggr_column_name).reset_index()

# print aggr_df_norm.head()

pos = [x for x in range(k)] # list(range(len(aggr_df_norm['0_0'])))
width = 0.05
legend_name = [ 'number of '+x for x in column_name ]
print len(legend_name)
color = ['maroon', 'gray', 'orange', 'black', 'green', 'yellow', 'pink', 'darkgoldenrod', 'magenta', 'red', 'white']

fig, ax = plt.subplots(figsize=(10,5))

# Plotting the bars

for i, v in enumerate(column_name):
    print aggr_df_norm[v]
    plt.bar([p + width*i for p in pos],
        #using df['pre_score'] data,
        aggr_df_norm[v],
        # of width
        width,
        # with alpha 0.5
        alpha=1.0,
        # with color
        color=color[i % len(color) ],
        # with label the first value in first_name
        # label='Cluster_'+aggr_df_norm['ClusterLabels'][i]+''
            )

# Set the y axis label
ax.set_ylabel('Percentage')
ax.set_xlabel('User Clusters')

# Set the chart's title
# ax.set_title('Users Clustered')

# Set the position of the x ticks
ax.set_xticks([p + 4 * width for p in pos])

# Set the labels for the x ticks
ax.set_xticklabels(aggr_df_norm['ClusterLabels'])
ax.set_position([0.1,0.1,0.5,0.8])
# Setting the x-axis and y-axis limits
# plt.xlim(min(pos)-width, max(pos)+width*4)
# plt.ylim([0, max(aggr_df_norm['pre_score'] + aggr_df_norm['mid_score'] + aggr_df_norm['post_score'])] )

# Adding the legend and showing the plot
plt.legend(legend_name, loc='center right', bbox_to_anchor=(1.35, 0.5))
plt.grid()
plt.show()