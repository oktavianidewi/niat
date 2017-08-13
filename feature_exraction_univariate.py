# code base = zoning_addfeat.py
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
        print column_name
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

# task 1 remove the highly correlated pi
def highly_correlated_pi(df, ai):
    # correlation between Active Interests and all features
    correlation = df[features+[ai]].corr()[ai].abs()
    n = zip(correlation.nlargest(3).keys(), correlation.nlargest(3))
    # avoid correlation = 1
    nlargest = [ (x,y) for x,y in n if x!=ai ]
    key, value = nlargest[0]
    return key

def correlated(df, ai):
    # correlation between Active Interests and all features
    topk = 1+1
    correlation = df[list(set(features)-set(zone_columns))+[ai]].corr()[ai].abs()
    correlation_real = df[list(set(features)-set(zone_columns))+[ai]].corr()[ai]
    n_real = zip(correlation.keys(), map(lambda x:round(x, 4), correlation))
    n = zip(correlation.nlargest(topk).keys(), correlation.nlargest(topk))
    # avoid correlation = 1
    nlargest = [ (x,y) for x,y in n if x!=ai ]
    return nlargest, n_real

def sentiment_correlated(df, ai):
    # correlation between Active Interests and all features
    topk = len(zone_columns_pp)
    correlation = df[zone_columns_pp+[ai]].corr()[ai]
    n = zip(correlation.nlargest(topk).keys(), map(lambda x:round(x, 4), correlation.nlargest(topk)) )
    # avoid correlation = 1
    nlargest = [ (x,y) for x,y in n if x!=ai ]
    return nlargest

def fiveZ(all_df):
    # 3 polarity and 2 subjectivity
    all_df['zone_polarity'] = 0
    all_df['zone_subjectivity'] = 0
    all_df['zone_polarity'].loc[ ((all_df['PostTextPolarity'] < 0)) ] = 'neg'
    # zone neu
    all_df['zone_polarity'].loc[ ((all_df['PostTextPolarity'] == 0))  ] = 'neu'
    # zone pos
    all_df['zone_polarity'].loc[ ((all_df['PostTextPolarity'] > 0)) ] = 'pos'
    # 2 subjectivity
    # zone sub
    all_df['zone_subjectivity'].loc[ ((all_df['PostTextSubjectivity'] > 0)) ] = 'sub'
    # zone obj
    all_df['zone_subjectivity'].loc[ ((all_df['PostTextSubjectivity'] == 0))  ] = 'obj'

    zone_dummies_pol = pd.get_dummies(all_df['zone_polarity'])
    zone_dummies_sub = pd.get_dummies(all_df['zone_subjectivity'])
    zone_dummies = zone_dummies_pol.join(zone_dummies_sub)
    dict_aggr = {'UserId':np.min, 'neg':np.sum, 'neu':np.sum, 'pos':np.sum, 'sub':np.sum, 'obj':np.sum}
    return zone_dummies, dict_aggr

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

def zonings(all_df, type):
    all_df['zone'] = 0
    if type == '4p':
        # 0 as positive , best
        # zone po
        all_df['zone'].loc[ ((all_df['PostTextPolarity'] >= 0) & (all_df['PostTextPolarity'] <= 1.0)) &
                            ((all_df['PostTextSubjectivity'] >= 0) & (all_df['PostTextSubjectivity'] <= 0.5)) ] = 'po'
        # zone ps
        all_df['zone'].loc[ ((all_df['PostTextPolarity'] >= 0) & (all_df['PostTextPolarity'] <= 1.0)) &
                            ((all_df['PostTextSubjectivity'] > 0.5) & (all_df['PostTextSubjectivity'] <= 1.0)) ] = 'ps'
        # zone ngo
        all_df['zone'].loc[ ((all_df['PostTextPolarity'] < 0) & (all_df['PostTextPolarity'] >= -1.0)) &
                            ((all_df['PostTextSubjectivity'] >= 0) & (all_df['PostTextSubjectivity'] <= 0.5)) ] = 'ngo'
        # zone ngs
        all_df['zone'].loc[ ((all_df['PostTextPolarity'] < 0) & (all_df['PostTextPolarity'] >= -1.0)) &
                            ((all_df['PostTextSubjectivity'] > 0.5) & (all_df['PostTextSubjectivity'] <= 1.0)) ] = 'ngs'

        all_df['ent_polarity'] = 0
        all_df['ent_subjectivity'] = 0
        all_df['ent_polarity'].loc[ ((all_df['PostTextPolarity'] < 0) & (all_df['PostTextPolarity'] >= -1.0)) ] = 'neg_ratio'
        all_df['ent_polarity'].loc[ ((all_df['PostTextPolarity'] >= 0) & (all_df['PostTextPolarity'] <= 1.0)) ] = 'pos_ratio'
        all_df['ent_subjectivity'].loc[ ((all_df['PostTextSubjectivity'] >= 0) & (all_df['PostTextPolarity'] <= 0.5)) ] = 'obj_ratio'
        all_df['ent_subjectivity'].loc[ ((all_df['PostTextSubjectivity'] > 0.5) & (all_df['PostTextPolarity'] <= 1.0)) ] = 'subj_ratio'

        ent_dummies = pd.get_dummies(all_df['ent_polarity']).join(pd.get_dummies(all_df['ent_subjectivity']))
        zone_dummies = pd.get_dummies(all_df['zone'])

        all_df = all_df.join(zone_dummies)
        all_df = all_df.join(ent_dummies)

        dict_aggr= { 'po':np.sum, 'ps':np.sum, 'ngo':np.sum, 'ngs':np.sum, 'neg_ratio':np.mean, 'pos_ratio':np.mean, 'obj_ratio':np.mean, 'subj_ratio':np.mean }
        ent_polarity = [ 'neg_ratio', 'pos_ratio' ]
        ent_subjectivity = [ 'obj_ratio', 'subj_ratio']
        aggr_df = all_df.groupby(['UserId'], sort=True).agg(dict_aggr).reset_index()
        # add entropy features
        aggr_df = entropy_features(aggr_df, ent_polarity, ent_subjectivity)
        zones = ['po', 'ps', 'ngo', 'ngs']

    elif type == '4n':
        # 0 as negative
        # zone po
        all_df['zone'].loc[ ((all_df['PostTextPolarity'] > 0) & (all_df['PostTextPolarity'] <= 1.0)) &
                            ((all_df['PostTextSubjectivity'] >= 0) & (all_df['PostTextSubjectivity'] <= 0.5)) ] = 'po'
        # zone ps
        all_df['zone'].loc[ ((all_df['PostTextPolarity'] > 0) & (all_df['PostTextPolarity'] <= 1.0)) &
                            ((all_df['PostTextSubjectivity'] > 0.5) & (all_df['PostTextSubjectivity'] <= 1.0)) ] = 'ps'
        # zone ngo
        all_df['zone'].loc[ ((all_df['PostTextPolarity'] <= 0) & (all_df['PostTextPolarity'] >= -1.0)) &
                            ((all_df['PostTextSubjectivity'] >= 0) & (all_df['PostTextSubjectivity'] <= 0.5)) ] = 'ngo'
        # zone ngs
        all_df['zone'].loc[ ((all_df['PostTextPolarity'] <= 0) & (all_df['PostTextPolarity'] >= -1.0)) &
                            ((all_df['PostTextSubjectivity'] > 0.5) & (all_df['PostTextSubjectivity'] <= 1.0)) ] = 'ngs'

        all_df['ent_polarity'] = 0
        all_df['ent_subjectivity'] = 0
        all_df['ent_polarity'].loc[ ((all_df['PostTextPolarity'] <= 0) & (all_df['PostTextPolarity'] >= -1.0)) ] = 'neg_ratio'
        all_df['ent_polarity'].loc[ ((all_df['PostTextPolarity'] > 0) & (all_df['PostTextPolarity'] <= 1.0)) ] = 'pos_ratio'
        all_df['ent_subjectivity'].loc[ ((all_df['PostTextSubjectivity'] >= 0) & (all_df['PostTextPolarity'] <= 0.5)) ] = 'obj_ratio'
        all_df['ent_subjectivity'].loc[ ((all_df['PostTextSubjectivity'] > 0.5) & (all_df['PostTextPolarity'] <= 1.0)) ] = 'subj_ratio'

        ent_dummies = pd.get_dummies(all_df['ent_polarity']).join(pd.get_dummies(all_df['ent_subjectivity']))
        zone_dummies = pd.get_dummies(all_df['zone'])

        all_df = all_df.join(zone_dummies)
        all_df = all_df.join(ent_dummies)

        dict_aggr= { 'po':np.sum, 'ps':np.sum, 'ngo':np.sum, 'ngs':np.sum, 'neg_ratio':np.mean, 'pos_ratio':np.mean, 'obj_ratio':np.mean, 'subj_ratio':np.mean }
        ent_polarity = [ 'neg_ratio', 'pos_ratio' ]
        ent_subjectivity = [ 'obj_ratio', 'subj_ratio']
        aggr_df = all_df.groupby(['UserId'], sort=True).agg(dict_aggr).reset_index()
        # add entropy features
        aggr_df = entropy_features(aggr_df, ent_polarity, ent_subjectivity)
        zones = ['po', 'ps', 'ngo', 'ngs']
    elif type == '2polp':
        # 2 zones 0 to negative
        all_df['zone_polarity'] = 0
        all_df['zone_polarity'].loc[ ((all_df['PostTextPolarity'] < 0)) ] = 'neg'
        all_df['zone_polarity'].loc[ ((all_df['PostTextPolarity'] >= 0)) ] = 'pos'

        all_df['ent_polarity'] = 0
        all_df['ent_subjectivity'] = 0
        all_df['ent_polarity'].loc[ all_df['PostTextPolarity'] < 0  ] = 'neg_ratio'
        all_df['ent_polarity'].loc[ ((all_df['PostTextPolarity'] >= 0) & (all_df['PostTextPolarity'] <= 1.0)) ] = 'pos_ratio'

        ent_dummies = pd.get_dummies(all_df['ent_polarity']).join(pd.get_dummies(all_df['ent_subjectivity']))
        zone_dummies = pd.get_dummies(all_df['zone_polarity'])
        dict_aggr = { 'neg':np.sum, 'pos':np.sum, 'neg_ratio':np.mean, 'pos_ratio':np.mean }

        all_df = all_df.join(zone_dummies)
        all_df = all_df.join(ent_dummies)

        ent_polarity = [ 'neg_ratio', 'pos_ratio' ]
        ent_subjectivity = 0
        aggr_df = all_df.groupby(['UserId'], sort=True).agg(dict_aggr).reset_index()
        # add entropy features
        aggr_df = entropy_features(aggr_df, ent_polarity, ent_subjectivity)
        zones = ['pos', 'neg']
    elif type == '2poln':
        # 2 zones 0 to negative
        all_df['zone_polarity'] = 0
        all_df['zone_polarity'].loc[ ((all_df['PostTextPolarity'] <= 0)) ] = 'neg'
        all_df['zone_polarity'].loc[ ((all_df['PostTextPolarity'] > 0)) ] = 'pos'

        all_df['ent_polarity'] = 0
        all_df['ent_subjectivity'] = 0
        all_df['ent_polarity'].loc[ all_df['PostTextPolarity'] <= 0  ] = 'neg_ratio'
        all_df['ent_polarity'].loc[ ((all_df['PostTextPolarity'] > 0) & (all_df['PostTextPolarity'] <= 1.0)) ] = 'pos_ratio'

        ent_dummies = pd.get_dummies(all_df['ent_polarity']).join(pd.get_dummies(all_df['ent_subjectivity']))
        zone_dummies = pd.get_dummies(all_df['zone_polarity'])
        dict_aggr = { 'neg':np.sum, 'pos':np.sum, 'neg_ratio':np.mean, 'pos_ratio':np.mean }

        all_df = all_df.join(zone_dummies)
        all_df = all_df.join(ent_dummies)

        ent_polarity = [ 'neg_ratio', 'pos_ratio' ]
        ent_subjectivity = 0
        aggr_df = all_df.groupby(['UserId'], sort=True).agg(dict_aggr).reset_index()
        # add entropy features
        aggr_df = entropy_features(aggr_df, ent_polarity, ent_subjectivity)
        zones = ['pos', 'neg']
    elif type == '3pol':
        # 3 zones
        all_df['zone_polarity'] = 0
        all_df['zone_polarity'].loc[ ((all_df['PostTextPolarity'] < 0)) ] = 'neg'
        # zone neu
        all_df['zone_polarity'].loc[ ((all_df['PostTextPolarity'] == 0))  ] = 'neu'
        # zone pos
        all_df['zone_polarity'].loc[ ((all_df['PostTextPolarity'] > 0)) ] = 'pos'

        all_df['ent_polarity'] = 0
        all_df['ent_subjectivity'] = 0
        all_df['ent_polarity'].loc[ ((all_df['PostTextPolarity'] < 0) & (all_df['PostTextPolarity'] >= -1.0)) ] = 'neg_ratio'
        all_df['ent_polarity'].loc[ all_df['PostTextPolarity'] == 0  ] = 'neu_ratio'
        all_df['ent_polarity'].loc[ ((all_df['PostTextPolarity'] > 0) & (all_df['PostTextPolarity'] <= 1.0)) ] = 'pos_ratio'

        ent_dummies = pd.get_dummies(all_df['ent_polarity']).join(pd.get_dummies(all_df['ent_subjectivity']))
        zone_dummies = pd.get_dummies(all_df['zone_polarity'])
        dict_aggr = { 'neg':np.sum, 'neu':np.sum, 'pos':np.sum, 'neg_ratio':np.mean, 'neu_ratio':np.mean, 'pos_ratio':np.mean }

        all_df = all_df.join(zone_dummies)
        all_df = all_df.join(ent_dummies)

        ent_polarity = [ 'neg_ratio', 'neu_ratio', 'pos_ratio' ]
        ent_subjectivity = 0
        aggr_df = all_df.groupby(['UserId'], sort=True).agg(dict_aggr).reset_index()
        # add entropy features
        aggr_df = entropy_features(aggr_df, ent_polarity, ent_subjectivity)
        zones = ['pos', 'neu', 'neg']
    elif type == '6':
        # 6 Zones
        # zone po
        all_df['zone'].loc[ ((all_df['PostTextPolarity'] > 0) & (all_df['PostTextPolarity'] <= 1.0)) &
                            ((all_df['PostTextSubjectivity'] >= 0) & (all_df['PostTextSubjectivity'] <= 0.5)) ] = 'po'
        # zone ps
        all_df['zone'].loc[ ((all_df['PostTextPolarity'] > 0) & (all_df['PostTextPolarity'] <= 1.0)) &
                            ((all_df['PostTextSubjectivity'] > 0.5) & (all_df['PostTextSubjectivity'] <= 1.0)) ] = 'ps'
        # zone nto
        all_df['zone'].loc[ ((all_df['PostTextPolarity'] == 0)) &
                            ((all_df['PostTextSubjectivity'] >= 0) & (all_df['PostTextSubjectivity'] <= 0.5)) ] = 'nto'
        # zone nts
        all_df['zone'].loc[ ((all_df['PostTextPolarity'] == 0)) &
                            ((all_df['PostTextSubjectivity'] > 0.5) & (all_df['PostTextSubjectivity'] <= 1.0)) ] = 'nts'
        # zone ngo
        all_df['zone'].loc[ ((all_df['PostTextPolarity'] < 0) & (all_df['PostTextPolarity'] >= -1.0)) &
                            ((all_df['PostTextSubjectivity'] >= 0) & (all_df['PostTextSubjectivity'] <= 0.5)) ] = 'ngo'
        # zone ngs
        all_df['zone'].loc[ ((all_df['PostTextPolarity'] < 0) & (all_df['PostTextPolarity'] >= -1.0)) &
                            ((all_df['PostTextSubjectivity'] > 0.5) & (all_df['PostTextSubjectivity'] <= 1.0)) ] = 'ngs'

        all_df['ent_polarity'] = 0
        all_df['ent_subjectivity'] = 0
        all_df['ent_polarity'].loc[ ((all_df['PostTextPolarity'] < 0) & (all_df['PostTextPolarity'] >= -1.0)) ] = 'neg_ratio'
        all_df['ent_polarity'].loc[ (all_df['PostTextPolarity'] == 0) ] = 'neu_ratio'
        all_df['ent_polarity'].loc[ ((all_df['PostTextPolarity'] > 0) & (all_df['PostTextPolarity'] <= 1.0)) ] = 'pos_ratio'
        all_df['ent_subjectivity'].loc[ ((all_df['PostTextSubjectivity'] >= 0) & (all_df['PostTextPolarity'] <= 0.5)) ] = 'obj_ratio'
        all_df['ent_subjectivity'].loc[ ((all_df['PostTextSubjectivity'] > 0.5) & (all_df['PostTextPolarity'] <= 1.0)) ] = 'subj_ratio'

        ent_dummies = pd.get_dummies(all_df['ent_polarity']).join(pd.get_dummies(all_df['ent_subjectivity']))
        zone_dummies = pd.get_dummies(all_df['zone'])

        all_df = all_df.join(zone_dummies)
        all_df = all_df.join(ent_dummies)

        dict_aggr= { 'po':np.sum, 'ps':np.sum,'nto':np.sum, 'nts':np.sum, 'ngo':np.sum, 'ngs':np.sum, 'neg_ratio':np.mean, 'neu_ratio':np.mean, 'pos_ratio':np.mean, 'obj_ratio':np.mean, 'subj_ratio':np.mean }
        ent_polarity = [ 'neg_ratio', 'neu_ratio', 'pos_ratio' ]
        ent_subjectivity = [ 'obj_ratio', 'subj_ratio']
        aggr_df = all_df.groupby(['UserId'], sort=True).agg(dict_aggr).reset_index()
        # add entropy features
        aggr_df = entropy_features(aggr_df, ent_polarity, ent_subjectivity)
        zones = ['po', 'ps', 'nto', 'nts', 'ngo', 'ngs']

    return aggr_df, zones

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
                print subjectivity, polarity, nama_zone
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
    # polarity_zone = [[-0.0, -0.442], [-0.442, -1.0], [0.0, 0.0], [0.0, 0.4], [0.4, 1.0]]
    # subjectivity_zone = [0.0, 0.5, 1.0]
    """
    0_0: obj-low_neg_polarity
    0_1: obj-high_neg_polarity
    0_2: obj-neutral
    0_3: obj-low_pos_polarity
    0_4: obj-high_pos_polarity
    1_0: subj-low_neg_polarity
    1_1: subj-high_neg_polarity
    1_2: subj-neutral
    1_3: subj-low_pos_polarity
    1_4: subj-high_pos_polarity
    """
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

        aggr_df.drop(['userId', 'UserID_y', 'UserID_x'], axis=1, inplace=True)
        # print 'data baru kolom', aggr_df.dtypes

    aggr_df['frequent_day_part'] = aggr_df['frequent_day_part'].map(
        {'Early_Morning': 0, 'Morning': 1, 'Afternoon': 2, 'Evening': 3})
    # aggr_df = all_df.groupby(['UserId'], sort=True).agg(dict_aggr).reset_index()
    # add entropy features
    aggr_df = entropy_features_(aggr_df, ent_)
    return aggr_df, column_zone_dummies #, target

# *************************************** #
# """
polarity_zone = [[-0.0, -0.442], [-0.442, -1.0], [0.0, 0.0], [0.0, 0.4], [0.4, 1.0]]
subjectivity_zone = [0.0, 0.5, 1.0]

# merge between 2 datasets
aggr_df_old, column_zone_dummies = dataset('old', polarity_zone, subjectivity_zone)
aggr_df_new, column_zone_dummies_new = dataset('new', polarity_zone, subjectivity_zone)
cols = list(aggr_df_new.columns.values)
aggr_df = aggr_df_new.append(aggr_df_old[cols], ignore_index=True)
# aggr_df = aggr_df_old

# Separate golden standard column name
target_dummies = pd.get_dummies(aggr_df['ActiveInterests'])
# print 'target dummies ', target_dummies
aggr_df = aggr_df.join(target_dummies)
target = sorted( list( set( target_dummies.columns.tolist() ) - set( ['Random'] ) ) )

zone_columns = column_zone_dummies
print cols

# List of Behavior Features
# behavior = ['Friends','NoPhotos','NoProfilePhotos','NoCoverPhotos','NoPosts','PostTextLengthMedian','SharedNewsSum','UploadVideoSum','UploadPhotoSum']
demographic = ['GenderCode']
categorical = ['frequent_day_part']
behavior = ['Friends','NoPhotos','NoProfilePhotos','NoCoverPhotos','NoPosts','CommentsCountSum','SharesCountSum','UploadVideoSum','UploadPhotoSum','SharedNewsSum','PostTextLengthMedian','LikesCountSum']

pi = ['Arts_and_Entertainment', 'Business_and_Services', 'Community_and_Organizations', 'Education_y', 'Health_and_Beauty', 'Home_Foods_and_Drinks', 'Law_Politic_and_Government','News_and_Media', 'Pets_And_Animals', 'Recreation_and_Sports', 'Regional', 'Religion_and_Spirituality', 'Restaurants_and_Bars', 'Technology_Computers_and_Internet', 'Transportation_Travel_and_Tourism']
binary_pi = ['split_Arts_and_Entertainment','split_Business_and_Services','split_Community_and_Organizations','split_Education_y','split_Health_and_Beauty','split_Home_Foods_and_Drinks','split_Law_Politic_and_Government','split_News_and_Media','split_Pets_And_Animals','split_Recreation_and_Sports','split_Regional','split_Religion_and_Spirituality','split_Restaurants_and_Bars','split_Technology_Computers_and_Internet','split_Transportation_Travel_and_Tourism']

# Preprocessing Sentiment Features
x_df, zone_columns_pp = log_modified(aggr_df[zone_columns])
logbehavior_df, logbehavior = log_modified(aggr_df[behavior])

# x_df, zone_columns_pp = robust_modified(aggr_df[zone_columns])
# x_df, zone_columns_pp = ratio_modified(aggr_df[zone_columns])
aggr_df = aggr_df.join(x_df[zone_columns_pp])
aggr_df = aggr_df.join(logbehavior_df[logbehavior])
# x_df, zone_columns_pp = no_modified(aggr_df[zone_columns])

# aggr_df.to_csv('best_zones_log.csv')
# print aggr_df
# quit()

# Feature collection
# features = zone_columns_pp + ['entropy_polarity', 'entropy_subjectivity'] + log_behavior # log_behavior #[] #pi #binary_pi #log_behavior #behavior + pi
features = zone_columns + ['entropy'] + logbehavior + demographic + categorical # log_behavior #[] #pi #binary_pi #log_behavior #behavior + pi
aggr_df['frequent_day_part'].fillna(aggr_df['frequent_day_part'].mean(), inplace=True)

# models = [LogisticRegression(), RandomForestClassifier(n_estimators=100), GaussianNB(), DecisionTreeClassifier()]
# names = ["Logistic Regression", "Random Forests Classifier", "Naive Bayes", "Decision Tree"]

# logistic regression for feature selection
models = [LogisticRegression()]
names = ["Logistic Regression"]
gabung = zip(models, names)

# task 2 : gimana caranya menilai feature importance ?
rows = len(gabung)
cols = len(target)
numCV = 3
print numCV,'-fold CV'
print names
pos = 0

logreg_result = {}
for model, name in gabung:
    # cols = 0
    print name
    print features
    cv_scores = []
    print 'MEAN ACCURACY \t MACRO F1 \t TARGET'
    for i, ai in enumerate(target):
        # filter with logistic regression
        logit = sm.Logit(aggr_df[target][ai], aggr_df[features])
        result = logit.fit()
        print ai
        # print result.summary()
        kk = pd.DataFrame({
            'coefficients': result.params,
            'std': result.bse,
            'pvalue': result.pvalues
        }).reset_index()
        kk = kk.loc[ kk['pvalue'] < 0.10 ]
        kk_value = kk[['index', 'coefficients']].values
        for y,z in kk_value:
            print y ,'\t', z
        logreg_result[ai] = kk_value
    print '\n'

print logreg_result
quit()
# Train multiple classifier
cls = [RandomForestClassifier(n_estimators=100), GaussianNB(), DecisionTreeClassifier()]
cls_names = ["Random Forests Classifier", "Naive Bayes", "Decision Tree"]
cls_gabung = zip(cls, cls_names)

for cls, cls_name in cls_gabung:
    cv_scores = []
    print cls_name
    print 'dengan filter'
    print 'MEAN ACCURACY \t MACRO F1 \t TARGET'
    for i, ai in enumerate(target):
        selected_features = logreg_result[ai]
        cv_model_accuracyf = cross_val_score(cls, aggr_df[selected_features], aggr_df[target][ai], cv=numCV,
                                            scoring='accuracy')
        cv_model_f1_macrof = cross_val_score(cls, aggr_df[selected_features], aggr_df[target][ai], cv=numCV,
                                            scoring='f1_macro')
        round_accuracyf = round(cv_model_accuracyf.mean(), 3)
        round_f1macrof = round(cv_model_f1_macrof.mean(), 3)
        print round_accuracyf, '\t', round_f1macrof, '\t', ai, '\t', selected_features

    print 'tanpa filter'
    for i, ai in enumerate(target):
        selected_features = logreg_result[ai]
        cv_model_accuracy = cross_val_score(cls, aggr_df[features], aggr_df[target][ai], cv=numCV,
                                            scoring='accuracy')
        cv_model_f1_macro = cross_val_score(cls, aggr_df[features], aggr_df[target][ai], cv=numCV,
                                            scoring='f1_macro')
        round_accuracy = round(cv_model_accuracy.mean(), 3)
        round_f1macro = round(cv_model_f1_macro.mean(), 3)
        print round_accuracy, '\t', round_f1macro, '\t', ai, '\t', features
plt.show()