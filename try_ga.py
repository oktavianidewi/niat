import json
import numpy as np
import pandas as pd
import load_all_dataset as ld
from sklearn.cross_validation import train_test_split, cross_val_score
from sklearn.naive_bayes import GaussianNB

def fitness_old(individu):
    # rumus F1
    # print individu
    polarity_zone = individu
    subjectivity_zone = [0.0, 0.5, 1.0]

    # Load the text dataset, drop null non-text, assign UserId
    all_df = ld.all_post()
    all_df = all_df[['postId', 'LikesCount', 'SharesCount', 'CommentsCount', 'PostTextLength', 'PostTextPolarity', 'PostTextSubjectivity']].dropna()
    all_df['UserId'] = all_df['postId'].str.split('_').str.get(0).astype(int)
    all_df['zone'] = 'a'

    for index_sub, subjectivity in enumerate(subjectivity_zone):
        # print subjectivity
        if index_sub < len(subjectivity_zone)-1:
            for index_pol, polarity in enumerate(polarity_zone):
                # print index_pol, polarity
                nama_zone = str(index_sub)+'_'+str(index_pol)
                if polarity[1] < 0.0:
                    # jika polarity < 0
                    all_df['zone'].loc[ ((all_df['PostTextPolarity'] < polarity[0]) & (all_df['PostTextPolarity'] >= polarity[1])) &
                                        ((all_df['PostTextSubjectivity'] >= subjectivity_zone[index_sub] ) & (all_df['PostTextSubjectivity'] <= subjectivity_zone[index_sub+1])) ] = nama_zone
                elif polarity[1] == 0.0:
                    # jika polarity = 0
                    all_df['zone'].loc[ ( all_df['PostTextPolarity'] == 0.0 ) &
                                        ((all_df['PostTextSubjectivity'] >= subjectivity_zone[index_sub] ) & (all_df['PostTextSubjectivity'] <= subjectivity_zone[index_sub+1])) ] = nama_zone
                else:
                    all_df['zone'].loc[ ((all_df['PostTextPolarity'] > polarity[0]) & (all_df['PostTextPolarity'] <= polarity[1])) &
                                        ((all_df['PostTextSubjectivity'] >= subjectivity_zone[index_sub] ) & (all_df['PostTextSubjectivity'] <= subjectivity_zone[index_sub+1])) ] = nama_zone

    zone_dummies = pd.get_dummies(all_df['zone'])
    column_zone_dummies = zone_dummies.columns.tolist()
    dict_aggr = {x:np.sum for x in column_zone_dummies}
    all_df = all_df.join(zone_dummies)
    aggr_df = all_df.groupby(['UserId'], sort=True).agg(dict_aggr).reset_index()

    # Load golden standard file
    gs_df = pd.read_csv('data/userlevel_all_features_1007.csv', header=0)

    # Merge zone file and golden standard
    aggr_df = pd.merge(aggr_df, gs_df, how='inner', left_on='UserId', right_on='UserNum')

    # Separate golden standard column name
    target_dummies = pd.get_dummies(aggr_df['ActiveInterests'])
    aggr_df = aggr_df.join(target_dummies)
    target = sorted( list( set( target_dummies.columns.tolist() ) - set( ['Random'] ) ) )

    models = [GaussianNB()]
    names = ["Gaussian Naive Bayes"]
    gabung = zip(models, names)
    numCV = 5
    fscore_arr = []
    for model, name in gabung:
        # print name
        for i, ai in enumerate(target):
            # cari F1 score
            # print ai
            fscore_arr.append( cross_val_score(model, aggr_df[column_zone_dummies], aggr_df[target][ai], cv=numCV, scoring='f1_macro') )
            # round_feature_important = map(lambda x:round(x, 4), rf.feature_importances_)
            # map(lambda x:round(x, 3), correlation)
        # print fscore_arr
    rerataf1 = round(np.mean(fscore_arr), 3)
    return rerataf1

def threshold_zones(all_df, polarity_zone, subjectivity_zone):
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
                elif polarity[1] == 0.0:
                    # jika polarity = 0
                    all_df['zone'].loc[(all_df['PostTextPolarity'] == 0.0) &
                                       ((all_df['PostTextSubjectivity'] >= subjectivity_zone[index_sub]) & (
                                       all_df['PostTextSubjectivity'] <= subjectivity_zone[index_sub + 1]))] = nama_zone
                else:
                    all_df['zone'].loc[
                        ((all_df['PostTextPolarity'] > polarity[0]) & (all_df['PostTextPolarity'] <= polarity[1])) &
                        ((all_df['PostTextSubjectivity'] >= subjectivity_zone[index_sub]) & (
                        all_df['PostTextSubjectivity'] <= subjectivity_zone[index_sub + 1]))] = nama_zone
    return all_df

def dataset(type, polarity_zone, subjectivity_zone):
    if type == 'old':
        # Load the text dataset, drop null non-text, assign UserId
        all_df = ld.all_post()
        all_df = all_df[['postId', 'LikesCount', 'SharesCount', 'CommentsCount', 'PostTextLength', 'PostTextPolarity', 'PostTextSubjectivity']].dropna()
        all_df['UserId'] = all_df['postId'].str.split('_').str.get(0).astype(int)
        all_df['zone'] = 'a'

        all_df = threshold_zones(all_df, polarity_zone, subjectivity_zone)

        zone_dummies = pd.get_dummies(all_df['zone'])
        column_zone_dummies = zone_dummies.columns.tolist()
        dict_aggr = {x:np.sum for x in column_zone_dummies}
        all_df = all_df.join(zone_dummies)
        aggr_df = all_df.groupby(['UserId'], sort=True).agg(dict_aggr).reset_index()

        # Load golden standard file
        gs_df = pd.read_csv('data/userlevel_all_features_1007.csv', header=0)

        # Merge zone file and golden standard
        aggr_df = pd.merge(aggr_df, gs_df, how='inner', left_on='UserId', right_on='UserNum')
        aggr_df = aggr_df[['UserId', 'ActiveInterests']+column_zone_dummies]

        # harusnya digabung dulu, baru activeinterests dibikin dummies
    elif type == 'new':
        # Load the new dataset, drop null non-text, assign UserId
        ds_file_array = ['data/english_foodgroup_new.json', 'data/english_TEDtranslate_new.json',
                         'data/english_traveladdiction_new.json']
        all_df = ld.new_dataset(ds_file_array)
        # print all_df.dtypes
        all_df = all_df[['UserID', 'LikesCount', 'SharesCount', 'CommentsCount', 'PostTextLength', 'PostTextPolarity','PostTextSubjectivity', 'ActiveInterests']].dropna()
        all_df['UserId'] = all_df['UserID']
        all_df['zone'] = 'a'
        all_df = threshold_zones(all_df, polarity_zone, subjectivity_zone)

        zone_dummies = pd.get_dummies(all_df['zone'])
        column_zone_dummies = zone_dummies.columns.tolist()
        dict_aggr = {x: np.sum for x in column_zone_dummies}
        dict_aggr.update({'ActiveInterests': np.min})
        all_df = all_df.join(zone_dummies)
        aggr_df = all_df.groupby(['UserId'], sort=True).agg(dict_aggr).reset_index()
    """
    target_dummies = pd.get_dummies(aggr_df['ActiveInterests'])
    aggr_df = aggr_df.join(target_dummies)
    target = sorted(list(set(target_dummies.columns.tolist()) - set(['Random'])))
    """

    return aggr_df, column_zone_dummies #, target


# belum di edit
def fitness(individu):
    # rumus F1
    # print individu
    polarity_zone = individu
    subjectivity_zone = [0.0, 0.5, 1.0]

    # merge between 2 datasets
    aggr_df_old, column_zone_dummies = dataset('old', polarity_zone, subjectivity_zone)
    aggr_df_new, column_zone_dummies_new = dataset('new', polarity_zone, subjectivity_zone)
    cols = list(aggr_df_old.columns.values)
    aggr_df = aggr_df_old.append(aggr_df_new[cols], ignore_index=True)

    # Separate golden standard column name
    target_dummies = pd.get_dummies(aggr_df['ActiveInterests'])
    # print 'target dummies ', target_dummies
    aggr_df = aggr_df.join(target_dummies)
    target = sorted( list( set( target_dummies.columns.tolist() ) - set( ['Random'] ) ) )

    # print 'total data: ', len(aggr_df)

    models = [GaussianNB()]
    names = ["Gaussian Naive Bayes"]
    # models = [LogisticRegression()]
    # names = ["Logistic Regression"]
    gabung = zip(models, names)
    numCV = 5
    fscore_arr = []
    for model, name in gabung:
        # print name
        for i, ai in enumerate(target):
            # cari F1 score
            # print ai
            fscore_arr.append( cross_val_score(model, aggr_df[column_zone_dummies], aggr_df[target][ai], cv=numCV, scoring='f1_macro') )
            # round_feature_important = map(lambda x:round(x, 4), rf.feature_importances_)
            # map(lambda x:round(x, 3), correlation)
        # print fscore_arr
    rerataf1 = round(np.mean(fscore_arr), 3)
    return rerataf1

# zones = [[-0.0, -1.0], [0.0, 0.0], [0.0, 1.0]]
# zones = [[-0.0, -0.371], [-0.371, -1.0], [0.0, 0.0], [0.0, 0.4], [0.4, 1.0]]
# print fitness(zones)
# quit()

filename = 'hasil_2016112523_1 bagi 2_0.8.json'
print filename
population_history = json.load(open(filename))
for gen, value in enumerate(population_history):
    if gen == 19:
        print 'Generasi-', gen
        hitung_fitness = [ fitness(item_history_populasi) for item_history_populasi in population_history[gen] ]
        index_fittest = hitung_fitness.index(max(hitung_fitness))
        print hitung_fitness
        print population_history[gen]
        print population_history[gen][index_fittest]
