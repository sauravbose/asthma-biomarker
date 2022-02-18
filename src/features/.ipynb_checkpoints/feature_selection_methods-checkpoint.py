#Database tools
import pandas as pd

#Math tools
import numpy as np
import itertools

#ML tools
from sklearn.feature_selection import chi2
from sklearn.feature_selection import f_classif
from skrebate import ReliefF, MultiSURF
from scipy import stats

def corr_fs(X_df,X_train_all,X_test_all):
    '''Correlation based feature selection (Eliminates features with |corr| > 0.9)'''
    correlations = {}
    col_names = X_df.columns.tolist()

    col_ids = list(range(len(col_names)))
    for col_a, col_b in itertools.combinations(col_ids, 2):
        corr,_ = stats.pearsonr(X_train_all[:, col_a], X_train_all[:, col_b])

        correlations[col_names[col_a] + '__' + col_names[col_b]] = corr

    corr_df = pd.DataFrame.from_dict(correlations, orient='index').reset_index()
    corr_df.columns = ['feature_pair', 'correlation']
    corr_df['abs_correlation'] = np.abs(corr_df.correlation)

    highly_correlated_pair = corr_df.loc[corr_df.abs_correlation > 0.9]

    feat_pair = highly_correlated_pair.feature_pair.values
    feat_pair_corr = highly_correlated_pair.values

    #Drop the second feature in every highly correlated pair
    feat_drop = [i.split('__')[1] for i in feat_pair]
    feat_keep = [i.split('__')[0] for i in feat_pair]

    feat_drop_idx = [list(col_names).index(i) for i in feat_drop]

    X_train = np.delete(X_train_all, feat_drop_idx, axis=1)
    X_test = np.delete(X_test_all, feat_drop_idx, axis=1)

    return feat_pair_corr, feat_keep, X_train, X_test


def chi2_fs(X_df,X_train_all,X_test_all,y_train,p_val_thresh):
    '''Chi2 statistical test for feature selection'''
    #Df with only continuous variables
    cont_data = X_df.loc[:, X_df.apply(lambda x: x.nunique()) >= 1000]

    #Find column indices of continuous features
    cont_data_id =  np.where(np.isin(X_df.columns, cont_data.columns))[0]

    cont_data_colnames = cont_data.columns

    #Remove continuous features for chi-sq
    X_train_fs = np.delete(X_train_all,cont_data_id,1)

    c,p = chi2(X_train_fs,y_train)

    feature_ids = np.where(p<=p_val_thresh)[0]

    #Df with no continuous variables. This is used for chi-sq feature selection
    data_fs = X_df.drop(cont_data.columns,axis=1)

    selected_features = np.append(cont_data_colnames,np.array(data_fs.columns[feature_ids]))
    selected_feature_id = np.where(np.isin(X_df.columns, selected_features))[0]

    #New X_train and X_test matrices
    X_train = X_train_all[:,selected_feature_id]
    X_test = X_test_all[:,selected_feature_id]

    X_train_df = pd.DataFrame(X_train,columns=selected_features)

    return selected_features, X_train_df, X_train, X_test

def anova_fs(X_df,X_train_all,X_test_all,y_train,p_val_thresh):
    '''ANOVA F statistical test for feature selection'''
    f,p = f_classif(X_train_all,y_train)
    feature_ids = np.where(p<=p_val_thresh)[0]


    selected_features = np.array(X_df.columns[feature_ids])

    #New X_train and X_test matrices
    X_train = X_train_all[:,feature_ids]
    X_test = X_test_all[:,feature_ids]

    X_train_df = pd.DataFrame(X_train,columns=selected_features)

    return selected_features, X_train_df, X_train, X_test

def relieff_fs(X_df,X_train_all,X_test_all,y_train):
    '''ReliefF for feature selection'''
    fs = ReliefF(discrete_threshold = 5, n_jobs=1)
    fs.fit(X_train_all, y_train)

    feature_scores = fs.feature_importances_
    feature_ids = np.where(feature_scores>=0)[0]
    selected_features = np.array(X_df.columns[feature_ids])

    #New X_train and X_test matrices
    X_train = X_train_all[:,feature_ids]
    X_test = X_test_all[:,feature_ids]

    return selected_features, feature_scores, X_train, X_test

def multisurf_fs(X_df,X_train_all,X_test_all,y_train):
    '''MultiSURF for feature selection'''
    fs = MultiSURF(discrete_threshold = 5, n_jobs=1)
    fs.fit(X_train_all, y_train)

    feature_scores = fs.feature_importances_
    feature_ids = np.where(feature_scores>=0)[0]
    selected_features = np.array(X_df.columns[feature_ids])

    #New X_train and X_test matrices
    X_train = X_train_all[:,feature_ids]
    X_test = X_test_all[:,feature_ids]

    return selected_features, feature_scores, X_train, X_test
