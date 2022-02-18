#System tools
import time, datetime
import os, io
import pickle, dill
import glob
import sys, getopt, copy
from joblib import Parallel, delayed
import multiprocessing

#Database tools
import sqlalchemy as sa
import psycopg2 as p
import pandas as pd

#Math tools
import numpy as np

#ML tools
from sklearn.preprocessing import normalize, scale, StandardScaler
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold

from sklearn.feature_selection import chi2
from sklearn.feature_selection import f_classif
from skrebate import ReliefF, MultiSURF

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import BernoulliNB, GaussianNB, MultinomialNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
from xgboost.sklearn import XGBClassifier

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import auc, roc_auc_score, roc_curve, precision_recall_curve, average_precision_score
from sklearn.metrics import make_scorer

from sklearn.model_selection import cross_validate
from sklearn.model_selection import LeaveOneOut

from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from hyperopt.pyll.stochastic import sample


#User defined modules
fsDirectory = '/mnt/isilon/masino_lab/boses1/Projects/GC-MS/src/features/'
# fsDirectory = '../features/'
sys.path.append(fsDirectory)
from feature_selection_methods import chi2_fs, anova_fs, relieff_fs, multisurf_fs

#Plotting tools
import matplotlib.pyplot as plt
import seaborn as sns

#Extract command line arguements
try:
    opts,args = getopt.getopt(sys.argv[1:],"a:f:c:e:",["ml_algo=","feature_selection=","class_balance_method=","eval_metric_name="])

except getopt.GetoptError:
    print("Invalid input arguments")
    sys.exit(2)

for opt,arg in opts:
    if opt in ["-a","--ml_algo"]:
        algorithm = arg

    elif opt in ["-f","--feature_selection"]:
        feature_selection_method = arg

    elif opt in ["-c","--class_balance_method"]:
        class_balance_method = arg

    elif opt in ["-e","--eval_metric_name"]:
        eval_metric_name = arg

file_path = '/mnt/isilon/masino_lab/boses1/Projects/GC-MS/src/'
# file_path = '../'

#Names for storing results
algorithm_name = algorithm + "_fs_" + feature_selection_method + "_cb_" + class_balance_method + "_eval_" + eval_metric_name
trials_file_name = algorithm + "_fs_" + feature_selection_method + "_cb_" + class_balance_method + "_eval_" + eval_metric_name + '_trials' + '.pik'
results_file_name = algorithm + "_fs_" + feature_selection_method + "_cb_" + class_balance_method + "_eval_" + eval_metric_name + '_results' + '.pik'
p_val_thresh = 0.05

#Uncomment for line by line memory profiling
# from memory_profiler import profile
# mem_data = open(results_file_name+'memory_profiler.log', 'w+')
# @profile(stream=mem_data)

#Uncomment for time profiling and memory profiling via mprof
# @profile
def load_data():

    with open(file_path + 'data/'+'ml_cv_data.pik', "rb") as f:
        cross_val_data = dill.load(f)

    return cross_val_data

cross_val_data = load_data()


#Define objective function to optimize
#Uncomment for line by line memory profiling
# @profile(stream=mem_data)

#Uncomment for time profiling and memory profiling via mprof
# @profile
def feature_selection(cross_val_data):
    for split in range(len(cross_val_data)):
        cross_val_data[split] = list(cross_val_data[split])

        train_x = cross_val_data[split][0]
        train_y = cross_val_data[split][1]
        test_x = cross_val_data[split][2]
        X_df = pd.DataFrame(train_x, columns = cross_val_data[split][4])

        feature_scores = 'N/A'

        if feature_selection_method == 'no':
            selected_features = X_df.columns

        elif feature_selection_method == 'anovaF':
            selected_features, X_train_df, train_x, test_x = anova_fs(X_df,train_x,test_x,train_y,p_val_thresh)

        elif feature_selection_method == 'reliefF':
            selected_features, feature_scores, train_x, test_x = relieff_fs(X_df,train_x,test_x,train_y)

        elif feature_selection_method == 'multisurf':
            selected_features, feature_scores, train_x, test_x = multisurf_fs(X_df,train_x,test_x,train_y)

        elif feature_selection_method == 'anova_reliefF':
            selected_features_anova, X_train_df, X_train_anova, X_test_anova = anova_fs(X_df,train_x,test_x,train_y,p_val_thresh)
            selected_features, feature_scores, train_x, test_x = relieff_fs(X_train_df,X_train_anova,X_test_anova,train_y)

        elif feature_selection_method == 'anova_multisurf':
            selected_features_anova, X_train_df, X_train_anova, X_test_anova = anova_fs(X_df,train_x,test_x,train_y,p_val_thresh)
            selected_features, feature_scores, train_x, test_x = multisurf_fs(X_train_df,X_train_anova,X_test_anova,train_y)

        cross_val_data[split][0] = train_x
        cross_val_data[split][2] = test_x
        cross_val_data[split][4] = selected_features

    return cross_val_data

rand_state = 23974623
cross_val_data = feature_selection(cross_val_data)

#Define parameter search space for hyperopt
if class_balance_method == 'class_weight':
    param_space = {'learning_rate': hp.uniform('learning_rate', 0.01, 0.5),
        'max_depth': hp.choice('max_depth', list(range(1,100))),
        'n_estimators': hp.choice('n_estimators', list(range(2,2000))),
        'min_child_weight': hp.choice('min_child_weight', np.append(range(1,100),[0.5])),
        'gamma': hp.choice('dtree_gamma', [hp.uniform('dtree_gamma_1', 0.1, 1),hp.choice('dtree_gamma_2', list(range(5,100)))]),
        'subsample': hp.uniform('subsample', 0.1, 1),
        'colsample_bytree':hp.uniform('colsample_bytree', 0.1, 1),
        'reg_alpha':hp.choice('reg_alpha', [0,1e-5, 1e-2, 0.1, 1, 10,100]),
        'reg_lambda':hp.choice('reg_lambda', [0,1e-5, 1e-2, 0.1, 1, 10,100]),
        'scale_pos_weight':hp.choice('scale_pos_weight', np.append(np.linspace(0,1,10),[0.376]))
        }
               
else:
    param_space = {'learning_rate': hp.uniform('learning_rate', 0.01, 0.5),
        'max_depth': hp.choice('max_depth', list(range(1,100))),
        'n_estimators': hp.choice('n_estimators', list(range(2,2000))),
        'min_child_weight': hp.choice('min_child_weight', np.append(range(1,100),[0.5])),
        'gamma': hp.choice('dtree_gamma', [hp.uniform('dtree_gamma_1', 0.1, 1),hp.choice('dtree_gamma_2', list(range(5,100)))]),
        'subsample': hp.uniform('subsample', 0.1, 1),
        'colsample_bytree':hp.uniform('colsample_bytree', 0.1, 1),
        'reg_alpha':hp.choice('reg_alpha', [0,1e-5, 1e-2, 0.1, 1, 10,100]),
        'reg_lambda':hp.choice('reg_lambda', [0,1e-5, 1e-2, 0.1, 1, 10,100])
        }

#Define objective function to optimize
#Uncomment for line by line memory profiling
# @profile(stream=mem_data)

#Uncomment for time profiling and memory profiling via mprof
# @profile
def xgb_objective(params):
    clf = XGBClassifier(n_jobs=-1, random_state = 3984623, **params)
    print(params)

    def compute_cv_metric(split, cross_val_data):
        train_x = cross_val_data[split][0]
        train_y = cross_val_data[split][1]
        test_x = cross_val_data[split][2]
        test_y = cross_val_data[split][3]

        clf.fit(train_x,train_y)

        y_pred_prob = clf.predict_proba(test_x)[:,1]

        # aucpr = average_precision_score(test_y, y_pred_prob[:,1])

        return list(zip(y_pred_prob, test_y))

    eval_result = Parallel()(delayed
                        (compute_cv_metric)(split, cross_val_data)
                            for split in range(len(cross_val_data)))

    y_pred_prob_arr = []
    y_test_arr = []
    for elem in eval_result:
        y_pred_prob_arr.append(elem[0][0])
        y_test_arr.append(elem[0][1])

    if np.isnan(y_pred_prob_arr).any() == True:
        eval_metric = -999

    else:
        if eval_metric_name == 'roc':
            eval_metric = roc_auc_score(y_test_arr, y_pred_prob_arr)

        elif eval_metric_name == 'apr':
            eval_metric = average_precision_score(y_test_arr, y_pred_prob_arr)

        elif eval_metric_name == 'accuracy':
            y_pred = [int(i>=0.5) for i in y_pred_prob_arr]
            eval_metric = accuracy_score(y_test_arr, y_pred)

    return {'loss':-eval_metric, 'params':params, 'prediction_proba': y_pred_prob_arr, 'y_test': y_test_arr, 'status':STATUS_OK}

rand_state = np.random.RandomState(314)

#Function to run and store a single trial
#Uncomment for line by line memory profiling
# @profile(stream=mem_data)

#Uncomment for time profiling and memory profiling via mprof
# @profile
def run_trials():
    trials_step = 1  # how many additional trials to do after loading saved trials.
    max_trials = 1  # initial max_trials.

    try:  # try to load an already saved trials object, and increase the max
        trials = pickle.load(open(file_path + 'train/trials/' + trials_file_name, "rb"))
        print("Found saved Trials! Loading...")
        max_trials = len(trials.trials) + trials_step
        print("Rerunning from {} trials to {} (+{}) trials".format(len(trials.trials), max_trials, trials_step))
    except:  # create a new trials object and start searching
        trials = Trials()

    #Optimize
    best = fmin(fn=xgb_objective, space=param_space, algo=tpe.suggest, max_evals=max_trials, trials=trials, rstate = rand_state)

    # save the trials object
    with open(file_path + 'train/trials/' + trials_file_name, "wb") as f:
        pickle.dump(trials, f, protocol=pickle.HIGHEST_PROTOCOL)

    return trials

#Run trials
#Uncomment for line by line memory profiling
# @profile(stream=mem_data)

#Uncomment for time profiling and memory profiling via mprof
# @profile
def get_trials(num_trials = 1000):
    for _ in range(num_trials):
        trials = run_trials()
    return trials

trials = get_trials()

#Sort the results in ascending order of loss (which is a negative AP)
bayes_trials_results = sorted(trials.results, key = lambda x: x['loss'])


results = {'algo':algorithm_name, 'cv_data':cross_val_data, 'trials': trials.trials, 'opt_param': bayes_trials_results[0]['params'],
           'param_sorted': bayes_trials_results
            }


def write_results(results):
    with open(file_path + 'train/results/' + results_file_name, "wb") as f:
        pickle.dump(results, f,protocol=pickle.HIGHEST_PROTOCOL)

write_results(results)
