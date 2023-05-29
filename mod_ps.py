import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from keras.models import Sequential
from keras.layers import Dense, Dropout, Input
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from keras.wrappers.scikit_learn import KerasClassifier
import copy
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from sklearn.model_selection import train_test_split



for i in range(1,101):
    conf_list = pd.read_csv("D:/plasmode/conf_list.csv", index_col=0)
    cov_list = pd.read_csv("D:/plasmode/cov_list.csv", index_col=0)
    
    simdata = pd.read_csv("D:/plasmode/data_{}.csv".format(i), index_col=0)

    simdata = simdata.dropna()

    data = pd.DataFrame(copy.deepcopy(simdata))

    n = map(str, conf_list['x'].to_list())
    conf_cols = data.filter(n)

    n = map(str, cov_list['x'].to_list())
    cov_cols = data.filter(n)

    t = data["EXPOSURE{}".format(i)]
    
    #tune on 50% of data
    X_train, X_test, y_train, y_test = train_test_split(cov_cols, t, test_size = 0.5, stratify = t, random_state = 43)

    #fit reference method, with all confounders
    logmod_con = LogisticRegression(penalty='none',solver='lbfgs', max_iter=1000)
    
    logmod_con.fit(conf_cols, t)

    #create 10 fold CV for all data drive methods
    folds = 10

    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=1001)
    
    #-------------LASSO-------------
    logistic = LogisticRegression(penalty='l1', max_iter=1000)
    
    alphas = np.arange(0.01, 0.31, 0.02)
    
    hyperparameters = dict(C=alphas,solver=['liblinear','saga'])
    
    clf_ps = RandomizedSearchCV(estimator = logistic, param_distributions = hyperparameters, 
                                cv=skf.split(X_train, y_train), verbose=0,
                                scoring='neg_brier_score', n_iter=100)
    
    logmod_ps = clf_ps.fit(X_train, y_train)

    logmod_ps.fit(cov_cols, t)

    #-------------MLP-------------
    def build_classifier(optimizer, kernel, units, hidden_layers, activation):
        classifier = Sequential()
        # First Hidden Layer
        # classifier.add(Input(shape=(features_all.shape[1],)))
        classifier.add(Dense(units=units, activation='sigmoid', input_dim=cov_cols.shape[1], kernel_initializer=kernel))
        # classifier.add(Dropout(rate=0.1))

        for i in range(int(hidden_layers)):
            # Add one hidden layer
            classifier.add(Dense(units=units, activation=activation, kernel_initializer=kernel))

        # Second  Hidden Layer
        # Output Layer
        classifier.add(Dense(1, activation='sigmoid', kernel_initializer=kernel))
        # Compiling the neural network
        classifier.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
        return classifier


    nn_model_ps = KerasClassifier(build_fn=build_classifier)

    para_nn = {'batch_size': [10, 32, 64],
               'epochs': [10, 100, 1000],
               'optimizer': ['adam', 'rmsprop', 'SGD'],
               'kernel': ['random_normal', 'random_uniform', 'truncated_normal'],
               'units': [8, 32, 64, 128],
               "hidden_layers": [2, 3, 5, 7],
               "activation": ['tanh', 'sigmoid','relu', 'selu']}

    random_search_nn_ps = RandomizedSearchCV(estimator=nn_model_ps,
                                             param_distributions =para_nn,
                                             cv=skf.split(X_train, y_train),
                                             return_train_score=True,
                                             scoring='neg_brier_score', 
                                             n_iter=100)

    random_search_nn_ps.fit(cov_cols, t)

    #-------------XgBoost-------------
    xgb_m = xgb.XGBClassifier(n_jobs=1, objective= 'binary:logistic') 
    
    para_xgb = {
        'n_estimators': [1000, 600, 300, 100],
        'min_child_weight': [1, 10, 50],
        'gamma': [0.5,  2, None],
        'subsample': [0.6, 0.8, 1.0],
        'learning_rate': [0.02, 0.1, 0.2, 0.5],
        'max_depth': [3, 5, 7, 12]
    }

    random_search_xgb_ps = RandomizedSearchCV(xgb_m, 
                                              param_distributions=para_xgb, 
                                              scoring='neg_brier_score',
                                              cv=skf.split(X_train, y_train),
                                              verbose=3, n_iter=100) 

    random_search_xgb_ps.fit(cov_cols, t)

    PS_logcon = logmod_con.predict_proba(conf_cols)[:, 1]

    PS_lasso = logmod_ps.predict_proba(cov_cols)[:, 1]

    PS_nn_model = random_search_nn_ps.predict_proba(cov_cols)[:, 1]

    PS_xgb_m = random_search_xgb_ps.predict_proba(cov_cols)[:, 1]

    cov_cols['ps_log'] = PS_logcon

    cov_cols['ps_lasso'] = PS_lasso

    cov_cols['ps_xgb'] = PS_xgb_m

    cov_cols['ps_nn'] = PS_nn_model

    cov_cols['treatment'] = t

    cov_cols['Y'] = data["EVENT{}".format(i)]

    cov_cols.to_csv(r'D:/plasmode/modelled/output_m{}.csv'.format(i), index = False)
