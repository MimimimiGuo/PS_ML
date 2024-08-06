import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from keras.models import Sequential
from keras.layers import Dense, Dropout, Input
# from bartpy.samplers import *
# from bartpy.featureselection import SelectNullDistributionThreshold, SelectSplitProportionThreshold
# from bartpy.sklearnmodel import SklearnModel
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from datetime import datetime
from xbart import XBART
from keras.wrappers.scikit_learn import KerasClassifier
import copy
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample

for i in range(0,20):
    simdata = pd.read_csv("D:/sim_rf/S_025_002/output_m{}.csv".format(i), index_col=0)
    data = pd.DataFrame(copy.deepcopy(simdata))

    if sum(data['Y']==1)/data.shape[0] <= 0.1:
        # separate minority and majority classes
        not_treat = data[data['Y'] == 0]
        treat = data[data['Y'] == 1]

        # upsample minority
        treat_upsampled = resample(treat,
                                   replace=True,  # sample with replacement
                                   n_samples=len(not_treat),  # match number in majority class
                                   random_state=27)  # reproducible results

        # combine majority and upsampled minority
        data = pd.concat([not_treat, treat_upsampled])


    y = data['Y']
    t = data['treatment']



    # features_all = data.iloc[:,0:112] #include all variables and treatment


    features_all_up = data.iloc[:,np.r_[0:15, 20:30, 35:45, 50:76]]
    features_all = simdata.iloc[:,np.r_[0:15, 20:30, 35:45, 50:76]]

    # all_con = data.iloc[:, np.r_[0:100,111]] #include treatment
    all_con_up = data.iloc[:, np.r_[0:15, 20:30, 35:45, 50:65, 76]]
    all_con = simdata.iloc[:, np.r_[0:15, 20:30, 35:45, 50:65, 76]]


    logmod_con = LogisticRegression(penalty='none',solver='lbfgs', max_iter=200)
    logmod_con.fit(all_con_up, y)


    x_train, x_test, y_train, y_test = train_test_split(features_all_up, t, test_size = 0.9, random_state = 4, stratify = t)


    logistic = LogisticRegression(penalty='l1', max_iter=200)
    alphas = np.arange(0.01, 0.31, 0.05)
    hyperparameters = dict(C=alphas,solver=['liblinear','saga'])
    clf = GridSearchCV(logistic, hyperparameters, cv=5, verbose=0, scoring='brier_score_loss')
    logmod = clf.fit(x_train, y_train)
    logmod.best_estimator_.get_params()
    logmod.fit(features_all_up, y)

    folds = 5
    param_comb = 5

    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=1001)

    def build_classifier(optimizer, kernel, units, hidden_layers, activation):
        classifier = Sequential()
        # First Hidden Layer
        # classifier.add(Input(shape=(features_all.shape[1],)))
        classifier.add(Dense(units=units, activation='sigmoid', input_dim=features_all_up.shape[1], kernel_initializer=kernel))
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


    nn_model = KerasClassifier(build_fn=build_classifier)

    # nn_model = Sequential()
    # nn_model.add(Dense(units=units, input_shape=(features_all.shape[1],), activation='sigmoid'))
    # nn_model.add(Dropout(0.1))
    # nn_model.add(Dense(units=units, activation='relu', kernel_initializer=kernel))
    # nn_model.add(Dropout(0.1))
    # nn_model.add(Dense(1, activation='sigmoid'))
    # nn_model.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['binary_accuracy'])
    # nn_model.fit(features_all, y, epochs=1, batch_size=10, verbose=1)

    para_nn = {'batch_size': [5, 8, 10],
               'epochs': [1, 10, 100,1000],
               'optimizer': ['adam', 'rmsprop', 'SGD'],
               'kernel': ['random_normal'],
               'units': [4, 8, 64, 128],
               "hidden_layers": [2, 3, 5, 7],
               "activation": ['tanh', 'sigmoid','relu', 'selu']}

    random_search_nn = RandomizedSearchCV(estimator=nn_model,
                                          param_distributions =para_nn,
                                          cv=skf.split(x_train, y_train),
                                          return_train_score=True,
                                          scoring='brier_score_loss', n_iter=10,
                                          n_jobs=-1)
    random_search_nn.fit(features_all_up, y)

    # bartmod = SklearnModel()  # Use default parameters
    # bartmod.fit(features_all, y)  # Fit the model

    # bartmod = XBART(num_trees=1,num_sweeps = 10, n_min = 2, burnin = 5, model="Probit")
    # y_bart = copy.deepcopy(y)
    # y_bart[y_bart==0] = -1
    # bartmod.fit(features_all_up, y_bart)
    #


    xgb_m = xgb.XGBClassifier(objective= 'binary:logistic') #, learning_rate=0.05, n_estimators=500, bootstrap=False  # min_child_weight=100 n_estimators=1500 , n_estimators=500 ,learning_rate=0.02
    # xgb_m.fit(features_all, y)
    para_xgb = {
        'n_estimators': [600, 300, 100],
        'min_child_weight': [1, 10, 50],
        'gamma': [0.5,  2, 5],
        'subsample': [0.6, 0.8, 1.0],
        'learning_rate': [0.02, 0.1, 0.2, 0.5],
        'max_depth': [3, 5, 7, 12]
    }
    folds = 5

    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=1001)

    random_search_xgb = RandomizedSearchCV(xgb_m, param_distributions=para_xgb, scoring='brier_score_loss',
                                       cv=skf.split(x_train, y_train), verbose=3, n_iter=10,
                                           n_jobs=-1) #random_state=1001, n_iter=param_comb

    # Here we go
    # start_time = timer(None)  # timing starts from this point for "start_time" variable
    random_search_xgb.fit(features_all_up, y)
    # timer(start_time)


    rf = RandomForestClassifier()# max_features=0.65, max_depth=5  # n_estimators = 550,n_estimators=300, max_features=0.65, bootstrap=False,
    para_rf = {
        'max_features': [0.6, 0.8, 1.0, "auto", "sqrt"],
        'max_depth': [3, 5, 7, 12, None],
        'criterion': ["gini", "entropy"],
        'n_estimators': [100, 300, 600],
        'bootstrap': [True, False]
    }


    random_search_rf = RandomizedSearchCV(rf, param_distributions=para_rf, scoring='brier_score_loss',
                                       cv=skf.split(x_train, y_train), verbose=3, n_iter=10,
                                          n_jobs=-1)#, n_iter=param_comb

    # Here we go
    # start_time = timer(None)  # timing starts from this point for "start_time" variable
    random_search_rf.fit(features_all_up, y)
    # timer(start_time)




    # features_all['treatment'] = 0
    # all_con.iloc[:,100] = 0
    PS_lasso = logmod.predict_proba(features_all)[:, 1]
    PS_nn_model = random_search_nn.predict_proba(features_all)[:, 1]
    PS_xgb_m = random_search_xgb.predict_proba(features_all)[:, 1]
    PS_rf = random_search_rf.predict_proba(features_all)[:, 1]    # data = pd.read_csv("output.csv")
    PS_logcon = logmod_con.predict_proba(all_con)[:,1]
    # PS_bart_1= bartmod.predict(features_all, return_mean=False)  # Make predictions on the train set
    # PS_bart = norm.cdf(PS_bart_1.mean(axis=1))  # Make predictions on the train set
    # PS_bart = model_bart.predict(data)
    simdata['pred_rf'] = pd.DataFrame(PS_rf)
    simdata['pred_log'] = pd.DataFrame(PS_lasso)
    simdata['pred_xgb'] = pd.DataFrame(PS_xgb_m)
    simdata['pred_nn'] = pd.DataFrame(PS_nn_model)
    simdata['pred_log_conf'] = pd.DataFrame(PS_logcon)
    # simdata['pred_bart'] = pd.DataFrame(PS_bart)

    simdata.to_csv(r'D:/sim_rf/S_025_002_DRS/output_m{}.csv'.format(i))
