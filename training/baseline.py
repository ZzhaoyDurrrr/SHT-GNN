from fancyimpute import SimpleFill, KNN, IterativeImputer, IterativeSVD, SoftImpute
import numpy as np
import pandas as pd
from sklearn import preprocessing
import pickle
import time

from utils.utils import construct_missing_X_from_mask

def baseline_imputation(data, args, log_path):
    t0 = time.time()

    # train_edge_mask
    train_edge_mask = data.train_edge_mask.numpy()
    X, X_incomplete = construct_missing_X_from_mask(train_edge_mask, data.df_X)
    
    t_load = time.time()
    X_filled = baseline_inpute(X_incomplete, args.method,args.level)
    t_impute = time.time()
    mask = np.isnan(X_incomplete)
    diff = X[mask] - X_filled[mask]
    mae = np.mean(np.abs(diff))
    rmse = np.sqrt(np.mean(diff**2))
    t_test = time.time()

    obj = dict()
    obj['args'] = args
    obj['rmse'] = rmse
    obj['mae'] = mae
    obj['load_time'] = t_load - t0
    obj['impute_time'] = t_impute - t_load
    obj['test_time'] = t_test - t_impute
    print('MSE: {:.3g}, MAE: {:.3g}'.format(rmse,mae))
    # pickle.dump(obj, open(log_path + 'result.pkl', "wb"))

def baseline_inpute(X_incomplete, method='mean',level=0):

    if method == 'mean':
        X_filled_mean = SimpleFill().fit_transform(X_incomplete)
        return X_filled_mean
    elif method == 'mice':
        max_iter = [3,10,50][level]
        X_filled_mice = IterativeImputer(max_iter=max_iter).fit_transform(X_incomplete)
        return X_filled_mice
    elif method == 'random_forest':
        iteration_num = 20
        return random_forest_impute(X_incomplete, iteration_num)
    else:
        raise NotImplementedError

from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer

def random_forest_impute(X_incomplete, iteration_num):
    initial_imputer = SimpleImputer(strategy='mean')
    X_filled_initial = initial_imputer.fit_transform(X_incomplete)

    for i in range(X_incomplete.shape[1]):
        missing_mask = np.isnan(X_incomplete[:, i])
        if missing_mask.sum() == 0:
            continue 

        X_train = X_filled_initial[~missing_mask]
        y_train = X_incomplete[~missing_mask, i]
        X_test = X_filled_initial[missing_mask]

        rf = RandomForestRegressor(n_estimators=iteration_num, random_state=42)
        rf.fit(X_train, y_train)
        X_filled_initial[missing_mask, i] = rf.predict(X_test)

    return X_filled_initial

