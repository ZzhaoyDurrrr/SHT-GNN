from sklearn.linear_model import LinearRegression
import numpy as np
import pickle
import joblib
import time
from os import path
from utils.utils import construct_missing_X_from_mask
from sklearn.impute import IterativeImputer

def linear_regression(data, args, log_path):
    t0 = time.time()
    train_edge_mask = data.train_edge_mask_single.numpy()
    train_edge_mask_origin = data.train_edge_mask_single.numpy()
    test_subjects = data.test_subjects
    train_subjects = data.train_subjects
    y = data.y.detach().numpy()
    
    train_y_mask = data.train_y_mask.clone().detach()
    validation_y_mask = data.validation_y_mask.clone().detach()
    
    y_train = y[train_y_mask]
    y_test = y[validation_y_mask]

    train_edge_mask = mask_columns_for_test(data.group_indices, test_subjects, train_edge_mask).reshape(data.nrow * data.ncol, 1)
    X_incomplete = construct_missing_X_from_mask(train_edge_mask, data.df_X)
    observation_list = choose_training_subject(data.group_indices, train_subjects)
    X_incomplete = X_incomplete.iloc[observation_list,:]
    X_incomplete_origin = construct_missing_X_from_mask(train_edge_mask_origin, data.df_X)

    level = 1
    max_iter = [3, 20, 50][level]  
    imputer = IterativeImputer(max_iter=max_iter)

    imputer.fit(X_incomplete)
    X = imputer.transform(X_incomplete_origin)
    t_impute = time.time()

    reg = LinearRegression().fit(X[train_y_mask, :], y_train)
    y_pred_test = reg.predict(X[validation_y_mask, :])
    t_reg = time.time()

    rmse = np.sqrt(np.mean((y_pred_test - y_test) ** 2))
    t_test = time.time()

    if path.exists(log_path + 'result.pkl'):
        obj = joblib.load(log_path + 'result.pkl')
        obj['args_linear_regression'] = args
    else:
        obj = dict()
        obj['args'] = args
    obj['rmse'] = rmse
    obj['reg_time'] = t_reg - t_impute
    obj['test_time'] = t_test - t_reg
    print('{}: rmse: {:.3g}'.format(args.method,rmse))
    pickle.dump(obj, open(log_path + 'result.pkl', "wb"))
    
    with open(log_path + 'Minmse.txt', 'w') as f:
        f.write(str(rmse))

def mask_columns_for_test(subject_indices, test_subjects, mask_matrix):
    for subject in test_subjects:
        for idx in subject_indices[subject]:
            mask_matrix[idx, :] = False 
    return mask_matrix

def choose_training_subject(subject_indices, test_subjects):
    observation_list = []
    for subject in test_subjects:
        for idx in subject_indices[subject]:
            observation_list.append(idx) 
    return observation_list