import pandas as pd
import os.path as osp
import inspect
import json
from torch_geometric.data import Data
from sklearn import preprocessing

import torch
import random
import numpy as np
from scipy.stats.qmc import Sobol
import pdb

from utils.utils import get_known_mask, mask_edge, mask_edge_T

def create_node(df, mode, node_dim):
    if mode == 0: 
        nrow, ncol = df.shape
        sample_node = np.zeros((nrow,node_dim))
        sample_node[np.arange(nrow), 0] = 1
        sobol_gen = Sobol(d = node_dim, scramble=True) 
        feature_node = sobol_gen.random(n = ncol)
        node = sample_node.tolist() + feature_node.tolist()  

    elif mode == 1: 
        nrow, ncol = df.shape
        sobol_gen = Sobol(d = node_dim, scramble=True)  
        feature_node = sobol_gen.random(n = ncol)
        sample_node = [[1] * node_dim for i in range(nrow)] 
        node = sample_node + feature_node.tolist()
    return node

def create_edge(df):
    n_row, n_col = df.shape
    edge_start = []
    edge_end = []
    for x in range(n_row):
        edge_start = edge_start + [x] * n_col 
        edge_end = edge_end + list(n_row+np.arange(n_col)) 
    edge_start_new = edge_start + edge_end
    edge_end_new = edge_end + edge_start
    return (edge_start_new, edge_end_new)

def create_edge_tensor(df):
    n_row, n_col = df.shape
    edge_start = torch.repeat_interleave(torch.arange(n_row), n_col)
    edge_end = torch.tile(torch.arange(n_col) + n_row, (n_row,))
    
    edge_start_new = torch.cat([edge_start, edge_end])
    edge_end_new = torch.cat([edge_end, edge_start])
    
    return edge_start_new, edge_end_new

def create_edge_attr(df):
    edge_attr = []
    edge_attr = df.values.reshape(-1,1).tolist()
    edge_attr = edge_attr + edge_attr
    return edge_attr

def get_group_indices(num_groups, records_per_group):
    group_indices = {}
    current_index = 0
    for k in range(num_groups):
        num_edges = records_per_group[k] * (records_per_group[k] - 1)
        group_indices[k] = list(range(current_index, current_index + num_edges))
        current_index += num_edges
    return group_indices

def generate_vectors(n_vectors, dim):
    vectors = np.random.randn(n_vectors, dim)
    vectors_norm = np.linalg.norm(vectors, axis=1, keepdims=True)
    normalized_vectors = vectors / vectors_norm
    return normalized_vectors

def get_data(df_X, df_y, mask, mask_y, train_edge_prob, train_y_prob, seed, normalize=True):
    
    df_X = df_X.fillna(df_X.mean())
    if normalize:
        x = df_X.values.astype(float)
        min_max_scaler = preprocessing.MinMaxScaler()
        x_scaled = min_max_scaler.fit_transform(x)
        df_X = pd.DataFrame(x_scaled)
    
    normalize_y = True
    if normalize_y:
        y = df_y.values.astype(float)
        y = y.reshape(-1,1)
        min_max_scaler = preprocessing.MinMaxScaler()
        df_y = min_max_scaler.fit_transform(y)

    edge_start, edge_end = create_edge_tensor(df_X)
    edge_index = torch.stack([edge_start, edge_end], dim=0)
    edge_attr = torch.tensor(create_edge_attr(df_X), dtype=torch.float)
    origin_edge_index = edge_index
    origin_edge_attr = edge_attr

    node_init = create_node(df_X, mode=1, node_dim=16)

    x = torch.tensor(node_init, dtype=torch.float)
    y = torch.tensor(df_y, dtype=torch.float)

    torch.manual_seed(seed)
   
    nrow = df_X.values.shape[0]
    ncol = df_X.values.shape[1]

    # load the original subject id column 
    group_list = pd.read_csv("..")
    subjects = list(group_list['Subject'])

    # load subject_indices.json, you can generate it in Data_preprocessing.py
    with open('subject_indices.json', 'r') as file:
        subject_indices = json.load(file)
    
    subject_indices = {int(k): v for k, v in subject_indices.items()}

    training_ratio = train_edge_prob
    all_subjects = list(subject_indices.keys())

    assert set(subjects) == set(all_subjects), "Error: subjects and all_subjects do not match!"

    mask = torch.tensor(mask.values).reshape(-1,1).squeeze(1)
    double_edge_mask = torch.cat((mask, mask),dim=0)
    edge_index, edge_attr = mask_edge(edge_index, edge_attr, double_edge_mask, True)
    
    mask_type = 'Subject'
    subject_ratio = 0.8
    if mask_type == 'MCAR':

        train_matrix_mask = get_known_mask(training_ratio, nrow * ncol)
        train_observation_matrix_mask = train_matrix_mask.reshape(nrow * ncol,1)
        train_observation_double_mask = torch.cat((train_observation_matrix_mask,train_observation_matrix_mask),dim=0)
        double_train_edge_mask = mask_edge_T(train_observation_double_mask, double_edge_mask, True).squeeze(1)
    
    elif mask_type == 'Subject': 
        np.random.shuffle(all_subjects)
        
        test_subjects = sorted(all_subjects[int(len(all_subjects) * subject_ratio):])
        train_subjects = sorted(all_subjects[:int(len(all_subjects) * subject_ratio)])
        
        train_mask = get_known_mask(training_ratio, nrow * ncol)
        train_mask_matrix = train_mask.reshape(nrow, ncol)

        train_edge_mask_matrix = mask_columns_for_test(subject_indices, test_subjects, train_mask_matrix.clone()).reshape(nrow * ncol, 1)
        test_input_matrix_mask = train_mask

        train_observation_double_mask = torch.cat((train_edge_mask_matrix, train_edge_mask_matrix),dim=0)
        double_train_edge_mask = train_observation_double_mask.squeeze(1)
        
        test_input_double_mask = torch.cat((test_input_matrix_mask, test_input_matrix_mask), dim=0)
        double_test_input_edge_mask = test_input_double_mask

    train_edge_mask = double_train_edge_mask
    train_edge_index, train_edge_attr = mask_edge(edge_index, edge_attr, double_train_edge_mask, True)
    test_input_edge_index, test_input_edge_attr = mask_edge(edge_index, edge_attr, double_test_input_edge_mask, True)

    train_labels = train_edge_attr[:int(train_edge_attr.shape[0]/2),0]
    test_edge_index, test_edge_attr = mask_edge(edge_index, edge_attr, ~double_train_edge_mask, True)
    test_labels = test_edge_attr[:int(test_edge_attr.shape[0]/2),0]

    # Step 1: Create subject -> observation mapping
    subject_to_obs = {}
    for idx, subject in enumerate(subjects):
        if subject not in subject_to_obs:
            subject_to_obs[subject] = []
        subject_to_obs[subject].append(idx)
        
    # Step 2: Create observation -> edge mapping
    obs_to_edges_otf = {}
    obs_to_edges_fto = {}
    for idx in range(nrow):
        obs_to_edges_otf[idx] = list(range(idx * ncol, (idx + 1) * ncol)) 
        obs_to_edges_fto[idx] = list(range(nrow * ncol + idx * ncol, nrow * ncol + (idx + 1) * ncol))   

    # Step 3: Combine the mappings
    subject_to_edges_otf = {}
    subject_to_edges_fto = {}
    for subject, observations in subject_to_obs.items():
        edges = []
        for obs in observations:
            edges.extend(obs_to_edges_otf[obs])
        subject_to_edges_otf[subject] = edges

    for subject, observations in subject_to_obs.items():
        edges = []
        for obs in observations:
            edges.extend(obs_to_edges_fto[obs])
        subject_to_edges_fto[subject] = edges

    # load Longitudinal_edge_index.csv, you can generate it in Data_preprocessing.py
    longitudinal_edge_index = pd.read_csv("Longitudinal_edge_index.csv")
    longitudinal_edge_index = torch.tensor(np.array(longitudinal_edge_index)).T.to(torch.int64)

    edge_size = longitudinal_edge_index.shape[1]
    longitudinal_edge_weight = torch.ones(edge_size) 

    # load Longitudinal_edge_index.csv, you can generate it in Data_preprocessing.py
    group_index_series = pd.read_csv("Subject_Indices_foredge.csv")['Subject']
    group_to_edges = {}

    for group_index, edge_weight in zip(group_index_series, longitudinal_edge_weight):
        if group_index not in group_to_edges:
            group_to_edges[group_index] = []
        
        group_to_edges[group_index].append(edge_weight)

    validation_prob = train_y_prob
    observed_y_mask = torch.tensor(mask_y.values).squeeze(1)
    mask_y = mask_y_for_test(subject_indices, test_subjects, observed_y_mask)
    
    validation_y_mask = ~mask_y

    train_y_mask = get_known_mask(validation_prob, nrow)
    train_y_mask = mask_y_for_test(subject_indices, test_subjects, train_y_mask)

    data = Data(x=x, y=y, edge_index=edge_index, edge_attr=edge_attr,
            origin_edge_index = origin_edge_index, origin_edge_attr = origin_edge_attr,
            subject_to_obs = subject_to_obs, train_edge_index=train_edge_index,
            train_edge_attr=train_edge_attr, train_edge_mask=train_edge_mask,train_labels=train_labels,
            test_edge_index=test_edge_index,test_edge_attr=test_edge_attr, 
            subject_to_edges_otf = subject_to_edges_otf, subject_to_edges_fto = subject_to_edges_fto,
            test_edge_mask=~train_edge_mask,test_labels=test_labels, train_edge_mask_single = train_edge_mask_matrix,
            df_X=df_X, group_indices = subject_indices, edge_attr_dim=train_edge_attr.shape[-1], 
            longitudinal_edge_index = longitudinal_edge_index, user_num=df_X.shape[0], group_to_edges = group_to_edges,
            longitudinal_edge_weight = longitudinal_edge_weight, train_y_mask =  train_y_mask, validation_y_mask = validation_y_mask,
            train_edge_mask_origin = double_train_edge_mask, test_input_edge_attr = test_input_edge_attr, test_input_edge_index = test_input_edge_index,
            train_subjects = train_subjects, test_subjects = test_subjects,
            nrow = nrow, ncol = ncol)

    return data

def load_data(args):

    # load the original dataframe --- the last column is response variable / the other columns are covariates
    dataframe = pd.read_csv('..')
    dataframe_y = pd.DataFrame(dataframe.iloc[:,-1])
    dataframe_x = pd.DataFrame(dataframe.iloc[:,:-1])
    
    # load the dataframe mask if data is not fully observed
    mask = pd.read_csv('...')
    mask_origin = mask
    mask_x = pd.DataFrame(mask.iloc[:,:-1])
    mask_y = pd.DataFrame(mask_origin.iloc[:,-1])

    # dataframe_x 
    data = get_data(dataframe_x, dataframe_y, mask_x, mask_y, args.train_edge_prob, args.train_y_prob, args.seed)

    return data

def mask_columns_for_test(subject_indices, test_subjects, mask_matrix):

    for subject in test_subjects:
        for idx in subject_indices[subject]:
            mask_matrix[idx, :] = False 
    return mask_matrix

def mask_y_for_test(subject_indices, test_subjects, mask_y):

    for subject in test_subjects:
        for idx in subject_indices[subject]:
            mask_y[idx] = False 
    return mask_y

def get_observation_indices(train_subjects, subject_indices):
    observation_list = []
    for subject in train_subjects:
        for idx in subject_indices[subject]:
            observation_list.append(idx)
    return observation_list


def generate_Jaccard_Distance(train_mask_matrix, nrow, ncol):
    train_mask_matrix = train_mask_matrix[:int(train_mask_matrix.shape[0]/2)].reshape(nrow, ncol)
    
    # load Longitudinal_edge_index.csv, you can generate it in All_Data_preprocessing.py
    pairs = pd.read_csv("Longitudinal_edge_index.csv")
    results = []
    for _, row in pairs.iterrows():
        i = row['Edge_start']
        j = row['Edge_end']
        distance = asymmetric_distance(train_mask_matrix[j], train_mask_matrix[i])
        results.append(distance)

    return results

def asymmetric_distance(mask1, mask2):
    additional_info = mask2 & ~mask1
    total_info = mask1 | mask2
    return additional_info.sum() / total_info.sum()

