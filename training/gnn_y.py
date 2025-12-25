import numpy as np
import torch
import torch.nn.functional as F
import pickle
import pandas as pd
import random

import csv
from training.WeightGraph import *
from models.gnn_model import get_gnn
from models.prediction_model import MLPNet, MLPNet_nobias
from utils.plot_utils import plot_curve
from utils.utils import build_optimizer, get_known_mask, mask_edge

def train_gnn_y(data, args, log_path, device):
    model = get_gnn(data, args).to(device)
    weight_model = WeightGraph(args.node_dim,args.node_dim, args.node_dim, F.relu).to(device)

    if args.impute_hiddens == '':
        impute_hiddens = []
    else:
        impute_hiddens = list(map(int, args.impute_hiddens.split('_')))
    if args.concat_states:
        input_dim = args.node_dim * len(model.convs) * 2
    else:
        input_dim = args.node_dim * 2
    impute_model = MLPNet(input_dim, 1,
                            hidden_layer_sizes=impute_hiddens,
                            hidden_activation=args.impute_activation,
                            dropout=args.dropout).to(device)

    impute_model = MLPNet(input_dim, 1,
                            hidden_layer_sizes=impute_hiddens,
                            hidden_activation=args.impute_activation,
                            dropout=args.dropout).to(device)

    if args.predict_hiddens == '':
        predict_hiddens = []
    else:
        predict_hiddens = list(map(int, args.predict_hiddens.split('_')))
    n_row, n_col = data.df_X.shape
    predict_model = MLPNet_nobias(n_col, 1,
                           hidden_layer_sizes=predict_hiddens,
                           dropout=args.dropout).to(device)

    trainable_parameters = list(model.parameters()) \
                           + list(impute_model.parameters()) \
                           + list(predict_model.parameters()) \
                          
    weight_parameters =  list(weight_model.parameters())
    
    # build optimizer
    scheduler, opt = build_optimizer(args, trainable_parameters)
    _, opt_weight = build_optimizer(args, weight_parameters)

    timestep_weight = data.longitudinal_edge_weight.clone().detach().to(device)

    Train_loss = []
    Validation_rmse = []
    MADGap_list = []
    Lr = []

    x = data.x.clone().detach().to(device)
    y = data.y.clone().detach().to(device)

    edge_missing_weight = torch.ones_like(timestep_weight)
    origin_edge_index = data.origin_edge_index.clone().detach().to(device)

    validation_y_mask = data.validation_y_mask.clone().detach().to(device)
    train_y_mask = data.train_y_mask.clone().detach().to(device)

    longitudinal_edge_index = data.longitudinal_edge_index.clone().detach().to(device)
    longitudinal_edge_weight = edge_missing_weight * timestep_weight

    test_input_edge_index = data.test_input_edge_index.clone().detach().to(device)
    test_input_edge_attr = data.test_input_edge_attr.clone().detach().to(device)

    longitudinal_edge_index_origin = longitudinal_edge_index.clone().detach()
    
    subject_to_edge_otf = data.subject_to_edges_otf
    subject_to_edge_fto = data.subject_to_edges_fto
    subject_to_edge = data.group_to_edges

    subject_to_obs = data.subject_to_obs
    
    # load dataframe with original pid vector named 'Subject' 
    subject_list = pd.read_csv("...")
    
    indicator_matrix = generate_indicator_matrix(subject_list)
    indicator_matrix = torch.tensor(indicator_matrix).to(device)
    
    unique_subject_set = set(subject_list['Subject'])

    batch_size_subject = 300
    observation_batch_size = 12000
    
    origin_edge_attr = data.origin_edge_attr.clone().detach().to(device)
    edge_index_batch_start, edge_index_batch_end = create_batch_edge_index(observation_batch_size, n_col)
    edge_index_batch = torch.stack([edge_index_batch_start, edge_index_batch_end], dim=0).to(device)
    edge_index_batch_otf = edge_index_batch[:,:int(edge_index_batch.shape[1]/2)]
    train_edge_mask = data.train_edge_mask_origin.clone().detach().to(device)
    
    weight_update_frequency = 50
    dynamic_epochs = 500

    for epoch in range(args.epochs):
        model.train()
        impute_model.train()
        weight_model.train()
        predict_model.train()

        # Selection_mask_batch is for all edges (not only observed)
        selected_subject = random.sample(unique_subject_set, batch_size_subject)
        # selection_mask_batch
        selection_index_batch = get_batch_edge_index(selected_subject, subject_to_edge_otf, subject_to_edge_fto).to(device)
        edge_attr_batch = origin_edge_attr[selection_index_batch,:]
        train_edge_mask_batch = train_edge_mask[selection_index_batch]
        
        observation_num = int(edge_attr_batch.shape[0] / (2 * n_col))
        excess_num = observation_batch_size - observation_num

        train_edge_index_batch_otf = edge_index_batch_otf[:,:int(observation_num * n_col)].clone()
        train_edge_index_batch_otf[1,:] = train_edge_index_batch_otf[1,:] - excess_num
        train_edge_index_batch_fto =  train_edge_index_batch_otf[[1, 0],:]
        train_edge_index_batch = torch.concat([train_edge_index_batch_otf, train_edge_index_batch_fto], dim=1)
        full_train_edge_index_batch = train_edge_index_batch.clone().detach().to(device)
        
        train_edge_attr_batch = edge_attr_batch[train_edge_mask_batch]
        train_edge_index_batch = train_edge_index_batch[:,train_edge_mask_batch]
        selected_observation = get_subject_to_obv(subject_to_obs, selected_subject)
        
        longitudinal_edge_index, longitudinal_edge_weight = generate_longitudinal_edge(selected_subject, subject_to_obs, subject_to_edge)
        longitudinal_edge_index = torch.tensor(longitudinal_edge_index, dtype=torch.int64).to(device)
        longitudinal_edge_weight = torch.tensor(longitudinal_edge_weight, dtype=torch.float32).to(device)

        for i in range(dynamic_epochs):
            start_time = time.time()  
            memory_before = torch.cuda.memory_allocated()
            x_batch = torch.concat([x[selected_observation,:], x[-n_col:,:]], dim=0)

            args.known = 0.7
            known_mask = get_known_mask(args.known, int(train_edge_attr_batch.shape[0] / 2)).to(device)
            double_known_mask = torch.cat((known_mask, known_mask), dim=0)
            known_edge_index, known_edge_attr = mask_edge(train_edge_index_batch, train_edge_attr_batch, double_known_mask, True)

            opt.zero_grad()
            opt_weight.zero_grad()

            longitudinal_edge_weight_batch = longitudinal_edge_weight.clone().detach()
            x_embd = model(x_batch, known_edge_attr, known_edge_index)

            if epoch % weight_update_frequency == 0:
                edge_logits = weight_model(x_embd[:observation_num].clone(), longitudinal_edge_index, mode=1)
                longitudinal_edge_weight_temp = edge_logits / edge_logits.mean()
                longitudinal_edge_weight = longitudinal_edge_weight_temp * longitudinal_edge_weight_batch
            else:
                longitudinal_edge_weight = longitudinal_edge_weight_batch.detach()
 
            x_embd_longitud = model.longitudinal_message_passing(x_embd[:observation_num], longitudinal_edge_index, longitudinal_edge_weight)        
            x_embd[:observation_num,:] = x_embd_longitud

            distance_matrix = 1.0 - cosine_similarity_matrix(x_embd[:observation_num,:])

            indicator_matrix_batch = indicator_matrix[selected_observation][:,selected_observation]
            plus_indicator_batch = torch.nonzero(indicator_matrix_batch == 1)
            minus_indicator_batch = torch.nonzero(indicator_matrix_batch == -1)
            remote_distance = (distance_matrix[plus_indicator_batch[:,0], plus_indicator_batch[:,1]]).sum()/plus_indicator_batch.shape[0]
            neighbour_distance = (distance_matrix[minus_indicator_batch[:,0], minus_indicator_batch[:,1]]).sum()/ minus_indicator_batch.shape[0]
            MADgap = remote_distance - neighbour_distance

            X = impute_model([x_embd[full_train_edge_index_batch[0,:int(observation_num * n_col)]], x_embd[full_train_edge_index_batch[1,:int(observation_num* n_col)]]])
            X = torch.reshape(X, [observation_num, n_col])

            memory_after = torch.cuda.memory_allocated()
            memory_used_in_iteration = (memory_after - memory_before) / (1024 ** 2)  # 转换为 MB
            print(f'Iteration {i}, GPU memory used: {memory_used_in_iteration:.2f} MB')
        
            y_label_batch = y[selected_observation]
            y_label_train_mask = train_y_mask[selected_observation]
            label_train = y_label_batch[y_label_train_mask]
            
            epoch_time = time.time() - start_time
            print(f'Epoch {i} completed in {epoch_time:.2f} seconds')

            pred = predict_model(X)[:, 0]
            pred_train = pred[y_label_train_mask]

            mse = F.mse_loss(label_train.squeeze(1), pred_train)
            loss = mse - MADgap * args.oversmooth_lambda
            if torch.isnan(loss):
                print("NaN detected in train_loss, skipping this step.")
                break  

            loss.backward()
            opt.step()
            if epoch % weight_update_frequency == 0:
                opt_weight.step()

            train_loss = loss.item()
            if scheduler is not None:
                scheduler.step()
            for param_group in opt.param_groups:
                Lr.append(param_group['lr'])
        
            if i % 1 == 0:
                model.eval()
                impute_model.eval()
                weight_model.eval()
                predict_model.eval()
                with torch.no_grad():
                    x_embd = model(x,test_input_edge_attr, test_input_edge_index)
                    
                    edge_logits_test = weight_model(x_embd[:n_row].clone(), longitudinal_edge_index_origin, mode=1)
                    longitudinal_edge_weight_test = edge_logits_test / edge_logits_test.mean()

                    x_embd_longitud = model.longitudinal_message_passing(x_embd[:n_row], longitudinal_edge_index_origin, longitudinal_edge_weight_test)        
                    x_embd[:n_row,:] = x_embd_longitud

                    X = impute_model([x_embd[origin_edge_index[0,:int(n_row * n_col)]], x_embd[origin_edge_index[1, :int(n_row * n_col)]]])
                    X = torch.reshape(X, [n_row, n_col])
                        
                    pred = predict_model(X)[:, 0]
                    pred_val = pred[validation_y_mask]
                    label_val = y[validation_y_mask]

                    val_rmse = torch.sqrt(F.mse_loss(pred_val, label_val.squeeze(1)))

                    Train_loss.append(train_loss)
                    Validation_rmse.append(val_rmse.item())
                    MADGap_list.append(MADgap.item())
                    print('External Epoch: ', epoch, "Internal Epoch: ", i)
                    print('Loss: ', train_loss)
                    print('Validation mse:', val_rmse.item())
                    print('MADGap:', MADgap.item())
                    print('Min Validation RMSE', np.array(Validation_rmse).min())
                    print('')
           
    obj = dict()
    obj['args'] = args
    obj['curves'] = dict()
    obj['curves']['train_loss'] = Train_loss
    obj['curves']['test_rmse'] = Validation_rmse
    

    pickle.dump(obj, open(log_path + 'result.pkl', "wb"))

    torch.save(model, log_path + 'model.pt')
    torch.save(impute_model, log_path + 'impute_model.pt')
    torch.save(predict_model, log_path + 'predict_model.pt')

    with open(log_path +'MADGap.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        for item in MADGap_list:
            writer.writerow([item])

    with open(log_path +'Rmse.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        for item in Validation_rmse:
            writer.writerow([item])

    with open(log_path + 'Minmse.txt', 'w') as f:
        f.write(str(np.array(Validation_rmse).min()))



    plot_curve(obj['curves'], log_path+'curves.png',keys=None, 
                clip=True, label_min=True, label_end=True)
    
def cosine_similarity_matrix(x):
    x_normalized = F.normalize(x, p=2, dim=1)
    similarity_matrix = torch.matmul(x_normalized, x_normalized.T)
    return similarity_matrix

def create_batch_edge_index(n_row, n_col):
    edge_start = torch.repeat_interleave(torch.arange(n_row), n_col)
    edge_end = torch.tile(torch.arange(n_col) + n_row, (n_row,))
    
    edge_start_new = torch.cat([edge_start, edge_end])
    edge_end_new = torch.cat([edge_end, edge_start])
    
    return edge_start_new, edge_end_new

def get_batch_edge_index(selected_subject, subject_to_edge_otf, subject_to_edge_fto):
    edge_index_list_otf = []
    edge_index_list_fto = []
    batch_subject = selected_subject
    for subject in batch_subject:
        edge_index_list_otf.extend(subject_to_edge_otf[subject])
    edge_index_list_otf = torch.tensor(np.array(edge_index_list_otf), dtype=torch.long)
    batch_subject = selected_subject
    for subject in batch_subject:
        edge_index_list_fto.extend(subject_to_edge_fto[subject])
    edge_index_list_fto = torch.tensor(np.array(edge_index_list_fto), dtype=torch.long)
    edge_index_list = torch.concat([edge_index_list_otf, edge_index_list_fto], dim=0)
    return edge_index_list

def get_batch_node_index(batch_size, selected_subject, subject_to_edge):
    edge_index_list = []
    for subject in selected_subject:
        edge_index_list.extend(subject_to_edge[subject])
    edge_index_list = torch.tensor(np.array(edge_index_list))
    return edge_index_list

def get_subject_to_obv(subject_to_obs, selected_subject):
    observation_list = []
    for subject in selected_subject:
        observation_list.extend(subject_to_obs[subject])
    observation_list = torch.tensor(np.array(observation_list),dtype=torch.long)
    return observation_list

def generate_longitudinal_edge(selected_subject, subject_to_obs, subject_to_edge):
    list_lengths = {key: len(subject_to_obs[key]) for key in selected_subject}

    edge_index = []
    edge_attr = []
    start_index = 0
    for key, length in list_lengths.items():
        for i in range(start_index, start_index + length - 1):
            edge_index.append([i, i + 1])
        
        start_index += length
        if length == 1:
            continue
        edge_attr_temp = subject_to_edge[key]
        edge_attr.extend(edge_attr_temp)
    edge_indices_array = np.array(edge_index).reshape(2,-1)
    edge_attr_array = np.array(edge_attr)
    return edge_indices_array, edge_attr_array


def generate_normalize_matrix(data):
    pid_list = list(data['Subject'])  

    filtered_pid_list = []
    for i in range(len(pid_list) - 1):
        if pid_list[i] == pid_list[i + 1]:
            filtered_pid_list.append(pid_list[i])

    unique_pids = list(set(filtered_pid_list))
    pid_count = {pid: filtered_pid_list.count(pid) for pid in unique_pids}
    N = len(filtered_pid_list)  
    matrix = np.zeros((N, N))
    for i, pid in enumerate(filtered_pid_list):
        count = pid_count[pid]
        for j in range(N):
            if filtered_pid_list[j] == pid:
                matrix[i, j] = 1 / count

    return matrix

def generate_indicator_matrix(group_data):
    group_data = group_data['Subject'].tolist()
    # Determine the size of the matrix N
    N = len(group_data)
    # Initialize the N*N matrix with zeros
    matrix = np.zeros((N, N))
    # Pointers to navigate through the matrix
    row_start = 0
    # Process the group data to fill the matrix
    unique_groups = sorted(set(group_data), key=lambda x: group_data.index(x))
    for group in unique_groups:
        count = group_data.count(group)
        block_matrix = create_block_matrix(count)
        matrix[row_start:row_start + count, row_start:row_start + count] = block_matrix
        row_start += count
    return matrix


def create_block_matrix(size):
    block = np.zeros((size, size))
    for i in range(1, size):
        block[i, i - 1] = -1
        for j in range(i-1):
            block[i, j] = 1
    return block