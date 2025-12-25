import time
import argparse
import sys
import os
import os.path as osp

import numpy as np
import torch
import pandas as pd

from training.linear_regression import linear_regression

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_types', type=str, default='EGSAGE_EGSAGE_EGSAGE')
    parser.add_argument('--post_hiddens', type=str, default=None) # default to be 1 hidden of node_dim
    parser.add_argument('--concat_states', action='store_true', default=False)
    parser.add_argument('--norm_embs', type=str, default=None) # default to be all true
    parser.add_argument('--aggr', type=str, default='mean')
    parser.add_argument('--node_dim', type=int, default=16)
    parser.add_argument('--edge_dim', type=int, default=16)
    parser.add_argument('--node_mode', type=int, default=1)
    parser.add_argument('--train_edge_prob', type=int, default=0.7)
    parser.add_argument('--edge_mode', type=int, default=1)  # 0: use it as weight 1: as input to mlp
    parser.add_argument('--gnn_activation', type=str, default='relu')
    parser.add_argument('--impute_hiddens', type=str, default='16')
    parser.add_argument('--impute_activation', type=str, default='relu')
    parser.add_argument('--predict_hiddens', type=str, default='')
    parser.add_argument('--epochs', type=int, default=20000)
    parser.add_argument('--opt', type=str, default='adam')
    parser.add_argument('--opt_scheduler', type=str, default='step')
    parser.add_argument('--opt_restart', type=int, default=0)
    parser.add_argument('--opt_decay_step', type=int, default=1000)
    parser.add_argument('--opt_decay_rate', type=float, default=0.9)
    parser.add_argument('--dropout', type=float, default=0.)
    parser.add_argument('--weight_decay', type=float, default=0.000)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--known', type=float, default=0.7) # 1 - edge dropout rate
    parser.add_argument('--known_y', type=float, default=0.7) # 1 - edge dropout rate
    parser.add_argument('--valid', type=float, default=0.) # valid-set ratio
    parser.add_argument('--oversmooth_lambda', type=float, default=0.) # valid-set ratio
    parser.add_argument('--log_dir', type=str, default='Y')
    parser.add_argument('--mode', type=str, default='4042')
    parser.add_argument('--penalty', type=str, default='')
    parser.add_argument('--iteration_num', type=int, default=20)
    parser.add_argument('--longimodel_types', type=str, default='Longit_Longit_Longit')
    parser.add_argument('--weight_model', type=str, default='All')
    parser.add_argument('--data', type=str, default='ADNI')  
    parser.add_argument('--method', type=str, default='mean')
    parser.add_argument('--train_y_prob', type=float, default=0.7)
    parser.add_argument('--train_edge', type=float, default=0.7)
    parser.add_argument('--train_y', type=float, default=0.7)
    parser.add_argument('--load_dir', type=str, default='0')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--time_interval', type=int, default=3)
    args = parser.parse_args()

    seed = 1
    np.random.seed(seed)
    torch.manual_seed(seed)

    from Longitudinal_Network.data_process.data_load import load_data
    for train_edge_prob in [0.7, 0.5, 0.3]:
         for train_y_prob in [0.7, 0.5, 0.3]:
            for time_interval in [3, 5, 7]:
                for method in ['mice','mean']:
                    args.train_edge_prob = train_edge_prob
                    args.train_y_prob = train_y_prob
                    args.method = method
                    args.time_interval = time_interval

                    data = load_data(args)
                    
                    args.data = 'GLOBEM'
                    log_path = './baselines/Cov{}_Y{}_{}/{}/'.format(args.train_edge_prob, args.train_y_prob, args.time_interval, args.method)

                    if not os.path.isdir(log_path):
                        os.makedirs(log_path)

                    cmd_input = 'python ' + ' '.join(sys.argv) + '\n'
                    with open(osp.join(log_path, 'cmd_input.txt'), 'a') as f:
                        f.write(cmd_input)
                    
                    linear_regression(data, args, log_path)

if __name__ == '__main__':
    main()