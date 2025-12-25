import time
import argparse
import sys
import os
import os.path as osp

import numpy as np
import torch
import pandas as pd

from training.gnn_y import train_gnn_y
from data_process.data_subparser import add_uci_subparser
from utils.utils import auto_select_gpu

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
    parser.add_argument('--train_edge_prob', type=float, default=0.7)
    parser.add_argument('--train_y_prob', type=float, default=0.7)
    parser.add_argument('--time_interval', type=int, default=3)
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
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--log_dir', type=str, default='Y')
    parser.add_argument('--mode', type=str, default='')
    parser.add_argument('--penalty', type=str, default='')
    parser.add_argument('--iteration_num', type=int, default=20)
    parser.add_argument('--longimodel_types', type=str, default='Longit_Longit_Longit')
    parser.add_argument('--weight_model', type=str, default='All')
    subparsers = parser.add_subparsers()
    add_uci_subparser(subparsers)
    
    args = parser.parse_args()
    args.model_types = 'EGSAGE_EGSAGE_EGSAGE'
    
    # select device
    if torch.cuda.is_available():
        cuda = auto_select_gpu()
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ['CUDA_VISIBLE_DEVICES'] = str(cuda)
        print('Using GPU {}'.format(os.environ['CUDA_VISIBLE_DEVICES']))
        device = torch.device('cuda:{}'.format(cuda))
    else:
        print('Using CPU')
        device = torch.device('cpu')

    from data_process.data_load import load_data
    
    # device = torch.device('cpu')
    args.seed = 2
    seed = args.seed
    np.random.seed(seed)
    torch.manual_seed(seed)

    dataname = 'ADNI'
    dataname = 'LGAPE'
    dataname = 'GLOBEM'

    for longimodel_types in ['Longit_Longit_Longit']:
        for train_edge_prob in [0.7,0.5]:
            for train_y_prob in [0.7,0.5]:
                for time_interval in [3,5,7]:
                    for oversmooth_lambda in [0.0001]:
                        args.data = dataname
                        args.dropout = 0.2
                        args.known = 0.8
                        args.node_dim = 16
                        args.edge_dim = 16
                        
                        args.train_edge_prob = train_edge_prob
                        args.model_types = 'EGSAGE_EGSAGE_EGSAGE'
                        args.longimodel_types = longimodel_types
                        args.oversmooth_lambda = oversmooth_lambda
                        args.train_y_prob =  train_y_prob
                        args.time_interval = time_interval

                        args.lr = 0.001

                        print(args)
                        data = load_data(args)
                        args.epochs = 5
                        log_path = './SLIGNN/{}/Cov{}_Y{}_{}_{}_G/'.format(args.longimodel_types, args.train_edge_prob, args.train_y_prob, 
                                                                         args.time_interval, args.oversmooth_lambda)
                        if not os.path.isdir(log_path):
                            os.makedirs(log_path)

                        cmd_input = 'python ' + ' '.join(sys.argv) + '\n'
                        with open(osp.join(log_path, 'cmd_input.txt'), 'a') as f:
                            f.write(cmd_input)

                        train_gnn_y(data, args, log_path, device)

if __name__ == '__main__':
    main()