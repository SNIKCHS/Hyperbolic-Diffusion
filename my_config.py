import argparse

from utils.train_utils import add_flags_from_config

config_args = {
        'lr': 1e-2,  # learning rate 1e-2
        'dropout': 0.0,  #'dropout probability'
        'cuda': 1,  # 'which cuda device to use (-1 for cpu training)')
        'device':'cuda',
        'epochs': 200,
        'weight_decay': 0.,
        'optimizer': 'RiemannianAdam',  #RiemannianAdam Adam
        'momentum': 0.999,
        'patience': 100,
        'seed': 1234,
        'log_freq': 1,
        'eval-freq': 1,
        'save': 0,
        'save-dir': None,
        'sweep-c': 0,
        'lr_reduce_freq': 2000, #20
        'gamma': 0.9, # 0.5
        'print-epoch': True,

        'min-epochs': 100,
        'grad_clip':10,
        'att_logit':'exp', #Specify logit for attention, can be any of [exp, sigmoid, tanh, ... from torch.<loigt>]

        'num_layers':4,
        'n_nodes': None,
        'task': 'rec',
        'model': 'HGCN',  # ['MLP','HNN','GCN','HGCN'
        'diff_model':'HGCN',
        'dim': 20,  # 隐层的dim

        'max_z': 10,   #'atom type'),
        'manifold': 'Hyperboloid',   # 'which manifold to use, can be any of [Euclidean, Hyperboloid, PoincareBall]'),
        'c': None,  # 'hyperbolic radius, set to None for trainable curvature'),
        'r': 2.,  # 'fermi-dirac decoder parameter for lp'),
        't': 1., # 'fermi-dirac decoder parameter for lp'),
        'pretrained-embeddings': None,  # 'path to pretrained embeddings (.npy file) for Shallow node classification'),
        'pos-weight': 0, # 'whether to upweight positive class in node classification tasks'),
        'bias': 0, # 'whether to use bias (1) or not (0)'),
        'act': 'selu', # 'which activation function to use (or None for no activation)'), silu,selu
        'n-heads': 4, # 'number of attention heads for graph attention networks, must be a divisor dim'),
        'alpha': 0.2, # 'alpha for leakyrelu in graph attention networks'),
        'double-precision': '0', # 'whether to use double precision'),
        'use_att': 1, # 'whether to use hyperbolic attention or not'),
        'local_agg': 1, # 'whether to local tangent space aggregation or not'),
        'att_type':'adjmask_dist',
        'encdec_share_curvature':False,
        'dataset': 'cora', # 'which dataset to use'),
        'val-prop': 0.05, # 'proportion of validation edges for link prediction'),
        'test-prop': 0.1, # 'proportion of test edges for link prediction'),
        'use-feats': 1, # 'whether to use node features or not'),
        'normalize-feats': 1, # 'whether to normalize input node features'),
        'normalize-adj': 1, # 'whether to row-normalize the adjacency matrix'),
        'split-seed': 1234, # 'seed for data splits (train/test/val)')
}

