import argparse

from utils.train_utils import add_flags_from_config

config_args = {
        'lr': 0.01,  # learning rate
        'dropout': 0.0,  #'dropout probability'
        'cuda': 1,  # 'which cuda device to use (-1 for cpu training)')
        'device':'cuda',
        'epochs': 300,
        'weight_decay': 0.,
        'optimizer': 'Adam',
        'momentum': 0.999,
        'patience': 100,
        'seed': 1234,
        'log_freq': 1,
        'eval-freq': 1,
        'save': 0,
        'save-dir': None,
        'sweep-c': 0,
        'lr_reduce_freq': 10,
        'gamma': 0.5,
        'print-epoch': True,

        'min-epochs': 100,
        'grad_clip':None,

        'num_layers':8,
        'feat_dim':11,  # 输入特征的维度 ，hyperbolid会自动+1
        'n_nodes': None,
        'task': 'reg',
        'model': 'HGCN',
        'dim': 12,  # 隐层的dim
        'n_atom_embed': 8, # 'atom embedding dimension'),
        'max_z': 100,   #'atom type'),
        'manifold': 'Hyperboloid',   # 'which manifold to use, can be any of [Euclidean, Hyperboloid, PoincareBall]'),
        'c': 1,  # 'hyperbolic radius, set to None for trainable curvature'),
        'r': 2.,  # 'fermi-dirac decoder parameter for lp'),
        't': 1., # 'fermi-dirac decoder parameter for lp'),
        'pretrained-embeddings': None,  # 'path to pretrained embeddings (.npy file) for Shallow node classification'),
        'pos-weight': 0, # 'whether to upweight positive class in node classification tasks'),
        'num-layers': 2, # 'number of hidden layers in encoder'),
        'bias': 0, # 'whether to use bias (1) or not (0)'),
        'act': 'relu', # 'which activation function to use (or None for no activation)'),
        'n-heads': 4, # 'number of attention heads for graph attention networks, must be a divisor dim'),
        'alpha': 0.2, # 'alpha for leakyrelu in graph attention networks'),
        'double-precision': '0', # 'whether to use double precision'),
        'use_att': 1, # 'whether to use hyperbolic attention or not'),
        'local_agg': 1, # 'whether to local tangent space aggregation or not'),

        'dataset': 'cora', # 'which dataset to use'),
        'val-prop': 0.05, # 'proportion of validation edges for link prediction'),
        'test-prop': 0.1, # 'proportion of test edges for link prediction'),
        'use-feats': 1, # 'whether to use node features or not'),
        'normalize-feats': 1, # 'whether to normalize input node features'),
        'normalize-adj': 1, # 'whether to row-normalize the adjacency matrix'),
        'split-seed': 1234, # 'seed for data splits (train/test/val)')
}

