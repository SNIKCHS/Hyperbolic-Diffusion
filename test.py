import torch
from schnetpack.datasets import QM9
import schnetpack as spk
import os

from torch.optim import Adam

from Models.GCN import GCN
from Models.HGCN import HGCN,HGNN
from my_config import config_args
import numpy as np
import logging
import time
import wandb
import json

from optimizers import RiemannianAdam


class obj(object):
    def __init__(self, dict_):
        self.__dict__.update(dict_)

no_wandb = False
# no_wandb = True
if no_wandb:
    mode = 'disabled'
else:
    mode = 'online'
kwargs = {'entity': 'elma', 'name': 'hgcn_geoopt', 'project': 'regression',
          'settings': wandb.Settings(_disable_stats=True), 'reinit': True, 'mode': mode}
wandb.init(**kwargs)

qm9data = QM9('./data/qm9.db', download=True, load_only=[QM9.U0])
qm9split = './data/qm9split'

train, val, test = spk.train_test_split(
    data=qm9data,
    num_train=30000,
    num_val=10000,
    split_file=os.path.join(qm9split, "split30000-10000.npz"),
)

train_loader = spk.AtomsLoader(train, batch_size=64, shuffle=False)
val_loader = spk.AtomsLoader(val, batch_size=64)


args = json.loads(json.dumps(config_args), object_hook=obj)
print(args.act)
device = torch.device('cuda')
model = HGCN(device,args)
model = model.to(device)
total = sum([param.nelement() for param in model.parameters()])
print("Number of parameter: %.2fM" % (total / 1e6))
euc_param = []
hyp_param = []
# for n,p in model.named_parameters():
#     # if n in ['centroids.centroid_embedding.weight','embedding.weight']:
#     if n in ['embedding.weight']:
#         hyp_param.append(p)
#     else:
#         euc_param.append(p)

# optimizer = Adam(params=iter(euc_param), lr=1e-4,weight_decay=args.weight_decay)
# Roptimizer = RiemannianAdam(curvatures=model.curvatures,params=iter(hyp_param), lr=1e-4,weight_decay=args.weight_decay)
# Roptimizer = RiemannianAdam(curvatures=[1],params=iter(hyp_param), lr=1e-4,weight_decay=args.weight_decay)
optimizer = Adam(params=model.parameters(), lr=1e-4,weight_decay=args.weight_decay)
lr_scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer,
    step_size=args.lr_reduce_freq,
    gamma=float(args.gamma)
)

# Train model
t_total = time.time()



for epoch in range(args.epochs):
    step = 0
    model.train()
    MAE_sum,  t = 0,  0
    counter = 0
    n_iterations = len(train_loader)
    n_test_iterations = len(val_loader)
    for input in (train_loader):
        for key in input:
            input[key] = input[key].to(device)

        t = time.time()
        optimizer.zero_grad()
        # Roptimizer.zero_grad()
        loss,MAE = model(input)
        if torch.isnan(loss):
            raise AssertionError
        step += 1
        str = " ".join(['Epoch: {:04d}'.format(epoch + 1),
                        'step:{:04d}/{:04d}'.format(step,n_iterations),
                        'MAE: {:.4f}'.format(MAE),
                        'lr: {}'.format(lr_scheduler.get_last_lr()[0]),
                        'time: {:.4f}s'.format(time.time() - t)
                        ])

        print(str)
        # print(model.manifold.k)
        wandb.log({"MAE ": MAE}, commit=True)
        # curvatures = list(model.get_submodule('curvatures'))
        # print('curvatures:',curvatures)

        loss.backward()
        MAE_sum += MAE
        if args.grad_clip is not None:
            grad_clip = float(args.grad_clip)
            all_params = list(model.parameters())
            for param in all_params:
                torch.nn.utils.clip_grad_value_(param, grad_clip)
        optimizer.step()
        # Roptimizer.step()
        # en_curvatures = model.get_submodule('encoder.curvatures')
        # for p in en_curvatures.parameters():
        #     p.data.clamp_(1e-8)
        # de_curvatures = model.get_submodule('decoder.curvatures')
        # for p in de_curvatures.parameters():
        #     p.data.clamp_(1e-8)
        lr_scheduler.step()
    if (epoch + 1) % args.log_freq == 0:
        str = " ".join(['Epoch: {:04d}'.format(epoch + 1),
                        'lr: {}'.format(lr_scheduler.get_last_lr()[0]),
                        'MAE: {:.4f}'.format(MAE / step),
                        'time: {:.4f}s'.format(time.time() - t)
                        ])
        print(str)
    with torch.no_grad():
        step,MAE_epoch = 0,0
        for input in (val_loader):
            for key in input:
                input[key] = input[key].to(device)
            loss,MAE = model(input)
            if torch.isnan(loss):
                raise AssertionError
            step += 1
            str = " ".join(['step: {:04d}'.format(step),
                            'MAE: {:.4f}'.format(MAE),
                            'time: {:.4f}s'.format(time.time() - t)
                            ])
            print(str)
            MAE_epoch += MAE.item()
        print('epoch:',epoch,'test_mean_MAE:',MAE_epoch/step)
        wandb.log({'epoch':epoch,"test_mean_MAE": MAE_epoch/step}, commit=True)
