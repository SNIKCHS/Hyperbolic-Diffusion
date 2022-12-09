import torch
from schnetpack.datasets import QM9
import schnetpack as spk
import os

from Models.HGCN import HGCN
from my_config import config_args
import optimizers
import numpy as np
import logging
import time
import wandb
import json

class obj(object):
    def __init__(self, dict_):
        self.__dict__.update(dict_)

# no_wandb = False
no_wandb = True
if no_wandb:
    mode = 'disabled'
else:
    mode = 'online'
kwargs = {'entity': 'elma', 'name': 'hgcn_gaussatt', 'project': 'regression',
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

train_loader = spk.AtomsLoader(train, batch_size=4, shuffle=False)
val_loader = spk.AtomsLoader(val, batch_size=4)


args = json.loads(json.dumps(config_args), object_hook=obj)
device = torch.device('cpu')
model = HGCN(device)
tot_params = sum([np.prod(p.size()) for p in model.parameters()])
logging.info(f"Total number of parameters: {tot_params}")
optimizer = getattr(optimizers, args.optimizer)(params=model.parameters(), lr=1e-4,
                                                weight_decay=args.weight_decay)
lr_scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer,
    step_size=args.lr_reduce_freq,
    gamma=float(args.gamma)
)

# Train model
t_total = time.time()
model = model.to(device)
step = 0

for epoch in range(args.epochs):
    model.train()
    loss_sum, n, t = 0, 0, 0.0
    counter = 0
    for input in (train_loader):
        for key in input:
            input[key] = input[key].to(device)

        t = time.time()
        optimizer.zero_grad()
        loss,MAE = model(input)
        if torch.isnan(loss):
            raise AssertionError
        step += 1
        print('step', step, ' loss:', loss,' MAE:',MAE, ' lr: ', lr_scheduler.get_last_lr())
        wandb.log({"MAE ": MAE}, commit=True)
        # curvatures = list(model.get_submodule('encoder.curvatures'))
        # print('encoder:',curvatures)
        # curvatures = list(model.get_submodule('decoder.curvatures'))
        # print('decoder:',curvatures)

        loss.backward()
        loss_sum += loss
        n += 1

        if args.grad_clip is not None:
            grad_clip = float(args.grad_clip)
            all_params = list(model.parameters())
            for param in all_params:
                torch.nn.utils.clip_grad_value_(param, grad_clip)
        optimizer.step()

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
                        'loss: {:.4f}'.format(loss_sum / n),
                        'time: {:.4f}s'.format(time.time() - t)
                        ])
        print(str)
