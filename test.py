import torch
import torch.nn as nn
from schnetpack.datasets import QM9
import schnetpack as spk
import os
from my_config import config_args
import optimizers
import numpy as np
import logging
import time
import wandb

from torch.nn import init
from schnetpack.nn import AtomDistances, HardCutoff
from schnetpack import Properties
from manifolds import Hyperboloid
# no_wandb = False
no_wandb = True
if no_wandb:
    mode = 'disabled'
else:
    mode = 'online'
kwargs = {'entity': 'elma', 'name': 'hgcn', 'project': 'regression',
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


class DenseAtt(nn.Module):
    def __init__(self, in_features, edge_dim=1):
        super(DenseAtt, self).__init__()
        self.att_mlp = nn.Sequential(
            nn.Linear(2 * in_features + edge_dim, 2 * in_features, bias=True),
            nn.SiLU(),
            nn.Linear(2 * in_features, in_features, bias=True),
            nn.SiLU(),
            nn.Linear(in_features, 1),
            nn.Sigmoid()
        )
        self.in_features = in_features

    def forward(self, x_left, x_right, distances, edge_mask):
        distances = distances * edge_mask
        x_cat = torch.concat((x_left, x_right, distances), dim=1)  # (b*n*n,2*dim+1)
        att = self.att_mlp(x_cat)  # (b*n_node*n_node,1)

        return att * edge_mask


class ResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block1 = nn.Sequential(
            # nn.LazyBatchNorm1d(),
            nn.Linear(dim, dim),
            nn.SiLU(),
        )
        self.block2 = nn.Sequential(
            # nn.LazyBatchNorm1d(),
            nn.Linear(dim, dim),
            nn.SiLU(),
        )

    def forward(self, x):
        h = self.block1(x)
        h = self.block2(h)
        h = h + x
        return h


def unsorted_segment_sum(data, segment_ids, num_segments, normalization_factor, aggregation_method: str):
    """Custom PyTorch op to replicate TensorFlow's `unsorted_segment_sum`.
        Normalization: 'sum' or 'mean'.
    """
    result_shape = (num_segments, data.size(1))
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result.scatter_add_(0, segment_ids, data)
    if aggregation_method == 'sum':
        result = result / normalization_factor

    if aggregation_method == 'mean':
        norm = data.new_zeros(result.shape)
        norm.scatter_add_(0, segment_ids, data.new_ones(data.shape))
        norm[norm == 0] = 1
        result = result / norm
    return result
def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight,gain=0.25)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


class GCLayer(nn.Module):
    def __init__(self, in_features, out_features, act):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.linear = nn.Linear(in_features, out_features, bias=True)

        self.normalization_factor = 100
        self.aggregation_method = 'sum'
        self.att = DenseAtt(out_features, edge_dim=1)
        self.node_mlp = nn.Sequential(
            nn.Linear(out_features, out_features),
            nn.LayerNorm(out_features),
            nn.SiLU(),
            nn.Linear(out_features, out_features))
        self.act = act

        self.ln = nn.LayerNorm(out_features)


    def forward(self, input):
        h, distances, edges, node_mask, edge_mask = input

        h = self.linear(h)
        h = self.Agg(h, distances, edges, node_mask, edge_mask)
        h = self.ln(h)
        h = self.act(h)
        output = (h, distances, edges, node_mask, edge_mask)
        return output


    def Agg(self, x, distances, edges, node_mask, edge_mask):
        row, col = edges  # 0,0,0...0,1 0,1,2..,0
        x_row = x[row]
        x_col = x[col]
        att = self.att(x_row, x_col, distances, edge_mask)  # (b*n_node*n_node,dim)

        agg = x_col * att

        out = unsorted_segment_sum(agg, row, num_segments=x.size(0),  # num_segments=b*n_nodes
                                   normalization_factor=self.normalization_factor,
                                   aggregation_method=self.aggregation_method)  # sum掉第二个n_nodes (b*n_nodes*n_nodes,dim)->(b*n_nodes,dim)

        out = self.node_mlp(out)
        out = out + x
        return out
class HGCLayer(nn.Module):
    def __init__(self, manifold, in_features, out_features, c_in, c_out, act):
        super().__init__()
        self.manifold = manifold
        self.in_features = in_features
        self.out_features = out_features
        self.c_in = c_in
        self.bias = nn.Parameter(torch.Tensor(1, out_features))
        self.linear = nn.Linear(in_features, out_features, bias=False)

        self.normalization_factor = 100
        self.aggregation_method = 'sum'
        self.att = DenseAtt(out_features, edge_dim=1)
        self.node_mlp = nn.Sequential(
            nn.Linear(out_features, out_features),
            nn.LayerNorm(out_features),
            nn.SiLU(),
            nn.Linear(out_features, out_features))

        self.c_out = c_out
        self.act = act
        if self.manifold.name == 'Hyperboloid':
            self.ln = nn.LayerNorm(out_features - 1)
        else:
            self.ln = nn.LayerNorm(out_features)
        self.reset_parameters()

    def reset_parameters(self):
        # init.xavier_uniform_(self.linear.weight, gain=0.01)
        init.constant_(self.bias, 0.1)

    def forward(self, input):
        h, distances, edges, node_mask, edge_mask = input

        h = self.HypLinear(h)

        h = self.HypAgg(h, distances, edges, node_mask, edge_mask)
        h = self.HNorm(h)
        h = self.HypAct(h)
        output = (h, distances, edges, node_mask, edge_mask)
        return output

    def HypLinear(self, x):
        x = self.manifold.logmap0(x, self.c_in)
        x = self.linear(x)
        x = self.manifold.proj_tan0(x, self.c_in)
        x = self.manifold.expmap0(x, self.c_in)
        bias = self.manifold.proj_tan0(self.bias.view(1, -1), self.c_in)
        hyp_bias = self.manifold.expmap0(bias, self.c_in)
        res = self.manifold.mobius_add(x, hyp_bias, self.c_in)
        return res

    def HypAgg(self, x, distances, edges, node_mask, edge_mask):
        x_tangent = self.manifold.logmap0(x, c=self.c_in)  # (b*n_node,dim)
        row, col = edges  # 0,0,0...0,1 0,1,2..,0
        x_tangent_row = x_tangent[row]
        x_tangent_col = x_tangent[col]

        x_local_tangent = self.manifold.logmap(x[row], x[col], c=self.c_in)  # (b*n_node*n_node,dim)  x_col落在x_row的切空间

        att = self.att(x_tangent_row, x_tangent_col, distances, edge_mask)  # (b*n_node*n_node,dim)

        agg = x_local_tangent * att

        out = unsorted_segment_sum(agg, row, num_segments=x_tangent.size(0),  # num_segments=b*n_nodes
                                   normalization_factor=self.normalization_factor,
                                   aggregation_method=self.aggregation_method)  # sum掉第二个n_nodes (b*n_nodes*n_nodes,dim)->(b*n_nodes,dim)

        out = self.node_mlp(out)
        support_t = self.manifold.proj_tan(out, x, self.c_in)
        output = self.manifold.expmap(support_t, x, c=self.c_in)

        return output

    def HypAct(self, x):
        xt = self.act(self.manifold.logmap0(x, c=self.c_in))
        xt = self.manifold.proj_tan0(xt, c=self.c_out)
        out = self.manifold.expmap0(xt, c=self.c_out)
        return out

    def HNorm(self, x):
        h = self.manifold.logmap0(x, self.c_in)
        if self.manifold.name == 'Hyperboloid':
            h[..., 1:] = self.ln(h[..., 1:].clone())
        else:
            h = self.ln(h)
        h = self.manifold.expmap0(h, c=self.c_in)
        return h


class HGCN(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device
        self.manifold = Hyperboloid()
        self.act = nn.SiLU()
        self.c = 1
        self.Layer = nn.Sequential(
            HGCLayer(self.manifold, 20, 128, self.c, self.c, self.act),
            HGCLayer(self.manifold, 128, 128, self.c, self.c, self.act),
            HGCLayer(self.manifold, 128, 128, self.c, self.c, self.act),
            HGCLayer(self.manifold, 128, 128, self.c, self.c, self.act),
            HGCLayer(self.manifold, 128, 128, self.c, self.c, self.act),
        )
        self.embedding = nn.Embedding(10, 20)
        self.distances = AtomDistances()
        self._edges_dict = {}
        self.out = nn.Sequential(
            nn.Linear(128, 64),
            nn.SiLU(),
            # ResBlock(64),
            nn.Linear(64, 1)
        )
        self.loss_fn = nn.MSELoss(reduction='mean')
        self.cutoff = HardCutoff()
        self.apply(weight_init)

    def forward(self, inputs):
        atomic_numbers = inputs[Properties.Z]  # (b,n_atom)
        positions = inputs[Properties.R]  # (b,n_atom,3)
        positions -= positions.mean(dim=1, keepdim=True)
        node_mask = inputs[Properties.atom_mask]  # (b,n_atom)
        u0 = inputs['energy_U0']
        batch_size, n_nodes = atomic_numbers.size()

        size = node_mask.size()
        edge_mask = node_mask.unsqueeze(2).expand(size[0], size[1], size[1])  # (b,n_atom,n_atom)
        edge_mask = edge_mask * edge_mask.permute(0, 2, 1)

        ar = torch.arange(atomic_numbers.size(1), device=atomic_numbers.device)[None, None, :].repeat(
            atomic_numbers.size(0), atomic_numbers.size(1), 1)  # (b,n_atom,n_atom)
        nbh = ar * edge_mask
        h = self.embedding(atomic_numbers)  # (b,n_atom,embed)

        distance = self.distances(positions, nbh.long(), neighbor_mask=edge_mask.bool())
        edges = self.get_adj_matrix(n_nodes, batch_size)

        h = h.view(batch_size * n_nodes, -1)
        h = self.manifold.proj_tan0(h, self.c)
        h = self.manifold.expmap0(h, self.c)
        distance = distance.view(batch_size * n_nodes * n_nodes, 1)
        node_mask = node_mask.view(batch_size * n_nodes, -1)
        edge_mask = edge_mask.view(batch_size * n_nodes * n_nodes, 1)
        edge_mask = self.cutoff(distance) * edge_mask

        input = (h, distance, edges, node_mask, edge_mask)
        output, distances, edges, node_mask, edge_mask = self.Layer(input)
        # output = self.manifold.logmap0(output, self.c)
        output = self.out(output) * node_mask
        output = output.view(batch_size, n_nodes).sum(1, keepdim=True)
        # print(output)
        # print(u0)
        loss = torch.sqrt(self.loss_fn(output, u0))
        MAE_loss = torch.nn.functional.l1_loss(output, u0, reduction='mean')
        return loss, MAE_loss

    def get_adj_matrix(self, n_nodes, batch_size):
        # 对每个n_nodes，batch_size只要算一次
        if n_nodes in self._edges_dict:
            edges_dic_b = self._edges_dict[n_nodes]
            if batch_size in edges_dic_b:
                return edges_dic_b[batch_size]
            else:
                # get edges for a single sample
                rows, cols = [], []
                for batch_idx in range(batch_size):
                    for i in range(n_nodes):
                        for j in range(n_nodes):
                            rows.append(i + batch_idx * n_nodes)
                            cols.append(j + batch_idx * n_nodes)
                edges = [torch.LongTensor(rows).to(self.device),
                         torch.LongTensor(cols).to(self.device)]
                edges_dic_b[batch_size] = edges
                return edges
        else:
            self._edges_dict[n_nodes] = {}
            return self.get_adj_matrix(n_nodes, batch_size)

class GCN(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device
        self.act = nn.SiLU()
        self.Layer = nn.Sequential(
            GCLayer( 20, 128,  self.act),
            GCLayer(128, 128, self.act),
            GCLayer(128, 128, self.act),
            GCLayer(128, 128, self.act),
            GCLayer(128, 128, self.act),
        )
        self.embedding = nn.Embedding(10, 20)
        self.distances = AtomDistances()
        self._edges_dict = {}
        self.out = nn.Sequential(
            nn.Linear(128, 64),
            nn.SiLU(),
            # ResBlock(64),
            nn.Linear(64, 1)
        )
        self.loss_fn = nn.MSELoss(reduction='mean')
        self.cutoff = HardCutoff()
        self.apply(weight_init)

    def forward(self, inputs):
        atomic_numbers = inputs[Properties.Z]  # (b,n_atom)
        positions = inputs[Properties.R]  # (b,n_atom,3)
        positions -= positions.mean(dim=1, keepdim=True)
        node_mask = inputs[Properties.atom_mask]  # (b,n_atom)
        u0 = inputs['energy_U0']
        batch_size, n_nodes = atomic_numbers.size()

        size = node_mask.size()
        edge_mask = node_mask.unsqueeze(2).expand(size[0], size[1], size[1])  # (b,n_atom,n_atom)
        edge_mask = edge_mask * edge_mask.permute(0, 2, 1)

        ar = torch.arange(atomic_numbers.size(1), device=atomic_numbers.device)[None, None, :].repeat(
            atomic_numbers.size(0), atomic_numbers.size(1), 1)  # (b,n_atom,n_atom)
        nbh = ar * edge_mask
        h = self.embedding(atomic_numbers)  # (b,n_atom,embed)

        distance = self.distances(positions, nbh.long(), neighbor_mask=edge_mask.bool())
        edges = self.get_adj_matrix(n_nodes, batch_size)

        h = h.view(batch_size * n_nodes, -1)

        distance = distance.view(batch_size * n_nodes * n_nodes, 1)
        node_mask = node_mask.view(batch_size * n_nodes, -1)
        edge_mask = edge_mask.view(batch_size * n_nodes * n_nodes, 1)
        edge_mask = self.cutoff(distance) * edge_mask

        input = (h, distance, edges, node_mask, edge_mask)
        output, distances, edges, node_mask, edge_mask = self.Layer(input)
        output = self.out(output) * node_mask
        output = output.view(batch_size, n_nodes).sum(1, keepdim=True)
        # print(output)
        # print(u0)
        loss = torch.sqrt(self.loss_fn(output, u0))
        MAE_loss = torch.nn.functional.l1_loss(output, u0,reduction='mean')
        return loss,MAE_loss

    def get_adj_matrix(self, n_nodes, batch_size):
        # 对每个n_nodes，batch_size只要算一次
        if n_nodes in self._edges_dict:
            edges_dic_b = self._edges_dict[n_nodes]
            if batch_size in edges_dic_b:
                return edges_dic_b[batch_size]
            else:
                # get edges for a single sample
                rows, cols = [], []
                for batch_idx in range(batch_size):
                    for i in range(n_nodes):
                        for j in range(n_nodes):
                            rows.append(i + batch_idx * n_nodes)
                            cols.append(j + batch_idx * n_nodes)
                edges = [torch.LongTensor(rows).to(self.device),
                         torch.LongTensor(cols).to(self.device)]
                edges_dic_b[batch_size] = edges
                return edges
        else:
            self._edges_dict[n_nodes] = {}
            return self.get_adj_matrix(n_nodes, batch_size)
import json

class obj(object):
    def __init__(self, dict_):
        self.__dict__.update(dict_)


args = json.loads(json.dumps(config_args), object_hook=obj)
device = torch.device('cpu')
model = HGCN(device)
total = sum([param.nelement() for param in model.parameters()])
print("Number of parameter: %.2fM" % (total / 1e6))

optimizer = getattr(optimizers, args.optimizer)(params=model.parameters(), lr=1e-4,
                                                weight_decay=args.weight_decay)
lr_scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer,
    step_size=args.lr_reduce_freq,
    gamma=float(args.gamma)
)
tot_params = sum([np.prod(p.size()) for p in model.parameters()])
logging.info(f"Total number of parameters: {tot_params}")

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
