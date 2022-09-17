"""Base model class."""

import numpy as np

from sklearn.metrics import roc_auc_score, average_precision_score
import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.layers import FermiDiracDecoder
import manifolds
import old.encoders as encoders
from old.decoders import RegressionDecoder, model2decoder
from utils.eval_utils import acc_f1
from schnetpack import Properties

class BaseModel(nn.Module):
    """
    Base model for graph embedding tasks.
    """

    def __init__(self, args):
        super(BaseModel, self).__init__()
        self.manifold_name = args.manifold
        if args.c is not None: # 双曲半径，设置为None表示可训练的曲率
            self.c = torch.tensor([args.c])
            if not args.cuda == -1:
                self.c = self.c.to(args.device)
        else:
            self.c = nn.Parameter(torch.Tensor([1.]))
        self.manifold = getattr(manifolds, self.manifold_name)()  # 选择相应的流形
        if self.manifold.name == 'Hyperboloid':
            args.feat_dim = args.feat_dim + 1
        self.nnodes = args.n_nodes
        self.encoder = getattr(encoders, args.model)(self.c, args)

    def encode(self, x, adj):
        if self.manifold.name == 'Hyperboloid':
            o = torch.zeros_like(x)
            x = torch.cat([o[:, 0:1], x], dim=1)
        h = self.encoder.encode(x, adj)
        return h

    def compute_metrics(self, embeddings, data, split):
        raise NotImplementedError

    def init_metric_dict(self):
        raise NotImplementedError

    def has_improved(self, m1, m2):
        raise NotImplementedError



class NCModel(BaseModel):
    """
    Base model for node classification task.
    """

    def __init__(self, args):
        super(NCModel, self).__init__(args)
        self.decoder = model2decoder[args.model](self.c, args)
        if args.n_classes > 2:
            self.f1_average = 'micro'
        else:
            self.f1_average = 'binary'
        if args.pos_weight:  # 是否在节点中增加正类的权重
            # self.weights = torch.Tensor([1., 1. / data['labels'][idx_train].mean()])
            pass
        else:
            self.weights = torch.Tensor([1.] * args.n_classes)
        if not args.cuda == -1:
            self.weights = self.weights.to(args.device)

    def decode(self, h, adj, idx):
        output = self.decoder.decode(h, adj)
        return F.log_softmax(output[idx], dim=1)

    def compute_metrics(self, embeddings, data, split):  # 计算loss，acc等
        idx = data[f'idx_{split}']
        output = self.decode(embeddings, data['adj_train_norm'], idx)
        loss = F.nll_loss(output, data['labels'][idx], self.weights)
        acc, f1 = acc_f1(output, data['labels'][idx], average=self.f1_average)
        metrics = {'loss': loss, 'acc': acc, 'f1': f1}
        return metrics

    def init_metric_dict(self):
        return {'acc': -1, 'f1': -1}

    def has_improved(self, m1, m2):
        return m1["f1"] < m2["f1"]


class LPModel(BaseModel):
    """
    Base model for link prediction task.
    """

    def __init__(self, args):
        super(LPModel, self).__init__(args)
        self.dc = FermiDiracDecoder(r=args.r, t=args.t)  # 类似于sigmoid的输出，r,t为超参数
        self.nb_false_edges = args.nb_false_edges
        self.nb_edges = args.nb_edges

    def decode(self, h, idx):
        if self.manifold_name == 'Euclidean':
            h = self.manifold.normalize(h)
        emb_in = h[idx[:, 0], :]
        emb_out = h[idx[:, 1], :]
        sqdist = self.manifold.sqdist(emb_in, emb_out, self.c)
        probs = self.dc.forward(sqdist)
        return probs

    def compute_metrics(self, embeddings, data, split):
        if split == 'train':
            edges_false = data[f'{split}_edges_false'][np.random.randint(0, self.nb_false_edges, self.nb_edges)]
        else:
            edges_false = data[f'{split}_edges_false']
        pos_scores = self.decode(embeddings, data[f'{split}_edges'])
        neg_scores = self.decode(embeddings, edges_false)
        loss = F.binary_cross_entropy(pos_scores, torch.ones_like(pos_scores))
        loss += F.binary_cross_entropy(neg_scores, torch.zeros_like(neg_scores))
        if pos_scores.is_cuda:
            pos_scores = pos_scores.cpu()
            neg_scores = neg_scores.cpu()
        labels = [1] * pos_scores.shape[0] + [0] * neg_scores.shape[0]
        preds = list(pos_scores.data.numpy()) + list(neg_scores.data.numpy())
        roc = roc_auc_score(labels, preds)
        ap = average_precision_score(labels, preds)
        metrics = {'loss': loss, 'roc': roc, 'ap': ap}
        return metrics

    def init_metric_dict(self):
        return {'roc': -1, 'ap': -1}

    def has_improved(self, m1, m2):
        return 0.5 * (m1['roc'] + m1['ap']) < 0.5 * (m2['roc'] + m2['ap'])

from schnetpack.datasets import QM9
from schnetpack.nn import AtomDistances
class RegModel(nn.Module):
    """
    分子的回归模型
    """
    def __init__(self, args):
        super(RegModel, self).__init__()
        self.manifold_name = args.manifold
        if args.c is not None:  # 双曲半径，设置为None表示可训练的曲率
            self.c = torch.tensor([args.c])
            if not args.cuda == -1:
                self.c = self.c.to(args.device)
        else:
            self.c = nn.Parameter(torch.Tensor([1.]))
        self.manifold = getattr(manifolds, self.manifold_name)()  # 选择相应的流形
        if self.manifold.name == 'Hyperboloid':
            args.feat_dim = args.feat_dim + 1
        self.distances = AtomDistances()
        self.encoder = getattr(encoders, args.model)(self.c, args)
        self.decoder = RegressionDecoder(self.c, args)
        self.embedding = nn.Embedding(args.max_z, args.n_atom_embed, padding_idx=0)


    def encode(self, inputs):

        atomic_numbers = inputs[Properties.Z]  # (b,n_atom)
        positions = inputs[Properties.R]       # (b,n_atom,3)
        cell = inputs[Properties.cell]         # useless
        cell_offset = inputs[Properties.cell_offset]         # useless
        neighbors = inputs[Properties.neighbors]   # (b,n_atom,n_atom-1)
        neighbor_mask = inputs[Properties.neighbor_mask]
        atom_mask = inputs[Properties.atom_mask]   # (b,n_atom)


        size = atom_mask.size()
        adj = atom_mask.unsqueeze(2).expand(size[0], size[1], size[1])
        adj = adj * adj.permute(0, 2, 1)
        n = atom_mask.sum(1).view(-1, 1, 1).expand(-1, size[1], size[1])
        adj = adj / n  # (b,n_atom,n_atom) atom_mask like ,归一化


        x = self.embedding(atomic_numbers)  # (b,n_atom,n_atom_embed)

        x = torch.concat([x,positions],dim=2)  # (b,n_atom,n_atom_embed+3)
        r_ij = self.distances(
            positions, neighbors, cell, cell_offset, neighbor_mask=neighbor_mask
        )
        # 暂时不使用pair-wise距离
        if self.manifold.name == 'Hyperboloid':
            o = torch.zeros_like(x)
            x = torch.cat([o[:,:, 0:1], x], dim=2)  # (b,n_atom,n_atom_embed+3+1)

        h = self.encoder.encode(x, adj)
        return h

    def decode(self, h, adj):
        output = self.decoder.decode(h, adj)
        return output

    def mse_loss(self,batch, result):
        # print('b:',batch)
        # print('u0:',result[QM9.U0])
        diff = batch - result[QM9.U0]

        err_sq = torch.mean(diff ** 2)
        return err_sq

    def compute_metrics(self, embeddings, data):  # 计算loss，acc等
        output = self.decode(embeddings, None)
        loss = self.mse_loss(output, data)
        acc, f1 = None,None
        metrics = {'loss': loss, 'acc': acc, 'f1': f1}
        return metrics

    def init_metric_dict(self):
        return {'acc': -1, 'f1': -1}

    def has_improved(self, m1, m2):
        return m1["f1"] < m2["f1"]