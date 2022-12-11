"""Euclidean layers."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter

from layers.att_layers import DenseAtt


def get_dim_act(args):
    """
    Helper function to get dimension and activation at every layer.
    :param args:
    :return:
    """
    if not args.act:
        act = lambda x: x
    else:
        act = getattr(F, args.act)
    acts = [act] * (args.num_layers)

    dims = [args.dim] * (args.num_layers+1)

    return dims, acts

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

class GCLayer(nn.Module):
    def __init__(self, in_features, out_features, act):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.linear = nn.Linear(in_features, out_features, bias=True)

        self.normalization_factor = 100
        self.aggregation_method = 'sum'
        self.att = DenseAtt(out_features, edge_dim=1)
        # self.node_mlp = nn.Sequential(
        #     nn.Linear(out_features, out_features),
        #     nn.LayerNorm(out_features),
        #     nn.SiLU(),
        #     nn.Linear(out_features, out_features))
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

        # out = self.node_mlp(out)
        out = out + x
        return out


class FermiDiracDecoder(Module):
    """Fermi Dirac to compute edge probabilities based on distances."""

    def __init__(self, r, t):
        super(FermiDiracDecoder, self).__init__()
        self.r = r
        self.t = t

    def forward(self, dist):
        probs = 1. / (torch.exp((dist - self.r) / self.t) + 1)
        return probs

'''
InnerProductDecdoer implemntation from:
https://github.com/zfjsail/gae-pytorch/blob/master/gae/model.py
'''
class InnerProductDecoder(nn.Module):
    """Decoder for using inner product for prediction."""

    def __init__(self, dropout=0, act=torch.sigmoid):
        super(InnerProductDecoder, self).__init__()
        self.dropout = dropout
        self.act = act

    def forward(self, emb_in, emb_out):
        cos_dist = emb_in * emb_out
        probs = self.act(cos_dist.sum(1))
        return probs
