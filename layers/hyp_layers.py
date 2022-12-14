"""Hyperbolic layers."""
import math
import geoopt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.modules.module import Module

from layers.att_layers import DenseAtt


def get_dim_act_curv(args):
    """
    Helper function to get dimension and activation at every layer.
    :param args:
    :return:
    """
    if not args.act:
        act = lambda x: x
    else:
        act = getattr(F, args.act)
    acts = [act] * (args.num_layers)  # len=args.num_layers
    # dims = [args.dim] * (args.num_layers + 1)  # len=args.num_layers+1
    dims = [args.dim] + [args.hidden_dim] * (args.num_layers)  # len=args.num_layers+1
    # dims = [args.dim] + [args.hidden_dim] * (args.num_layers)  # len=args.num_layers+1

    if args.c is None:
        # create list of trainable curvature parameters
        manifolds = [geoopt.Lorentz(learnable=True) for _ in range(args.num_layers + 1)]
    else:
        # fixed curvature
        manifolds = [geoopt.Lorentz(args.c, learnable=False) for _ in range(args.num_layers + 1)]

    return dims, acts, manifolds


class HNNLayer(nn.Module):
    """
    Hyperbolic neural networks layer.
    """

    def __init__(self, manifold, in_features, out_features, c_in,c_out, dropout, act, use_bias):
        super(HNNLayer, self).__init__()
        self.linear = HypLinear(manifold, in_features, out_features, c_in, dropout, use_bias)
        self.hyp_act = HypAct(manifold, c_in, c_out, act)

    def forward(self, x):
        # print('x:', torch.any(torch.isnan(x)))
        h = self.linear.forward(x)
        # print('linear:', torch.any(torch.isnan(h)))
        h = self.hyp_act.forward(h)
        # print('hyp_act:', torch.any(torch.isnan(h)))
        return h


class HyperbolicGraphConvolution(nn.Module):
    """
    Hyperbolic graph convolution layer.
    """

    def __init__(self, manifold, in_features, out_features, c_in, c_out, dropout, act, use_bias, use_att,local_agg, att_type='adjmask_dist', att_logit=torch.exp, decode=False):
        super(HyperbolicGraphConvolution, self).__init__()
        self.linear = HypLinear(manifold, in_features, out_features, c_in, dropout, use_bias)
        self.agg = HypAgg(manifold, c_in, use_att, out_features, dropout, att_type=att_type, att_logit=att_logit, decode=decode,local_agg=local_agg)
        self.hyp_act = HypAct(manifold, c_in, c_out, act)

    def forward(self, input):
        x, adj = input

        h = self.linear.forward(x)
        # print('linear:',torch.any(torch.isnan(h)))
        h = self.agg.forward(h, adj)
        # print('agg:', torch.any(torch.isnan(h)))
        h = self.hyp_act.forward(h)
        # print('hyp_act:', torch.any(torch.isnan(h)))
        output = (h, adj)

        return output


class HypLinear(nn.Module):
    """
    Hyperbolic linear layer.
    input in manifold
    output in manifold
    """

    def __init__(self, manifold, in_features, out_features, c, dropout, use_bias):
        super(HypLinear, self).__init__()
        self.manifold = manifold
        self.in_features = in_features
        self.out_features = out_features
        self.c = c
        self.dropout = dropout
        self.use_bias = use_bias
        self.bias = nn.Parameter(torch.Tensor(out_features))
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.reset_parameters()

    def reset_parameters(self):
        init.xavier_uniform_(self.weight, gain=math.sqrt(2))
        init.constant_(self.bias, 0)

    def forward(self, x):
        drop_weight = F.dropout(self.weight, self.dropout, training=self.training)

        mv = self.manifold.mobius_matvec(drop_weight, x, self.c)  # x???log???????????????drop_weight?????????exp???manifold

        res = self.manifold.proj(mv, self.c)

        if self.use_bias:
            bias = self.manifold.proj_tan0(self.bias.view(1,1, -1), self.c)
            hyp_bias = self.manifold.expmap0(bias, self.c)
            hyp_bias = self.manifold.proj(hyp_bias, self.c)
            res = self.manifold.mobius_add(res, hyp_bias, c=self.c)
            res = self.manifold.proj(res, self.c)
        return res

    def extra_repr(self):
        return 'in_features={}, out_features={}, c={}'.format(
            self.in_features, self.out_features, self.c
        )


class HypAgg(Module):
    """
    Hyperbolic aggregation layer.
    """

    def __init__(self, manifold, c, use_att, in_features, dropout, att_type='adjmask_dist', att_logit=None, decode=False,local_agg = True):
        super(HypAgg, self).__init__()
        self.manifold = manifold
        self.c = c
        self.in_features = in_features
        self.dropout = dropout
        self.use_att = use_att
        self.local_agg = local_agg
        if self.use_att:
            self.att = DenseAtt(in_features, dropout)
        self.node_mlp = nn.Sequential(
            nn.Linear(2*in_features, in_features),
            nn.SiLU(),
            nn.Linear(in_features, in_features))


    def forward(self, x, adj):
        x_tangent = self.manifold.logmap0(x, c=self.c)  # (b,n_atom,n_embed)
        if self.use_att:
            if self.local_agg:
                b = x.size(0)
                n = x.size(1)
                dim = x.size(2)
                _,mask = adj
                x_provide = x.unsqueeze(2).expand(-1, -1, n, -1).reshape(b, -1,dim)  # (b,n_atom,n_embed) expand (b,n_atom,1,n_embed)->(b,n_atom*n_atom,n_embed) ???????????????
                x_map = x.unsqueeze(1).expand(-1, n, -1, -1).reshape(b, -1, dim)  # (b,n_atom,n_embed) expand (b,1,n_atom,n_embed)->(b,n_atom*n_atom,n_embed) ??????????????????
                x_local_tangent = self.manifold.logmap(x_provide, x_map, c=self.c).reshape(b, n, n,dim)  # (b,n_atom,n_atom,n_embed) ???????????????????????????????????????????????????????????????
                x_local_tangent = torch.clamp(x_local_tangent, min=-1e3, max=1e3)
                x_local_self_tangent = self.manifold.logmap(x, x, c=self.c) # (b,n_atom,n_embed)
                adj_att = self.att(x_local_tangent,x_local_self_tangent, adj)
                att_rep = adj_att.unsqueeze(-1) * x_local_tangent * mask.unsqueeze(-1)

                support_t = torch.sum(att_rep, dim=2)

                support_t = self.node_mlp(torch.concat([x_local_self_tangent,support_t],dim=2))+x_local_self_tangent
                support_t = self.manifold.proj_tan(support_t, x, self.c)

                # support_t = torch.clamp(support_t, min=-1e6, max=1e6)
                output = self.manifold.proj(self.manifold.expmap(support_t, x, c=self.c), c=self.c)

                return output
            else:
                adj_att = self.att(x_tangent,None, adj)  # (b,atom_num,atom_num)
                support_t = torch.matmul(adj_att, x_tangent)
                support_t = self.manifold.proj_tan0(support_t, self.c)
                support_t = torch.clamp(support_t, min=-1e6, max=1e6)
                output = self.manifold.proj(self.manifold.expmap0(support_t, c=self.c), c=self.c)
                # print(output)
        else:
            support_t = torch.bmm(adj[1], x_tangent) # (b,n_atom,n_atom) (b,n_atom,n_embed)
            output = self.manifold.proj(self.manifold.expmap0(support_t, c=self.c), c=self.c)

        return output

    def extra_repr(self):
        return 'c={}'.format(self.c)


class HypAct(Module):
    """
    Hyperbolic activation layer.
    input in manifold
    output in manifold
    """

    def __init__(self, manifold, c_in, c_out, act):
        super(HypAct, self).__init__()
        self.manifold = manifold
        self.c_in = c_in
        self.c_out = c_out
        self.act = act

    def forward(self, x):
        xt = self.act(self.manifold.logmap0(x, c=self.c_in))
        # print('act:', xt[0])
        xt = self.manifold.proj_tan0(xt, c=self.c_out)
        # print('proj_tan0:', xt[0])
        # print('proj_tan0:', torch.any(torch.isnan(xt)))
        out = self.manifold.proj(self.manifold.expmap0(xt, c=self.c_out), c=self.c_out)
        return out

    def extra_repr(self):
        return 'c_in={}, c_out={}'.format(
            self.c_in, self.c_out
        )
