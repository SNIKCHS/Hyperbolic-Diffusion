"""Hyperbolic layers."""
import math

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
    dims = [args.feat_dim] + ([args.dim] * (args.num_layers))  # len=args.num_layers+1
    n_curvatures = args.num_layers  # len=args.num_layers 出去后会+1

    if args.c is None:
        # create list of trainable curvature parameters
        curvatures = nn.ParameterList([nn.Parameter(torch.Tensor([1.])) for _ in range(n_curvatures)])
    else:
        # fixed curvature
        curvatures = [torch.tensor([args.c]) for _ in range(n_curvatures)]
        if not args.cuda == -1:
            curvatures = [curv.to(args.device) for curv in curvatures]
    return dims, acts, curvatures


class HNNLayer(nn.Module):
    """
    Hyperbolic neural networks layer.
    """

    def __init__(self, manifold, in_features, out_features, c_in,c_out, dropout, act, use_bias):
        super(HNNLayer, self).__init__()
        self.linear = HypLinear(manifold, in_features, out_features, c_in, dropout, use_bias)
        self.hyp_act = HypAct(manifold, c_in, c_out, act)

    def forward(self, x):
        h = self.linear.forward(x)
        h = self.hyp_act.forward(h)
        return h


class HyperbolicGraphConvolution(nn.Module):
    """
    Hyperbolic graph convolution layer.
    """

    def __init__(self, manifold, in_features, out_features, c_in, c_out, dropout, act, use_bias, use_att, local_agg):
        super(HyperbolicGraphConvolution, self).__init__()
        self.linear = HypLinear(manifold, in_features, out_features, c_in, dropout, use_bias)
        self.agg = HypAgg(manifold, c_in, out_features, dropout, use_att, local_agg)
        self.hyp_act = HypAct(manifold, c_in, c_out, act)

    def forward(self, input):
        x, adj = input

        h = self.linear.forward(x)
        # print('linear',h)
        h = self.agg.forward(h, adj)
        # print('agg', h)
        h = self.hyp_act.forward(h)
        # print('hyp_act', h)
        output = h, adj


        return output


class HypLinear(nn.Module):
    """
    Hyperbolic linear layer.
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

        mv = self.manifold.mobius_matvec(drop_weight, x, self.c)  # x先log到切空间与drop_weight相乘再exp到manifold

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

    def __init__(self, manifold, c, in_features, dropout, use_att, local_agg):
        super(HypAgg, self).__init__()
        self.manifold = manifold
        self.c = c

        self.in_features = in_features
        self.dropout = dropout
        self.local_agg = local_agg
        self.use_att = use_att
        if self.use_att:
            self.att = DenseAtt(in_features, dropout)

        self.fusion = nn.Sequential(
            nn.Linear(2*in_features,2*in_features),
            nn.Linear(2*in_features,in_features),
        )

    def forward(self, x, adj):
        # print('x:',torch.isnan(x).any())
        x_tangent = self.manifold.logmap0(x, c=self.c) # (b,n_atom,n_embed)
        # print('x_tangent:', torch.isnan(x_tangent).any())
        if self.use_att:
            if self.local_agg:
                # x_local_tangent = []
                # # (b,n_atom,n_embed) expand (b,n_atom,1,n_embed)->(b,n_atom,n_atom,n_embed) 提供切空间
                # # (b,n_atom,n_embed) expand (b,1,n_atom,n_embed)->(b,n_atom,n_atom,n_embed) 要映射的向量
                # for i in range(x.size(1)):
                #     temp = self.manifold.logmap(x[:,i], x, c=self.c)  # 把所有原子投影到第i个原子的切空间  (b,n_atom,n_embed)
                #     x_local_tangent.append(temp)
                #
                # x_local_tangent = torch.stack(x_local_tangent, dim=1) # (b,n_atom,n_atom,n_embed)
                b = x.size(0)
                n = x.size(1)
                dim = x.size(2)
                x_provide = x.unsqueeze(2).expand(-1,-1,n,-1).reshape(b,-1,dim)
                x_map = x.unsqueeze(1).expand(-1,n,-1,-1).reshape(b,-1,dim)
                x_local_tangent = self.manifold.logmap(x_provide, x_map, c=self.c).reshape(b,n,n,dim) # (b,n_atom*n_atom,n_embed)
                # print('x_local_tangent:', torch.isnan(x_local_tangent).any())
                x_tangent_self = self.manifold.logmap(x,x, c=self.c) # (b,n_atom,n_embed)
                adj_att = self.att(x_local_tangent,x_tangent_self, adj).unsqueeze(-1)  # (b,atom_num,atom_num,1)

                adj_att = adj_att.expand(-1,-1,-1,x_local_tangent.size()[3])

                att_rep = adj_att * x_local_tangent # (b,n_atom,n_atom,n_embed)
                # print('att_rep:', torch.isnan(att_rep).any())
                # print('att_rep',att_rep)
                support_t = torch.sum(att_rep, dim=2)# (b,n_atom,n_embed)
                # print('support_t:', torch.isnan(support_t).any())
                support_t = self.fusion(torch.concat([support_t,x_tangent_self],dim=2))
                support_t = self.manifold.proj_tan(support_t,x,self.c)
                # print('att_rep:', att_rep)
                support_t = torch.clamp(support_t,min=-1e6,max=1e6)
                output = self.manifold.proj(self.manifold.expmap(support_t, x, c=self.c), c=self.c)  #需要对每个support_t[:,i]从x[:,i]的切空间映射回流形
                # print(output)
                # print('output:', torch.isnan(output).any())
                return output
            else:
                adj_att = self.att(x_tangent, adj)  # (b,atom_num,atom_num)
                support_t = torch.matmul(adj_att, x_tangent)+x_tangent
                support_t = self.manifold.proj_tan0(support_t,self.c)
        else:
            support_t = torch.spmm(adj, x_tangent)
        output = self.manifold.proj(self.manifold.expmap0(support_t, c=self.c), c=self.c)
        # print(output)
        return output

    def extra_repr(self):
        return 'c={}'.format(self.c)


class HypAct(Module):
    """
    Hyperbolic activation layer.
    """

    def __init__(self, manifold, c_in, c_out, act):
        super(HypAct, self).__init__()
        self.manifold = manifold
        self.c_in = c_in
        self.c_out = c_out
        self.act = act

    def forward(self, x):
        xt = self.act(self.manifold.logmap0(x, c=self.c_in))
        xt = self.manifold.proj_tan0(xt, c=self.c_out)
        return self.manifold.proj(self.manifold.expmap0(xt, c=self.c_out), c=self.c_out)

    def extra_repr(self):
        return 'c_in={}, c_out={}'.format(
            self.c_in, self.c_out
        )
