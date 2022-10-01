import torch
from schnetpack import Properties
from torch import nn
from layers.hyp_layers import get_dim_act_curv,HNNLayer,HyperbolicGraphConvolution
from layers.layers import get_dim_act,GraphConvolution, Linear
from manifolds.hyperboloid import Hyperboloid
import torch.nn.functional as F
import manifolds

class Encoder(nn.Module):
    """
    Encoder abstract class.
    """

    def __init__(self,args):
        super(Encoder, self).__init__()
        if args.manifold == 'Hyperboloid':
            n_atom_embed = args.feat_dim-4
        else:
            n_atom_embed = args.feat_dim - 3
        self.embedding = nn.Embedding(args.max_z, n_atom_embed, padding_idx=0)

    def forward(self,pos,h,adj):
        h = self.embedding(h)  # (b,n_atom,n_atom_embed)
        h = torch.concat([pos, h], dim=2)  # (b,n_atom,n_atom_embed+3)
        if self.manifold.name == 'Hyperboloid':
            o = torch.zeros_like(h)
            h = torch.cat([o[:, :, 0:1], h], dim=2)  # (b,n_atom,feat_dim)
        return self.encode(h,adj)


    def encode(self, h, adj):
        if self.encode_graph:
            input = (h, adj)
            output , _ = self.layers(input)
        else:
            output = self.layers(h)
        return output

class MLP(Encoder):
    """
    Multi-layer perceptron.
    """

    def __init__(self, args):
        super(MLP, self).__init__(args)
        assert args.num_layers > 0
        dims, acts = get_dim_act(args)
        layers = []
        for i in range(len(dims) - 1):
            in_dim, out_dim = dims[i], dims[i + 1]
            act = acts[i]
            layers.append(Linear(in_dim, out_dim, args.dropout, act, args.bias))
        self.layers = nn.Sequential(*layers)
        self.encode_graph = False

class HNN(Encoder):
    """
    Hyperbolic Neural Networks.
    """

    def __init__(self, args):
        super(HNN, self).__init__(args)

        self.manifold = getattr(manifolds, args.manifold)()
        assert args.num_layers > 1
        dims, acts, self.curvatures = get_dim_act_curv(args)
        self.curvatures.append(nn.Parameter(torch.Tensor([1]).to(args.device)))
        hnn_layers = []
        for i in range(len(dims) - 1):
            c_in, c_out = self.curvatures[i], self.curvatures[i + 1]
            in_dim, out_dim = dims[i], dims[i + 1]
            act = acts[i]
            hnn_layers.append(
                    HNNLayer(self.manifold, in_dim, out_dim, c_in, c_out, args.dropout, act, args.bias)
            )
        self.layers = nn.Sequential(*hnn_layers)
        self.encode_graph = False

    def encode(self, x, adj):
        x_hyp = self.manifold.proj(self.manifold.expmap0(self.manifold.proj_tan0(x, self.curvatures[0]), c=self.curvatures[0]), c=self.curvatures[0])
        output = super(HNN, self).encode(x_hyp, adj)
        output = self.manifold.logmap0(output, self.curvatures[-1])
        output = self.manifold.proj_tan0(output,  self.curvatures[-1])
        return output

class GCN(Encoder):
    """
    Graph Convolution Networks.
    """

    def __init__(self, args):
        super(GCN, self).__init__(args)
        assert args.num_layers > 0
        dims, acts = get_dim_act(args)
        gc_layers = []
        for i in range(len(dims) - 1):
            in_dim, out_dim = dims[i], dims[i + 1]
            act = acts[i]
            gc_layers.append(GraphConvolution(in_dim, out_dim, args.dropout, act, args.bias))
        self.layers = nn.Sequential(*gc_layers)
        self.encode_graph = True

class HGCAE(Encoder):
    """
    Hyperbolic Graph Convolutional Auto-Encoders.
    """

    def __init__(self, args): #, use_cnn
        super(HGCAE, self).__init__(args)
        self.manifold = getattr(manifolds, args.manifold)()
        assert args.num_layers > 0
        dims, acts, self.curvatures = get_dim_act_curv(args)
        if args.c is None:
            self.curvatures.append(nn.Parameter(torch.Tensor([1]).to(args.device)))
        else:
            self.curvatures.append(torch.tensor([args.c]).to(args.device))
        hgc_layers = []
        for i in range(len(dims) - 1):
            c_in, c_out = self.curvatures[i], self.curvatures[i + 1]
            in_dim, out_dim = dims[i], dims[i + 1]
            act = acts[i]
            hgc_layers.append(
                    HyperbolicGraphConvolution(
                            self.manifold, in_dim, out_dim, c_in, c_out, args.dropout, act, args.bias, args.use_att,args.local_agg,
                            att_type=args.att_type, att_logit=args.att_logit
                    )
            )
        self.layers = nn.Sequential(*hgc_layers)
        self.encode_graph = True

    def encode(self, x, adj):
        x_hyp = self.manifold.proj(
                self.manifold.expmap0(self.manifold.proj_tan0(x, self.curvatures[0]), c=self.curvatures[0]),
                c=self.curvatures[0])
        output = super(HGCAE, self).encode(x_hyp, adj)
        output = self.manifold.logmap0(output, self.curvatures[-1])
        output = self.manifold.proj_tan0(output, self.curvatures[-1])
        return output

