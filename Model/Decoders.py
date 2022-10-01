import manifolds
import torch.nn as nn
import torch.nn.functional as F
import torch
from layers.layers import GraphConvolution, Linear, get_dim_act

class Decoder(nn.Module):
    """
    Decoder abstract class
    """

    def __init__(self, c,args):
        super(Decoder, self).__init__()
        self.c = c

        self.out = Linear(args.feat_dim,args.max_z+3,None,None,True)

    def decode(self, x, adj):
        '''
        output
        - nc : probs
        - rec : input_feat
        '''
        if self.decode_adj:
            input = (x, adj)
            output, _ = self.decoder.forward(input)
        else:
            output = self.decoder.forward(x)

        # if self.c is not None:
        #     output = self.manifold.logmap0(output, self.curvatures[-1])
        #     output = self.manifold.proj_tan0(output,  self.curvatures[-1])
        output = self.out(output)
        return output





class GCNDecoder(Decoder):
    """
    Graph Convolution Decoder.
    """

    def __init__(self, c, args):
        super(GCNDecoder, self).__init__(c,args)

        assert args.num_layers > 0
        dims, acts = get_dim_act(args)
        dims = dims[::-1]
        # acts = acts[::-1]
        acts = acts[::-1][:-1] + [lambda x: x]  # Last layer without act
        gc_layers = []
        for i in range(len(dims) - 1):
            in_dim, out_dim = dims[i], dims[i + 1]
            act = acts[i]
            gc_layers.append(GraphConvolution(in_dim, out_dim, args.dropout, act, args.bias))
        self.decoder = nn.Sequential(*gc_layers)

        self.decode_adj = True


class LinearDecoder(Decoder):
    """
    MLP Decoder for Hyperbolic/Euclidean
    """

    # NOTE : self.c is fixed, not trainable
    def __init__(self, c, args):
        super(LinearDecoder, self).__init__(c,args)
        self.manifold = getattr(manifolds, args.manifold)()
        self.input_dim = args.dim
        self.output_dim = args.n_classes
        self.bias = args.bias
        self.decoder = Linear(self.input_dim, self.output_dim, args.dropout, lambda x: x, self.bias)
        self.decode_adj = False


    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}, c={}'.format(
            self.input_dim, self.output_dim, self.bias, self.c
        )


import layers.hyp_layers as hyp_layers

class HGCAEDecoder(Decoder):
    """
    Decoder for HGCAE
    """

    def __init__(self, c, args):
        super(HGCAEDecoder, self).__init__(c,args)
        self.manifold = getattr(manifolds, args.manifold)()


        assert args.num_layers > 0

        dims, acts, _ = hyp_layers.get_dim_act_curv(args)
        dims = dims[::-1] # 倒序
        acts = acts[::-1][:-1] + [lambda x: x]  # Last layer without act
        self.curvatures = self.c[::-1]

        if not args.encdec_share_curvature and args.num_layers == args.num_dec_layers:  # do not share and enc-dec mirror-shape
            num_c = len(self.curvatures)
            self.curvatures = self.curvatures[:1]
            if args.c is None:
                self.curvatures += [nn.Parameter(torch.Tensor([1]).to(args.device))] * (num_c - 1)
            else:
                self.curvatures += [torch.tensor([args.c])] * (num_c - 1)
                if not args.cuda == -1:
                    self.curvatures = [curv.to(args.device) for curv in self.curvatures]

        # self.curvatures = self.curvatures[:-1] + [None]

        hgc_layers = []
        num_dec_layers = args.num_dec_layers
        for i in range(num_dec_layers):
            c_in, c_out = self.curvatures[i], self.curvatures[i + 1]
            in_dim, out_dim = dims[i], dims[i + 1]
            act = acts[i]
            hgc_layers.append(
                hyp_layers.HyperbolicGraphConvolution(
                    self.manifold, in_dim, out_dim, c_in, c_out, args.dropout, act, args.bias, args.use_att,args.local_agg,
                    att_type=args.att_type, att_logit=args.att_logit, decode=True
                )
            )

        self.decoder = nn.Sequential(*hgc_layers)
        self.decode_adj = True

    def decode(self, x, adj):
        output = super(HGCAEDecoder, self).decode(x, adj)
        return output


class HNNDecoder(Decoder):
    """
    Decoder for HNN
    """

    def __init__(self, c, args):
        super(HNNDecoder, self).__init__(c,args)
        self.manifold = getattr(manifolds, args.manifold)()

        assert args.num_layers > 0

        dims, acts, _ = hyp_layers.get_dim_act_curv(args)
        dims = dims[::-1]
        acts = acts[::-1][:-1] + [lambda x: x]  # Last layer without act

        self.curvatures = c[::-1]

        if not args.encdec_share_curvature and args.num_layers == args.num_dec_layers:  # do not share and enc-dec mirror-shape
            num_c = len(self.curvatures)
            self.curvatures = self.curvatures[:1]
            if args.c is None:
                self.curvatures += [nn.Parameter(torch.Tensor([1]).to(args.device))] * (num_c - 1)
            else:
                self.curvatures += [torch.tensor([args.c])] * (num_c - 1)
                if not args.cuda == -1:
                    self.curvatures = [curv.to(args.device) for curv in self.curvatures]

        hnn_layers = []
        num_dec_layers = args.num_dec_layers
        for i in range(num_dec_layers):
            in_dim, out_dim = dims[i], dims[i + 1]
            act = acts[i]
            c_in, c_out = self.curvatures[i], self.curvatures[i + 1]

            hnn_layers.append(
                hyp_layers.HNNLayer(
                    self.manifold, in_dim, out_dim, c_in, c_out, args.dropout, act, args.bias
                )
            )

        self.decoder = nn.Sequential(*hnn_layers)
        self.decode_adj = False

    def decode(self, x, adj):
        output = super(HNNDecoder, self).decode(x, adj)
        return output


model2decoder = {
    'GCN': GCNDecoder,
    'HNN': HNNDecoder,
    'HGCAE': HGCAEDecoder,
    'MLP': LinearDecoder,
}

