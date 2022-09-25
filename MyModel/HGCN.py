import torch
from schnetpack import Properties
from torch import nn
from layers import hyp_layers
from manifolds.hyperboloid import Hyperboloid
import torch.nn.functional as F

class Encoder(nn.Module):
    """
    Hyperbolic-GCN.
    """

    def __init__(self, c, args):
        super(Encoder, self).__init__()
        self.c = c
        self.manifold = Hyperboloid()
        assert args.num_layers > 1
        dims, acts, self.curvatures = hyp_layers.get_dim_act_curv(args)  # 从参数中获取每一层的维度、激活和曲率
        self.curvatures.append(self.c)
        hgc_layers = []

        for i in range(args.num_layers):
            c_in, c_out = self.curvatures[i], self.curvatures[i + 1]
            in_dim, out_dim = dims[i], dims[i + 1]
            act = acts[i]
            hgc_layers.append(
                hyp_layers.HyperbolicGraphConvolution(  # 设置每一层的卷积操作 return h, adj
                    self.manifold, in_dim, out_dim, c_in, c_out, args.dropout, act, args.bias, args.use_att,
                    args.local_agg
                )
            )
        self.layers = nn.Sequential(*hgc_layers)
        self.encode_graph = True

    def encode(self, x, adj):
        x_tan = self.manifold.proj_tan0(x, self.curvatures[0])  # 把x（欧式向量）投影到原点的切空间
        x_hyp = self.manifold.expmap0(x_tan, c=self.curvatures[0])  # 在原点把x_tan指数映射
        input = (x_hyp, adj)
        output, _ = self.layers.forward(input)  # 产生了NaN
        output = self.manifold.logmap0(output, self.c)
        output = self.manifold.proj_tan0(output, self.c)
        return output


class Decoder(nn.Module):
    """
    MLP Decoder for Hyperbolic/Euclidean node classification models.
    """

    def __init__(self, c, args):
        super(Decoder, self).__init__()
        self.c = c

        self.manifold = Hyperboloid()
        self.input_dim = args.dim
        self.output_dim = 1
        self.bias = args.bias
        self.cls = nn.Sequential(
            nn.Linear(self.input_dim, self.input_dim, self.bias),
            nn.Linear(self.input_dim, self.input_dim, self.bias),
            nn.Linear(self.input_dim, self.input_dim, self.bias),
            nn.Linear(self.input_dim, self.output_dim, self.bias),
        )

        self.decode_adj = False

    def decode(self, x, adj):
        x = self.manifold.proj_tan0(self.manifold.logmap0(x, c=self.c), c=self.c)
        probs = self.cls.forward(x)
        probs = torch.sum(probs, dim=1)
        return probs

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}, c={}'.format(
            self.input_dim, self.output_dim, self.bias, self.c
        )


class RegModel(nn.Module):
    """
    分子的回归模型

    数据先proj_tan0再expmap0
    """

    def get_dim_act_curv(self,args):
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
            curvatures = nn.ParameterList([nn.Parameter(torch.Tensor([1])) for _ in range(n_curvatures)])
        else:
            # fixed curvature
            curvatures = [torch.tensor([args.c]) for _ in range(n_curvatures)]
            if not args.cuda == -1:
                curvatures = [curv.to(args.device) for curv in curvatures]
        return dims, acts, curvatures

    def __init__(self, args):
        super(RegModel, self).__init__()
        self.manifold_name = args.manifold
        if args.c is not None:  # 双曲半径，设置为None表示可训练的曲率
            self.c = torch.tensor([args.c])
            if not args.cuda == -1:
                self.c = self.c.to(args.device)
        else:
            self.c = nn.Parameter(torch.Tensor([1.]))
        self.manifold = Hyperboloid()  # 选择相应的流形
        self.embedding = nn.Embedding(args.max_z, args.n_atom_embed, padding_idx=0)
        if self.manifold.name == 'Hyperboloid':
            args.feat_dim = args.feat_dim + 1

        """
        Encoder
        """
        dims, acts, self.curvatures = self.get_dim_act_curv(args)  # 从参数中获取每一层的维度、激活和曲率
        self.curvatures.append(self.c)
        hgc_layers = []

        for i in range(args.num_layers):
            c_in, c_out = self.curvatures[i], self.curvatures[i + 1]
            in_dim, out_dim = dims[i], dims[i + 1]
            act = acts[i]
            # hgc_layers.append(
            #     hyp_layers.HyperbolicGraphConvolution(  # 设置每一层的卷积操作 return h, adj
            #         self.manifold, in_dim, out_dim, c_in, c_out, args.dropout, act, args.bias, args.use_att,
            #         args.local_agg
            #     )
            # )
            hgc_layers.append(
                hyp_layers.HNNLayer(
                    self.manifold, in_dim, out_dim, c_in,c_out, args.dropout, act, args.bias
                )
            )
        self.encoder = nn.Sequential(*hgc_layers)
        self.decoder = nn.Sequential(
            nn.Linear(args.dim, args.dim, args.bias),
            nn.Linear(args.dim, args.dim, args.bias),
            nn.Linear(args.dim, args.dim, args.bias),
            nn.Linear(args.dim, 1, args.bias),
        )

    def encode(self, x, adj):

        x_tan = self.manifold.proj_tan0(x, self.curvatures[0])  # 把x（欧式向量）投影到原点的切空间
        x_hyp = self.manifold.expmap0(x_tan, c=self.curvatures[0])  # 在原点把x_tan指数映射
        input = (x_hyp, adj)
        output = self.encoder(x_hyp)  # 产生了NaN
        output = self.manifold.logmap0(output, self.c)
        output = self.manifold.proj_tan0(output, self.c)

        return output

    def decode(self, h, adj):
        return self.decoder(h)


    def forward(self,inputs):
        atomic_numbers = inputs[Properties.Z]  # (b,n_atom)
        positions = inputs[Properties.R]  # (b,n_atom,3)
        atom_mask = inputs[Properties.atom_mask]  # (b,n_atom)

        size = atom_mask.size()
        adj = atom_mask.unsqueeze(2).expand(size[0], size[1], size[1])
        adj = adj * adj.permute(0, 2, 1)
        n = atom_mask.sum(1).view(-1, 1, 1).expand(-1, size[1], size[1])
        adj = adj / n  # (b,n_atom,n_atom) atom_mask like ,归一化

        x = self.embedding(atomic_numbers)  # (b,n_atom,n_atom_embed)
        x = torch.concat([x, positions], dim=2)  # (b,n_atom,n_atom_embed+3)
        if self.manifold.name == 'Hyperboloid':
            o = torch.zeros_like(x)
            x = torch.cat([o[:, :, 0:1], x], dim=2)  # (b,n_atom,feat_dim)

        h = self.encode(x, adj)
        output = self.decode(h,adj)

        return torch.sum(output,dim=1)




