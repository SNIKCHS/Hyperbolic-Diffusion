import torch
import Model.Decoders as Decoders
import Model.Encoders as Encoders
from schnetpack import Properties
from torch import nn
from layers import hyp_layers
from manifolds.hyperboloid import Hyperboloid



class HyperbolicAE(nn.Module):

    def __init__(self, args):
        super(HyperbolicAE, self).__init__()
        self.manifold_name = args.manifold

        self.manifold = Hyperboloid()  # 选择相应的流形
        self.embedding = nn.Embedding(args.max_z, args.n_atom_embed, padding_idx=0)
        if self.manifold.name == 'Hyperboloid':
            args.feat_dim = args.feat_dim + 1

        self.encoder = getattr(Encoders, args.model)(args)
        c = self.encoder.curvatures if hasattr(self.encoder, 'curvatures') else args.c
        self.decoder = Decoders.model2decoder[args.model](c, args)



    def forward(self, inputs):
        atomic_numbers = inputs[Properties.Z]  # (b,n_atom)
        positions = inputs[Properties.R]  # (b,n_atom,3)
        positions -= positions.mean(dim=1,keepdim = True)
        atom_mask = inputs[Properties.atom_mask]  # (b,n_atom)

        size = atom_mask.size()
        adj = atom_mask.unsqueeze(2).expand(size[0], size[1], size[1]) # (b,n_atom,n_atom)
        adj = adj * adj.permute(0, 2, 1) # (b,n_atom,n_atom)
        n = atom_mask.sum(1).view(-1, 1, 1).expand(-1, size[1], size[1])
        # adj = adj / n  # (b,n_atom,n_atom) atom_mask like ,归一化

        x = self.embedding(atomic_numbers)  # (b,n_atom,n_atom_embed)
        x = torch.concat([x, positions], dim=2)  # (b,n_atom,n_atom_embed+3)
        if self.manifold.name == 'Hyperboloid':
            o = torch.zeros_like(x)
            x = torch.cat([o[:, :, 0:1], x], dim=2)  # (b,n_atom,feat_dim)

        h = self.encoder.encode(x, adj)
        output = self.decoder.decode(h, adj)
        atomic_numbers_one_hot = self.one_hot(atomic_numbers)
        target = torch.concat([atomic_numbers_one_hot, positions.float()], dim=2)
        return self.compute_loss(target,output)

    def compute_loss(self,x,x_hat):
        """
        auto-encoder的损失
        :param x: encoder的输入 [原子序数,原子坐标]
        :param x_hat: decoder的输出 (b,n_atom,4)
        :return: loss
        """

        loss_fun = nn.MSELoss(reduction='sum')
        loss = loss_fun(x,x_hat)/ x.size(0)
        return loss

    def one_hot(self, label, N=20):  # 对标签进行独热编码

        return (torch.arange(N).to('cuda') == label.unsqueeze(2).repeat(1, 1, N)).int()

class HyperbolicDiffusion(nn.Module):

    def __init__(self, args,encoder,decoder):
        super(HyperbolicDiffusion, self).__init__()
        self.manifold_name = args.manifold
        self.manifold = Hyperboloid()  # 选择相应的流形
        self.embedding = nn.Embedding(args.max_z, args.n_atom_embed, padding_idx=0)
        if self.manifold.name == 'Hyperboloid':
            args.feat_dim = args.feat_dim + 1

        self.encoder = encoder
        self.decoder = decoder

    def forward(self, inputs):
        atomic_numbers = inputs[Properties.Z]  # (b,n_atom)
        positions = inputs[Properties.R]  # (b,n_atom,3)
        positions -= positions.mean(dim=1,keepdim = True)
        atom_mask = inputs[Properties.atom_mask]  # (b,n_atom)

        size = atom_mask.size()
        adj = atom_mask.unsqueeze(2).expand(size[0], size[1], size[1]) # (b,n_atom,n_atom)
        adj = adj * adj.permute(0, 2, 1) # (b,n_atom,n_atom)
        n = atom_mask.sum(1).view(-1, 1, 1).expand(-1, size[1], size[1])
        # adj = adj / n  # (b,n_atom,n_atom) atom_mask like ,归一化

        x = self.embedding(atomic_numbers)  # (b,n_atom,n_atom_embed)
        x = torch.concat([x, positions], dim=2)  # (b,n_atom,n_atom_embed+3)
        if self.manifold.name == 'Hyperboloid':
            o = torch.zeros_like(x)
            x = torch.cat([o[:, :, 0:1], x], dim=2)  # (b,n_atom,feat_dim)

        h = self.encoder.encode(x, adj)
        output = self.decoder.decode(h, adj)
        atomic_numbers_one_hot = self.one_hot(atomic_numbers)
        target = torch.concat([atomic_numbers_one_hot, positions.float()], dim=2)
        return self.compute_loss(target,output)

    def compute_loss(self,x,x_hat):


        loss_fun = nn.MSELoss(reduction='sum')
        loss = loss_fun(x,x_hat)/ x.size(0)
        return loss

    def one_hot(self, label, N=20):  # 对标签进行独热编码

        return (torch.arange(N).to('cuda') == label.unsqueeze(2).repeat(1, 1, N)).int()