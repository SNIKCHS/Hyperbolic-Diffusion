import numpy as np
from schnetpack.nn import MollifierCutoff, HardCutoff
import torch
from schnetpack.nn import AtomDistances

import Model.Decoders as Decoders
import Model.Encoders as Encoders
from schnetpack import Properties
from torch import nn

import manifolds
from layers.layers import Linear
from utils import utils
from manifolds.hyperboloid import Hyperboloid
import math
from torch.nn import functional as F


class HyperbolicAE(nn.Module):

    def __init__(self, args):
        super(HyperbolicAE, self).__init__()

        self.manifold = getattr(manifolds, args.manifold)()  # 选择相应的流形
        self.encoder = getattr(Encoders, args.model)(args)
        c = self.encoder.curvatures if hasattr(self.encoder, 'curvatures') else args.c
        self.decoder = Decoders.model2decoder[args.model](c, args)
        self.args = args
        self.distances = AtomDistances()
        self.cutoff = HardCutoff()

    def forward(self, inputs):
        atomic_numbers = inputs[Properties.Z]  # (b,n_atom)
        positions = inputs[Properties.R]  # (b,n_atom,3)
        positions -= positions.mean(dim=1, keepdim=True)
        atom_mask = inputs[Properties.atom_mask]  # (b,n_atom)


        size = atom_mask.size()
        mask = atom_mask.unsqueeze(2).expand(size[0], size[1], size[1])  # (b,n_atom,n_atom)
        mask = mask * mask.permute(0, 2, 1)  # (b,n_atom,n_atom) mask
        # n = atom_mask.sum(1).view(-1, 1, 1).expand(-1, size[1], size[1])
        # mask = mask / n  # (b,n_atom,n_atom) atom_mask like ,归一化

        ar = torch.arange(atomic_numbers.size(1), device=atomic_numbers.device)[None, None, :].repeat(atomic_numbers.size(0),atomic_numbers.size(1),1)  # (b,n_atom,n_atom)
        nbh = ar * mask  # 邻接关系 存index

        dist = self.distances(positions,nbh.long(),neighbor_mask=mask.bool())  #(b,n_atom,n_atom)
        mask = self.cutoff(dist) * mask

        h = self.encoder(positions, atomic_numbers, (dist,mask))

        mu = torch.mean(h,dim=-1)
        logvar = torch.log(torch.std(h,dim=-1))
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp().pow(2))/ positions.size(0)
        KLD = 0.01*torch.clamp(KLD,min=0,max=1e2)
        output = self.decoder.decode(h, (dist,mask))
        target = (atomic_numbers, positions.float())
        return self.compute_loss(target, output),KLD

    def compute_loss(self, x, x_hat):
        """
        auto-encoder的损失
        :param x: encoder的输入 [原子序数,原子坐标]
        :param x_hat: decoder的输出 (b,n_atom,4)
        :return: loss
        """
        atomic_numbers, positions = x
        positions_pred,atomic_numbers_pred = x_hat[...,:3],x_hat[...,3:]
        # positions_pred = self.manifold.logmap0(positions_pred,self.decoder.curvatures[-1])
        n_type = atomic_numbers_pred.size(-1)

        atom_loss_f = nn.CrossEntropyLoss(reduction='sum')
        pos_loss_f = nn.MSELoss(reduction='sum')
        # loss = (atom_loss_f(atomic_numbers_pred.view(-1,n_type),atomic_numbers.view(-1))+pos_loss_f(positions_pred,positions)) / positions.size(0)
        loss = (atom_loss_f(atomic_numbers_pred.view(-1, n_type), atomic_numbers.view(-1))) / positions.size(0)
        # print(atomic_numbers_pred.view(-1, n_type)[:3], atomic_numbers.view(-1)[:3])
        return loss


def cosine_beta_schedule(timesteps, s=0.008, raise_to_power: float = 1):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 2
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas = np.clip(betas, a_min=0, a_max=0.999)
    alphas = 1. - betas
    alphas_cumprod = np.cumprod(alphas, axis=0)

    if raise_to_power != 1:
        alphas_cumprod = np.power(alphas_cumprod, raise_to_power)

    return alphas_cumprod


def clip_noise_schedule(alphas2, clip_value=0.001):
    """
    For a noise schedule given by alpha^2, this clips alpha_t / alpha_t-1. This may help improve stability during
    sampling.
    """
    alphas2 = np.concatenate([np.ones(1), alphas2], axis=0)

    alphas_step = (alphas2[1:] / alphas2[:-1])

    alphas_step = np.clip(alphas_step, a_min=clip_value, a_max=1.)
    alphas2 = np.cumprod(alphas_step, axis=0)

    return alphas2


def polynomial_schedule(timesteps: int, s=1e-4, power=3.):
    """
    A noise schedule based on a simple polynomial equation: 1 - x^power.
    """
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas2 = (1 - np.power(x / steps, power)) ** 2

    alphas2 = clip_noise_schedule(alphas2, clip_value=0.001)

    precision = 1 - 2 * s

    alphas2 = precision * alphas2 + s

    return alphas2


def sum_except_batch(x):
    return x.view(x.size(0), -1).sum(-1)


def gaussian_KL(q_mu, q_sigma, p_mu, p_sigma, node_mask):
    """Computes the KL distance between two normal distributions.

        Args:
            q_mu: Mean of distribution q.
            q_sigma: Standard deviation of distribution q.
            p_mu: Mean of distribution p.
            p_sigma: Standard deviation of distribution p.
        Returns:
            The KL distance, summed over all dimensions except the batch dim.
        """
    return sum_except_batch(
        (
                torch.log(p_sigma / q_sigma)
                + 0.5 * (q_sigma ** 2 + (q_mu - p_mu) ** 2) / (p_sigma ** 2)
                - 0.5
        ) * node_mask
    )


def gaussian_KL_for_dimension(q_mu, q_sigma, p_mu, p_sigma, d):
    """Computes the KL distance between two normal distributions.

        Args:
            q_mu: Mean of distribution q.
            q_sigma: Standard deviation of distribution q.
            p_mu: Mean of distribution p.
            p_sigma: Standard deviation of distribution p.
        Returns:
            The KL distance, summed over all dimensions except the batch dim.
        """
    mu_norm2 = sum_except_batch((q_mu - p_mu) ** 2)
    assert len(q_sigma.size()) == 1
    assert len(p_sigma.size()) == 1
    return d * torch.log(p_sigma / q_sigma) + 0.5 * (d * q_sigma ** 2 + mu_norm2) / (p_sigma ** 2) - 0.5 * d


def cdf_standard_gaussian(x):
    return 0.5 * (1. + torch.erf(x / math.sqrt(2)))


class PredefinedNoiseSchedule(torch.nn.Module):
    """
    Predefined noise schedule. Essentially creates a lookup array for predefined (non-learned) noise schedules.
    """

    def __init__(self, noise_schedule, timesteps, precision):
        super(PredefinedNoiseSchedule, self).__init__()
        self.timesteps = timesteps

        if noise_schedule == 'cosine':
            alphas2 = cosine_beta_schedule(timesteps)
        elif 'polynomial' in noise_schedule:
            splits = noise_schedule.split('_')
            assert len(splits) == 2
            power = float(splits[1])
            alphas2 = polynomial_schedule(timesteps, s=precision, power=power)
        else:
            raise ValueError(noise_schedule)

        print('alphas2', alphas2)

        sigmas2 = 1 - alphas2

        log_alphas2 = np.log(alphas2)
        log_sigmas2 = np.log(sigmas2)

        log_alphas2_to_sigmas2 = log_alphas2 - log_sigmas2

        print('gamma', -log_alphas2_to_sigmas2)

        self.gamma = torch.nn.Parameter(
            torch.from_numpy(-log_alphas2_to_sigmas2).float(),
            requires_grad=False)

    def forward(self, t):
        t_int = torch.round(t * self.timesteps).long()
        return self.gamma[t_int]


class ResBlock(nn.Module):
    def __init__(self, attn=False):
        super().__init__()
        self.block1 = nn.Sequential(
            # nn.LazyBatchNorm1d(),
            nn.Linear(400, 400),
            nn.SiLU(),
        )
        self.block2 = nn.Sequential(
            # nn.LazyBatchNorm1d(),
            nn.Linear(400, 400),
            nn.SiLU(),
        )
        if attn:
            # self.attn = AttnBlock(out_ch)
            pass
        else:
            self.attn = nn.Identity()


    def forward(self, x):
        h = self.block1(x)
        h = self.block2(h)
        h = h + x
        h = self.attn(h)
        return h
class DenoiseNet(nn.Module):
    def __init__(self, args):
        super().__init__()
        args.dim += 1
        self.act = getattr(F, args.act)
        # self.net = getattr(Encoders, args.diff_model)(args)
        self.net = nn.Sequential(
            nn.Linear(args.dim, 400),
            ResBlock(),
            ResBlock(),
            ResBlock(),
            ResBlock(),
            nn.Linear(400, 400),
            nn.SiLU(),
            nn.Linear(400, 400),
            nn.SiLU(),
            nn.Linear(400, args.dim),
        )

        self.out = Linear(args.dim, args.dim - 1, None, None, True)

    def forward(self, t, z_t, adj):
        t = t.repeat(1,z_t.size(1)).unsqueeze(2)
        z_t = torch.cat([t, z_t], dim=2)
        noise = self.net(z_t)

        return self.out(noise)


class HyperbolicDiffusion(nn.Module):

    def __init__(self, args, encoder, decoder,T = 1000,beta_1=1e-4, beta_T=0.02, timesteps: int = 1000,loss_type='l2', noise_schedule='cosine', noise_precision=1e-4):
        super(HyperbolicDiffusion, self).__init__()
        self.manifold = Hyperboloid()  # 选择相应的流形
        self.dim = args.dim
        self.n_dims = args.dim
        self.denoise_net = DenoiseNet(args)
        self.T = T

        self.register_buffer(
            'betas', torch.linspace(beta_1, beta_T, T).double())
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)
        self.register_buffer(
            'sqrt_alphas_bar', torch.sqrt(alphas_bar))
        self.register_buffer(
            'sqrt_one_minus_alphas_bar', torch.sqrt(1. - alphas_bar))

        self.encoder = encoder
        self.decoder = decoder

    def forward(self, inputs):
        atomic_numbers = inputs[Properties.Z]  # (b,n_atom)
        positions = inputs[Properties.R]  # (b,n_atom,3)
        positions -= positions.mean(dim=1, keepdim=True)
        atom_mask = inputs[Properties.atom_mask]  # (b,n_atom)

        size = atom_mask.size()
        adj = atom_mask.unsqueeze(2).expand(size[0], size[1], size[1])  # (b,n_atom,n_atom)
        adj = adj * adj.permute(0, 2, 1)  # (b,n_atom,n_atom)


        h = self.encoder(positions,atomic_numbers, adj)
        loss = self.compute_loss(h, atom_mask.unsqueeze(2), t0_always=True, adj=adj)
        return loss



    def compute_loss(self, h, node_mask, t0_always, adj):

        t = torch.randint(self.T,size=(h.shape[0],), device=h.device)
        noise = torch.randn_like(h)
        x_t = (
                extract(self.sqrt_alphas_bar, t, h.shape) * h +
                extract(self.sqrt_one_minus_alphas_bar, t, h.shape) * noise)
        pred_noise = self.denoise_net(t.view(h.shape[0],1),x_t,adj)
        loss = F.mse_loss(pred_noise, noise, reduction='mean')

        return loss

        # for t in range(self.T):
        #     t = torch.ones((h.shape[0],),device=h.device,dtype=torch.int64)*t
        #     noise = torch.randn_like(h) * node_mask
        #
        #     x_t = (
        #             extract(self.sqrt_alphas_bar, t, h.shape) * h +
        #             extract(self.sqrt_one_minus_alphas_bar, t, h.shape) * noise)
        #     print(x_t[:1,:2])
        #
        #
        # return 0


def extract(v, t, x_shape):
    """
    Extract some coefficients at specified timesteps, then reshape to
    [batch_size, 1, 1, 1, 1, ...] for broadcasting purposes.
    """
    out = torch.gather(v, dim=0, index=t).float()
    return out.view([t.shape[0]] + [1] * (len(x_shape) - 1))




