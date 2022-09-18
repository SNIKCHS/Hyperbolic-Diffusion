"""Hyperboloid manifold."""

import torch

from manifolds.base import Manifold
from utils.math_utils import arcosh, cosh, sinh 


class Hyperboloid(Manifold):
    """
    Hyperboloid manifold class.

    We use the following convention: -x0^2 + x1^2 + ... + xd^2 = -K

    c = 1 / K is the hyperbolic curvature. 
    """

    def __init__(self):
        super(Hyperboloid, self).__init__()
        self.name = 'Hyperboloid'
        self.eps = {torch.float32: 1e-7, torch.float64: 1e-15}
        self.min_norm = 1e-15
        self.max_norm = 1e6

    def minkowski_dot(self, x, y, keepdim=True):
        # (b,1,embeds)*(b,n_atom,embeds)

        res = torch.sum(x * y, dim=-1) - (2 * x[..., 0] * y[..., 0]).squeeze()  #(b,n_atom)
        if keepdim:
            res = res.view(res.shape + (1,)) #(b,n_atom,1)
        return res

    def minkowski_norm(self, u, keepdim=True):
        dot = self.minkowski_dot(u, u, keepdim=keepdim)
        return torch.sqrt(torch.clamp(dot, min=self.eps[u.dtype]))

    def sqdist(self, x, y, c):
        K = 1. / c
        prod = self.minkowski_dot(x, y)
        theta = torch.clamp(-prod / K, min=1.0 + self.eps[x.dtype])
        sqdist = K * arcosh(theta) ** 2
        # clamp distance to avoid nans in Fermi-Dirac decoder
        return torch.clamp(sqdist, max=50.0)

    def proj(self, x, c):
        K = 1. / c
        d = x.size(-1) - 1
        y = x.narrow(-1, 1, d)  # 切片，得到x[:,:,1:]

        y_sqnorm = torch.norm(y, p=2, dim=2, keepdim=True) ** 2
        mask = torch.ones_like(x)
        mask[:,:, 0] = 0
        vals = torch.zeros_like(x)
        vals[:,:, 0:1] = torch.sqrt(torch.clamp(K + y_sqnorm, min=self.eps[x.dtype]))
        return vals + mask * x   # mask * x = x1:d

    def proj_tan(self, u, x, c):
        K = 1. / c
        d = x.size(-1) - 1
        ux = torch.sum(x.narrow(-1, 1, d) * u.narrow(-1, 1, d), dim=2, keepdim=True)
        mask = torch.ones_like(u)
        mask[:,:, 0] = 0
        vals = torch.zeros_like(u)
        vals[:,:, 0:1] = ux / torch.clamp(x[:,:, 0:1], min=self.eps[x.dtype])
        return vals + mask * u

    def proj_tan0(self, u, c):
        narrowed = u.narrow(-1, 0, 1) # x[:,:,0]
        vals = torch.zeros_like(u)
        vals[:,:, 0:1] = narrowed
        return u - vals

    """
    把u从x的切空间指数映射到流形
    """
    def expmap(self, u, x, c):
        K = 1. / c
        sqrtK = K ** 0.5
        normu = self.minkowski_norm(u)
        normu = torch.clamp(normu, max=self.max_norm)
        theta = normu / sqrtK
        theta = torch.clamp(theta, min=self.min_norm)
        result = cosh(theta) * x + sinh(theta) * u / theta
        return self.proj(result, c)

    def logmap(self, x, y, c):
        """
        把y从流形映射到x的切空间
        :param x:
        :param y:
        :param c:
        :return:
        """

        if len(x.size())<3 :
            x = x.unsqueeze(1)# x:(b,1,embeds)
        K = 1. / c

        xy = torch.clamp(self.minkowski_dot(x, y) + K, max=-self.eps[x.dtype]) - K  # (b,n_atom,1)

        u = y + xy * x * c  # (b,n_atom,embeds)
        normu = self.minkowski_norm(u)
        normu = torch.clamp(normu, min=self.min_norm)
        dist = self.sqdist(x, y, c) ** 0.5
        result = dist * u / normu

        return self.proj_tan(result, x, c)

    def expmap0(self, u, c):
        K = 1. / c
        sqrtK = K ** 0.5
        d = u.size(-1) - 1
        x = u.narrow(-1, 1, d)  # u[:,:,1:]
        x_norm = torch.norm(x, p=2, dim=2, keepdim=True)
        x_norm = torch.clamp(x_norm, min=self.min_norm)
        theta = x_norm / sqrtK
        res = torch.ones_like(u)

        res[:,:, 0:1] = (sqrtK * cosh(theta)).view(res[:,:, 0:1].size())
        res[:,:, 1:] = (sqrtK * sinh(theta) * x / x_norm).view(res[:,:, 1:].size())


        return self.proj(res, c)

    def logmap0(self, x, c):
        K = 1. / c
        sqrtK = K ** 0.5
        d = x.size(-1) - 1
        y = x.narrow(-1, 1, d)

        y_norm = torch.norm(y, p=2, dim=2, keepdim=True)
        y_norm = torch.clamp(y_norm, min=self.min_norm)
        res = torch.zeros_like(x)
        theta = torch.clamp(x[:,:, 0:1] / sqrtK, min=1.0 + self.eps[x.dtype])
        res[:,:, 1:] = sqrtK * arcosh(theta) * y / y_norm
        return res

    def mobius_add(self, x, y, c):
        """
        x:被加数
        y:偏差
        """
        u = self.logmap0(y, c)
        v = self.ptransp0(x, u, c)
        return self.expmap(v, x, c)

    def mobius_matvec(self, m, x, c):
        u = self.logmap0(x, c)
        mu = u @ m.transpose(-1, -2)
        return self.expmap0(mu, c)

    # 把u从x的切空间平行传输到y的切空间
    def ptransp(self, x, y, u, c):
        logxy = self.logmap(x, y, c)
        logyx = self.logmap(y, x, c)
        sqdist = torch.clamp(self.sqdist(x, y, c), min=self.min_norm)
        alpha = self.minkowski_dot(logxy, u) / sqdist
        res = u - alpha * (logxy + logyx)
        return self.proj_tan(res, y, c)

    """
    把u从原点的切空间平行传输到x的切空间
    """
    def ptransp0(self, x, u, c):
        K = 1. / c
        sqrtK = K ** 0.5
        x0 = x.narrow(-1, 0, 1)
        d = x.size(-1) - 1
        y = x.narrow(-1, 1, d)  # x[:,:,1:]
        y_norm = torch.clamp(torch.norm(y, p=2, dim=2, keepdim=True), min=self.min_norm)
        y_normalized = y / y_norm  # x[:,:,1:]
        v = torch.ones_like(x)
        v[:,:, 0:1] = - y_norm
        v[:,:, 1:] = (sqrtK - x0) * y_normalized
        alpha = torch.sum(y_normalized * u[:,:, 1:], dim=2, keepdim=True) / sqrtK  # (b,n_atom,1)
        res = u - alpha * v
        return self.proj_tan(res, x, c)

    def to_poincare(self, x, c):
        K = 1. / c
        sqrtK = K ** 0.5
        d = x.size(-1) - 1
        return sqrtK * x.narrow(-1, 1, d) / (x[:,:, 0:1] + sqrtK)
