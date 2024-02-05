import torch

from examples.operator.pde.diff_ops import VectorizedLaplacian


class NegativeLinearFokkerPlanck:
    def __init__(self,
                 local_potential_ftn,
                 scale=1.,
                 laplacian_eps=1e-5):
        self.laplacian_eps = laplacian_eps
        self.laplacian = VectorizedLaplacian(eps=laplacian_eps)
        self.local_potential_ftn = local_potential_ftn
        self.scale = scale

    def __call__(self, f, xs, importance=None):
        # xs: (B, D)
        if importance is None:
            lap_f, grad_f, fs = self.laplacian(f, xs, return_grad=True)  # (B, L), (B, L, D), (B, L)
        else:
            g = lambda x: importance(x).sqrt() * f(x)
            lap_g, grad_g, gs = self.laplacian(g, xs, return_grad=True)  # (B, L), (B, L, D), (B, L)
            sqrt_ws = importance(xs).sqrt()
            lap_f, grad_f, fs = lap_g / sqrt_ws, grad_g / sqrt_ws.unsqueeze(-1), gs / sqrt_ws
        lap_pot, grad_pot, pot = self.laplacian(lambda x: self.local_potential_ftn(x).view(-1, 1), xs.view(xs.shape[0], -1), return_grad=True)  # (B, 1), (B, D), (B, 1)
        Kf_t = - (lap_f +
                  torch.einsum('bd,bld->bl', grad_pot, grad_f) +
                  torch.einsum('bl,b->bl', fs, lap_pot.squeeze(1))
                  )  # (B, L)
        return - self.scale * Kf_t, fs


def sin_of_cos_potential(xs, cs):
    return torch.sin((torch.cos(xs) * torch.tensor(cs, device=xs.device).view(1, -1)).sum(-1))
