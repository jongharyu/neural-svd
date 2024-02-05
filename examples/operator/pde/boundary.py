import torch
import torch.nn as nn


class DirichletBoundaryMask(nn.Module):
    # Dirichlet condition; from (Jin et al., 2021)
    def __init__(self, boundary_ftns):
        super().__init__()
        self.boundary_ftns = boundary_ftns

    def forward(self, x):
        mask = torch.stack([1 - torch.exp(-ftn(x)) for ftn in self.boundary_ftns], dim=1).prod(dim=-1)
        return mask.reshape(-1, 1).to(x.device)


class DirichletBoundaryMaskBox(nn.Module):
    # Dirichlet condition (for box shaped boundary)
    def __init__(self, lim, mode='dir_box_sqrt'):
        super().__init__()
        self.lim = lim
        self.mode = mode
        assert mode in ['dir_box_sqrt', 'dir_box_exp']

    def forward(self, x):
        """Makes boundary conditions for network (fixed box)."""
        # Force the wavefunction to zero at the boundaries of the box defined by [-lim, lim]^d
        mask = torch.ones(x.shape[0]).to(x.device)
        x = torch.clamp(x, min=-self.lim, max=self.lim).reshape(x.shape[0], -1)
        for i in range(x.shape[1]):
            if self.mode == 'dir_box_sqrt':  # (Pfau et al. 2018)
                mask = mask * torch.maximum(((2 * self.lim ** 2 - x[:, i] ** 2).sqrt() - self.lim) / self.lim,
                                            torch.Tensor([0.]).to(x.device))
            elif self.mode == 'dir_box_exp':  # (Jin et al. 2022)
                mask = mask * (1 - torch.exp(- (self.lim - x[:, i]))) * (1 - torch.exp(- (x[:, i] + self.lim)))
                # mask *= (1 - torch.exp(- (x[:, i] ** 2 - self.lim ** 2) / self.lim ** 2)) # * (1 - torch.exp(- (x[:, i] + self.lim)))
        return mask.reshape(-1, 1).to(x.device)  # (B, 1)


class ExponentialMask(nn.Module):
    def __init__(self, output_dim, init_scale=1000, boundary_mask=None):
        super().__init__()
        self.output_dim = output_dim
        self.scales = nn.Parameter(init_scale * torch.ones(output_dim))
        self.boundary_mask = boundary_mask

    def forward(self, x):
        # x: (B, D)
        r = torch.norm(x, p=2, dim=-1).view(-1, 1)
        mask = torch.exp(- r / self.scales.view(1, -1))  # (B, output_dim)
        if self.boundary_mask is not None:
            return mask * self.boundary_mask(x)
        else:
            return mask
