from contextlib import nullcontext

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SiamNetwork(nn.Module):
    def __init__(self,
                 backbone,
                 projector,
                 online_head_rep=None,
                 online_head_emb=None,
                 online_heads_trunc=None,
                 mu=1.0,
                 regularize_mode=None,
                 not_all_together=False,
                 separation=False,
                 batch_l2norm=False,  # if True, when separation is False, normalize f by its batch l2-norm
                 momentum=0.9,
                 posthoc_scaling=False):
        super().__init__()
        self.backbone = backbone  # I/O dim (?, D)
        self.projector = projector  # I/O dim (D, L)
        self.online_head_rep = online_head_rep  # I/O dim (D, C)
        self.online_head_emb = online_head_emb  # I/O dim (L, C)
        self.online_heads_trunc = online_heads_trunc  # I/O dim (L, C)

        assert regularize_mode in ['l2_ball', 'l2_sphere', 'clip', 'tanh']
        self.mu = mu
        self.regularize_mode = regularize_mode
        self.not_all_together = not_all_together

        # track the batch l2-norm of f_emb (not f_ord) to normalize
        self.separation = separation
        self.batch_l2norm = batch_l2norm
        if batch_l2norm:
            assert not separation, "batch l2-normalization is applicable only when separation is not True"
        self.momentum = momentum
        self.l2norm = None  # (L, )
        self.local_scales_logits = None
        self.global_scale = None
        if separation:
            feature_dim = projector.output_dim
            self.scales_param = nn.Parameter(torch.linspace(mu / feature_dim, mu, feature_dim).flip(0).unsqueeze(0),
                                             requires_grad=True)
        self.posthoc_scaling = posthoc_scaling

    def update_l2norm(self, phi):
        if not self.training:
            # during inference
            batch_l2norm = self.l2norm
        else:
            # during training
            batch_l2norm = phi.norm(dim=0) / np.sqrt(phi.shape[0])
            with torch.no_grad():
                if self.l2norm is None:
                    self.l2norm = batch_l2norm
                else:
                    self.l2norm = torch.sqrt(self.momentum * self.l2norm ** 2 +
                                             (1 - self.momentum) * batch_l2norm ** 2)
        return batch_l2norm

    @property
    def scales(self):
        scales = torch.sqrt(torch.abs(self.scales_param))
        return normalize(scales, np.sqrt(self.mu), 'l2_ball')

    def forward(self, z1, z2=None,
                weights=None,  # for "graph"
                classification=False,
                freeze_model=False):
        if z2 is None:
            with torch.no_grad() if freeze_model else nullcontext():
                f1_rep = self.backbone(z1)
                f1_emb = self.projector(f1_rep)
                if self.separation or self.batch_l2norm:
                    l2norm = self.update_l2norm(f1_emb)
                    if self.separation:
                        f1_emb /= l2norm.clamp(min=1e-6)
                    if self.batch_l2norm:
                        l2norm_all = (l2norm ** 2).sum().sqrt()  # sum over feature dimension
                        if l2norm_all > np.sqrt(self.mu):
                            f1_emb *= np.sqrt(self.mu) / l2norm_all.clamp(min=1e-6)
                else:
                    if self.posthoc_scaling:
                        f1_emb = np.sqrt(self.mu) * normalize(f1_emb, 1., self.regularize_mode)
                    else:
                        f1_emb = normalize(f1_emb, np.sqrt(self.mu), self.regularize_mode)

            if classification:
                if weights is None:
                    weights = 1.
                logits_rep = self.online_head_rep(f1_rep.detach())
                logits_emb = self.online_head_emb(weights * f1_emb.detach())
                logits_trunc = {}
                for dim in self.online_heads_trunc:
                    logits_trunc[dim] = self.online_heads_trunc[dim](weights * f1_emb[:, :int(dim)].detach())
                if self.separation:
                    f1_emb *= self.scales.to(z1.device)
                return f1_rep, f1_emb, logits_rep, logits_emb, logits_trunc
            else:
                if self.separation:
                    f1_emb *= self.scales.to(z1.device)
                return f1_rep, f1_emb
        else:
            with torch.no_grad() if freeze_model else nullcontext():
                if self.not_all_together:
                    f1_rep, f1_emb = self.forward(z1)
                    f2_rep, f2_emb = self.forward(z2)
                else:
                    f_rep, f_emb = self.forward(torch.cat([z1, z2], dim=0))
                    f1_rep, f2_rep = f_rep.chunk(2, dim=0)
                    f1_emb, f2_emb = f_emb.chunk(2, dim=0)

            if classification:
                if weights is None:
                    weights = 1.
                logits_rep = self.online_head_rep(f1_rep.detach())
                logits_emb = self.online_head_emb(weights * f1_emb.detach())
                logits_trunc = {}
                for dim in self.online_heads_trunc:
                    logits_trunc[dim] = self.online_heads_trunc[dim](weights * f1_emb[:, :int(dim)].detach())
                if self.separation:
                    f1_emb *= self.scales.to(z1.device)
                return f1_rep, f1_emb, f2_rep, f2_emb, logits_rep, logits_emb, logits_trunc
            else:
                return f1_rep, f1_emb, f2_rep, f2_emb


class HeteroNetwork(nn.Module):
    def __init__(self,
                 backbones,
                 projectors,
                 online_heads=None,
                 mu=1.0,
                 regularize_mode=None):
        super().__init__()
        self.mu = mu
        self.backbones = nn.ModuleDict({'x': backbones[0], 'y': backbones[1]})
        self.projectors = nn.ModuleDict({'x': projectors[0], 'y': projectors[1]})
        self.online_heads = nn.ModuleDict({'x': online_heads[0], 'y': online_heads[1]}) if online_heads else None

        self.output_dims = {
            key: self.backbones[key].output_dim
            if isinstance(self.projectors[key], nn.Identity) else self.projectors[key].output_dim
            for key in self.projectors
        }
        assert regularize_mode in ['l2_ball', 'l2_sphere', 'clip', 'tanh']
        self.regularize_mode = regularize_mode

    def forward(self, x, y):
        return [*self.forward_single(x, 'x'), *self.forward_single(y, 'y')]

    def forward_single(self, x, x_or_y, classify=False):
        assert x_or_y in ['x', 'y']
        fx_rep = self.backbones[x_or_y](x)
        fx_emb = self.projectors[x_or_y](fx_rep)
        fx_emb = normalize(fx_emb, np.sqrt(self.mu), self.regularize_mode)
        if classify:
            logits = self.online_heads[x_or_y](fx_emb.detach())
            return fx_rep, fx_emb, logits
        else:
            return fx_rep, fx_emb


def normalize(z, r_up, regularize_mode):
    if r_up > 0:
        if regularize_mode == 'l2_ball':
            # normalize each row to have l2-norm <= r_up
            mask = (torch.norm(z, p=2, dim=-1) < r_up).float().unsqueeze(1)  # (B, 1)
            return mask * z + (1 - mask) * r_up * F.normalize(z, p=2, dim=1)
        elif regularize_mode == 'l2_sphere':
            # normalize each row to have l2-norm = r_up
            return r_up * F.normalize(z, p=2, dim=1)
        elif regularize_mode == 'clip':
            # clip each coord to be |coord| <= r_up
            return torch.clip(z, min=-r_up, max=r_up)
        elif regularize_mode == 'tanh':
            # apply tanh to make |output| <= r_up
            return r_up * nn.Tanh()(z)
        else:
            raise NotImplementedError
    else:
        return z
