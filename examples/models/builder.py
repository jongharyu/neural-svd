import os

import numpy as np
import torch
import torch.nn as nn
import torchvision

from examples.models.mlp import get_mlp
from examples.models.resnet import _resnet
from examples.models.siam import SiamNetwork
from tools.generic import fix_dataparallel_keys


def load_checkpoint(pretrained_path, model, optimizer=None, lr_scheduler=None, scaler=None, args=None):
    if os.path.isfile(pretrained_path):
        print("=> Loading checkpoint '{}'".format(pretrained_path))
        ckpt = torch.load(pretrained_path, map_location='cpu')
        try:
            state_dict = fix_dataparallel_keys(ckpt['state_dict'])
        except:
            state_dict = fix_dataparallel_keys(ckpt['model'])
        if args is not None:
            args.start_epoch = ckpt['epoch']
        missing_keys, unexpected_kes = model.load_state_dict(state_dict, strict=False)  # shouldn't have missing keys
        assert not missing_keys
        print(f"=> Successfully loaded pre-trained model '{pretrained_path}'")
        if optimizer is not None and 'optimizer' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer'])
            print(f"=> Successfully loaded optimizer!")
        if lr_scheduler is not None and 'iter' in ckpt:
            lr_scheduler.iter = ckpt['iter']
            print(f"=> Successfully set lr_scheduler!")
        if scaler is not None and 'scaler' in ckpt:
            scaler.load_state_dict(ckpt['scaler'])
            print(f"=> Successfully loaded grad_scaler!")
        else:
            print(f"=> There is no stored scaler!: {ckpt.keys()}")
    else:
        raise ValueError(f"=> No checkpoint found at '{args.pretrained}'")


def get_resnet_backbone(arch, castrate=True):
    if arch == 'resnet50':
        backbone = torchvision.models.resnet50(zero_init_residual=True)
        backbone.feature_dim = 2048
    else:
        backbone = _resnet(arch, zero_init_residual=True)
    if castrate:
        backbone.output_dim = backbone.feature_dim
        backbone.fc = torch.nn.Identity()

    return backbone


class LinearProbe(nn.Module):
    def __init__(self, pretrained: SiamNetwork, trunc_dims, num_classes,
                 hidden_dims=None, sort=False):
        super().__init__()
        self.pretrained = pretrained
        self.trunc_dims = trunc_dims if trunc_dims else [pretrained.projector.output_dim]
        self.heads = nn.ModuleDict()
        if hidden_dims is None or len(hidden_dims) == 0:
            self.heads['rep'] = nn.Linear(in_features=pretrained.backbone.output_dim,
                                          out_features=num_classes)  # representation -> label
            self.heads['emb'] = nn.Linear(in_features=pretrained.projector.output_dim,
                                          out_features=num_classes)  # embedding -> label
            for dim in trunc_dims:
                self.heads[f'trunc({dim})'] = nn.Linear(in_features=np.abs(dim),
                                                        out_features=num_classes)  # truncated embedding -> label
        else:
            sizes = hidden_dims + [num_classes]
            self.heads['rep'] = get_mlp(sizes=[pretrained.backbone.output_dim] + sizes,
                                        bias=True, use_bn=True, last_layer_bn=False)  # representation -> label
            self.heads['emb'] = get_mlp(sizes=[pretrained.projector.output_dim] + sizes,
                                        bias=True, use_bn=True, last_layer_bn=False)  # embedding -> label
            for dim in trunc_dims:
                self.heads[f'trunc({dim})'] = get_mlp(sizes=[np.abs(dim)] + sizes,
                                                      bias=True, use_bn=True, last_layer_bn=False)  # truncated embedding -> label
        self.output_dim = num_classes
        self.sort = sort
        self.spectrum = None
        self.sort_indices = None

    def register_spectrum(self, spectrum):
        self.spectrum = torch.Tensor(spectrum[1:])
        self.sort_indices = np.argsort(spectrum[1:])[::-1]
        print(f"Eigenvalues registered and will be {'sorted' if self.sort else 'not sorted'}, "
              f"and the sorted eigvals are {spectrum[self.sort_indices]}")

    def forward(self, x, normalize=False):
        f_rep, f_emb = self.pretrained(x, classification=False, freeze_model=True)
        if normalize:
            f_emb /= torch.sqrt(self.spectrum).view(1, -1).to(x.device)
        if self.sort and self.sort_indices is not None:
            f_emb = f_emb[..., self.sort_indices.copy()]
        logits = {}
        logits['rep'] = self.heads['rep'](f_rep.detach())
        logits['emb'] = self.heads['emb'](f_emb.detach())
        for dim in self.trunc_dims:
            if dim > 0:
                logits[f'trunc({dim})'] = self.heads[f'trunc({dim})'](f_emb[:, :dim].detach())
            else:
                logits[f'trunc({dim})'] = self.heads[f'trunc({dim})'](f_emb[:, dim:].detach())

        return logits
