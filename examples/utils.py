import csv
import datetime
import os

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from mpl_toolkits.axes_grid1 import make_axes_locatable


def get_loss_descriptor(args):
    if args.loss.name in ['nestedlora', 'neuralsvd']:
        loss_name = f'{args.loss.name}' \
                    f'{"_seq" if args.loss.neuralsvd.sequential else "_jnt"}' \
                    f'{"_sort" if args.sort else ""}' \
                    f'{f"_step{args.loss.neuralsvd.step}" if (args.loss.neuralsvd.step > 1 and not args.loss.neuralsvd.sequential) else ""}'
    elif args.loss.name == 'neuralef':
        if args.loss.neuralef.unbiased:
            if args.loss.neuralef.include_diag:
                assert args.loss.neuralef.batchnorm_mode == 'none'
                loss_name = 'Sanger_diag1bn0'
            else:
                assert args.loss.neuralef.batchnorm_mode != 'none'
                loss_name = f'muEG_diag1bn{args.loss.neuralef.batchnorm_mode}'
        else:
            if args.loss.neuralef.include_diag:
                assert args.loss.neuralef.batchnorm_mode == 'none'
                loss_name = 'alphaEGdiag_diag0bn0'
            else:
                assert args.loss.neuralef.batchnorm_mode != 'none'
                loss_name = f'alphaEG_diag0bn{args.loss.neuralef.batchnorm_mode}'
        loss_name = f'{loss_name}_l2bn{args.loss.neuralef.batchnorm_mode}'
    elif args.loss.name == 'spin':
        loss_name = f'spin_decay{args.loss.spin.decay}'
    else:
        raise NotImplementedError
    return loss_name


def get_log_file(log_dir, fieldnames):
    log_filename = os.path.join(log_dir, f'log_{datetime.datetime.now().isoformat()}.csv')
    log_file = open(log_filename, mode='w')
    log_writer = csv.DictWriter(log_file, fieldnames=fieldnames)
    log_writer.writeheader()
    return log_file, log_writer


def get_optimizer(args, model):
    if args.optimizer == 'rmsprop':
        optimizer = torch.optim.RMSprop(
            model.parameters(),
            lr=args.lr,
            alpha=args.rmsprop_decay,
            eps=1e-10,
            weight_decay=0,
            momentum=args.momentum,
        )
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.lr,
            eps=args.adam_eps
        )
    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=args.momentum
        )
    else:
        raise NotImplementedError
    return optimizer


def plot_orth(data, cmap='gray_r', linewidth=1, figsize=(5, 5), colorbar=False):
    fig, ax = plt.subplots(figsize=figsize)
    mesh = ax.pcolormesh(data[::-1, :], edgecolors='gray', linewidth=linewidth, cmap=cmap, vmin=-1, vmax=1)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    # ax_.yticks(-0.52 + np.arange(10))
    ax.tick_params(labelbottom=False, labelleft=False,)
    if colorbar:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(mesh, cax=cax, orientation='vertical')
    plt.show()


class GaussianFourierFeatureTransform(nn.Module):
    """
    An implementation of Gaussian Fourier feature mapping.

    "Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains":
       https://arxiv.org/abs/2006.10739
       https://people.eecs.berkeley.edu/~bmild/fourfeat/index.html
       https://colab.research.google.com/github/ndahlquist/pytorch-fourier-feature-networks/blob/master/demo.ipynb

    Given an input of size [batches, input_dim],
     returns a tensor of size [batches, mapping_size*2].
    """
    def __init__(self, input_dim, mapping_size=256, scale=10, deterministic=False, append_raw=False):
        super().__init__()
        self.input_dim = input_dim
        self.deterministic = deterministic
        if deterministic:
            # Deterministic integer modulation
            # size = (input_dim, input_dim * mapping_size)
            self._B = nn.Parameter(
                scale * torch.cat([i * torch.eye(input_dim) for i in range(1, mapping_size + 1)], dim=0).T,
                requires_grad=False,
            )
            self._mapping_size = input_dim * mapping_size
        else:
            # Gaussian random projection
            self._B = nn.Parameter(
                2 * torch.pi * scale * torch.randn((input_dim, mapping_size)).float(),
                requires_grad=False,
            )
            self._mapping_size = mapping_size
        self.feature_dim = 2 * self._mapping_size
        self.append_raw = append_raw
        if append_raw:
            self.feature_dim += input_dim

    def forward(self, x):
        assert x.dim() in [2, 3], 'Expected 2D or 3D input (got {}D input)'.format(x.dim())
        if x.dim() == 3:
            batches, n_particles, dims = x.shape
            assert n_particles * dims == self.input_dim,\
                "Expected input to have {} dims (got {} dims)".format(self.input_dim, (n_particles, dims))
            # Make shape compatible for matmul with _B.
            # From [B, N, D] to [B, N * D].
            x = x.reshape(batches, n_particles * dims)
        else:
            batches, dims = x.shape
            assert dims == self.input_dim,\
                "Expected input to have {} dims (got {} dims)".format(self.input_dim, dims)
        proj = x @ self._B.to(x.device)  # (batches, mapping_size)
        feature = torch.cat([torch.sin(proj), torch.cos(proj)], dim=1)
        if self.append_raw:
            feature = torch.cat([feature, x], dim=1)
        return feature
