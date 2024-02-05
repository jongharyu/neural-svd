from torch import nn as nn

from examples.models.mlp import get_mlp_eigfuncs
from examples.operator.pde.boundary import DirichletBoundaryMaskBox, ExponentialMask
from examples.utils import GaussianFourierFeatureTransform


class WaveFunctions(nn.Module):
    def __init__(self, base, boundary_mask, hard_mul_const=1.):
        super().__init__()
        self.base = base
        self.boundary_mask = boundary_mask
        self.hard_mul_const = hard_mul_const

    def forward(self, x):
        return self.hard_mul_const * self.base(x) * self.boundary_mask(x)


def get_wavefunctions(args):
    feature_map = None
    if args.use_fourier_feature:
        feature_map = GaussianFourierFeatureTransform(
            input_dim=args.ndim * args.n_particles,
            mapping_size=args.fourier_mapping_size,
            scale=args.fourier_scale,
            deterministic=args.fourier_deterministic,
            append_raw=args.fourier_append_raw,
        )
    base_model = get_mlp_eigfuncs(
        input_dim=args.ndim * args.n_particles,
        neigs=args.neigs,
        mlp_hidden_dims=args.mlp_hidden_dims,
        nonlinearity=args.nonlinearity,
        parallel=args.parallel,
        feature_map=feature_map,
    )
    if args.apply_boundary:
        assert args.boundary_mode in ['dir_box_sqrt', 'dir_box_exp']
        # currently implemented only for 1) boxes and 2) zero Dirichlet boundary condition
        boundary_mask = DirichletBoundaryMaskBox(lim=args.lim, mode=args.boundary_mode)
        # boundary_mask = DirichletBoundaryMask(
        #     boundary_ftns=[lambda x: x[:, i] - args.lim for i in range(args.ndim)] +
        #                   [lambda x: x[:, i] + args.lim for i in range(args.ndim)]
        # )
    else:
        boundary_mask = lambda x: 1.
    if args.apply_exp_mask:
        boundary_mask = ExponentialMask(
            output_dim=args.neigs,
            init_scale=args.exp_mask_init_scale,
            boundary_mask=boundary_mask,
        )

    model = WaveFunctions(base_model, boundary_mask=boundary_mask, hard_mul_const=args.hard_mul_const)
    return model
