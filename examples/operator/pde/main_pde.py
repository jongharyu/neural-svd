# -*- coding: utf-8 -*-
import os
import random
from distutils.util import strtobool

import configargparse
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.laplace import Laplace

from examples import opts
from examples.operator.pde import get_wavefunctions
from examples.operator import train_operator
from examples.operator.pde.problems import get_problem
from examples.opts import parse_loss_configs
from examples.utils import get_log_file, get_loss_descriptor
from methods.general import get_evd_method

plt.style.use('ggplot')
IMPLEMENTED_LOSSES = ('neuralsvd', 'neuralef', 'spin')


def get_args():
    parser = configargparse.ArgumentParser(description='Solve PDEs')
    opts.mlp_opts(parser)
    opts.fourier_opts(parser)
    opts.loss_opts(parser)
    opts.operator_opts(parser)
    # base configs
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
    parser.add_argument('--num_workers', default=16, type=int)
    # evaluation configs
    parser.add_argument('--log_dir', type=str, default='./log')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--print_freq', default=1000, type=int, metavar='N', help='print frequency')
    parser.add_argument('--eval_freq', default=50000, type=int, metavar='N', help='print frequency')
    parser.add_argument('--print_local_energies', action='store_true')
    # optimization configs
    parser.add_argument('--num_iters', default=1000000, type=int)
    parser.add_argument('--optimizer', default='rmsprop')
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--lr', default=1e-5, type=float, help='learning rate')
    parser.add_argument('--rmsprop_decay', default=0.999, type=float, help='Decay rate of moving averages')
    parser.add_argument('--momentum', default=0., type=float)  # for sgd, rmsprop
    parser.add_argument('--adam_eps', default=1e-7, type=float, help='epsilon parameter for adam')
    parser.add_argument('--use_lr_scheduler', action='store_true')
    parser.add_argument('--ema_decay', default=0.99, type=float, help='EMA decay parameter')
    parser.add_argument('--use_amp', action='store_true')
    # problem configs
    parser.add_argument('--problem', default='sch', choices=['sch', 'fp'])
    parser.add_argument('--ndim', default=2, type=int, help='Dimension of space')
    parser.add_argument('--lim', help='Limit of box for visualization and validation')
    # Schrödinger's equation
    parser.add_argument('--potential_type', default='hydrogen',
                        choices=['infinite_well',
                                 'harmonic_oscillator',
                                 'cosine',
                                 'hydrogen',
                                 'hydrogen_mol_ion',
                                 'quantum_chemistry'])
    parser.add_argument('--mol_name', type=str)
    # hydrogen / hydrogen ion
    parser.add_argument('--charge', default=1.0, type=float, help='Nuclear charge of atom')
    parser.add_argument('--hydrogen_mol_ion_R', default=1.0, type=float)
    # pde solver configs
    parser.add_argument('--laplacian_eps', default=0.1, type=float, help='Finite difference step for Laplacian')
    parser.add_argument('--hard_mul_const', default=1., type=float)
    # boundary mask configs
    parser.add_argument('--apply_boundary', default=True, type=strtobool, help='Force zero boundary condition')
    parser.add_argument('--boundary_mode', default='sqrt', type=str,
                        choices=['dir_box_sqrt', 'dir_box_exp'])
    # exponential mask configs
    parser.add_argument('--apply_exp_mask', default=False, type=strtobool)
    parser.add_argument('--exp_mask_init_scale', default=1000., type=float)
    # validation configs
    parser.add_argument('--val_eps', default=0.1, type=float)
    # sampler configs
    parser.add_argument('--sampling_mode', type=str, choices=['gaussian', 'laplacian', 'uniform'])
    parser.add_argument('--sampling_scale', default=16, type=float)
    # parsing
    args = parser.parse_args()
    args = parse_loss_configs(args, IMPLEMENTED_LOSSES)
    return args


def get_dataloader(args, device):
    # training data and importance
    if args.sampling_mode == 'gaussian':
        def make_batch_ftn_train():
            return args.sampling_scale * torch.randn((args.batch_size, args.n_particles, args.ndim))
        mvn = MultivariateNormal(
            loc=torch.zeros(args.n_particles * args.ndim).to(device),
            covariance_matrix=args.sampling_scale ** 2 * torch.eye(args.n_particles * args.ndim).to(device),
        )
        def importance_train(x):
            # x: (B, n_particles, dim)
            return mvn.log_prob(x.view(x.shape[0], -1)).exp().view(-1, 1)
    elif args.sampling_mode == 'laplacian':
        def make_batch_ftn_train():
            return Laplace(
                torch.Tensor(np.zeros((args.batch_size, args.n_particles, args.ndim))).to(device),
                torch.Tensor(args.sampling_scale * np.ones((args.batch_size, args.n_particles, args.ndim))).to(device)
            ).sample()
        def importance_train(x):
            # x: (B, n_particles, dim)
            return Laplace(
                torch.Tensor(np.zeros(args.n_particles * args.ndim)).to(device),
                torch.Tensor(args.sampling_scale * np.ones(args.n_particles * args.ndim)).to(device)
            ).log_prob(x.view(x.shape[0], -1)).sum(-1).exp().view(-1, 1)
    elif args.sampling_mode == 'uniform':
        def make_batch_ftn_train():
            return args.sampling_scale * (2 * torch.rand((args.batch_size, args.n_particles, args.ndim)) - 1)
        def importance_train(x):
            # x: (B, n_particles, dim)
            return (1 / (2 * args.sampling_scale) ** args.ndim * torch.ones(x.shape[0], 1)).to(x.device).float()
    # validation data (grid)
    # warning: does not scale with ndim AND n_particles
    if args.ndim in [1, 2] and args.n_particles == 1:
        x = np.arange(-args.lim, args.lim, args.val_eps)
        xxs = np.meshgrid(*(args.ndim * [x]))
        val_data = np.array(list(zip(*[xx.flatten() for xx in xxs])))
        val_data = torch.tensor(val_data).to(device).float()
        def batch_ftn_val():
            for i in range(int(np.ceil(len(val_data) / float(args.batch_size)))):
                yield val_data[i * args.batch_size: min((i + 1) * args.batch_size, len(val_data))], 0.
        def importance_val(x):
            return (1 / (2 * args.lim) ** args.ndim * torch.ones(x.shape[0], 1)).to(x.device).float().view(-1, 1)
    else:
        val_data = None
        batch_ftn_val = None
        importance_val = None
    return make_batch_ftn_train, val_data, batch_ftn_val, importance_train, importance_val


def get_log_dir(args):
    problem = args.problem
    if problem == 'sch':
        if args.potential_type == 'quantum_chemistry':
            problem = f'sch_{args.mol_name}_ndim{args.ndim}'
        else:
            problem = f'sch_{args.potential_type}_ndim{args.ndim}'
            if args.potential_type == 'hydrogen_mol_ion':
                problem += f'_R{args.hydrogen_mol_ion_R}'
    else:
        problem = f'fp_ndim{args.ndim}'
    loss_name = get_loss_descriptor(args)
    assert args.sampling_mode in ['gaussian', 'laplacian', 'uniform']
    return os.path.join(
        args.log_dir,
        f'{problem}_ss{args.operator_scale},{args.operator_shift}',
        f'{loss_name}_rsdl{args.residual_weight}'
        f'_neigs{args.neigs}'
        f'_{args.nonlinearity}_wn{int(args.weight_normalization)}_p{int(args.parallel)}'
        f'_bdd{int(args.apply_boundary)}'
        f'{f"_{args.boundary_mode}" if args.apply_boundary else ""}'
        f'_exp{int(args.apply_exp_mask)}'
        f'{f",{args.exp_mask_init_scale}" if args.apply_exp_mask else ""}'
        f'_lap{args.laplacian_eps}'
        f'_fourier{int(args.use_fourier_feature)}'
        f'{f",raw{int(args.fourier_append_raw)},size{args.fourier_mapping_size},scale{args.fourier_scale}" if args.use_fourier_feature else ""}'
        f'_{args.sampling_mode},scale{args.sampling_scale}'
        f'_bs{args.batch_size}_niters{args.num_iters}'
        f'_{args.optimizer}_lr{args.lr}'
        f'_ema{args.ema_decay}_hard{args.hard_mul_const}'
        f'{"_sch" if args.use_lr_scheduler else ""}'
        f'{"_sort" if args.sort else ""}'
        f'_seed{args.seed}'
    )


def main(args):
    # set random seed
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f'There are {torch.cuda.device_count()} GPU(s) available.')
        print(f'Device name: {torch.cuda.get_device_name(0)}')
    else:
        print('No GPU available, using the CPU instead.')
        device = torch.device("cpu")
    # set log dir
    args.log_dir = get_log_dir(args)
    if os.path.exists(args.log_dir) and not args.overwrite:
        raise ValueError(f"{args.log_dir} already exists and overwrite is not permitted")
    os.makedirs(args.log_dir, exist_ok=True)
    print(f'creating file {args.log_dir}')
    log_file, log_writer = get_log_file(
        args.log_dir,
        fieldnames=['iter', 'train_loss', 'avg_train_loss', 'time']
    )
    # set args
    if args.lim == 'pi':
        args.lim = np.pi
    else:
        args.lim = float(args.lim)
    # define problem (operator / ground truth spectrum)
    operator, ground_truth_spectrum = get_problem(args, device)
    # define base model
    model = get_wavefunctions(args)
    # set dataloader
    make_batch_ftn_train, val_data, batch_ftn_val, importance_train, importance_val = get_dataloader(args, device)
    # train
    method = get_evd_method(args, args.loss.name, model).to(device)
    all_eigvals, all_norms = train_operator(
        args,
        method,
        operator,
        make_batch_ftn_train,
        val_data,
        batch_ftn_val,
        log_writer,
        log_file,
        device,
        importance_train=importance_train,
        importance_val=importance_val,
        ground_truth_spectrum=ground_truth_spectrum
    )
    np.savez(f'{args.log_dir}/stats.npz', all_eigvals=all_eigvals, all_norms=all_norms)


if __name__ == '__main__':
    args = get_args()
    main(args)
