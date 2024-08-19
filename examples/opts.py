import copy
from collections import defaultdict
from distutils.util import strtobool

import numpy as np

from tools.generic import Namespace


def mlp_opts(parser):
    group = parser.add_argument_group("MLP")
    group.add('--mlp_hidden_dims',
              type=str,
              default='32,32,10',
              metavar='MLP',
              help='MLP hidden dimensions')
    group.add('--nonlinearity',
              type=str,
              default='relu')
    group.add('--weight_normalization',
              type=strtobool,
              default=False)
    group.add('--parallel',
              type=strtobool,
              default=False)


def fourier_opts(parser):
    group = parser.add_argument_group("Fourier features")
    # feature configs
    group.add('--use_fourier_feature',
              action='store_true')
    group.add('--fourier_mapping_size',
              default=128,
              type=int)
    group.add('--fourier_scale',
              default=1.,
              type=float)
    group.add('--fourier_deterministic',
              action='store_true')
    group.add('--fourier_append_raw',
              action='store_true')


def operator_opts(parser):
    parser.add_argument('--operator_scale', default=1., type=float)
    parser.add_argument('--operator_shift', default=0., type=float)


def loss_opts(parser):
    parser.add_argument('--loss_name', type=str, required=True)
    parser.add_argument('--neigs', type=int, required=True)
    parser.add_argument('--sort', type=strtobool, default=False)
    parser.add_argument('--post_align', default=False, action='store_true')
    parser.add_argument('--residual_weight', default=0., type=float)

    group = parser.add_argument_group("NestedLoRA")
    group.add('--neuralsvd.step', default=1, type=int)
    group.add('--neuralsvd.sequential', default=False, type=strtobool, help="if True, apply the sequential nesting")
    group.add('--neuralsvd.set_first_mode_const', type=strtobool, default=True)  # cdk only

    group = parser.add_argument_group("NeuralEigenfunctions/NeuralEigenmaps")
    group.add('--neuralef.unbiased', type=strtobool, default=False)
    group.add('--neuralef.batchnorm_mode', type=str, default='unbiased', choices=['biased', 'unbiased'])
    group.add('--neuralef.include_diag', type=strtobool, default=False)
    # neigenmaps
    group.add('--neuralef.reg_weight', type=float, default=1.0)
    group.add('--neuralef.stop_grad', type=strtobool, default=True)  # cdk only
    group.add('--neuralef.symmetric', type=strtobool, default=True)  # cdk only

    group = parser.add_argument_group("SpIN")
    group.add('--spin.decay', default=0.01, type=float, help='Decay rate of moving averages for SpIN')
    group.add('--spin.show_plots', default=True, help='Show pyplot plots. 2D slices at z=0 are used for ndim=3.')
    group.add('--spin.use_pfor', type=strtobool, default=True, help='Use parallel_for.')
    group.add('--spin.per_example', default=False, help='Use a different strategy for computing covariance Jacobian')

    group = parser.add_argument_group("SimCLR")  # cdk only
    group.add('--simclr.normalize', type=strtobool, default=False)
    group.add('--simclr.temperature', type=float, default=0.1)
    group.add('--simclr.num_negatives', type=float, default=0.1)

    group = parser.add_argument_group("BarlowTwins")  # cdk only
    group.add('--barlowtwins.reg_weight', type=float, default=1.0)


def reg_opts(parser):
    group = parser.add_argument_group("Regularization")
    group.add('--mu', type=float, default=0.)
    group.add('--regularize_mode', type=str, default='l2_ball',
              choices=['l2_ball', 'l2_sphere', 'clip', 'tanh'])
    group.add('--posthoc_scaling', action='store_true')


def dist_learning_opts(parser):
    group = parser.add_argument_group("Distributed training")
    group.add('--world_size', default=1, type=int,
              help='number of nodes for distributed training')
    group.add('--rank', default=0, type=int,
              help='node rank for distributed training')
    group.add('--dist_url', default='tcp://127.0.0.1', type=str,
              help='url used to set up distributed training')
    group.add('--dist_port', default='1234', type=str,
              help='port used to set up distributed training')
    group.add('--dist_backend', default='nccl', type=str,
              help='distributed backend')


def amp_opts(parser):
    group = parser.add_argument_group("Mixed precision training")
    group.add('--disable_amp', default=False, action='store_true')
    group.add('--clip_grad_norm', default=False, action='store_true')
    group.add('--clip_grad_norm_value', default=1., type=float)


def proj_opts(parser):
    group = parser.add_argument_group("Projector")
    group.add('--projector_dims', type=str, default='2048,2048')
    group.add('--projector_use_bn', type=strtobool, default=True)
    group.add('--projector_last_layer_bn', type=strtobool, default=True)
    group.add('--projector_activation', type=str, default='relu')


def parse_loss_configs(args, implemented_losses):
    loss_configs_dict = dict(loss=defaultdict(dict))
    loss_configs_dict['loss']['name'] = args.loss_name
    del args.loss_name
    for key, value in copy.deepcopy(args).__dict__.items():
        if key.startswith(implemented_losses):
            loss_name, loss_config_name = key.split('.')
            loss_configs_dict['loss'][loss_name][loss_config_name] = value
            del args.__dict__[key]
    losses_configs = Namespace(loss_configs_dict)

    for key, value in losses_configs.__iter__():
        vars(args)[key] = value

    return args
