import math
from functools import partial

import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import weight_norm

from methods.utils import parse_str


class Erf(nn.Module):
    def __init__(self):
        super(Erf, self).__init__()

    def forward(self, x):
        return x.erf()


class SinAndCos(nn.Module):
    def __init__(self):
        super(SinAndCos, self).__init__()

    def forward(self, x):
        assert x.shape[1] % 2 == 0
        x1, x2 = x.chunk(2, dim=1)
        return torch.cat([torch.sin(x1), torch.cos(x2)], 1)


class Sine(nn.Module):
    def __init__(self, w0=1.):
        super().__init__()
        self.w0 = w0

    def forward(self, x):
        return torch.sin(self.w0 * x)


def init_mlp(model, w_var_list, b_var_list):
    if not isinstance(w_var_list, list):
        w_var_list = [w_var_list]
    if not isinstance(b_var_list, list):
        b_var_list = [b_var_list]
    i = 0
    for m in model.modules():
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            with torch.no_grad():
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
                m.weight.normal_(0, math.sqrt(w_var_list[i] / fan_in))
                if m.bias is not None:
                    if math.sqrt(b_var_list[i]) > 0:
                        m.bias.normal_(0, math.sqrt(b_var_list[i]))
                    else:
                        m.bias.fill_(0.)
                i += 1
                if i >= len(w_var_list):
                    i = 0

        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.running_mean, 0)
            nn.init.constant_(m.running_var, 1)


def get_activation(nonlinearity='relu'):
    if nonlinearity == 'relu':
        activation_ftn = partial(nn.ReLU, inplace=True)
    elif nonlinearity.startswith('lrelu'):
        negative_slope = float(nonlinearity.replace("lrelu", ""))
        activation_ftn = partial(nn.LeakyReLU, negative_slope=negative_slope)
    elif nonlinearity.startswith('elu'):
        elu_alpha = float(nonlinearity.replace("elu", ""))
        activation_ftn = partial(nn.ELU, alpha=elu_alpha)
    elif nonlinearity == 'tanh':
        activation_ftn = nn.Tanh
    elif nonlinearity == 'erf':
        activation_ftn = Erf
    elif nonlinearity == 'sin_and_cos':
        activation_ftn = SinAndCos
    elif nonlinearity == 'siren':
        activation_ftn = Sine
    elif nonlinearity == 'linear':
        activation_ftn = nn.Identity
    elif nonlinearity == 'softplus':
        activation_ftn = nn.Softplus
    else:
        raise NotImplementedError
    return activation_ftn


def get_mlp_eigfuncs(
        input_dim,
        neigs,
        mlp_hidden_dims,
        nonlinearity,
        bias=True,
        weight_normalization=False,
        parallel=False,
        feature_map=None,
        debug=False,
):
    mlp_hidden_dims = parse_str(mlp_hidden_dims)
    if not parallel:
        # mlp_hidden_dims[0] *= neigs  # for a fair (?) comparison with NeuralEF...
        mlp_sizes = [input_dim if feature_map is None else feature_map.feature_dim] + mlp_hidden_dims + [neigs]
        model = get_mlp(
            mlp_sizes,
            bias=bias,
            nonlinearity=nonlinearity,
            use_bn=False,
            weight_normalization=weight_normalization,
            feature_map=feature_map,
        )
    else:
        model = ParallelMLP(
            input_dim=input_dim,
            mlp_hidden_dims=mlp_hidden_dims,
            output_dim=1,
            num_copies=neigs,
            bias=bias,
            nonlinearity=nonlinearity,
            weight_normalization=weight_normalization,
            feature_map=feature_map,
            debug=debug,
        )
    return model


def get_mlp(
        sizes,
        bias=True,
        nonlinearity='relu',
        use_bn=True,
        weight_normalization=False,
        last_layer_bn=True,
        feature_map=None,
):
    activation = get_activation(nonlinearity)
    # BarlowTwins uses sizes=[output_dim, 8192, 8192, 8192], bias=False
    # BarlowTwins--HSIC paper uses sizes=[2048, 512, feature_dim], feature_dim=128, bias=True
    if len(sizes) == 1:
        if not use_bn:
            model = nn.Identity()
        else:
            if use_bn and last_layer_bn:
                model = nn.BatchNorm1d(sizes[0])
    else:
        num_layers = len(sizes) - 1
        layers = [] if feature_map is None else [feature_map]
        for i in range(num_layers):
            layer = nn.Linear(sizes[i], sizes[i + 1], bias=bias)
            if weight_normalization:
                layer = weight_norm(layer)
            layers.append(layer)  # Note: BarlowTwins sets this bias always False
            if use_bn and ((i < num_layers - 1) or
                           ((i == num_layers - 1) and last_layer_bn)):
                layers.append(nn.BatchNorm1d(sizes[i + 1]))
            if i < num_layers - 1:
                layers.append(activation())
            # NOTE: there is no activation at the end!

        model = nn.Sequential(*layers)
    model.output_dim = sizes[-1]
    return model


class ParallelMLP(nn.Module):
    def __init__(
            self,
            input_dim,
            mlp_hidden_dims,
            output_dim,
            num_copies,
            nonlinearity,
            bias=False,
            weight_normalization=False,
            feature_map=None,
            debug=False,
    ):
        super().__init__()
        self.feature_map = feature_map
        ws = nn.ParameterList()
        bs = nn.ParameterList()
        hdim_prev = input_dim if feature_map is None else feature_map.feature_dim
        for hdim in mlp_hidden_dims + [output_dim]:
            if not debug:
                ws.append(nn.Parameter(math.sqrt(2. / hdim_prev) * torch.randn(num_copies, hdim, hdim_prev)))
                if bias:
                    bs.append(nn.Parameter(torch.zeros([num_copies, hdim, 1])))
            else:
                ws.append(nn.Parameter(.1 * torch.ones([num_copies, hdim, hdim_prev])))
                if bias:
                    bs.append(nn.Parameter(.1 * torch.ones([num_copies, hdim, 1])))
            hdim_prev = hdim
        self.ws = ws
        self.bs = bs
        self.activation = get_activation(nonlinearity)
        self.bias = bias
        self.weight_normalization = weight_normalization

    def norm(self, w):
        return torch.linalg.norm(w, dim=(-1, -2), keepdims=True) if self.weight_normalization else 1.

    def forward(self, x):
        if self.feature_map is not None:
            x = self.feature_map(x)
        h = torch.einsum(
            'lhd,bd->lhb',
            self.ws[0] / self.norm(self.ws[0]),
            x
        ) + (self.bs[0] if self.bias else 0.)
        h = self.activation()(h)
        for i in range(1, len(self.ws)):
            h = torch.einsum(
                'lhp,lpb->lhb',
                self.ws[i] / self.norm(self.ws[0]),
                h
            ) + (self.bs[i] if self.bias else 0.)
            if i < len(self.ws) - 1:
                h = self.activation()(h)
        return h.permute(2, 0, 1).squeeze()  # (B, L, O)


class ParallelLinear(nn.Module):
    def __init__(self, in_features, out_features, num_copies):
        super(ParallelLinear, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.randn(num_copies, out_features, in_features)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(num_copies, out_features, 1)))

        for i in range(num_copies):
            nn.init.normal_(self.weight[i], 0, math.sqrt(2. / in_features))
        nn.init.zeros_(self.bias)

    def forward(self, x):
        if x.dim() == 2:
            return torch.tensordot(self.weight, x, [[2], [1]]) + self.bias
        else:
            return self.weight @ x + self.bias


class MultidimParallelMLP(nn.Module):
    def __init__(self, in_features, out_features, num_copies, num_layers, hidden_size=64, nonlinearity='relu'):
        super(MultidimParallelMLP, self).__init__()

        if nonlinearity == 'relu':
            nonlinearity = nn.ReLU
        elif 'lrelu' in nonlinearity:
            nonlinearity = partial(nn.LeakyReLU, float(nonlinearity.replace("lrelu", "")))
        elif nonlinearity == 'erf':
            nonlinearity = Erf
        elif nonlinearity == 'sin_and_cos':
            nonlinearity = SinAndCos
        else:
            raise NotImplementedError

        if num_layers == 1:
            self.fn = nn.Sequential(
                ParallelLinear(in_features, out_features, num_copies))
        else:
            layers = [ParallelLinear(in_features, hidden_size, num_copies),
                      nonlinearity(),
                      ParallelLinear(hidden_size, out_features, num_copies)]
            for _ in range(num_layers - 2):
                layers.insert(2, nonlinearity())
                layers.insert(2, ParallelLinear(hidden_size, hidden_size, num_copies))
            self.fn = nn.Sequential(*layers)

    def forward(self, x):
        return self.fn(x).permute(2, 1, 0)


class Parallel(nn.Module):
    def __init__(self, models):
        super(Parallel, self).__init__()
        self.models = nn.ModuleList(models)

    def forward(self, x):
        return torch.cat([model(x) for model in self.models], 1)


class ParallelMLPSlow(nn.Module):
    def __init__(self, sizes, nonlinearity='relu'):
        super(ParallelMLPSlow, self).__init__()
        sizes = sizes.copy()
        activation_ftn = get_activation(nonlinearity)
        num_copies = sizes[-1]
        sizes[-1] = 1
        num_layers = len(sizes) - 1
        layers = []
        for i in range(num_layers):
            layers.append(ParallelLinear(sizes[i], sizes[i + 1], num_copies))
            if i < num_layers - 1:
                layers.append(activation_ftn())
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x).permute(2, 1, 0)
