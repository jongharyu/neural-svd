import time

import numpy as np
import termplotlib as tpl
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.linalg import eigh
from tqdm import tqdm


def check_only_one_arg(*args):
    return np.array([int(arg is not None) for arg in args]).sum() == 1


def get_sqrt_weight_func(importance_train, importance_val):
    def sqrt_weight_func(x):
        sqrt_ws_train = 1.
        sqrt_ws_val = 1.
        if importance_train is not None:
            sqrt_ws_train = importance_train(x).sqrt()
        if importance_val is not None:
            sqrt_ws_val = importance_val(x).sqrt()
        return sqrt_ws_train, sqrt_ws_val
    return sqrt_weight_func


def compute_spectrum_evd(
        model,
        dataloader,
        operator,
        importance_train=None,
        importance_val=None,
        set_first_mode_const=False,
        post_align=False,
        normalize=False,
        sort=False,
        gpu=None,
        device=None,
):
    assert check_only_one_arg(gpu, device)
    sqrt_weight_func = get_sqrt_weight_func(importance_train, importance_val)
    start = time.time()
    n = 0
    cov = 0.
    quad = 0.
    eigfuncs = []
    for (x, _) in tqdm(dataloader):
        if isinstance(x, list):
            x = x[0]
        if gpu is not None:
            x = x.cuda(gpu, non_blocking=True)
        if device is not None:
            x = x.to(device)
        sqrt_ws_train, sqrt_ws_val = sqrt_weight_func(x)
        if sqrt_ws_train.shape != sqrt_ws_val.shape:
            print("Warning: shape mismatch in sqrt_ws_train and sqrt_ws_val",
                  x.shape, sqrt_ws_train.shape, sqrt_ws_val.shape)
            continue
        sqrt_ws = sqrt_ws_train / sqrt_ws_val
        Tphi, phi = operator(model, x, importance=importance_train)
        Tphi = Tphi.detach()
        phi = phi.detach()
        eigfuncs.append(sqrt_ws_train * phi)
        phi = sqrt_ws * phi
        Tphi = sqrt_ws * Tphi
        if set_first_mode_const:
            phi = nn.ConstantPad1d((1, 0), 1)(phi)  # (batch_size, feature_dim + 1)
            Tphi = nn.ConstantPad1d((1, 0), 1)(Tphi)  # (batch_size, feature_dim + 1)
        phi = torch.nan_to_num(phi)
        Tphi = torch.nan_to_num(Tphi)
        Tphi[torch.where(torch.all(torch.isclose(x, torch.zeros_like(x[0])), dim=1))] *= 0.
        cov += phi.T @ phi  # (L, L)
        quad += phi.T @ Tphi  # (L, L)
        n += len(x)
    cov /= n
    quad /= n
    end = time.time()
    print(f"Took {end - start}s to compute spectrum with data of size {n}")
    # collect outputs
    outputs = dict()
    outputs['eigfuncs'] = eigfuncs = torch.cat(eigfuncs, dim=0).cpu().numpy()
    outputs['cov'] = cov = cov.cpu().numpy()
    outputs['quad'] = quad = quad.cpu().numpy()
    outputs['eigvals'] = eigvals = np.diag(quad) / np.diag(cov)  # Rayleigh quotient based estimators
    outputs['norms'] = norms = np.diag(cov)  # norm based estimator (specifically for NestedLoRA)
    if normalize:
        outputs['cov'] = cov / (np.sqrt(norms[:, np.newaxis]) @ np.sqrt(norms[:, np.newaxis]).T)
        outputs['eigfuncs'] = eigfuncs / np.sqrt(norms).reshape(1, -1)
    if sort:
        sort_indices = np.argsort(eigvals)[::-1]
        outputs['eigvals'] = outputs['eigvals'][sort_indices]
        outputs['eigfuncs'] = outputs['eigfuncs'][:, sort_indices, ...]
        outputs['cov'] = outputs['cov'][:, sort_indices][sort_indices, :]
        outputs['quad'] = outputs['quad'][:, sort_indices][sort_indices, :]
        outputs['norms'] = outputs['norms'][sort_indices]
    if post_align:
        outputs['eigfuncs_aligned'], outputs['eigvals_aligned'], outputs['cov_aligned'] = post_alignment(
            outputs['eigfuncs'], outputs['cov'], outputs['quad']
        )
    return outputs


@torch.no_grad()
def compute_spectrum_svd(
        model,
        dataloader,
        gpu=None,
        device=None,
        sort=False,
        set_first_mode_const=False
):
    # TODO: add importance if there is a usage
    # TODO: add operator if there is a usage (currently, this only applies to NestedLoRAForCDK)
    # TODO: if the operator is a differential operator computed with autograd, torch.no_grad should be removed
    print("Computing spectrum...")
    assert check_only_one_arg(gpu, device)
    start = time.time()
    with torch.no_grad():
        n = 0
        matrix_x = 0.
        matrix_y = 0.
        for i, (x, y, cls) in enumerate(tqdm(dataloader)):
            if gpu is not None:
                x = x.cuda(gpu, non_blocking=True)
                y = y.cuda(gpu, non_blocking=True)
            if device is not None:
                x = x.to(device)
                y = y.to(device)
            f, g = model(x, y)  # (batch_size, feature_dim)
            if set_first_mode_const:
                f = nn.ConstantPad1d((1, 0), 1)(f)  # (batch_size, feature_dim + 1)
                g = nn.ConstantPad1d((1, 0), 1)(g)  # (batch_size, feature_dim + 1)
            matrix_x += f.T @ f
            matrix_y += g.T @ g
            n += x.shape[0]
        matrix_x = matrix_x / n
        matrix_y = matrix_y / n

    end = time.time()
    diag_x = torch.diag(matrix_x).unsqueeze(1)
    diag_y = torch.diag(matrix_y).unsqueeze(1)
    spectrum = (diag_x * diag_y).sqrt()
    orthogonality_x = matrix_x / (diag_x @ diag_x.T).sqrt()
    orthogonality_y = matrix_y / (diag_y @ diag_y.T).sqrt()
    print(f"Took {end - start}s to compute spectrum with data of size {n}")

    spectrum = spectrum.squeeze().cpu().numpy()
    orthogonality_x = orthogonality_x.cpu().numpy()
    orthogonality_y = orthogonality_y.cpu().numpy()
    if sort:
        sort_indices = np.argsort(spectrum)[::-1]
        spectrum = spectrum[sort_indices]
        orthogonality_x = orthogonality_x[sort_indices, :][:, sort_indices]
        orthogonality_y = orthogonality_y[sort_indices, :][:, sort_indices]

    return spectrum, orthogonality_x, orthogonality_y  # (feature_dim, ), (feature_dim, feature_dim)


def post_alignment(eigfuncs, cov, quad):
    eigvals_cov, eigvecs_cov = eigh(cov)
    whitening = eigvecs_cov @ np.diag(1 / np.sqrt(eigvals_cov)) @ eigvecs_cov.T
    eigvals, V = eigh(whitening @ quad @ whitening)
    eigvals = np.sqrt(eigvals[::-1])
    V = V[:, ::-1]
    eigfuncs = eigfuncs @ (V.T @ whitening).T
    orthogonality = np.eye(quad.shape[0])
    return eigfuncs, eigvals, orthogonality


def plot_orth(ax, data, cmap='gray', linewidth=0.005):
    mesh = ax.pcolormesh(data[::-1, :], edgecolors='black', linewidth=linewidth, cmap=cmap, vmin=0, vmax=1)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.tick_params(labelbottom=False, labelleft=False,)
    return mesh


def plot_and_save_spectrum(
        spectrum,
        orthogonality,
        orthogonality_p=None,
        log_dir=None,
        tag=None,
        termplot=True,
        ground_truth_spectrum=None,
        ylim=(0, 1),
):
    if termplot:
        try:
            fig = tpl.figure()
            for key in spectrum:
                if spectrum[key] is not None:
                    fig.plot(np.arange(1, len(spectrum[key]) + 1), spectrum[key],
                             label=f'sum={spectrum[key].sum():.2f}', width=80, height=20)
            fig.show()
        except:
            print("Warning: Something went wrong with termplot")

    ncols = 2 if orthogonality_p is None else 3
    fig, axes = plt.subplots(nrows=1, ncols=ncols, figsize=(5 * ncols, 4))
    for key in spectrum:
        if spectrum[key] is not None:
            axes[0].plot(spectrum[key], marker='o', label=f'{key}(sum={spectrum[key].sum():.2f})')
            axes[0].set_xlim([0, len(spectrum[key]) - 1])
    if ground_truth_spectrum is not None:
        axes[0].plot(ground_truth_spectrum, marker='x', label='ground truth')
    axes[0].legend()
    axes[0].set_title(f'Spectrum')
    axes[0].set_ylim(ylim)
    if ground_truth_spectrum is not None:
        axes[0].set_ylim([0, np.max(ground_truth_spectrum)])
    axes[0].grid('on')

    mesh = plot_orth(axes[1], np.abs(orthogonality))
    axes[1].set_title(r'Orthogonality ($f(x)$)')
    axes[1].grid('off')
    if ncols == 3:
        mesh = plot_orth(axes[2], np.abs(orthogonality_p))
        axes[2].set_title(r'Orthogonality ($g(y)$)')
        axes[2].grid('off')
    divider = make_axes_locatable(axes[-1])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(mesh, cax=cax, orientation='vertical')

    fig.suptitle(tag)
    fig.tight_layout()
    plt.savefig(f'{log_dir}/spectrum_{tag}.png')
    plt.close()
