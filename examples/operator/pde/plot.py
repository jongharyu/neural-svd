import numpy as np
from matplotlib import pyplot as plt


def plot_1d_eigfuncs(xs, eigfuncs, log_dir, tag=''):
    fig, axes = plt.subplots(1, eigfuncs.shape[1], figsize=(eigfuncs.shape[1] * 2, 2.5))
    for i in range(eigfuncs.shape[1]):
        axes[i].plot(xs, eigfuncs[:, i])
        axes[i].grid(False)
        axes[i].axis('off')
    fig.suptitle(tag)
    fig.tight_layout()
    plt.savefig(f'{log_dir}/eigfuncs_{tag}.png', bbox_inches='tight')


def plot_2d_eigfuncs(eigfuncs, log_dir, tag=''):
    fig, axes = plt.subplots(1, eigfuncs.shape[1], figsize=(eigfuncs.shape[1] * 2, 2.5))
    for i in range(eigfuncs.shape[1]):
        n = int(np.sqrt(eigfuncs[:, i].shape[0]))
        axes[i].imshow(eigfuncs[:, i].reshape(n, n), interpolation='none', cmap='coolwarm')
        axes[i].grid(False)
        axes[i].axis('off')
    fig.suptitle(tag)
    fig.tight_layout()
    plt.savefig(f'{log_dir}/eigfuncs_{tag}.png', bbox_inches='tight')
