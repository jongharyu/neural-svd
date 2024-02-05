import numpy as np
import torch
from matplotlib import pyplot as plt

from methods.utils import remove_inf


def plot_hist_ratios_wrapper(rs_pxy_train, rs_pxpy_train, rs_pxy_test, rs_pxpy_test, tag='', filepath='hist.png'):
    rs_pxy_train = remove_inf(torch.cat(rs_pxy_train).numpy())
    rs_pxpy_train = remove_inf(torch.cat(rs_pxpy_train).numpy())
    rs_pxy_test = remove_inf(torch.cat(rs_pxy_test).numpy())
    rs_pxpy_test = remove_inf(torch.cat(rs_pxpy_test).numpy())

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))

    # plot hist(ratios)
    axes[0][0] = plot_hist_ratios(axes[0][0], rs_pxy_train, rs_pxpy_train)
    axes[0][0].set_xlabel(r'ratios $\frac{p(x,y)}{p(x)p(y)}$')
    axes[0][0].set_title('train')
    axes[0][1] = plot_hist_ratios(axes[0][1], rs_pxy_test, rs_pxpy_test)
    axes[0][1].set_xlabel(r'ratios $\frac{p(x,y)}{p(x)p(y)}$')
    axes[0][1].set_title('test')

    # plot hist(log_ratios)
    axes[1][0] = plot_hist_ratios(axes[1][0], np.log(np.maximum(rs_pxy_train, 1e-5)),
                                  np.log(np.maximum(rs_pxpy_train, 1e-5)))
    axes[1][0].set_xlabel(r'log ratios $\log\frac{p(x,y)}{p(x)p(y)}$')
    axes[1][0].set_title('train')
    axes[1][1] = plot_hist_ratios(axes[1][1], np.log(np.maximum(rs_pxy_test, 1e-5)),
                                  np.log(np.maximum(rs_pxpy_test, 1e-5)))
    axes[1][1].set_xlabel(r'log ratios $\log\frac{p(x,y)}{p(x)p(y)}$')
    axes[1][1].set_title('test')

    fig.suptitle(f'Histogram ({tag})')
    fig.tight_layout()
    plt.savefig(filepath)
    plt.close()


def plot_hist_ratios(ax, rs_pxy, rs_pxpy):
    ax.hist(rs_pxy, bins=100, density=False, label='joint', alpha=0.5, color='blue')
    # ax.set_yscale('log')
    ax_ = ax.twinx()
    ax_.hist(rs_pxpy, bins=100, density=False, label='indep', alpha=0.5, color='red')
    # ax_.set_yscale('log')
    # ask matplotlib for the plotted objects and their labels
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax_.get_legend_handles_labels()
    ax_.legend(lines1 + lines2, labels1 + labels2)
    ax.set_ylabel('joint', color='blue')
    ax_.set_ylabel('indep', color='red')
    ax.grid('on')
    return ax