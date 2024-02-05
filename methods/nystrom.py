import math
import time

import numpy as np
import torch


class Nystrom:
    # only for fixed kernels
    def __init__(self, kernel, xs, dim, emp_kernel=None):
        self.kernel = kernel
        self.xs = xs
        self.dim = dim

        # construct eigenfunctions via EVD
        self.eigvals, self.eigvecs, self.training_time = self.evd(xs, kernel, dim, emp_kernel)
        self.eigvals = torch.from_numpy(self.eigvals.copy()).to(xs.device)
        self.eigvecs = torch.from_numpy(self.eigvecs.copy()).to(xs.device)

    def __call__(self, xnew):
        # projection via Nystrom approximation
        return self.kernel(xnew, self.xs) @ self.eigvecs / self.eigvals / math.sqrt(self.xs.shape[0])

    @staticmethod
    def evd(xs, kernel, dim, emp_kernel=None):
        start = time.time()
        if emp_kernel is None:
            assert kernel is not None, "If emp_kernel is not provided, kernel must be provided"
            emp_kernel = kernel(xs, xs)  # (B, B)

        # eigenvalue decomposition
        eigvals, eigvecs = np.linalg.eigh(emp_kernel.data.cpu().numpy())
        eigvals = eigvals[::-1][:dim] / xs.shape[0]
        eigvecs = eigvecs[:, ::-1][:, :dim]

        training_time = time.time() - start
        print(f'Time elapsed: {training_time}s')

        return eigvals, eigvecs, training_time


def run_nystrom(kernel, neigs, train_data, val_data, log_dir, emp_kernel=None):
    nystrom = Nystrom(kernel, train_data, neigs, emp_kernel)
    eigvals = nystrom.eigvals.cpu().numpy()
    eigfuncs = nystrom(val_data).cpu().numpy()
    np.savez(f'{log_dir}/eigvals.npz', eigvals=eigvals, eigfuncs=eigfuncs)
    return eigvals, eigfuncs, nystrom.training_time
