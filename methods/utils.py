from typing import List

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

centralize = lambda phi: phi - torch.mean(phi, 0)  # centralize over batch dimension (when rows of phi are features)


def extract_tensor(x, mask, start_dim=0):
    assert x.shape[start_dim:] == mask.shape
    return x.flatten(start_dim=start_dim)[..., torch.any(mask.flatten().unsqueeze(1), dim=1)]


def off_diagonal(x, start_dim=0):
    # return a flattened view of the off-diagonal elements of a batch of square matrices
    # (the last two dimensions must form square matrices)
    batch_shape = x.shape[:start_dim]
    n, m = x.shape[start_dim:]
    assert n == m
    return x.flatten(start_dim=start_dim)[..., :-1].view(*batch_shape, n - 1, n + 1)[..., 1:].flatten(start_dim=start_dim)


def remove_inf(a):
    a_finite = a[np.isfinite(a)]
    a_min, a_max = a_finite.min(), a_finite.max()
    a = np.minimum(np.maximum(a, a_min), a_max)
    return a


def parse_str(dims_str: str) -> List:
    return list(map(int, dims_str.split(','))) if dims_str != '' else []


class BatchL2NormalizedFunctions(nn.Module):
    def __init__(self, base_model, neigs, momentum=0.9, batchnorm_mode='unbiased'):
        super().__init__()
        self.base_model = base_model
        self.momentum = momentum
        # batchnorm mode only affects test phase
        assert batchnorm_mode in ['biased', 'unbiased']
        self.initialized = False
        self.batchnorm_mode = batchnorm_mode
        self._norm_biased = nn.Parameter(torch.ones(1, neigs), requires_grad=False)  # (neigs, ); default for NeuralEF
        self._norm_unbiased = nn.Parameter(torch.ones(1, neigs), requires_grad=False)  # (neigs, )

    def forward(self, x):
        output = self.base_model(x).squeeze()
        norm_dims = (0, ) if len(output.shape) == 2 else (0, -1)
        if self.training:
            norm = batch_l2norm = output.norm(dim=norm_dims, keepdim=True) / np.sqrt(output.shape[0])
            self.update_norm(batch_l2norm)
        else:
            norm = self._norm_biased if self.batchnorm_mode else self._norm_unbiased
        return output / norm

    @torch.no_grad()
    def update_norm(self, batch_l2norm):
        if not self.initialized:
            self.initialized = True
            self._norm_biased.data = batch_l2norm
            self._norm_unbiased.data = batch_l2norm
        else:
            self._norm_biased.data = self.momentum * self._norm_biased + \
                                     (1 - self.momentum) * batch_l2norm
            self._norm_unbiased.data = torch.sqrt(self.momentum * self._norm_unbiased ** 2 +
                                                  (1 - self.momentum) * batch_l2norm ** 2)

    def register_norm(self, data):
        batch_size = len(data)
        while True:
            try:
                self.register_norm_batch(data, batch_size)
                break
            except:
                batch_size = batch_size // 2

    @torch.no_grad()
    def register_norm_batch(self, data, batch_size):
        num_iters = len(data) // batch_size + (len(data) % batch_size != 0)
        squared_norm = 0.
        for it in tqdm(range(num_iters)):
            idx = range(batch_size * it, min(batch_size * (it + 1), len(data)))
            squared_norm += self.base_model(data[idx]).norm(dim=0) ** 2
        self._norm_biased.data = self._norm_unbiased.data = torch.sqrt(squared_norm / len(data))
