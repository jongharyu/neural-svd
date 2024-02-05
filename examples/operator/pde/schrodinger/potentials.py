import numpy as np
import torch


def hydrogen_potential(x, charge=1.):
    x = x.reshape(x.shape[0], -1)
    # hydrogen atom H
    return - (charge / x.norm(dim=1, p=2)).reshape(-1, 1)


def hydrogen_mol_ion_potential(x, R, charge=2.):
    x = x.reshape(x.shape[0], -1)
    # hydrogen molecule ion H_2^+
    # nuclei locations: (0, ..., R), (0, ..., -R) (aligned over the last axis)
    e = torch.zeros((x.shape[-1], )).to(x.device)
    e[-1] = 1.
    return hydrogen_potential(x - R * e, charge) + hydrogen_potential(x + R * e, charge)


def infinite_well_potential(x):
    return torch.zeros((x.shape[0], ), device=x.device)


def harmonic_oscillator_potential(x, k=1.):
    x = x.reshape(x.shape[0], -1)
    # only symmetric
    return (k * x.norm(dim=1, p=2) ** 2).reshape(-1, 1)


def cosine_potential(x, cs):
    return (torch.cos(x.view(x.shape[0], -1)) * torch.tensor(cs, device=x.device).view(1, -1)).sum(-1)


# for quantum chemistry (from deepqmc_torch)
def nuclear_energy(mol):
    coords, charges = mol.coords, mol.charges
    coulombs = charges[:, None] * charges / (coords[:, None] - coords).norm(dim=-1)
    return coulombs.triu(1).sum()


def nuclear_potential(rs, mol):
    dists = (rs[:, :, None] - mol.coords).norm(dim=-1)
    return -(mol.charges / dists).sum(dim=(-1, -2))


def electronic_potential(rs):
    i, j = np.triu_indices(rs.shape[-2], k=1)
    dists = (rs[:, :, None] - rs[:, None, :])[:, i, j].norm(dim=-1)
    return (1 / dists).sum(dim=-1)


def local_potential_energy(rs, mol):
    # rs: (batch_size, n_elec, 3)
    Es_nuc = nuclear_energy(mol)
    Vs_nuc = nuclear_potential(rs, mol)
    Vs_el = electronic_potential(rs)
    return Es_nuc + Vs_nuc + Vs_el
