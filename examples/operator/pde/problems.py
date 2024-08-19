from functools import partial
import numpy as np
import torch

from examples import OperatorWrapper
from examples.operator.pde.schrodinger.potentials import (
    infinite_well_potential,
    harmonic_oscillator_potential,
    cosine_potential,
    hydrogen_potential,
    hydrogen_mol_ion_potential,
    local_potential_energy,
)
from examples.operator.pde.schrodinger import NegativeHamiltonian
from examples.operator.pde.others import (
    NegativeLinearFokkerPlanck,
    sin_of_cos_potential,
)
from examples.operator.pde.schrodinger.ground_truths import Hydrogen2D, Hydrogen3D, HarmonicOscillator, InfiniteWell2D
from examples.operator.pde.schrodinger.molecule import Molecule


def get_problem(args, device):
    # define problem
    ground_truth_spectrum = None
    if args.problem == 'sch':  # Schrödinger's equation
        args.n_particles = 1
        scale_kinetic = 1.
        if args.potential_type == 'infinite_well':
            assert args.ndim == 2
            local_potential_ftn = infinite_well_potential
            ground_truth_spectrum = - InfiniteWell2D(L=2 * args.lim).get_eigvals(args.neigs)
        elif args.potential_type == 'harmonic_oscillator':
            local_potential_ftn = partial(harmonic_oscillator_potential, k=1.)
            ground_truth_spectrum = - HarmonicOscillator(k=1., ndim=args.ndim).get_eigvals(args.neigs)
        elif args.potential_type == 'cosine':
            # (Han, Lu, and Zhou, 2020)
            # https://github.com/frankhan91/BSDEEigen/blob/master/equation.py
            assert args.lim == np.pi
            assert not args.apply_boundary
            assert args.use_fourier_feature
            assert args.fourier_deterministic
            assert not args.use_gaussian_sampling
            assert args.ndim in [1, 2, 5, 10]
            if args.ndim == 1:
                cs = [1.0]
            elif args.ndim == 2:
                assert args.neigs <= 25
                cs = [0.814723686393179, 0.905791937075619]
                ground_truth_spectrum = - np.array([
                                                       -0.591624518674115, 0.623365592493771, 0.662887867122419,
                                                       0.891545971509540, 0.982541637674317,
                                                       1.877877978290306, 2.146058357306075, 2.197531748842203,
                                                       2.465712127857973, 3.699555061533076,
                                                       3.701057706578779, 3.756708397099993, 3.758994296902169,
                                                       4.954067447329610, 4.955570092375313,
                                                       4.971698508267879, 4.973984408070056, 5.239878887283648,
                                                       5.242164787085825, 5.273721217881508,
                                                       5.275223862927211, 8.047887977307184, 8.049390622352888,
                                                       8.050173877109360, 8.051676522155063
                                                   ][:args.neigs])
            elif args.ndim == 5:
                cs = [0.162944737278636, 0.181158387415124, 0.025397363258701, 0.182675171227804, 0.126471849245082]
                ground_truth_spectrum = np.array([0.054018930536326] + (args.neigs - 1) * [0.])
            else:
                cs = [0.162944737278636, 0.181158387415124, 0.025397363258701, 0.182675171227804, 0.126471849245082,
                      0.019508080999882, 0.055699643773410, 0.109376303840997, 0.191501367086860, 0.192977707039855]
                ground_truth_spectrum = np.array([0.098087448866409] + (args.neigs - 1) * [0.])
            local_potential_ftn = partial(cosine_potential, cs=cs)
        elif args.potential_type == 'hydrogen':
            local_potential_ftn = partial(hydrogen_potential, charge=args.charge)
            if args.ndim == 2:
                ground_truth_spectrum = - Hydrogen2D(charge=args.charge).get_eigvals(args.neigs)
            elif args.ndim == 3:
                ground_truth_spectrum = - Hydrogen3D(charge=args.charge).get_eigvals(args.neigs)
        elif args.potential_type == 'hydrogen_mol_ion':
            local_potential_ftn = partial(hydrogen_mol_ion_potential, R=args.hydrogen_mol_ion_R, charge=2 * args.charge)
        elif args.potential_type == 'quantum_chemistry':
            assert args.ndim in [2, 3]
            mol = Molecule.from_name(args.mol_name)
            mol.coords = mol.coords.to(device)
            mol.charges = mol.charges.to(device)
            if args.ndim == 2:
                mol.coords = mol.coords[:, :2]
            local_potential_ftn = partial(local_potential_energy, mol=mol)
            args.n_particles = (mol.charges.sum() - mol.charge).type(torch.int).item()
            scale_kinetic = .5
        else:
            raise NotImplementedError
        operator = NegativeHamiltonian(
            local_potential_ftn=local_potential_ftn,
            scale_kinetic=scale_kinetic,
            laplacian_eps=args.laplacian_eps,
            n_particles=args.n_particles,
        )
    elif args.problem == 'fp':  # Fokker--Planck
        # (Han, Lu, and Zhou, 2020)
        # https://github.com/frankhan91/BSDEEigen/blob/master/equation.py
        args.n_particles = 1
        assert args.lim == np.pi
        assert not args.apply_boundary
        assert args.use_fourier_feature
        assert args.fourier_deterministic
        assert not args.use_gaussian_sampling
        assert args.ndim in [1, 2, 5, 10]
        if args.ndim == 1:
            cs = [1.0]
        elif args.ndim == 2:
            cs = [1.0, 1.0]
        elif args.ndim == 5:
            cs = [1.0, 0.8, 0.6, 0.4, 0.2]
        else:
            cs = [0.1, 0.3, 0.2, 0.5, 0.2, 0.1, 0.3, 0.4, 0.2, 0.2]
        ground_truth_spectrum = np.array([0.] + (args.neigs - 1) * [0.])
        operator = NegativeLinearFokkerPlanck(
            local_potential_ftn=partial(sin_of_cos_potential, cs=cs),
            scale=args.scale_operator,
            laplacian_eps=args.laplacian_eps,
        )
    else:
        raise NotImplementedError

    operator = OperatorWrapper(
        operator,
        scale=args.operator_scale,
        shift=args.operator_shift,
    )
    ground_truth_spectrum = (args.operator_scale * ground_truth_spectrum + args.operator_shift
                             if ground_truth_spectrum is not None else None)
    return operator, ground_truth_spectrum
