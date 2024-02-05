import numpy as np
from scipy.special import gammaln, hyp1f1, binom, genlaguerre, sph_harm, gamma, hyp2f1, lpmv


class ToyProblem:
    def __init__(self):
        pass

    def get_qnums(self, neigs):
        raise NotImplementedError

    def get_eigvals(self, neigs):
        raise NotImplementedError

    def eigfunc(self, *args):
        raise NotImplementedError

    def get_degeneracy(self, neigs):
        eigvals = self.get_eigvals(neigs)
        return self._get_degeneracy(eigvals)

    @staticmethod
    def _get_degeneracy(eigvals):
        cnt = 1
        eigval_prev = eigvals[0]
        degeneracy = [0]
        for eigval in eigvals[1:]:
            if eigval == eigval_prev:
                cnt += 1
            else:
                degeneracy.append(cnt)
                eigval_prev = eigval
                cnt = 1
        else:
            if cnt > 1:
                degeneracy.append(cnt)
        return np.array(degeneracy).cumsum()


class InfiniteWell2D(ToyProblem):
    def __init__(self, L=1.):
        super().__init__()
        self.L = L

    def get_qnums(self, neigs):
        qnums = []
        for n in range(1, 100):  # TODO: fix
            for i in range(1, n):
                qnums.append((n, i))
                qnums.append((i, n))
            qnums.append((n, n))
        return qnums[:neigs]

    def get_eigvals(self, neigs):
        return np.array(sorted([
            (nx ** 2) + (ny ** 2)
            for nx in range(1, neigs + 1)
            for ny in range(1, neigs + 1)
        ])[:neigs]) * np.pi ** 2 / self.L ** 2

    def eigfunc(self, nx, ny, x, y):
        L = self.L
        return 2 / L * np.sin(nx * np.pi * x / L) * np.sin(ny * np.pi * y / L)


class HarmonicOscillator(ToyProblem):
    def __init__(self, k=1., ndim=2):
        super().__init__()
        assert ndim == 2, f"dim={ndim} not implemented"
        self.k = k
        self.ndim = ndim

    def get_qnums(self, neigs):
        assert self.ndim == 2
        qnums = np.vstack([np.array([(i, n - i) for i in range(n + 1)]) for n in range(100)])  # TODO: fix
        return qnums[:neigs]

    def get_eigvals(self, neigs):
        ndim = self.ndim
        k = self.k

        num_degeneracy = lambda n: int(binom(ndim + n - 1, n))
        nend = 0
        num_states = 0
        while 1:
            num_states += num_degeneracy(nend)
            nend += 1
            if num_states >= neigs:
                break
        return np.sqrt(k) * np.concatenate([num_degeneracy(n) * [2 * n + ndim] for n in range(nend + 1)])

    def eigfunc(self, *args):
        assert self.ndim == 2
        return self._eigfunc_2d(*args)

    def _eigfunc_1d(self, n, x, b=1.):
        deg = np.zeros(n + 1)
        deg[-1] = 1
        return (1 / np.sqrt(2 ** n * np.exp(gammaln(n + 1))) *
                (b / np.pi) ** (1 / 4) *
                np.exp(- b * x ** 2 / 2) *
                np.polynomial.Hermite(deg)(np.sqrt(b) * x)
                )

    def _eigfunc_2d(self, nx, ny, x, y, b=1.):
        return (self._eigfunc_1d(nx, x, b) *
                self._eigfunc_1d(ny, y, b))


class Hydrogen2D(ToyProblem):
    def __init__(self, charge=1.):
        super().__init__()
        self.charge = charge  # charge of nucleus

    def get_qnums(self, neigs):
        nmax = int(np.ceil(np.sqrt(neigs)))
        qnums = [(n, l) for n in range(0, nmax + 1) for l in range(-n, n + 1)]
        return qnums[:neigs]

    def get_eigvals(self, neigs):
        # E(n;Z) = - Z^2 / [4 * (n+1/2)^2]
        # Quantum numbers: n = 0, 1, ...;
        #                  l = -n, -n+1, ... n
        # degeneracy: 2n+1. Use k^2 as an upper bound to \sum 2n+1.
        max_n = int(np.ceil(np.sqrt(neigs))) + 1
        qnums = []
        for n in range(0, max_n):
            for _ in range(2 * n + 1):
                qnums.append(n)
        qnums = np.array(qnums)
        ground_truth = - self.charge ** 2 / (4 * (qnums[:neigs] + 0.5) ** 2)
        return ground_truth

    def eigfunc(self, n, l, r, th):
        beta = 1 / (n + .5)
        abs_l = np.abs(l)
        radial = np.exp(np.log(beta)
                        - gammaln(2 * abs_l + 1)
                        + .5 * (gammaln(n + abs_l + 1) - np.log(2 * n + 1) - gammaln(n - abs_l + 1))
                        + abs_l * np.log(beta * r)
                        - beta * r / 2) * hyp1f1(-n + abs_l, 2 * abs_l + 1, beta * r)
        if l > 0:
            angular = 1 / np.sqrt(np.pi) * np.cos(l * th)
        elif l < 0:
            angular = 1 / np.sqrt(np.pi) * np.sin(l * th)
        else:
            angular = 1 / np.sqrt(2 * np.pi)

        return radial * angular


class Hydrogen3D(ToyProblem):
    def __init__(self, charge=1.):
        super().__init__()
        self.charge = charge  # charge of nucleus

    def get_qnums(self, neigs):
        nmax = int(np.ceil(np.sqrt(neigs)))
        qnums = [(n, l, m) for n in range(0, nmax + 1) for l in range(0, n) for m in range(-l, l + 1)]
        return qnums[:neigs]

    def get_eigvals(self, neigs):
        # E(n;Z) = - Z^2 / (4 * n^2)
        # Quantum numbers: n = 1, 2, ...;
        #                  l = 0, 1, ..., n-1;
        #                  m = -l, -l+1, ... l
        # degeneracy: n^2. Use k^3 as an upper bound to \sum n^2.
        max_n = int(np.ceil(neigs ** (1. / 3))) + 1
        qnums = []
        for n in range(1, max_n):
            for _ in range(n * n):
                qnums.append(n)
        qnums = np.array(qnums)
        ground_truth = - self.charge ** 2 / (4 * qnums[:neigs] ** 2)
        return ground_truth

    def eigfunc(self, n, l, m, r, th, phi):
        a0 = 2 / self.charge
        rho = 2 * r / (n * a0)
        radial = (
                np.sqrt((2 / (n * a0)) ** 3 / (2 * n)) *
                (rho ** l) *
                np.exp(.5 * (- rho + gammaln(n - l) - gammaln(n + l + 1))) *
                genlaguerre(n - l - 1, 2 * l + 1)(rho)
        )
        angular = real_sph_harm(np.array([m, l]), np.stack([phi, th]))
        # if m > 0:
        #     angular = (real_sph_harm(m, l, th, phi) - sph_harm(-m, l, th, phi)) / 2.
        # elif m < 0:
        #     angular = (sph_harm(m, l, th, phi) + sph_harm(-m, l, th, phi)) / (2. * 1j)
        # else:
        #     angular = sph_harm(0, l, th, phi)
        return radial * angular


def cartesian_to_polar(x, y):
    # Calculating radius
    r = np.sqrt(x * x + y * y)
    # Calculating angle (theta) in radian
    th = np.arctan2(y, x)
    return r, th


def cartesian_to_spherical(x, y, z):
    """Converts a cartesian coordinate (x, y, z) into a spherical one (radius, theta, phi)."""
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    th = np.arctan2(np.sqrt(x ** 2 + y ** 2), z)
    phi = np.arctan2(y, x)
    return r, th, phi


def legendre_function(mu, lamb, z):
    # Legendre function of the first kind
    # assume |z-1| < 1
    return 1 / gamma(1 - mu) * ((1 + z) / (1 - z)) ** (mu / 2) * hyp2f1(-lamb, lamb + 1, 1 - mu, (1 - z) / 2)


def sph_harm(ells, ths):
    """
    Parameters
    ----------
    orders = [l_1, l_2, ..., l_{D-1}]
        |l_1| \le l_2 \le ... \le l_{D-1}
    ths: np.array (D-1, n)
        radians

    Returns
    -------
    ys: np.array (n,)

    Notes
    -----
    The negative of l_1 corresponds to the usual order m of spherical harmonics.
    ths_1 corresponds to the azimuthal angle.
    l_{D-1} corresponds to the degree of the harmonic as a homogeneous polynomial.
    """
    assert len(ells) == ths.shape[0]
    assert np.abs(ells[0]) <= ells[1], ells[:2]
    assert np.all(ells[2:] - ells[1:][:-1] >= 0)
    def unit_func(j, m, l, th):
        if j == 2:
            return np.sqrt((2 * l + 1) / 2 * gamma(l + m + 1) / gamma(l - m + 1)) * \
                   lpmv(-m, l, np.cos(th))
        else:
            return np.sqrt((2 * l + j - 1) / 2 * gamma(l + m + j - 1) / gamma(l - m + 1)) * \
                   (np.sin(th) ** ((2 - j) / 2)) * legendre_function(-(m + (j - 2) / 2), l + (j - 2) / 2, np.cos(th))

    d = len(ells) + 1
    n = ths.shape[1]
    temp = np.zeros((d - 1, n), dtype=complex)
    temp[0] = np.exp(1j * ells[0] * ths[0])
    for j in range(1, d - 1):
        temp[j] = unit_func(j + 1, ells[j - 1], ells[j], ths[j])

    return temp.prod(axis=0) / np.sqrt(2 * np.pi)


def real_sph_harm(ells, ths):
    """
    See docstring of sph_harm.
    """
    ells = ells.copy()
    order_positive = ells[0] > 0
    ells[0] = -np.abs(ells[0])
    ys = sph_harm(ells, ths)
    if ells[0] == 0:
        return ys.real
    else:
        sign = 1 if ells[0] % 2 == 0 else -1
        return np.sqrt(2) * sign * (ys.imag if order_positive else ys.real)
