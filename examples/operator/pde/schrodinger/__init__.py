from examples.operator.pde.diff_ops import VectorizedLaplacian


class NegativeHamiltonian:
    def __init__(self,
                 local_potential_ftn,
                 scale_kinetic=1.,
                 laplacian_eps=1e-5,
                 n_particles=1):
        self.laplacian_eps = laplacian_eps
        self.laplacian = VectorizedLaplacian(eps=laplacian_eps)
        self.local_potential_ftn = local_potential_ftn
        self.scale_kinetic = scale_kinetic
        self.n_particles = n_particles

    def __call__(self, f, xs, importance=None, threshold=1e5):
        # threshold is to detect an anomaly in the hamiltonian
        lap, grad, fs = self.laplacian(f, xs, importance)
        kinetic = - self.scale_kinetic * lap
        potential = self.local_potential_ftn(xs.reshape((xs.shape[0], self.n_particles, -1))).view(-1, 1) * fs
        hamiltonian = kinetic + potential
        return - hamiltonian, fs
