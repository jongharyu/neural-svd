import torch
import torch.nn as nn

from methods.utils import BatchL2NormalizedFunctions


def compute_gram(f, Tf=None):
    if Tf is None:
        Tf = f
    return torch.einsum('bl...,bm...->lm', f, Tf) / f.shape[0]  # (L, L); O(B * L ** 2 * O)


class NeuralEigenfunctionsLossFunction(torch.autograd.Function):
    @staticmethod
    @torch.cuda.amp.custom_fwd
    def forward(
            ctx,
            phi,
            Tphi,
            phi1,
            Tphi1,
            phi2,
            Tphi2,
            unbiased,
            diagonal,
    ):
        """
        note: ideally, (phi1, Tphi1) and (phi2, Tphi2) must be independent
              however, the original NeuralEF uses phi1=phi2=phi, Tphi1=Tphi2=Tphi
            phi: (B, L) or (B, L, O)
            Tphi: (B, L) or (B, L, O)
            phi1: (B1, L) or (B1, L, O)
            Tphi1: (B1, L) or (B1, L, O)
            phi2: (B2, L) or (B2, L, O)
            Tphi2: (B2, L) or (B2, L, O)
        """
        variance_term = - Tphi / phi.shape[0]  # (B, L) or (B, L, O)
        if unbiased:
            # the mu-EigenGame version of NeuralEF
            gram_phi1 = compute_gram(phi1)  # (L, L); (i, j)-th entry = (phi_i.T @ phi_j)
            gram_phi2 = compute_gram(phi2)  # (L, L); (i, j)-th entry = (phi_i.T @ phi_j)
            coeff_phi1 = gram_phi1.triu(diagonal=diagonal)
            coeff_phi2 = gram_phi2.triu(diagonal=diagonal)
        else:
            # the original NeuralEF
            quad_phi1 = compute_gram(phi1, Tphi1)  # (L, L); (i, j)-th entry = (phi_i.T @ K @ phi_j)
            quad_phi2 = compute_gram(phi2, Tphi2)  # (L, L); (i, j)-th entry = (phi_i.T @ K @ phi_j)
            coeff_phi1 = (quad_phi2.triu(diagonal=diagonal) / (quad_phi2.diag() + 1e-5).view(-1, 1))  # (i,l) = <l|K|i> / <i|K|i> if i < l, 0 otherwise
            coeff_phi2 = (quad_phi1.triu(diagonal=diagonal) / (quad_phi1.diag() + 1e-5).view(-1, 1))  # (i,l) = <l|K|i> / <i|K|i> if i < l, 0 otherwise
        align_term_phi1 = torch.einsum('bl...,lm->bm...', Tphi1, coeff_phi1) / phi1.shape[0]
        align_term_phi2 = torch.einsum('bl...,lm->bm...', Tphi2, coeff_phi2) / phi2.shape[0]
        ctx.save_for_backward(variance_term, align_term_phi1, align_term_phi2)
        return (phi * variance_term).sum() + .5 * ((phi1 * align_term_phi1).sum() + (phi2 * align_term_phi2).sum())

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(
            ctx,
            *grad_outputs
    ):
        variance_term, align_term_phi1, align_term_phi2 = ctx.saved_tensors
        return 4 * variance_term, None, 2 * align_term_phi1, None, 2 * align_term_phi2, None, None, None


class NeuralEigenfunctions(nn.Module):
    # covers both explicit, PD kernel and self-adjoint linear operator
    def __init__(
        self,
        model,
        neigs,
        batchnorm_mode,
        sort=False,
        unbiased=False,
        include_diag=False,
    ):
        self.name = 'neuralef'
        super().__init__()
        if batchnorm_mode != 'none':
            self.model = BatchL2NormalizedFunctions(model, neigs, batchnorm_mode=batchnorm_mode)
        else:
            self.model = model
        self.unbiased = unbiased  # if True, becomes mu-EigenGame
        self.diagonal = 0 if include_diag else 1  # (when unbiased is True) if True, becomes GHA (Sanger)
        self.sort = sort
        self.eigvals = None
        self.sort_indices = None

    def forward(self, *args):
        output = self.model(*args)
        if self.sort_indices is not None and self.training:
            return output[:, self.sort_indices, ...]
        else:
            return output

    def register_eigvals(self, eigvals):
        print("NOTE: eigenvalues have been registered!")
        self.eigvals = torch.Tensor(eigvals)  # (L, )
        self.sort_indices = torch.sort(self.eigvals)[1].flip(0)

    def reset_eigvals(self):
        print("NOTE: eigenvalues have been reset!")
        self.eigvals = None
        self.sort_indices = None

    def _compute_loss(self, phi, Tphi, phi1, Tphi1, phi2, Tphi2):
        return NeuralEigenfunctionsLossFunction.apply(
            phi,
            Tphi,
            phi1,
            Tphi1,
            phi2,
            Tphi2,
            self.unbiased,
            self.diagonal,
        )

    def compute_loss_kernel(
            self,
            get_approx_kernel_op,
            x,
            importance,
            split_batch: bool,
            *args,
            **kwargs,  # to subsume operator_inverse
    ):
        model = self.model
        if split_batch:
            x1, x2 = torch.chunk(x, 2)
            Kphi1, phi1 = get_approx_kernel_op(x2)(model, x1, importance=importance)
            Kphi2, phi2 = get_approx_kernel_op(x1)(model, x2, importance=importance)
            phi = torch.cat([phi1, phi2])
            Kphi = torch.cat([Kphi1, Kphi2])
            loss = self._compute_loss(phi, Kphi, phi1, Kphi1, phi2, Kphi2)
        else:
            Kphi, phi = get_approx_kernel_op(x)(model, x, importance=importance)
            loss = self._compute_loss(phi, Kphi, phi, Kphi, phi, Kphi)
        return loss, dict(f=phi, Tf=Kphi, eigvals=None)

    def compute_loss_operator(
            self,
            operator,
            x,
            importance=None,
            *args,
            **kwargs,  # to subsume operator_inverse
    ):
        model = self.model
        Tphi, phi = operator(model, x, importance=importance)  # (B, L)
        phi1, phi2 = torch.chunk(phi, 2)
        Tphi1, Tphi2 = torch.chunk(Tphi, 2)
        loss = self._compute_loss(phi, Tphi, phi1, Tphi1, phi2, Tphi2)
        return loss, dict(f=phi, Tf=Tphi, eigvals=None)
