import torch
import torch.nn as nn

from methods.spin import Covariance, spin_step, jac_model_params, moving_average


class SpINxLossFunction(nn.Module):
    def __init__(self, neigs):
        super().__init__()
        self.neigs = neigs
        self.trace_weights = nn.Parameter(torch.ones(neigs), requires_grad=False)  # C * torch.arange(neigs + 1, 0, -1)

    def forward(self, phi, Tphi, phi1):
        sigma = (phi1.T @ phi1) / phi1.shape[0]  # (L, L); sigma
        pi = (phi.T @ Tphi) / phi.shape[0]  # (L, L); pi
        chol, chol_inv, Lambda, eigvals = spin_step(sigma, pi)

        # compute losses
        loss_trace = (self.trace_weights * eigvals).sum()
        residuals = (Tphi @ chol_inv.T) - (phi @ chol_inv.T) @ torch.diag(eigvals)
        loss_residuals = (residuals ** 2).mean(axis=0)  # (L, )
        losses = torch.hstack([loss_trace, loss_residuals])  # (L + 1, )
        return losses, sigma


class SpINxLossFunctionKernel(nn.Module):
    def __init__(self, model, neigs, get_approx_kernel_op, importance, split_batch):
        super().__init__()
        self.model = model
        self.spinx_loss_ftn = SpINxLossFunction(neigs)
        self.get_approx_kernel_op = get_approx_kernel_op
        self.importance = importance
        self.split_batch = split_batch

    def forward(
            self,
            x,
            *args,
            **kwargs,
    ):
        model = self.model
        if self.split_batch:
            x1, x2 = torch.chunk(x, 2)
            Kphi1, phi1 = self.get_approx_kernel_op(x2)(model, x1, importance=self.importance)
            phi2 = model(x2)
            phi = torch.cat([phi1, phi2])
            losses = self.spinx_loss_ftn(phi1, Kphi1, phi)[0]
        else:
            Kphi, phi = self.get_approx_kernel_op(x)(model, x, importance=self.importance)
            losses = self.spinx_loss_ftn(phi, Kphi, phi)[0]
        return losses


class SpINxLossFunctionOperator(nn.Module):
    def __init__(self, model, neigs, operator, importance, split_batch):
        super().__init__()
        self.model = model
        self.spinx_loss_ftn = SpINxLossFunction(neigs)
        self.operator = operator
        self.importance = importance
        self.split_batch = split_batch

    def forward(
            self,
            x,
            *args,
            **kwargs,
    ):
        model = self.model
        Tphi, phi = self.operator(model, x, importance=self.importance)  # (B, L)
        losses = self.spinx_loss_ftn(phi, Tphi)[0]
        return losses


class SpINx(nn.Module):
    def __init__(self, model, neigs, decay):
        self.name = 'spinx'
        super().__init__()
        self.model = model
        self.neigs = neigs
        self.decay = decay

        self.sigma_avg = nn.Parameter(torch.zeros(neigs, neigs), requires_grad=False)
        self.chol = nn.Parameter(torch.zeros(neigs, neigs), requires_grad=False)

        self.spinx_loss_ftn = SpINxLossFunction(neigs)
        self.weights = torch.ones(neigs + 1)

    def _moving_average(self, xprev, xnew):
        return moving_average(xprev, xnew.detach(), self.decay)

    def _compute_loss(self, phi, Tphi, phi1, weights):
        losses, sigma = self.spinx_loss_ftn(phi, Tphi, phi1)
        self.sigma_avg = self._moving_average(
            self.sigma_avg,
            sigma,
        )
        self.chol = torch.linalg.cholesky(self.sigma_avg + 1e-3 * torch.eye(self.sigma_avg.shape[0]).to(sigma))
        return (losses * weights / self.neigs).sum()

    def compute_losses_operator(
            self,
            operator,
            x,
            importance,
            *args,
            **kwargs,
    ):
        model = self.model
        Tphi, phi = operator(model, x, importance=importance)  # (B, L)
        losses = self.spinx_loss_ftn(phi, Tphi)[0]
        return losses

    def _update_weights(self, jac_losses):
        ntk_weights = torch.zeros(self.neigs + 1)
        for jac in jac_losses.values():
            ntk_weights += (jac ** 2).view(self.neigs + 1, -1).sum(dim=-1)
        self.weights = torch.sqrt(ntk_weights.sum() / ntk_weights).detach()

    def update_weights_kernel(
            self,
            get_approx_kernel_op,
            x,
            importance,
            split_batch
    ):
        jac_losses = jac_model_params(
            SpINxLossFunctionKernel(self.model, self.neigs, get_approx_kernel_op, importance, split_batch),
            x,
            use_vmap=False,
        )  # {param_name: L + 1, P...)}
        self._update_weights(jac_losses)

    def update_weights_operator(
            self,
            operator,
            x,
            importance,
            split_batch
    ):
        jac_losses = jac_model_params(
            SpINxLossFunctionOperator(self.model, self.neigs, operator, importance, split_batch),
            x,
            use_vmap=False,
        )  # {param_name: (L + 1, P...)}
        self._update_weights(jac_losses)

    def compute_loss_kernel(
            self,
            get_approx_kernel_op,
            x,
            importance,
            split_batch: bool,
            *args,
            **kwargs,
    ):
        model = self.model
        if split_batch:
            x1, x2 = torch.chunk(x, 2)
            Kphi1, phi1 = get_approx_kernel_op(x2)(model, x1, importance=importance)
            phi2 = model(x2)
            phi = torch.cat([phi1, phi2])
            loss = self._compute_loss(phi1, Kphi1, phi, self.weights)
            phi, Kphi = phi1, Kphi1
        else:
            Kphi, phi = get_approx_kernel_op(x)(model, x, importance=importance)
            loss = self._compute_loss(phi, Kphi, phi, self.weights)
        self.phi, self.Tphi = phi, Kphi
        return loss, dict(f=phi, Tf=Kphi, eigvals=None)

    def compute_loss_operator(
            self,
            operator,
            x,
            importance,
            *args,
            **kwargs,
    ):
        model = self.model
        Tphi, phi = operator(model, x, importance=importance)  # (B, L)
        loss = self._compute_loss(phi, Tphi, phi, self.weights)
        self.phi, self.Tphi = phi, Tphi
        return loss, dict(f=phi, Tf=Tphi, eigvals=None)

    def forward(self, x):
        # a wrapper to output orthonormalized eigenfunction
        return torch.triangular_solve(
            self.model(x).T,
            self.chol,
            upper=False,
        )[0].T
