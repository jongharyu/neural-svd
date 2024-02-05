import torch
import torch.nn as nn
from torch.func import functional_call, jacrev, vmap


def moving_average(xprev, xnew, decay):
    if xprev is None:
        if len(xnew.shape) == 2 and xnew.shape[0] == xnew.shape[1]:
            xprev = torch.eye(xnew.shape[0])
        else:
            xprev = torch.zeros_like(xnew)
    return (1 - decay) * xprev.to(xnew.device) + decay * xnew


def jac_model_params(model, x, use_vmap):
    if use_vmap:
        return {key: value.squeeze(1) for (key, value) in vmap(
            jacrev(functional_call, argnums=1),
            in_dims=(None, None, 0)
        )(
            model,
            dict(model.named_parameters()),
            x.unsqueeze(1)
        ).items()}  # {param_name: (B, L, P...)}
    else:
        return jacrev(functional_call, argnums=1)(
            model,
            dict(model.named_parameters()),
            x
        )  # {param_name: (B, L, P...)}


def spin_step(sigma, pi):
    chol = torch.linalg.cholesky(sigma + 1e-3 * torch.eye(sigma.shape[0]).to(sigma))
    chol_inv = torch.linalg.inv(chol)  # "choli"
    Lambda = chol_inv @ pi @ chol_inv.T  # "rq"
    eigvals = torch.diagonal(Lambda)
    return chol, chol_inv, Lambda, eigvals


class SpINFunction(torch.autograd.Function):
    @staticmethod
    @torch.cuda.amp.custom_fwd
    def forward(
            ctx,
            sigma_avg,
            pi,
    ):
        chol, chol_inv, Lambda, eigvals = spin_step(sigma_avg, pi)
        loss = torch.trace(Lambda)

        # compute components for gradient computation
        diag_chol_inv = torch.diag(torch.diagonal(chol_inv))  # "dl"
        triu = torch.triu(Lambda @ diag_chol_inv)
        gsigma = chol_inv.T @ triu  # "gxx"; compute the second term in the masked gradient
        gpi = - chol_inv.T @ diag_chol_inv  # "gobj"

        ctx.save_for_backward(gsigma, gpi)
        return loss, eigvals, chol, gsigma, gpi

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(
            ctx,
            *grad_outputs
    ):
        """
        Args:
            ctx: The context object to retrieve saved tensors
            grad_outputs: The gradients of the loss with respect to the outputs
        """
        gsigma, gpi = ctx.saved_tensors
        return gsigma, gpi


class Covariance(torch.autograd.Function):
    @staticmethod
    @torch.cuda.amp.custom_fwd
    def forward(
            ctx,
            x,
            y,
    ):
        ctx.save_for_backward(x, y)
        return x.T @ y / x.shape[0]

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(
            ctx,
            *grad_outputs
    ):
        """
        This custom backward for covariance is crucial for the correct behavior for SpIN
        Args:
            ctx: The context object to retrieve saved tensors
            grad_outputs: The gradients of the loss with respect to the outputs
        """
        x, y = ctx.saved_tensors
        return y @ grad_outputs[0] / y.shape[0], x @ grad_outputs[0] / x.shape[0]


class SpIN(nn.Module):
    def __init__(self, model, neigs, decay, use_vmap=True):
        """
        :param decay:
            0.0 = the moving average is constant (less update)
            1.0 = the moving average has no memory (more update)
        """
        self.name = 'spin'
        super().__init__()
        self.model = model
        self.neigs = neigs
        self.decay = decay
        self.use_vmap = use_vmap

        self.sigma_avg = nn.Parameter(torch.zeros(neigs, neigs), requires_grad=False)
        self.chol = nn.Parameter(torch.zeros(neigs, neigs), requires_grad=False)
        self.j_avg = nn.ParameterDict({
            name.replace('.', '_'): nn.Parameter(torch.zeros(neigs, neigs, *p.shape), requires_grad=False)
            for (name, p) in model.named_parameters()
        })

    def _moving_average(self, xprev, xnew):
        return moving_average(xprev, xnew.detach(), self.decay)

    def jac_model_params(self, x):
        return jac_model_params(self.model, x, self.use_vmap)

    def _compute_loss(
            self,
            phi_sigma,  # for sigma
            Tphi,
            jac_phi,
            phi,
    ):
        # warning: Tphi, jac_phi, phi must be computed using the same batch
        # note: phi can be same as phi_sigma
        sigma = (phi_sigma.T @ phi_sigma) / phi_sigma.shape[0]  # (L, L); sigma
        pi = Covariance.apply(phi, Tphi)  # (L, L); pi
        # pi = (phi.T @ Tphi) / phi.shape[0]  # (L, L); this does not work
        # compute EWMA(sigma)
        self.sigma_avg.data = self._moving_average(
            self.sigma_avg.data,
            sigma,
        )
        # compute the loss, which comes with cholesky(EWMA(sigma)) and gsigma
        loss, eigvals, self.chol.data, gsigma, gpi = SpINFunction.apply(self.sigma_avg.data, pi)
        """
        note: here,
            gsigma = "sigma_back" or "A" in the writeup,
            self.j_avg = "sigma_jac_avg"
        """
        # compute the second masked gradient term
        for i, (name, p) in enumerate(self.model.named_parameters()):
            if p.requires_grad:
                j_new = 2 * torch.einsum(
                    'bl...,bm->ml...',
                    jac_phi[name],  # jacobian of phi wrt this parameter
                    phi,
                ) / phi.shape[0]
                name_ = name.replace('.', '_')
                self.j_avg[name_].data = self._moving_average(
                    self.j_avg[name_].data,
                    j_new,  # (L, L, P)
                )  # "s"
                p.grad = torch.einsum('lm,lm...->...', gsigma, self.j_avg[name_].data)
        # warning: to compute and accumulate the first masked gradient term, loss.backward() must be called outside
        return loss, eigvals

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
            jac_phi1 = self.jac_model_params(x1)
            phi2 = model(x2)
            phi = torch.cat([phi1, phi2])
            loss, eigvals = self._compute_loss(phi, Kphi1, jac_phi1, phi1)
            phi, Kphi = phi1, Kphi1
        else:
            Kphi, phi = get_approx_kernel_op(x)(model, x, importance=importance)
            jac_phi = self.jac_model_params(x)
            loss, eigvals = self._compute_loss(phi, Kphi, jac_phi, phi)
        return loss, dict(f=phi, Tf=Kphi, eigvals=eigvals)

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
        jac_phi = self.jac_model_params(x)
        loss, eigvals = self._compute_loss(phi, Tphi, jac_phi, phi)
        return loss, dict(f=phi, Tf=Tphi, eigvals=eigvals)

    def forward(self, x):
        # a wrapper to output orthonormalized eigenfunction
        return torch.triangular_solve(
            self.model(x).T,
            self.chol,
            upper=False,
        )[0].T
