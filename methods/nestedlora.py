from typing import Tuple

import numpy as np
import torch
import torch.nn as nn

from methods.utils import off_diagonal, BatchL2NormalizedFunctions


def compute_lambda(f):
    return torch.einsum('bl...,bm...->lm', f, f) / f.shape[0]  # (L, L)


class ScaledFunctions(nn.Module):
    def __init__(self, base_model, init_scales=None):
        super().__init__()
        self.base_model = base_model
        self.scales_param = nn.Parameter(init_scales, requires_grad=True)

    @property
    def scales(self):
        return torch.abs(self.scales_param)

    def forward(self, x, scale=False):
        if scale:
            return self.base_model(x) * self.scales.to(x.device)
        else:
            return self.base_model(x)


class CauchySchwarzResidual(nn.Module):
    @staticmethod
    def forward(f, Tf, f1, Tf1, f2, Tf2):
        # f and Tf must be statistically independent
        # (f1, Tf1) and (f2, Tf2) must be statistically independent
        return ((f ** 2).mean(0).sum(-1) * (Tf ** 2).mean(0).sum(-1) -
                (f1 * Tf1).mean(0).sum(-1) * (f2 * Tf2).mean(0).sum(-1))


def get_joint_nesting_masks(weights: np.ndarray, set_first_mode_const: bool = False):
    vector_mask = list(np.cumsum(list(weights)[::-1])[::-1])
    if set_first_mode_const:
        vector_mask = [vector_mask[0]] + vector_mask
    vector_mask = torch.tensor(np.array(vector_mask)).float()
    matrix_mask = torch.minimum(vector_mask.unsqueeze(1), vector_mask.unsqueeze(1).T).float()
    return vector_mask, matrix_mask


def get_sequential_nesting_masks(L, set_first_mode_const: bool = False):
    if set_first_mode_const:
        L += 1
    vector_mask = torch.ones(L)
    matrix_mask = torch.triu(torch.ones(L, L))
    return vector_mask, matrix_mask


def compute_loss_metric(f, g, matrix_mask):
    lam_f = compute_lambda(f)
    lam_g = compute_lambda(g)
    # compute loss_metric = E_{p(x)p(y)}[(f^T(x) g(y))^2]
    # f: (B1, L)
    # g: (B2, L)
    # lam_f, lam_g: (L, L)
    return (matrix_mask * lam_f * lam_g).sum(), lam_f, lam_g  # O(L ** 2)


class NestedLoRALossFunctionEVD(torch.autograd.Function):
    @staticmethod
    @torch.cuda.amp.custom_fwd
    def forward(
            ctx: torch.autograd.function.FunctionCtx,
            f,
            Tf,
            f1,
            f2,
            vector_mask,
            matrix_mask,
    ):
        """
        the reduction assumed here is `mean` (i.e., we take average over batch)
            f: (B, L) or (B, L, O)
            Tf: (B, L) or (B, L, O)
            f1: (B1, L) or (B1, L, O)
            f2: (B2, L) or (B2, L, O)
        warning: f1 and f2 must be independent
        """
        ctx.vector_mask = vector_mask = vector_mask.to(f.device)
        ctx.matrix_mask = matrix_mask = matrix_mask.to(f.device)
        loss_metric, lam_f1, lam_f2 = compute_loss_metric(f1, f2, matrix_mask)
        ctx.save_for_backward(f, Tf, f1, f2, lam_f1, lam_f2)
        # compute loss_operator = -2 * E_{p(x)}[\sum_{l=1}^L f_l^T(x) (Tf_l)(x)]
        loss_operator = - 2 * torch.einsum('l,bl...,bl...->b', vector_mask, f, Tf).mean()  # O(B1 * L * O)
        loss = loss_operator + loss_metric
        return loss

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(
            ctx: torch.autograd.function.FunctionCtx,
            grad_output: torch.Tensor
    ) -> Tuple[torch.Tensor, ...]:
        """
        Args:
            ctx: The context object to retrieve saved tensors
            grad_output: The gradient of the loss with respect to the output
        """
        f, Tf, f1, f2, lam_f1, lam_f2 = ctx.saved_tensors
        operator_f = - (4 / f.shape[0]) * torch.einsum('l,bl...->bl...', ctx.vector_mask, Tf)
        metric_f1 = (2 / f1.shape[0]) * torch.einsum('lm,lm,bl...->bm...', ctx.matrix_mask, lam_f2, f1)
        metric_f2 = (2 / f2.shape[0]) * torch.einsum('lm,lm,bl...->bm...', ctx.matrix_mask, lam_f1, f2)
        return grad_output * operator_f, None, grad_output * metric_f1, grad_output * metric_f2, None, None, None


class NestedLoRALossFunctionSVD(torch.autograd.Function):
    # TODO: implement the matrix-valued kernel version
    @staticmethod
    @torch.cuda.amp.custom_fwd
    def forward(
            ctx: torch.autograd.function.FunctionCtx,
            f,
            Tg,
            g,
            Tadjf,
            vector_mask,
            matrix_mask,
    ):
        """
        the reduction assumed here is `mean` (i.e., we take average over batch)
            f: (B1, L)
            Tg: (B1, L)
            g: (B2, L)
            Tadjf: (B2, L)
        """
        ctx.vector_mask = vector_mask = vector_mask.to(f.device)
        ctx.matrix_mask = matrix_mask = matrix_mask.to(f.device)
        loss_metric, lam_f, lam_g = compute_loss_metric(f, g, matrix_mask)
        ctx.save_for_backward(f, Tg, g, Tadjf, lam_f, lam_g)
        # O(B^2 * L) version only
        # compute loss_operator = -2 * E_{p(x)}[f^T(x) (Tg)(x)]
        loss_operator = - 2 * torch.einsum('l,bl,bl->b', vector_mask, f, Tg).mean()  # O(B1 * L)
        loss = loss_operator + loss_metric
        return loss

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(
            ctx: torch.autograd.function.FunctionCtx,
            grad_output: torch.Tensor
    ) -> Tuple[torch.Tensor, ...]:
        """
        Args:
            ctx: The context object to retrieve saved tensors
            grad_output: The gradient of the loss with respect to the output
        """
        f, Tg, g, Tadjf, lam_f, lam_g = ctx.saved_tensors
        # for grad(f)
        operator_f = - (2 / f.shape[0]) * torch.einsum('l,bl->bl', ctx.vector_mask, Tg)
        metric_f = (2 / f.shape[0]) * torch.einsum('bi,il,il->bl', f, ctx.matrix_mask, lam_g)
        grad_f = operator_f + metric_f
        # for grad(g)
        operator_g = - (2 / g.shape[0]) * torch.einsum('l,bl->bl', ctx.vector_mask, Tadjf)
        metric_g = (2 / g.shape[0]) * torch.einsum('bi,il,il->bl', g, ctx.matrix_mask, lam_f)
        grad_g = operator_g + metric_g
        return grad_output * grad_f, None, grad_output * grad_g, None, None, None


class NestedLoRA(nn.Module):
    def __init__(
            self,
            model,
            neigs,
            step=1,
            sort=False,
            sequential=False,
            separation=False,
            separation_mode=None,
            separation_init_scale=1.,
            residual_weight=0.,
    ):
        self.name = 'nestedlora'
        super().__init__()
        self.neigs = neigs
        self.sort = sort
        self.eigvals = None
        self.sort_indices = None
        self.sequential = sequential
        if sequential:
            self.vector_mask, self.matrix_mask = get_sequential_nesting_masks(self.neigs)
        else:
            end_indices = list(range(step, neigs + 1, step))
            if neigs not in end_indices:
                end_indices.append(neigs)
            step_weights = np.zeros(neigs)
            step_weights[np.array(end_indices) - 1] = 1.
            step_weights = torch.tensor(step_weights / sum(step_weights))
            self.vector_mask, self.matrix_mask = get_joint_nesting_masks(step_weights)
        if separation:
            init_scales = separation_init_scale * torch.linspace(1. / neigs, 1, neigs).flip(0)
            if separation_mode == 'bn':
                base_model = BatchL2NormalizedFunctions(model, neigs)
            elif separation_mode == 'id':
                base_model = model
            else:
                raise ValueError
            self.model = ScaledFunctions(base_model, init_scales=init_scales)
            raise NotImplementedError
        else:
            self.model = model
        self.residual_weight = residual_weight

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

    def _compute_loss(
            self,
            *args,
            evd=True,
    ) -> torch.Tensor:
        if evd:
            return NestedLoRALossFunctionEVD.apply(
                *args,
                self.vector_mask,
                self.matrix_mask,
            )
        else:
            return NestedLoRALossFunctionSVD.apply(
                *args,
                self.vector_mask,
                self.matrix_mask,
            )

    def compute_loss_kernel(
            self,
            get_approx_kernel_op,
            x,
            importance,
            split_batch: bool,
            operator_inverse: bool,
            evd: bool = True,
    ):
        if evd:
            if split_batch:
                x1, x2 = torch.chunk(x, 2)
                if not operator_inverse:
                    Kf1, f1 = get_approx_kernel_op(x2)(self, x1, importance=importance)
                    f2 = self(x2)
                    loss = self._compute_loss(f1, Kf1, f1, f2, evd=True)
                    if self.residual_weight > 0.:
                        Kf2, f2 = get_approx_kernel_op(x1)(self, x2, importance=importance)
                        loss += self.residual_weight * CauchySchwarzResidual()(f1, Kf2, f1, Kf1, f2, Kf2)
                        # TODO: needs to be double checked if (f1, Kf1) and (f2, Kf2) are independent
                    f = f1
                    Kf = Kf1
                else:
                    Kf1, f1 = get_approx_kernel_op(x2)(self, x1, importance=importance)
                    Kf2, f2 = get_approx_kernel_op(x2)(self, x2, importance=importance)
                    loss = self._compute_loss(Kf1, f1, Kf1, Kf2, evd=True)
                    if self.residual_weight > 0.:
                        loss += self.residual_weight * CauchySchwarzResidual()(f1, Kf2, f1, Kf1, f2, Kf2)
                        # TODO: needs to be double checked if (f1, Kf1) and (f2, Kf2) are independent
                    f = torch.cat([f1, f2])
                    Kf = torch.cat([Kf1, Kf2])
            else:
                Kf, f = get_approx_kernel_op(x)(self, x, importance=importance)
                if not operator_inverse:
                    f1, f2 = torch.chunk(f, 2)
                    loss = self._compute_loss(f, Kf, f1, f2, evd=True)
                else:
                    Kf1, Kf2 = torch.chunk(Kf, 2)
                    loss = self._compute_loss(Kf, f, Kf1, Kf2, evd=True)
                if self.residual_weight > 0.:
                    f1, f2 = torch.chunk(f, 2)
                    Kf1, Kf2 = torch.chunk(Kf, 2)
                    loss += self.residual_weight * CauchySchwarzResidual()(f1, Kf2, f1, Kf1, f2, Kf2)
            return loss, dict(f=f, Tf=Kf, eigvals=None)
        else:
            raise NotImplementedError

    def compute_loss_operator(
            self,
            operator,
            x,
            importance=None,
            operator_inverse: bool = False,
            evd: bool = True,
    ):
        if evd:
            Tf, f = operator(self, x, importance=importance)
            if not operator_inverse:
                f1, f2 = torch.chunk(f, 2)
                loss = self._compute_loss(f, Tf, f1, f2, evd=True)
            else:
                Tf1, Tf2 = torch.chunk(Tf, 2)
                loss = self._compute_loss(Tf, f, Tf1, Tf2, evd=True)
            if self.residual_weight > 0.:
                f1, f2 = torch.chunk(f, 2)
                Tf1, Tf2 = torch.chunk(Tf, 2)
                loss += self.residual_weight * CauchySchwarzResidual()(f1, Tf2, f1, Tf1, f2, Tf2)
            return loss, dict(f=f, Tf=Tf, eigvals=None)
        else:
            raise NotImplementedError


class NestedLoRALossFunctionForCDK(torch.autograd.Function):
    @staticmethod
    @torch.cuda.amp.custom_fwd
    def forward(
            ctx: torch.autograd.function.FunctionCtx,
            f,
            g,
            vector_mask,
            matrix_mask,
            set_first_mode_const=True,
            batch_weights=None,
    ):
        """
        the reduction assumed here is `mean` (i.e., we take average over batch)
            f: (B, L)
            g: (B, L)
        """
        if set_first_mode_const:
            pad = nn.ConstantPad1d((1, 0), 1)
            f = pad(f)
            g = pad(g)
        if batch_weights is not None:
            f *= batch_weights
            g *= batch_weights
        ctx.vector_mask = vector_mask = vector_mask.to(f.device)
        ctx.matrix_mask = matrix_mask = matrix_mask.to(f.device)
        loss_metric, lam_f, lam_g = compute_loss_metric(f, g, matrix_mask)
        ctx.save_for_backward(f, g, lam_f, lam_g)
        ctx.set_first_mode_const = set_first_mode_const
        # along feature dim only
        # compute loss_operator = -2 * E_{p(x,y)}[f^T(x) g(y)]
        loss_operator = - 2 * torch.einsum('l,bl,bl->b', vector_mask, f, g).mean()  # O(B1 * L)
        loss = loss_operator + loss_metric
        gram_matrix = f @ g.T  # (B, B); each entry is (f^T(x_i) g(y_j))
        rs_joint = gram_matrix.diag()
        rs_indep = off_diagonal(gram_matrix)
        return loss, loss_operator, loss_metric, rs_joint, rs_indep

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(
            ctx: torch.autograd.function.FunctionCtx,
            grad_output: torch.Tensor,
            *args
    ) -> Tuple[torch.Tensor, ...]:
        """
        Args:
            ctx: The context object to retrieve saved tensors
            grad_output: The gradient of the loss with respect to the output
        """
        f, g, lam_f, lam_g = ctx.saved_tensors
        # for grad(f)
        operator_f = - (2 / f.shape[0]) * torch.einsum('l,bl->bl', ctx.vector_mask, g)
        metric_f = (2 / f.shape[0]) * torch.einsum('il,il,bi->bl', ctx.matrix_mask, lam_g, f)
        grad_f = operator_f + metric_f
        # for grad(g)
        operator_g = - (2 / g.shape[0]) * torch.einsum('l,bl->bl', ctx.vector_mask, f)
        metric_g = (2 / g.shape[0]) * torch.einsum('il,il,bi->bl', ctx.matrix_mask, lam_f, g)
        grad_g = operator_g + metric_g
        if ctx.set_first_mode_const:
            grad_f = grad_f[:, 1:]
            grad_g = grad_g[:, 1:]
        return grad_output * grad_f, grad_output * grad_g, None, None, None, None


class NestedLoRAForCDK(nn.Module):
    def __init__(
            self,
            model,
            neigs,
            step=1,
            sequential=False,
            set_first_mode_const=True,
    ):
        self.name = 'nestedlora'
        super().__init__()
        self.neigs = neigs
        self.sequential = sequential
        if sequential:
            self.vector_mask, self.matrix_mask = get_sequential_nesting_masks(self.neigs, set_first_mode_const)
        else:
            end_indices = list(range(step, neigs + 1, step))
            if neigs not in end_indices:
                end_indices.append(neigs)
            step_weights = np.zeros(neigs)
            step_weights[np.array(end_indices) - 1] = 1.
            step_weights = torch.tensor(step_weights / sum(step_weights))
            self.vector_mask, self.matrix_mask = get_joint_nesting_masks(step_weights, set_first_mode_const)
        self.set_first_mode_const = set_first_mode_const

        self.model = model

    def forward(self, *args):
        return self.model(*args)

    def compute_loss(
            self,
            f,
            g,
            batch_weights=None,
    ) -> torch.Tensor:
        return NestedLoRALossFunctionForCDK.apply(
            f,
            g,
            self.vector_mask,
            self.matrix_mask,
            self.set_first_mode_const,
            batch_weights,
        )
