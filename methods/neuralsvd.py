import numpy as np
import torch
import torch.nn as nn

from methods.nestedlora import get_sequential_nesting_masks, get_joint_nesting_masks, compute_loss_metric, ScaledFunctions
from methods.utils import off_diagonal, extract_tensor, BatchL2NormalizedFunctions


class NeuralSVDLoss(nn.Module):
    def __init__(
            self,
            model,
            step=1,
            neigs=0,
            separation=False,
            separation_mode=None,
            separation_init_scale=1.,
            stop_grad=False,
            along_batch_dim=True,
            weight_order=1.,
            sequential=False,
            oneshot=False,
            unittest=False,
    ):
        super().__init__()
        self.name = 'neuralsvd'
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
        # parameters for the nested objective
        self.stop_grad = stop_grad
        self.along_batch_dim = along_batch_dim
        self.neigs = neigs
        self.oneshot = oneshot
        self.sequential = sequential
        if sequential:
            assert oneshot
            self.vector_mask, self.matrix_mask = get_sequential_nesting_masks(self.neigs)
        else:
            if step == neigs:
                self.vector_mask = torch.ones(neigs)
                self.matrix_mask = torch.ones(neigs, neigs)
            else:
                # TODO: fix for step > 1
                assert step == 1, "currently, the code will not work for step > 1"
                self.end_indices = list(range(step, neigs + 1, step))
                if neigs not in self.end_indices:
                    self.end_indices.append(neigs)
                step_weights = [weight_order ** (n - neigs // step + 1)
                                for n in np.arange(neigs // step + (neigs % step != 0))]
                self.step_weights = torch.tensor(step_weights / sum(step_weights))
                self.vector_mask, self.matrix_mask = get_joint_nesting_masks(self.step_weights)
        if oneshot:
            assert not stop_grad
            assert not along_batch_dim

        self.unittest = unittest  # if True, check if oneshot nested loss is equivalent to stepwise nested loss

    def forward(self, *args):
        return self.model(*args)

    # def __call__(self, f, g, *args):
    #     # args can be either Kg (linear operator) or emp_kernel (integral kernel operator)
    #     if self.unittest:
    #         return self.test_oneshot(f, g, *args)
    #
    #     # compute a nested objective
    #     if self.oneshot:
    #         return self.nested_schmidt_norm_oneshot(f, g, *args)
    #     else:
    #         return self.nested_schmidt_norm_stepwise(f, g, *args)

    def test_oneshot(self, *args):
        if not self.along_batch_dim:
            print("DEBUGGING: we are checking if oneshot computation is correct...")
            nested_loss_oneshot = self.nested_schmidt_norm_oneshot(*args)
            nested_loss_stepwise = self.nested_schmidt_norm_stepwise(*args)
            assert torch.isclose(nested_loss_oneshot, nested_loss_stepwise), \
                (nested_loss_oneshot.item(), nested_loss_stepwise.item())
            return nested_loss_oneshot

    def compute_loss_operator(
            self,
            operator,
            x,
            importance,
            operator_inverse: bool,
            evd: bool = True,
    ):
        model = self.model
        if evd:
            Tf, f = operator(model, x, importance=importance)
            if not operator_inverse:
                f1, f2 = torch.chunk(f, 2)
                Tf1, Tf2 = torch.chunk(Tf, 2)
                grad_weights = self.nested_grad_weights_evd(f1, f2, Tf1)
                f1.backward(grad_weights)
                loss = self.nested_schmidt_norm_oneshot(f1.detach(), f2.detach(), Tf1.detach())
            else:
                raise NotImplementedError
            return loss, dict(f=f, Tf=Tf, eigvals=None)
        else:
            raise NotImplementedError

    def _compute_loss(self, f, Tf, f1, f2):
        Tf1, Tf2 = torch.chunk(Tf, 2)
        return self.nested_schmidt_norm_oneshot(f1, f2, Tf1)

    def nested_grad_weights_evd(
            self,
            f,
            fp,
            Kf,
    ):
        """
        usage:
            grad_weights = self.nested_grad_weights_evd(f, fp, Kf)
            f.backward(grad_weights)
        the reduction assumed here is `mean` (i.e., we take average over batch)
        f: (B1, L)
        fp: (B2, L)
        Kf: (B1, L)
        """
        self.vector_mask = self.vector_mask.to(f.device)
        self.matrix_mask = self.matrix_mask.to(f.device)

        # for grad(f)
        operator_f = - torch.einsum('l,bl->bl', self.vector_mask, Kf)
        metric_f = torch.einsum('bi,il,il->bl', f, self.matrix_mask, (fp.T @ fp) / fp.shape[0])
        grad_weights_f = operator_f + metric_f
        return 4 * grad_weights_f / f.shape[0]

    def nested_grad_weights_svd(
            self,
            f,
            Kg,
            g,
            Kadjf,
    ):
        """
        usage:
            grad_weights_f, grad_weights_g = self.nested_grad_weights_svd(f, Kg, g, Kadjf)
            f.backward(grad_weights_f)
            g.backward(grad_weights_g)
        the reduction assumed here is `mean` (i.e., we take average over batch)
        f: (B1, L)
        Kg: (B1, L)
        g: (B2, L)
        Kadjf: (B2, L)
        """
        self.vector_mask = self.vector_mask.to(f.device)
        self.matrix_mask = self.matrix_mask.to(f.device)

        # for grad(f)
        operator_f = - torch.einsum('l,bl->bl', self.vector_mask, Kg)
        metric_f = torch.einsum('bi,il,il->bl', f, self.matrix_mask, (g.T @ g) / g.shape[0])
        grad_weights_f = operator_f + metric_f
        # for grad(g)
        operator_g = - torch.einsum('l,bl->bl', self.vector_mask, Kadjf)
        metric_g = torch.einsum('bi,il,il->bl', g, self.matrix_mask, (f.T @ f) / f.shape[0])
        grad_weights_g = operator_g + metric_g
        return 2 * grad_weights_f / f.shape[0], 2 * grad_weights_g / g.shape[0]

    def nested_grad_weights_with_scales(self, phi, psi, Kpsi, scales, evd: bool):
        # TODO: (now deprecated) needs to be revised
        self.vector_mask = self.vector_mask.to(phi.device)
        self.matrix_mask = self.matrix_mask.to(phi.device)

        # for grad(phi)
        operator_phi = - torch.einsum('l,l,bl->bl',
                                      self.vector_mask, scales ** 2, Kpsi)
        metric_phi = torch.einsum('li,bi,i,l,il->bl',
                                  self.matrix_mask, phi, scales ** 2, scales ** 2, (psi.T @ psi) / psi.shape[0])
        grad_weights_phi = (operator_phi + metric_phi) / phi.shape[0]
        grad_weights_scales = torch.einsum('l,bl,bl->l', 1 / scales, phi, grad_weights_phi)
        if evd:
            return 4 * grad_weights_phi, 4 * grad_weights_scales
        else:
            # TODO: implement SVD case
            raise NotImplementedError
        pass

    def nested_schmidt_norm_oneshot(
            self,
            f,
            g,
            Kg
    ):
        # the reduction assumed here is `mean` (i.e., we take average over batch)
        # f: (B1, L)
        # g: (B2, L)
        # Kg: (B1, L)
        self.vector_mask = self.vector_mask.to(f.device)
        self.matrix_mask = self.matrix_mask.to(f.device)

        # O(B^2 * L) version only
        # compute loss_operator = -2 * E_{p(x)}[f^T(x) (Kg)(x)]
        loss_operator = - 2 * torch.einsum('l,bl,bl->b', self.vector_mask, f, Kg).mean()  # O(B1 * L)
        loss_metric = compute_loss_metric(f, g, self.matrix_mask)
        loss = loss_operator + loss_metric

        return loss

    def nested_schmidt_norm_stepwise(
            self,
            f,
            g,
            Kg
    ):
        loss_nested = 0.
        prev_last_dim = 0
        for i in self.end_indices:
            if self.stop_grad:
                partial_f = torch.cat([f[:, :prev_last_dim].detach(),
                                         f[:, prev_last_dim:i]], dim=-1)
                partial_g = torch.cat([g[:, :prev_last_dim].detach(),
                                         g[:, prev_last_dim:i]], dim=-1)
                partial_Kg = torch.cat([Kg[:, :prev_last_dim].detach(),
                                          Kg[:, prev_last_dim:i]], dim=-1)
            else:
                partial_f = f[:, :i]
                partial_g = g[:, :i]
                partial_Kg = Kg[:, :i]

            loss_, *_ = self.schmidt_norm(partial_f, partial_g, partial_Kg)
            loss_nested += self.step_weights[min(i, self.neigs) - 1] * loss_
            prev_last_dim = i

        return loss_nested

    def schmidt_norm(
            self,
            f,
            g,
            Kg
    ):
        # the reduction assumed here is `sum` (i.e., we take summation over batch)
        # f: (B1, L)
        # g: (B2, L)
        # Kg: (B1, L)
        quadforms = f * Kg  # (B1, L)
        sqnorms = f ** 2  # (B1, L)
        loss_operator = - 2 * quadforms.sum(dim=-1).mean(dim=0)
        if self.along_batch_dim:
            # unbiased version
            gram = f @ g.T  # (B1, B2); each entry is (f^T(x_i) g(y_j))
            loss_metric = (gram ** 2).mean()
        else:
            loss_metric = compute_loss_metric(f, g)
        loss = loss_operator + loss_metric
        return loss, quadforms, sqnorms


class NeuralSVDLossFixedKernel(NeuralSVDLoss):
    def nested_schmidt_norm_oneshot(self, f, g, emp_kernel):
        return super().nested_schmidt_norm_oneshot(f, g, (emp_kernel @ g) / g.shape[0])

    def nested_schmidt_norm_stepwise(
            self,
            f,
            g,
            emp_kernel
    ):
        loss_nested = 0.
        prev_last_dim = 0
        for i in self.end_indices:
            if self.stop_grad:
                partial_f = torch.cat([f[:, :prev_last_dim].detach(),
                                         f[:, prev_last_dim:i]], dim=-1)
                partial_g = torch.cat([g[:, :prev_last_dim].detach(),
                                         g[:, prev_last_dim:i]], dim=-1)
            else:
                partial_f = f[:, :i]
                partial_g = g[:, :i]

            loss_ = self.schmidt_norm(partial_f, partial_g, emp_kernel)
            loss_nested += self.step_weights[min(i, self.neigs) - 1] * loss_
            prev_last_dim = i

        return loss_nested

    def schmidt_norm(
            self,
            f,
            g,
            emp_kernel
    ):
        # the reduction assumed here is `sum` (i.e., we take summation over batch)
        # f: (B1, L)
        # g: (B2, L)
        # emp_kernel: (B1, B2)
        if self.along_batch_dim:
            # unbiased version
            gram_matrix = f @ g.T  # (B1, B2); each entry is (f^T(x_i) g(y_j))
            loss = ((emp_kernel - gram_matrix) ** 2).mean()
        else:
            # along feature dim
            # compute loss_operator = -2 * E_{p(x)p(y)}[k(x,y) f^T(x) g(y)]
            loss_operator = - 2 * ((f @ g.T) * emp_kernel).mean()  # O(B1 * B2 * L) + O(B1 * B2)
            loss_metric = compute_loss_metric(f, g)
            loss = loss_operator + loss_metric
        return loss


class NeuralSVDLossCDK:
    # NeuralSVD loss for a canonical dependence kernel
    # specifically for maximal correlation, the objective is known as the negative H-score
    def __init__(
            self,
            step=1,
            feature_dim=0,
            stop_grad=False,
            along_batch_dim=True,
            weight_order=1.,
            sequential=False,
            oneshot=False,
            unittest=False,
            set_first_mode_const=True,  # for maximal correlation kernel, the first modes are constant functions
            ratio_upper_bound=np.inf,
            ratio_lower_bound=0.,
            include_joint=True,  # include paired samples in the denominator (e.g., the "original" max. corr.)
    ):
        # parameters for maximal correlation kernel
        self.set_first_mode_const = set_first_mode_const
        self.include_joint = include_joint
        self.ratio_upper_bound = ratio_upper_bound
        self.ratio_lower_bound = ratio_lower_bound

        # parameters for the nested objective
        self.stop_grad = stop_grad
        self.along_batch_dim = along_batch_dim
        self.neigs = feature_dim
        self.oneshot = oneshot
        self.sequential = sequential
        if sequential:
            assert oneshot
            self.vector_mask, self.matrix_mask = get_sequential_nesting_masks(self.neigs,
                                                                              self.set_first_mode_const)
        else:
            self.end_indices = list(range(step, feature_dim + 1, step))
            if feature_dim not in self.end_indices:
                self.end_indices.append(feature_dim)
            step_weights = [weight_order ** (n - feature_dim // step + 1) for n in
                            np.arange(feature_dim // step + (feature_dim % step != 0))]
            self.step_weights = torch.tensor(step_weights / sum(step_weights))
            self.vector_mask, self.matrix_mask = get_joint_nesting_masks(self.step_weights, self.set_first_mode_const)
        if oneshot:
            assert not stop_grad
            assert not along_batch_dim

        self.unittest = unittest  # if True, check if oneshot nested loss is equivalent to stepwise nested loss

    def __call__(
            self,
            f,
            g,
            joint_mask=None,  # for "cross-domain retrieval"
            batch_weights=None,  # for "graph"
    ):
        if self.unittest:
            return self.test_oneshot(f, g, joint_mask, batch_weights)

        # compute a nested objective
        if self.oneshot:
            nested_loss, *_, rs_joint, rs_indep = \
                self.nested_schmidt_norm_oneshot(f, g, batch_weights=batch_weights)
            return nested_loss, rs_joint, rs_indep
        else:
            nested_loss, *_, rs_joint, rs_indep = \
                self.nested_schmidt_norm_stepwise(f, g, joint_mask, batch_weights)
            return nested_loss, rs_joint, rs_indep

    def test_oneshot(self, f, g, joint_mask, batch_weights):
        if not self.along_batch_dim and joint_mask is None:
            if self.ratio_upper_bound == np.inf and self.ratio_lower_bound == 0.:
                nested_loss_oneshot, nested_loss_operator_oneshot, nested_loss_metric_oneshot, *_ = \
                    self.nested_schmidt_norm_oneshot(f, g, batch_weights)
                nested_loss_stepwise, nested_loss_operator_stepwise, nested_loss_metric_stepwise, *_ = \
                    self.nested_schmidt_norm_stepwise(f, g, joint_mask, batch_weights)
                print("DEBUGGING: we are checking if oneshot computation is correct...")
                assert torch.isclose(nested_loss_operator_oneshot, nested_loss_operator_stepwise), (
                    nested_loss_operator_oneshot, nested_loss_operator_stepwise)
                assert torch.isclose(nested_loss_metric_oneshot, nested_loss_metric_stepwise), (
                    nested_loss_metric_oneshot, nested_loss_metric_stepwise)
                assert torch.isclose(nested_loss_oneshot, nested_loss_stepwise), (
                    nested_loss_oneshot, nested_loss_stepwise)

    def nested_grad_weights_evd(
            self,
            f,
            fp,
            batch_weights=None
    ):
        """
        usage:
            grad_weights = self.nested_grad_weights(f, fp)
            f.backward(grad_weights)
        the reduction assumed here is `mean` (i.e., we take average over batch)
        f: (B, L)
        fp: (B, L)
        """
        self.vector_mask = self.vector_mask.to(f.device)
        self.matrix_mask = self.matrix_mask.to(f.device)

        if self.set_first_mode_const:
            pad = nn.ConstantPad1d((1, 0), 1)
            f = pad(f)
            fp = pad(fp)
        if batch_weights is not None:
            f *= batch_weights
            fp *= batch_weights

        # for grad(f)
        operator_f = - torch.einsum('l,bl->bl', self.vector_mask, fp)
        metric_f = torch.einsum('bi,il,il->bl', f, self.matrix_mask, (fp.T @ fp) / fp.shape[0])
        grad_weights_f = 4 * (operator_f + metric_f) / f.shape[0]
        if self.set_first_mode_const:
            return grad_weights_f[:, 1:]
        else:
            return grad_weights_f

    def nested_grad_weights_svd(
            self,
            f,
            g,
            batch_weights=None
    ):
        """
        usage:
            grad_weights_f, grad_weights_g = self.nested_grad_weights(f, g)
            f.backward(grad_weights_f)
            g.backward(grad_weights_g)
        the reduction assumed here is `mean` (i.e., we take average over batch)
        f: (B, L)
        g: (B, L)
        """
        self.vector_mask = self.vector_mask.to(f.device)
        self.matrix_mask = self.matrix_mask.to(f.device)

        if self.set_first_mode_const:
            pad = nn.ConstantPad1d((1, 0), 1)
            f = pad(f)
            g = pad(g)
        if batch_weights is not None:
            f *= batch_weights
            g *= batch_weights

        # for grad(f)
        operator_f = - torch.einsum('l,bl->bl', self.vector_mask, g)
        metric_f = torch.einsum('bi,il,il->bl', f, self.matrix_mask, (g.T @ g) / g.shape[0])
        grad_weights_f = operator_f + metric_f
        grad_weights_f = 2 * grad_weights_f / f.shape[0]
        # for grad(g)
        operator_g = - torch.einsum('l,bl->bl', self.vector_mask, f)
        metric_g = torch.einsum('bi,il,il->bl', g, self.matrix_mask, (f.T @ f) / f.shape[0])
        grad_weights_g = operator_g + metric_g
        grad_weights_g = 2 * grad_weights_g / g.shape[0]
        if self.set_first_mode_const:
            return grad_weights_f[:, 1:], grad_weights_g[:, 1:]
        else:
            return grad_weights_f, grad_weights_g

    def nested_schmidt_norm_oneshot(
            self,
            f,
            g,
            batch_weights=None
    ):
        # the reduction assumed here is `mean` (i.e., we take average over batch)
        # f: (B, L)
        # g: (B, L)
        self.vector_mask = self.vector_mask.to(f.device)
        self.matrix_mask = self.matrix_mask.to(f.device)

        if self.set_first_mode_const:
            pad = nn.ConstantPad1d((1, 0), 1)
            f = pad(f)
            g = pad(g)
        if batch_weights is not None:
            f *= batch_weights
            g *= batch_weights

        # along feature dim only
        # compute loss_operator = -2 * E_{p(x,y)}[f^T(x) g(y)]
        loss_operator = - 2 * torch.einsum('l,bl,bl->b', self.vector_mask, f, g).mean()  # O(B1 * L)
        loss_metric = compute_loss_metric(f, g, self.matrix_mask)
        loss = loss_operator + loss_metric

        gram_matrix = f @ g.T  # (B, B); each entry is (f^T(x_i) g(y_j))
        rs_joint = gram_matrix.diag()
        rs_indep = off_diagonal(gram_matrix)

        return loss, loss_operator, loss_metric, rs_joint, rs_indep

    def nested_schmidt_norm_stepwise(
            self,
            f,
            g,
            joint_mask=None,
            batch_weights=None
    ):
        loss_nested = 0.
        loss_operator_nested = 0.
        loss_metric_nested = 0.
        prev_last_dim = 0
        assert f.shape[1] in self.end_indices, \
            f"{self.end_indices} must include the dimensionality of the feature {f.shape[1]}"

        rs_joint, rs_indep = None, None

        for i in self.end_indices:
            if self.stop_grad:
                partial_f = torch.cat([f[:, :prev_last_dim].detach(),
                                         f[:, prev_last_dim:i]], dim=-1)
                partial_g = torch.cat([g[:, :prev_last_dim].detach(),
                                         g[:, prev_last_dim:i]], dim=-1)
            else:
                partial_f = f[:, :i]
                partial_g = g[:, :i]

            loss_, loss_operator_, loss_metric_, loss_correction_, correction_triggered_, rs_joint, rs_indep = \
                self.schmidt_norm(partial_f, partial_g,
                                  correction=True,
                                  joint_mask=joint_mask,
                                  batch_weights=batch_weights)
            if correction_triggered_ != 'none':
                loss_nested += loss_correction_
            else:
                loss_nested += loss_
            loss_operator_nested += loss_operator_
            loss_metric_nested += loss_metric_
            prev_last_dim = i

        return loss_nested, loss_operator_nested, loss_metric_nested, rs_joint, rs_indep

    def schmidt_norm(
            self,
            f,
            g,
            correction=False,
            clip_negative=False,
            joint_mask=None,  # for "cross-domain retrieval"
            batch_weights=None,  # for "graph"
    ):
        # the reduction assumed here is `mean` (i.e., we take mean over batch)
        # phi, psi: (B, L)
        if self.set_first_mode_const:
            pad = nn.ConstantPad1d((1, 0), 1)
            f = pad(f)
            g = pad(g)
        if batch_weights is not None:
            f *= batch_weights
            g *= batch_weights

        if self.along_batch_dim:
            gram_matrix = f @ g.T  # (B, B); each entry is (f^T(x_i) g(y_j))
            if joint_mask is None:
                rs_joint = gram_matrix.diag()
                if self.include_joint:
                    rs_indep = gram_matrix.flatten()
                else:
                    rs_indep = off_diagonal(gram_matrix)
            else:
                rs_joint = extract_tensor(gram_matrix, joint_mask)
                if self.include_joint:
                    rs_indep = gram_matrix.flatten()
                else:
                    rs_indep = extract_tensor(gram_matrix, ~joint_mask)

            if clip_negative:
                rs_joint = torch.clamp(rs_joint, min=0.)
                rs_indep = torch.clamp(rs_indep, min=0.)

            # loss_operator = -2 * E_{p(x,y)}[f^T(x) g(y)] (correlation)
            loss_operator = - 2 * rs_joint.mean()  # scalar
            # loss_metric = E_{p(x)p(y)}[(f^T(x) g(y))^2]
            # complexity = O(L * B^2)
            # along batch dimension; fast if B << L
            # since we compute this term using a single batch psi,
            # we should exclude the "paired" samples on the diagonal
            loss_metric = (rs_indep ** 2).mean()  # scalar
        else:
            rs_joint = (f * g).sum(dim=-1)  # (B, )
            if clip_negative:
                rs_joint = torch.clamp(rs_joint, min=0.)
            loss_operator = - 2 * rs_joint.mean()  # scalar
            # complexity = O(B * L^2 + L^3) + O(L)
            loss_metric = compute_loss_metric(f, g)
            rs_indep = torch.tensor([0.])

        loss = loss_operator + loss_metric

        upper_correction_triggered = False
        lower_correction_triggered = False
        loss_correction = 0.
        if correction:
            # methods for nonnegative correction (to avoid overfitting)
            loss_metric_pxy = 0.
            loss_operator_pxpy = 0.
            if self.ratio_upper_bound < np.inf:
                loss_metric_pxy = (rs_joint ** 2).mean()  # for upper correction
                upper_correction_triggered = (loss_metric - loss_metric_pxy / self.ratio_upper_bound) < 0.
            if self.ratio_lower_bound > 0.:
                if self.along_batch_dim:
                    loss_operator_pxpy = -2 * rs_indep.mean()  # for lower correction
                else:
                    loss_operator_pxpy = -2 * f.mean(-1) @ g.mean(-1)
                lower_correction_triggered = (loss_operator - loss_operator_pxpy * self.ratio_lower_bound) > 0.

            if upper_correction_triggered and not lower_correction_triggered:
                loss_correction = (loss_metric_pxy / self.ratio_upper_bound + loss_operator)
            elif not upper_correction_triggered and lower_correction_triggered:
                loss_correction = (loss_operator_pxpy * self.ratio_lower_bound + loss_metric)
            elif upper_correction_triggered and lower_correction_triggered:
                loss_correction = (loss_metric_pxy / self.ratio_upper_bound + loss_operator) + \
                                  (loss_operator_pxpy * self.ratio_lower_bound + loss_metric)
            else:
                loss_correction = loss_operator + loss_metric  # add a scalar to a scalar

        correction_triggered = 'none'
        if correction:
            if upper_correction_triggered and not lower_correction_triggered:
                correction_triggered = 'upper'
            elif not upper_correction_triggered and lower_correction_triggered:
                correction_triggered = 'lower'
            elif upper_correction_triggered and lower_correction_triggered:
                correction_triggered = 'both'

        return loss, loss_operator, loss_metric, \
            loss_correction, correction_triggered, rs_joint, rs_indep
