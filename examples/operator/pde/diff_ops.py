import torch


class VectorizedLaplacian:
    def __init__(self, eps=1e-5):
        self.eps = eps
        self.laplacian = self.approx_laplacian if self.eps > 0 else self.exact_laplacian

    def __call__(self, f, xs, importance=None, return_grad=False):
        if importance is None:
            lap, grad, fs = self.laplacian(f, xs, return_grad=return_grad)
        else:
            g = lambda x: importance(x).sqrt() * f(x)
            lap_g, grad_g, gs = self.laplacian(g, xs, return_grad=return_grad)
            sqrt_ws = torch.clamp(importance(xs).sqrt(), min=1e-5)
            # Warning: division with sqrt_ws could be problematic if sqrt_ws is close to zero
            lap = lap_g / sqrt_ws
            fs = gs / sqrt_ws
            if return_grad:
                grad = grad_g / sqrt_ws
            else:
                grad = grad_g
        return lap, grad, fs

    def approx_laplacian(self, f, xs, return_grad=False):
        # input:
        #   xs: (B, *) matrix
        #   f: L-dim. function
        # output: (B, L) matrix or (B, ) if L=1
        # grad: (B, L, D) matrix or (B, D) if L=1
        if xs.dim() > 2:
            xis = [xi.requires_grad_() for xi in xs.flatten(start_dim=1).t()]
            xs_flat = torch.stack(xis, dim=1)  # (B, D)
        else:
            xs_flat = xs
        fs = f(xs_flat.view_as(xs))  # (B, L)
        D = xs_flat.shape[1]
        lap = - 2 * D * fs
        grads = []
        for i in range(D):
            epsi = torch.zeros((1, D)).to(xs.device)
            epsi[0, i] = self.eps
            f_plus = f(xs_flat + epsi)
            f_minus = f(xs_flat - epsi)
            lap += (f_plus + f_minus)
            if return_grad:
                grads.append(f_plus - f_minus)  # (B, L)
        lap = lap / (self.eps ** 2)
        if return_grad:
            grad = torch.stack(grads, dim=-1) / (2 * self.eps)  # (B, L, D)
            grad = grad.squeeze(1)
        return lap, grad if return_grad else 0., fs.squeeze(1)

    def exact_laplacian(self, f, xs, return_grad=False):
        # Reference: https://github.com/vsitzmann/siren/blob/master/diff_operators.py
        xis = [xi.requires_grad_() for xi in xs.flatten(start_dim=1).t()]
        xs_flat = torch.stack(xis, dim=1)  # (B, D)
        fs = f(xs_flat.view_as(xs))  # (B, L)
        (fs_g, *other) = fs if isinstance(fs, tuple) else (fs, ())
        lap, grad = vectorized_laplacian(fs_g, xs_flat)
        return lap, grad.view(fs_g.shape[0], fs.shape[-1], -1), fs_g  # (B, L), (B, L, D), (B, L)


def vectorized_laplacian(ys, x):
    # input:
    #   ys: (B, L), x: (B, *)
    # output:
    #   (B, L)
    lap_list, grad_list = zip(*[laplacian(ys[..., i], x) for i in range(ys.shape[-1])])
    return torch.cat(lap_list, dim=-1), torch.cat(grad_list, dim=-1)


def laplacian(y, x):
    # input:
    #   y: (B, ), x: (B, *)
    # output:
    #   (B, )
    grad = gradient(y, x)
    return divergence(grad, x), grad


def divergence(y, x):
    div = 0.
    for i in range(y.shape[-1]):
        div += torch.autograd.grad(y[..., i], x, torch.ones_like(y[..., i]), create_graph=True)[0][..., i:i+1]
    return div


def gradient(y, x, grad_outputs=None):
    if grad_outputs is None:
        grad_outputs = torch.ones_like(y)
    grad = torch.autograd.grad(y, [x], grad_outputs=grad_outputs, create_graph=True)[0]
    return grad


# Reference: https://discuss.pytorch.org/t/how-to-calculate-laplacian-sum-of-2nd-derivatives-in-one-step/41667/4
class ExactLaplacian:
    def __init__(self, create_graph=False, keep_graph=None, return_grad=False):
        self.create_graph = create_graph
        self.keep_graph = keep_graph
        self.return_grad = return_grad

    def __call__(self, f, xs):
        xis = [xi.requires_grad_() for xi in xs.flatten(start_dim=1).t()]
        xs_flat = torch.stack(xis, dim=1)
        ys = f(xs_flat.view_as(xs))
        (ys_g, *other) = ys if isinstance(ys, tuple) else (ys, ())
        ones = torch.ones_like(ys_g)
        (dy_dxs,) = torch.autograd.grad(ys_g, xs_flat, ones, create_graph=True)
        lap_ys = sum(
            torch.autograd.grad(
                dy_dxi, xi, ones, retain_graph=True, create_graph=self.create_graph
            )[0]
            for xi, dy_dxi in zip(xis, (dy_dxs[..., i] for i in range(len(xis))))
        )
        if not (self.create_graph if self.keep_graph is None else self.keep_graph):
            ys = (ys_g.detach(), *other) if isinstance(ys, tuple) else ys.detach()
        result = lap_ys, ys
        if self.return_grad:
            result += (dy_dxs.detach().view_as(xs),)
        return result
