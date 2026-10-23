"""Float64 Torch conditional least squares; all residual/Jacobian algebra stays on device.

The optimizer is bounded damped Gauss--Newton (Levenberg--Marquardt), not SGD.
Bootstrap replicas use boundary multiplicities on the original device-resident data.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import torch


def resolve_device(device: str) -> torch.device:
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    resolved = torch.device(device)
    if resolved.type not in ('cpu', 'cuda'):
        raise ValueError('fitting device must be cpu, cuda, cuda:N, or auto')
    if resolved.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA fitting requested but no CUDA device is visible to this process')
    return resolved


class Conditional:
    """Cache fixed geometry once; derive the normalized density and its exact Jacobian."""
    def __init__(self, N, X, nodes=256, device='cuda'):
        self.device = resolve_device(device)
        self.nodes = nodes
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != 9 or not np.isfinite(X).all():
            raise ValueError('X must be a finite [rows, 9] array')
        boundary, inverse = np.unique(X[:, 1:], axis=0, return_inverse=True)
        self.inverse = torch.as_tensor(inverse, device=self.device)
        self.rows = len(X)
        self.boundaries = len(boundary)
        x = self.tensor(X)
        b = self.tensor(boundary)
        self.u = 2 * N * x[:, 7] * torch.cos(x[:, 0] - x[:, 8])
        self.left = torch.remainder(x[:, 0] + x[:, 1] + x[:, 5] + x[:, 6] + math.pi, 2 * math.pi) - math.pi
        self.right = torch.remainder(x[:, 0] - x[:, 2] - x[:, 3] - x[:, 4] + math.pi, 2 * math.pi) - math.pi
        self.v = torch.cos(self.left) + torch.cos(self.right)
        self.A = b[:, 0] + b[:, 4] + b[:, 5]
        self.B = b[:, 1] + b[:, 2] + b[:, 3]
        self.Cr = 2 * N * b[:, 6] * torch.cos(b[:, 7])
        self.Ci = -2 * N * b[:, 6] * torch.sin(b[:, 7])
        self.Dr = torch.cos(self.A) + torch.cos(self.B)
        self.Di = torch.sin(self.A) - torch.sin(self.B)
        self.harmonics = self.tensor(np.arange(1, 7))
        n = self.harmonics[:, None]
        self.cos_left = torch.cos(n * self.left)
        self.cos_right = torch.cos(n * self.right)
        # Harmonic addition allows grid Fourier sums to use small matrix products
        # instead of repeatedly evaluating cos(n * phi) at millions of grid points.
        self.cos_A, self.sin_A = torch.cos(n * self.A), torch.sin(n * self.A)
        self.cos_B, self.sin_B = torch.cos(n * self.B), torch.sin(n * self.B)
        self.fixed_alpha = None
        self._fixed_data = None
        self._fixed_grid = {}
        self._grid_nodes = None
        self._grid_features = None

    def tensor(self, value):
        return torch.as_tensor(value, dtype=torch.float64, device=self.device)

    def _grid(self, nodes):
        if self._grid_nodes != nodes:
            theta = -math.pi + (torch.arange(nodes, device=self.device, dtype=torch.float64) + .5) * (2 * math.pi / nodes)
            c, s = torch.cos(theta), torch.sin(theta)
            # Keep at most one grid cache. Large refinement grids are streamed.
            features = None
            if self.boundaries * nodes <= 2_000_000:
                features = (self.Cr[:, None] * c - self.Ci[:, None] * s,
                            self.Dr[:, None] * c - self.Di[:, None] * s)
            self._grid_nodes = nodes
            self._grid_features = (theta, c, s, torch.cos(self.harmonics[:, None] * theta),
                                   torch.sin(self.harmonics[:, None] * theta), features)
        return self._grid_features

    def _gaussian(self, phi, alpha, value):
        count = max(1, int(math.ceil(math.sqrt(2 * math.log(2e15) / value) / (2 * math.pi))) + 1)
        offsets = torch.arange(-count, count + 1, device=self.device, dtype=torch.float64)
        flux = phi[None] + 2 * math.pi * offsets.reshape((-1,) + (1,) * phi.ndim)
        square = flux.square()
        terms = -.5 * alpha * square
        log_weight = torch.logsumexp(terms, dim=0) + .5 * torch.log(2 * math.pi * alpha)
        derivative = -.5 * (torch.softmax(terms, dim=0) * square).sum(dim=0) + .5 / alpha
        return log_weight, derivative

    def _data_villain(self, alpha, value):
        if value == 0:
            zero = torch.zeros_like(self.u)
            return zero, zero
        if value < .5:
            q = torch.exp(-self.harmonics.square() / (2 * alpha))
            coeff = torch.stack((2 * q, q * self.harmonics.square() / alpha.square()))
            left = coeff @ self.cos_left
            right = coeff @ self.cos_right
            wl, wr = 1 + left[0], 1 + right[0]
            return torch.log(wl) + torch.log(wr), left[1] / wl + right[1] / wr
        weights, derivatives = [], []
        for start in range(0, self.rows, 262144):
            left = self._gaussian(self.left[start:start + 262144], alpha, value)
            right = self._gaussian(self.right[start:start + 262144], alpha, value)
            weights.append(left[0] + right[0])
            derivatives.append(left[1] + right[1])
        return torch.cat(weights), torch.cat(derivatives)

    def _grid_villain(self, start, stop, alpha, value, grid):
        theta, _, _, cn, sn, _ = grid
        if value == 0:
            zero = torch.zeros((stop - start, len(theta)), dtype=torch.float64, device=self.device)
            return zero, zero
        if value < .5:
            q = torch.exp(-self.harmonics.square() / (2 * alpha))
            coeff = torch.stack((2 * q, q * self.harmonics.square() / alpha.square()))[:, None, :]
            # Shape [value/derivative, boundaries, nodes].
            left = ((coeff * self.cos_A[:, start:stop].T) @ cn
                    - (coeff * self.sin_A[:, start:stop].T) @ sn)
            right = ((coeff * self.cos_B[:, start:stop].T) @ cn
                     + (coeff * self.sin_B[:, start:stop].T) @ sn)
            wl, wr = 1 + left[0], 1 + right[0]
            return torch.log(wl) + torch.log(wr), left[1] / wl + right[1] / wr
        left_phi = torch.remainder(theta + self.A[start:stop, None] + math.pi, 2 * math.pi) - math.pi
        right_phi = torch.remainder(theta - self.B[start:stop, None] + math.pi, 2 * math.pi) - math.pi
        left = self._gaussian(left_phi, alpha, value)
        right = self._gaussian(right_phi, alpha, value)
        return left[0] + right[0], left[1] + right[1]

    @torch.no_grad()
    def evaluate(self, parameters, exact=True, nodes=None):
        p = self.tensor(parameters)
        if not exact:
            real = p[0] * self.Cr + p[1] * self.Dr
            imag = p[0] * self.Ci + p[1] * self.Di
            magnitude = torch.sqrt(real.square() + imag.square())
            i0e = torch.special.i0e(magnitude)
            norm = torch.log(i0e) + magnitude + math.log(2 * math.pi)
            factor = torch.special.i1e(magnitude) / i0e / magnitude.clamp_min(1e-300)
            means = torch.stack((factor * (real * self.Cr + imag * self.Ci),
                                 factor * (real * self.Dr + imag * self.Di)), dim=1)
            pred = torch.exp(p[0] * self.u + p[1] * self.v - norm[self.inverse])
            jac = pred[:, None] * (torch.stack((self.u, self.v), dim=1) - means[self.inverse])
            return pred, jac
        nodes = nodes or self.nodes
        alpha_value = float(p[1].item())
        if alpha_value < 0:
            raise ValueError('alpha must be nonnegative')
        grid = self._grid(nodes)
        norms = torch.empty(self.boundaries, dtype=torch.float64, device=self.device)
        means = torch.empty((self.boundaries, 3), dtype=torch.float64, device=self.device)
        chunk = max(1, min(self.boundaries, 262144 // nodes))
        for start in range(0, self.boundaries, chunk):
            stop = min(start + chunk, self.boundaries)
            _, c, s, _, _, cached = grid
            if cached is None:
                u = self.Cr[start:stop, None] * c - self.Ci[start:stop, None] * s
                v = self.Dr[start:stop, None] * c - self.Di[start:stop, None] * s
            else:
                u, v = cached[0][start:stop], cached[1][start:stop]
            key = (nodes, start, stop)
            if self.fixed_alpha is not None and key in self._fixed_grid:
                weight, derivative = (value.to(self.device) for value in self._fixed_grid[key])
            else:
                weight, derivative = self._grid_villain(start, stop, p[1], alpha_value, grid)
                if self.fixed_alpha is not None:
                    # Large normalization grids use host-backed fixed-weight blocks.
                    # Values are computed once on device, then reused by transfer.
                    if self.boundaries * nodes <= 2_000_000:
                        self._fixed_grid[key] = (weight, derivative)
                    else:
                        self._fixed_grid[key] = (weight.cpu(), derivative.cpu())
            log_weight = p[0] * u + p[2] * v + weight
            probability = torch.softmax(log_weight, dim=1)
            norms[start:stop] = torch.logsumexp(log_weight, dim=1) + math.log(2 * math.pi / nodes)
            means[start:stop] = torch.stack(((probability * u).sum(1),
                                            (probability * derivative).sum(1),
                                            (probability * v).sum(1)), dim=1)
        if self.fixed_alpha is not None and self._fixed_data is not None:
            weight, derivative = self._fixed_data
        else:
            weight, derivative = self._data_villain(p[1], alpha_value)
            if self.fixed_alpha is not None:
                self._fixed_data = (weight, derivative)
        pred = torch.exp(p[0] * self.u + p[2] * self.v + weight - norms[self.inverse])
        scores = torch.stack((self.u, derivative, self.v), dim=1)
        return pred, pred[:, None] * (scores - means[self.inverse])

    def predict(self, parameters, exact=True, nodes=None):
        return self.evaluate(parameters, exact, nodes)[0]


@dataclass
class Solution:
    x: torch.Tensor
    fun: torch.Tensor
    jac: torch.Tensor
    mse: float
    success: bool
    iterations: int
    message: str
    trace: list[dict]


@torch.no_grad()
def least_squares(model, target, initial, weights=None, *, exact=True, signed=False,
                  fixed_alpha=None, maxiter=200):
    """Projected damped Gauss--Newton with analytic Jacobians and GPU small solves."""
    x = model.tensor(initial).clone()
    target = model.tensor(target)
    weights = torch.ones_like(target) if weights is None else model.tensor(weights)
    denominator = weights.sum()
    if denominator.item() <= 0:
        raise ValueError('least squares requires positive total weight')
    indices = [0, 2] if fixed_alpha is not None else list(range(len(x)))
    if fixed_alpha is not None:
        x[1] = fixed_alpha
    lower = model.tensor([0., 0., -math.inf] if exact else ([1e-12] if len(x) == 1 else [0., -math.inf if signed else 0.]))
    scale = model.tensor([1., .3, 1.] if exact else [1.] * len(x))[indices]
    x = torch.maximum(x, lower)
    damping = 1e-3
    pred, full_jac = model.evaluate(x, exact)
    residual = pred - target
    cost = float(((weights * residual.square()).sum() / denominator).item())
    success, message = False, 'maximum iterations reached'
    iteration = -1
    trace = []
    from tqdm import trange
    for iteration in trange(maxiter, desc=getattr(model, "fit_label", "fit iterations"), leave=False):
        jac = full_jac[:, indices] * scale
        gradient = jac.T @ (weights * residual) / denominator
        hessian = jac.T @ (weights[:, None] * jac) / denominator
        blocked = (x[indices] <= lower[indices] + 1e-14) & (gradient > 0)
        gradient = gradient.masked_fill(blocked, 0.)
        trace.append({'iteration': iteration + 1, 'mse': cost, 'damping': damping,
                      'projected_gradient_max': float(gradient.abs().max().item())})
        if float(gradient.abs().max().item()) < 1e-12:
            success, message = True, 'projected gradient tolerance'
            break
        hessian[blocked, :] = 0.
        hessian[:, blocked] = 0.
        diagonal = torch.diag(hessian).clamp_min(1e-14)
        accepted = False
        for _ in range(18):
            system = hessian + damping * torch.diag(diagonal)
            system[blocked, blocked] = 1.
            step = torch.linalg.solve(system, -gradient)
            candidate = x.clone()
            candidate[indices] = torch.maximum(x[indices] + scale * step, lower[indices])
            change = (candidate[indices] - x[indices]) / scale
            if float(change.abs().max().item()) < 1e-11 * (1 + float((x[indices] / scale).abs().max().item())):
                success, message = True, 'step tolerance'
                break
            trial_pred, trial_jac = model.evaluate(candidate, exact)
            trial_residual = trial_pred - target
            trial_cost = float(((weights * trial_residual.square()).sum() / denominator).item())
            if math.isfinite(trial_cost) and trial_cost < cost:
                improvement = cost - trial_cost
                x, residual, full_jac = candidate, trial_residual, trial_jac
                cost = trial_cost
                damping = max(damping / 3, 1e-12)
                accepted = True
                if improvement < 1e-11 * max(cost, 1e-30):
                    success, message = True, 'cost tolerance'
                break
            damping *= 5
        if success:
            break
        if not accepted:
            message = 'no decreasing bounded step'
            break
    return Solution(x, residual, full_jac[:, indices], cost, success, iteration + 1, message, trace)


def _statistics(solution, weights, size):
    weights = torch.ones_like(solution.fun) if weights is None else weights
    jac = solution.jac * torch.sqrt(weights)[:, None]
    _, singular, vt = torch.linalg.svd(jac, full_matrices=False)
    dimension = solution.jac.shape[1]
    rank = int((singular > singular[0] * 1e-10).sum().item()) if singular[0].item() else 0
    covariance = correlation = None
    errors = [None] * dimension
    if rank == dimension and size > dimension:
        covariance = (vt.T / singular.square()) @ vt * (weights * solution.fun.square()).sum() / (size - dimension)
        std = torch.diag(covariance).clamp_min(0).sqrt()
        correlation = covariance / (std[:, None] * std[None, :]).clamp_min(1e-300)
        errors = std.cpu().tolist()
    return {'rank': rank, 'jacobian_singular_values': singular.cpu().tolist(),
            'covariance': None if covariance is None else covariance.cpu().tolist(),
            'correlation': None if correlation is None else correlation.cpu().tolist(),
            'errors': errors}



def fit_two_fixed(N, X, freq, alpha, p0=(.5, .5), device='cuda'):
    if not math.isfinite(alpha) or alpha < 0:
        raise ValueError('fixed alpha must be finite and nonnegative')
    model = Conditional(N, X, device=device)
    target = model.tensor(freq)
    model.fit_label = "2plaq old"
    old_solution = least_squares(model, target, p0, exact=False)
    if not old_solution.success:
        raise RuntimeError('old fit failed: ' + old_solution.message)
    def pack(sol, names):
        stats = _statistics(sol, None, len(freq))
        errors = stats.pop('errors')
        values = sol.x.cpu().tolist()
        if len(names) == 2 and len(values) == 3:
            values = [values[0], values[2]]
        return {**dict(zip(names, values)),
                **{name + '_err': err for name, err in zip(names, errors)},
                **stats, 'mse': sol.mse, 'success': sol.success,
                'iterations': sol.iterations, 'message': sol.message, 'trace': sol.trace}
    old = pack(old_solution, ['beta', 'alpha_eff'])
    model.fit_label = "2plaq fixed alpha"
    model.fixed_alpha = float(alpha)
    attempts = []
    best = None
    for a1 in (0., old['alpha_eff']):
        sol = least_squares(model, target, [old['beta'], alpha, a1], fixed_alpha=alpha)
        attempts.append({'initial': [old['beta'], alpha, a1], 'success': sol.success,
                         'mse': sol.mse, 'message': sol.message, 'trace': sol.trace})
        if sol.success and (best is None or sol.mse < best.mse):
            best = sol
    if best is None:
        raise RuntimeError(f'all fixed alpha fits failed: {attempts}')
    while model.nodes < 16384:
        current = model.predict(best.x)
        fine = model.predict(best.x, nodes=2 * model.nodes)
        delta = float(((fine-current).abs().max()/fine.max().clamp_min(1e-300)).item())
        model.nodes *= 2
        model._fixed_grid.clear()
        best = least_squares(model, target, best.x, fixed_alpha=alpha, maxiter=600)
        if not best.success:
            raise RuntimeError('fixed alpha fit failed during normalization refinement')
        if delta < 1e-8:
            break
    else:
        raise RuntimeError('normalization did not converge at 16384 nodes')
    new = pack(best, ['beta', 'alpha1'])
    new.update(alpha=float(alpha), error_scope='conditional_on_fixed_alpha',
               normalization_nodes=model.nodes, normalization_relative_change=delta,
               attempts=attempts, backend='torch', device=str(model.device))
    return {'old': old, 'new': new, 'backend': 'torch', 'device': str(model.device)}


class OnePlaquette:
    def __init__(self, X, device):
        self.device = resolve_device(device)
        self.fit_label = "1plaq alpha"
        self.X = self.tensor(X)
    def tensor(self, value):
        return torch.as_tensor(value, dtype=torch.float64, device=self.device)
    def evaluate(self, parameters, exact=True):
        alpha = parameters[0].clamp_min(1e-12)
        da = torch.remainder(self.X[:, 1] + math.pi, 2*math.pi)-math.pi
        # The numerator uses the original unwrapped (s, da) coordinates.
        energy = .5 * (self.X[:, 1] + 2*math.pi*self.X[:, 0]).square()
        value = float(alpha.item())
        if value < .5:
            n = self.tensor(np.arange(1, max(2, int(math.ceil(math.sqrt(2*value*math.log(2e15)))) + 1)))
            q = torch.exp(-n.square()/(2*alpha))
            c = torch.cos(da[:, None]*n)
            weight = 1 + 2*(c*q).sum(1)
            dlog = (c*q*n.square()/alpha.square()).sum(1)/weight - .5/alpha
            lognorm = torch.log(weight) - .5*torch.log(2*math.pi*alpha)
        else:
            count = max(1, int(math.ceil(math.sqrt(2*math.log(2e15)/value)/(2*math.pi)))+1)
            square = .5*(da[:, None]+2*math.pi*self.tensor(np.arange(-count,count+1))).square()
            terms = -alpha*square
            lognorm = torch.logsumexp(terms,1)
            dlog = -(torch.softmax(terms,1)*square).sum(1)
        pred = torch.exp(-alpha*energy-lognorm)
        return pred, (pred*(-energy-dlog))[:, None]


def fit_one(X, freq, p0=.5, device='cuda'):
    model = OnePlaquette(X, device)
    sol = least_squares(model, freq, [p0], exact=False)
    if not sol.success:
        raise RuntimeError('1plaq fit failed: '+sol.message)
    stats = _statistics(sol, None, len(freq))
    return {'alpha': float(sol.x[0].item()), 'alpha_err': stats['errors'][0],
            'mse': sol.mse, 'backend': 'torch', 'device': str(model.device),
            'iterations': sol.iterations, 'success': True, 'trace': sol.trace}


def prediction_mse(N, X, freq, parameters, nodes=256, device='cuda'):
    """GPU diagnostic at fixed final couplings, without any parameter refit."""
    model = Conditional(N, X, nodes=nodes, device=device)
    current = model.predict(parameters)
    while nodes < 16384:
        nodes *= 2
        fine = model.predict(parameters, nodes=nodes)
        delta = float(((fine-current).abs().max()/fine.max().clamp_min(1e-300)).item())
        if delta < 1e-8:
            return {'mse': float((fine-model.tensor(freq)).square().mean().item()),
                    'normalization_nodes': nodes, 'normalization_relative_change': delta}
        current = fine
    raise RuntimeError('corrected-alpha normalization did not converge at 16384 nodes')
