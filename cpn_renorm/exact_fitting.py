"""Exact periodic Gaussian weights and boundary-normalized conditional fits."""
from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares
from scipy.special import logsumexp

from .fitting import fit_two_plaq


def log_villain(phi: np.ndarray, alpha: float, representation: str = 'auto') -> np.ndarray:
    """Log weight up to an angle-independent constant; W_0 = 1 by continuity."""
    phi = (np.asarray(phi, float) + np.pi) % (2 * np.pi) - np.pi
    if alpha < 0 or not np.isfinite(alpha):
        raise ValueError('alpha must be finite and nonnegative')
    if alpha == 0:
        return np.zeros_like(phi)
    if representation == 'fourier' or (representation == 'auto' and alpha < 0.5):
        count = max(1, int(np.ceil(np.sqrt(2 * alpha * np.log(2e15)))))
        weight = np.ones_like(phi)
        for n in range(1, count + 1):
            weight += 2 * np.exp(-n * n / (2 * alpha)) * np.cos(n * phi)
        if np.any(weight <= 0):
            raise FloatingPointError('Fourier cancellation; use Gaussian representation')
        return np.log(weight)
    if representation not in ('auto', 'gaussian'):
        raise ValueError('unknown representation')
    count = max(1, int(np.ceil(np.sqrt(2 * np.log(2e15) / alpha) / (2 * np.pi))) + 1)
    terms = [-0.5 * alpha * (phi + 2 * np.pi * n) ** 2 for n in range(-count, count + 1)]
    # Match the Fourier convention (constant Fourier coefficient equals one).
    return logsumexp(terms, axis=0) + 0.5 * np.log(2 * np.pi * alpha)


class Conditional:
    """X rows: angle, six oriented boundary angles, overlap magnitude and phase."""
    def __init__(self, N: int, X: np.ndarray, nodes: int = 256):
        self.N, self.X, self.nodes = N, np.asarray(X, float), nodes
        self.boundaries, self.inverse = np.unique(self.X[:, 1:], axis=0, return_inverse=True)

    def log_weight(self, theta, boundary, parameters, exact=True):
        beta = parameters[0]
        matter = 2 * self.N * beta * boundary[..., 6] * np.cos(theta - boundary[..., 7])
        left = theta + boundary[..., 0] + boundary[..., 4] + boundary[..., 5]
        right = theta - boundary[..., 1] - boundary[..., 2] - boundary[..., 3]
        if not exact:
            return matter + parameters[1] * (np.cos(left) + np.cos(right))
        return (matter + log_villain(left, parameters[1]) + log_villain(right, parameters[1])
                + parameters[2] * (np.cos(left) + np.cos(right)))

    def predict(self, parameters, exact=True, nodes=None):
        nodes = nodes or self.nodes
        grid = -np.pi + (np.arange(nodes) + 0.5) * (2 * np.pi / nodes)
        norm = np.empty(len(self.boundaries))
        for start in range(0, len(norm), 64):
            boundary = self.boundaries[start:start + 64, None, :]
            weights = self.log_weight(grid[None, :], boundary, parameters, exact)
            norm[start:start + 64] = logsumexp(weights, axis=1) + np.log(2 * np.pi / nodes)
        weights = self.log_weight(self.X[:, 0], self.X[:, 1:], parameters, exact)
        return np.exp(weights - norm[self.inverse])



def fit_two_fixed(N, X, freq, alpha, p0=(.5,.5)):
    if not np.isfinite(alpha) or alpha < 0:
        raise ValueError('fixed alpha must be finite and nonnegative')
    model = Conditional(N, X)
    freq = np.asarray(freq, float)
    old = fit_two_plaq(N, X, freq, p0)
    attempts, best = [], None
    for a1 in (0., old['alpha_eff']):
        initial = [old['beta'], a1]
        result = least_squares(lambda q: model.predict([q[0], alpha, q[1]])-freq,
                               initial, bounds=([0,-np.inf],[np.inf,np.inf]),
                               ftol=1e-11, xtol=1e-11, gtol=1e-12, max_nfev=600)
        attempts.append({'initial': initial, 'success': bool(result.success),
                         'mse': float(np.mean(result.fun**2)), 'evaluations': result.nfev,
                         'optimality': float(result.optimality), 'message': result.message})
        if result.success and (best is None or result.cost < best.cost): best = result
    if best is None: raise RuntimeError(f'all fixed alpha fits failed: {attempts}')
    while model.nodes < 16384:
        p = [best.x[0], alpha, best.x[1]]
        current, fine = model.predict(p), model.predict(p, nodes=2*model.nodes)
        delta = float(np.max(abs(fine-current))/max(np.max(fine),1e-300))
        model.nodes *= 2
        best = least_squares(lambda q: model.predict([q[0],alpha,q[1]])-freq,
                             best.x, bounds=([0,-np.inf],[np.inf,np.inf]),
                             ftol=1e-11, xtol=1e-11, gtol=1e-12, max_nfev=600)
        if not best.success: raise RuntimeError('normalization refinement fit failed')
        if delta < 1e-8: break
    else: raise RuntimeError('normalization did not converge at 16384 nodes')
    _, singular, vt = np.linalg.svd(best.jac, full_matrices=False)
    rank = int(np.sum(singular > singular[0]*1e-10)) if singular[0] else 0
    cov = ((vt.T/singular**2)@vt * np.sum(best.fun**2)/(len(freq)-2)
           if rank == 2 and len(freq)>2 else None)
    errors = [None,None] if cov is None else np.sqrt(np.maximum(np.diag(cov),0)).tolist()
    return {'old': old, 'new': {'beta': float(best.x[0]), 'alpha': float(alpha),
            'alpha1': float(best.x[1]), 'beta_err': errors[0], 'alpha1_err': errors[1],
            'mse': float(np.mean(best.fun**2)), 'success': True, 'rank': rank,
            'covariance': None if cov is None else cov.tolist(),
            'jacobian_singular_values': singular.tolist(), 'evaluations': best.nfev,
            'optimality': float(best.optimality), 'message': best.message,
            'error_scope': 'conditional_on_fixed_alpha', 'attempts': attempts,
            'normalization_nodes': model.nodes, 'normalization_relative_change': delta,
            'backend': 'scipy', 'device': 'cpu'}, 'backend': 'scipy', 'device': 'cpu'}


def prediction_mse(N, X, freq, parameters, nodes=256):
    """Check normalization again at the corrected alpha without refitting."""
    model = Conditional(N, X, nodes)
    current = model.predict(parameters)
    while nodes < 16384:
        nodes *= 2
        fine = model.predict(parameters, nodes=nodes)
        delta = float(np.max(abs(fine-current))/max(np.max(fine),1e-300))
        if delta < 1e-8:
            return {'mse': float(np.mean((fine-freq)**2)), 'normalization_nodes': nodes,
                    'normalization_relative_change': delta}
        current = fine
    raise RuntimeError('corrected-alpha normalization did not converge at 16384 nodes')
