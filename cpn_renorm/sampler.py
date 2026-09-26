from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import torch

ActionKind = Literal["half", "full"]


@dataclass(frozen=True)
class Couplings:
    beta: float
    beta1: float
    alpha: float
    alpha1: float = 0.0
    mod: int = 0


def _roll(x: torch.Tensor, shift: int, dim: int) -> torch.Tensor:
    return torch.roll(x, shift, dim)


def _log_i0(x: torch.Tensor) -> torch.Tensor:
    return torch.log(torch.special.i0e(x)) + torch.abs(x)


class BatchedHMCSampler:
    """Batched constrained HMC for half/full refined Villain actions.

    Leading dimension is an independent-chain dimension. PBC forces are fully
    analytic and vectorized. OBC is used only by the small frozen-boundary fits
    and deliberately uses autograd as a correctness reference.
    """

    def __init__(self, *, chains: int, Lx: int, Ly: int, N: int,
                 couplings: Couplings, kind: ActionKind = "half",
                 periodic: bool = True, device: str | torch.device = "cpu",
                 dtype: torch.dtype = torch.float64, seed: int = 0,
                 epsilon: float = 0.05, trajectory_length: float = 1.0,
                 n_leapfrog: int | None = None,
                 mass_a: float = 1.0, mass_z: float = 1.0,
                 frozen_z: torch.Tensor | None = None,
                 frozen_a: torch.Tensor | None = None):
        if N <= 1 or chains < 1 or Lx < 2 or Ly < 2:
            raise ValueError("invalid lattice shape")
        if kind == "full" and couplings.alpha < 0:
            raise ValueError("full Villain alpha must be non-negative")
        if couplings.mod not in (0, 1):
            raise ValueError("mod must be 0 or 1")
        self.chains, self.Lx, self.Ly, self.N = chains, Lx, Ly, N
        self.c = couplings
        self.kind, self.periodic = kind, periodic
        self.device, self.dtype = torch.device(device), dtype
        self.cdtype = torch.complex128 if dtype == torch.float64 else torch.complex64
        # ``n_leapfrog`` remains an API-only compatibility aid for callers that
        # construct a sampler directly.  Configuration files intentionally no
        # longer accept it.  The trajectory length is the invariant quantity.
        if n_leapfrog is not None:
            trajectory_length = float(epsilon) * int(n_leapfrog)
        if epsilon <= 0 or trajectory_length <= 0:
            raise ValueError("epsilon and trajectory_length must be positive")
        self.trajectory_length = float(trajectory_length)
        self.epsilon = float(epsilon)
        self.n_leapfrog = 1
        self.set_epsilon(epsilon)
        self.mass_a, self.mass_z = float(mass_a), float(mass_z)
        self.generator = torch.Generator(device=self.device).manual_seed(int(seed))
        real = torch.randn((chains, Lx, Ly, N), generator=self.generator,
                           device=self.device, dtype=dtype)
        imag = torch.randn((chains, Lx, Ly, N), generator=self.generator,
                           device=self.device, dtype=dtype)
        self.z = torch.complex(real, imag)
        self.z /= torch.linalg.vector_norm(self.z, dim=-1, keepdim=True)
        self.a = (2 * math.pi * torch.rand((chains, Lx, Ly, 2),
                  generator=self.generator, device=self.device, dtype=dtype) - math.pi)
        self.s = torch.zeros((chains, Lx, Ly), device=self.device, dtype=torch.int64)
        self.frozen_z = self._mask(frozen_z, (chains, Lx, Ly))
        self.frozen_a = self._mask(frozen_a, (chains, Lx, Ly, 2))
        self.attempted = torch.zeros(chains, dtype=torch.int64, device=self.device)
        self.accepted = torch.zeros_like(self.attempted)
        self.metro_attempted = torch.zeros_like(self.attempted)
        self.metro_accepted = torch.zeros_like(self.attempted)
        self.last_accept_prob = torch.zeros(chains, dtype=dtype, device=self.device)

    def _mask(self, value: torch.Tensor | None, shape: tuple[int, ...]) -> torch.Tensor:
        if value is None:
            return torch.zeros(shape, dtype=torch.bool, device=self.device)
        value = torch.as_tensor(value, dtype=torch.bool, device=self.device)
        return torch.broadcast_to(value, shape).clone()

    @property
    def U(self) -> torch.Tensor:
        return torch.exp(1j * self.a)

    @property
    def accept_rate(self) -> torch.Tensor:
        return self.accepted / self.attempted.clamp_min(1)

    @property
    def metro_accept_rate(self) -> torch.Tensor:
        return self.metro_accepted / self.metro_attempted.clamp_min(1)

    def set_epsilon(self, candidate: float) -> float:
        """Set a candidate step size while preserving the exact trajectory length."""
        if not math.isfinite(candidate) or candidate <= 0:
            raise ValueError("epsilon candidate must be finite and positive")
        self.n_leapfrog = max(1, int(math.ceil(self.trajectory_length / candidate)))
        self.epsilon = self.trajectory_length / self.n_leapfrog
        return self.epsilon

    def reset_diagnostics(self) -> None:
        """Clear acceptance counters before production measurements."""
        self.attempted.zero_()
        self.accepted.zero_()
        self.metro_attempted.zero_()
        self.metro_accepted.zero_()
        self.last_accept_prob.zero_()

    def _spin_inner(self, dim: int) -> torch.Tensor:
        return torch.sum(torch.conj(_roll(self.z, -1, dim)) * self.z, dim=-1)

    def plaquette_da(self) -> torch.Tensor:
        a = self.a
        if self.periodic:
            return (a[..., 0] + _roll(a, -1, 1)[..., 1]
                    - _roll(a, -1, 2)[..., 0] - a[..., 1])
        return (a[:, :-1, :-1, 0] + a[:, 1:, :-1, 1]
                - a[:, :-1, 1:, 0] - a[:, :-1, :-1, 1])

    def villain_flux(self) -> torch.Tensor:
        da = self.plaquette_da()
        ss = self.s if self.periodic else self.s[:, :-1, :-1]
        return da + 2 * math.pi * ss

    def _matter_action(self) -> torch.Tensor:
        z, U, c = self.z, self.U, self.c
        if self.periodic:
            ix, iy = self._spin_inner(1), self._spin_inner(2)
            cp = (torch.real(U[..., 0] * ix).sum((1, 2))
                  + torch.real(U[..., 1] * iy).sum((1, 2)))
            inners = (ix, iy)
        else:
            ix = torch.sum(torch.conj(z[:, 1:]) * z[:, :-1], dim=-1)
            iy = torch.sum(torch.conj(z[:, :, 1:]) * z[:, :, :-1], dim=-1)
            cp = (torch.real(U[:, :-1, :, 0] * ix).sum((1, 2))
                  + torch.real(U[:, :, :-1, 1] * iy).sum((1, 2)))
            inners = (ix, iy)
        out = -2.0 * self.N * c.beta * cp
        if c.beta1 == 0:
            return out
        if c.mod == 0:
            deformation = sum((torch.abs(v).square() - 1).sum((1, 2)) for v in inners)
            return out - self.N * c.beta1 * deformation
        xterms = [_log_i0(2 * self.N * c.beta1 * torch.abs(v)).sum((1, 2))
                  for v in inners]
        return out - math.copysign(1.0, c.beta1) * sum(xterms)

    def action(self) -> torch.Tensor:
        out = self._matter_action()
        if self.kind == "half":
            return out - self.c.alpha * torch.cos(self.plaquette_da()).sum((1, 2))
        flux = self.villain_flux()
        return out + (0.5 * self.c.alpha * flux.square()
                      - self.c.alpha1 * torch.cos(flux)).sum((1, 2))

    def _project(self, z: torch.Tensor, field: torch.Tensor) -> torch.Tensor:
        radial = torch.real(torch.sum(torch.conj(z) * field, dim=-1, keepdim=True))
        return field - z * radial

    def _z_force_pbc(self) -> torch.Tensor:
        z, U, c = self.z, self.U, self.c
        result = torch.zeros_like(z)
        for dim, mu in ((1, 0), (2, 1)):
            # Force formulas use <z(x)|z(x+mu)>, whereas the action helper is
            # its conjugate <z(x+mu)|z(x)>.
            inner = torch.conj(self._spin_inner(dim))
            base = (_roll(U[..., mu], 1, dim)[..., None] * _roll(z, 1, dim)
                    + torch.conj(U[..., mu])[..., None] * _roll(z, -1, dim))
            result += 2 * self.N * c.beta * base
            if c.beta1 == 0:
                continue
            if c.mod == 0:
                coeff = c.beta1 * inner
            else:
                radius = torch.abs(inner)
                arg = 2 * self.N * abs(c.beta1) * radius
                ratio = torch.special.i1e(arg) / torch.special.i0e(arg).clamp_min(
                    torch.finfo(self.dtype).tiny)
                phase = torch.where(radius > 0, inner / radius, torch.zeros_like(inner))
                coeff = math.copysign(1.0, c.beta1) * ratio * phase
            deform = (_roll(coeff, 1, dim)[..., None] * _roll(z, 1, dim)
                      + torch.conj(coeff)[..., None] * _roll(z, -1, dim))
            result += 2 * self.N * abs(c.beta1) * deform if c.mod else 2 * self.N * deform
        return result

    def _a_force_pbc(self) -> torch.Tensor:
        c, U = self.c, self.U
        force = torch.zeros_like(self.a)
        force[..., 0] = 2 * self.N * c.beta * torch.imag(
            torch.conj(U[..., 0]) * torch.conj(self._spin_inner(1)))
        force[..., 1] = 2 * self.N * c.beta * torch.imag(
            torch.conj(U[..., 1]) * torch.conj(self._spin_inner(2)))
        if self.kind == "half":
            q = c.alpha * torch.sin(self.plaquette_da())
        else:
            f = self.villain_flux()
            q = c.alpha * f + c.alpha1 * torch.sin(f)
        force[..., 0] -= q - _roll(q, 1, 2)
        force[..., 1] -= _roll(q, 1, 1) - q
        return force

    def forces(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.periodic:
            fz, fa = self._z_force_pbc(), self._a_force_pbc()
        else:
            # Patch sampling is normally wrapped in no_grad; OBC deliberately
            # uses autograd as its force reference and must locally re-enable it.
            with torch.enable_grad():
                z0, a0 = self.z, self.a
                self.z = z0.detach().requires_grad_(True)
                self.a = a0.detach().requires_grad_(True)
                gz, ga = torch.autograd.grad(self.action().sum(), (self.z, self.a))
                fz, fa = -gz, -ga
                self.z, self.a = z0, a0
        fz = self._project(self.z, fz)
        fz = fz.masked_fill(self.frozen_z[..., None], 0)
        fa = fa.masked_fill(self.frozen_a, 0)
        return fz, fa

    def _momenta(self) -> tuple[torch.Tensor, torch.Tensor]:
        pa = math.sqrt(self.mass_a) * torch.randn(
            self.a.shape, generator=self.generator, device=self.device, dtype=self.dtype)
        rz = torch.randn(self.z.shape, generator=self.generator,
                         device=self.device, dtype=self.dtype)
        iz = torch.randn(self.z.shape, generator=self.generator,
                         device=self.device, dtype=self.dtype)
        pz = math.sqrt(self.mass_z) * torch.complex(rz, iz)
        pa.masked_fill_(self.frozen_a, 0)
        pz.masked_fill_(self.frozen_z[..., None], 0)
        return self._project(self.z, pz), pa

    def _kinetic(self, pz: torch.Tensor, pa: torch.Tensor) -> torch.Tensor:
        return (0.5 * pa.square().sum((1, 2, 3)) / self.mass_a
                + 0.5 * torch.abs(pz).square().sum((1, 2, 3)) / self.mass_z)

    def hmc_step(self, epsilon: float | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        if epsilon is None:
            eps, leaps = self.epsilon, self.n_leapfrog
        else:
            leaps = max(1, int(math.ceil(self.trajectory_length / float(epsilon))))
            eps = self.trajectory_length / leaps
        z_old, a_old, s_old = self.z.clone(), self.a.clone(), self.s.clone()
        pz, pa = self._momenta()
        old_h = self.action() + self._kinetic(pz, pa)
        fz, fa = self.forces()
        pz += 0.5 * eps * fz
        pa += 0.5 * eps * fa
        pz = self._project(self.z, pz)
        for leap in range(leaps):
            self.a = self.a + eps * pa / self.mass_a
            pnorm = torch.linalg.vector_norm(pz, dim=-1)
            theta = eps * pnorm / self.mass_z
            safe = pnorm > 1e-15
            unit = torch.where(safe[..., None], pz / pnorm.clamp_min(1e-15)[..., None],
                               torch.zeros_like(pz))
            ct, st = torch.cos(theta)[..., None], torch.sin(theta)[..., None]
            pz, self.z = ct * pz - st * self.z * pnorm[..., None], ct * self.z + st * unit
            self.z /= torch.linalg.vector_norm(self.z, dim=-1, keepdim=True)
            pz = self._project(self.z, pz)
            fz, fa = self.forces()
            scale = 0.5 if leap == leaps - 1 else 1.0
            pz += scale * eps * fz
            pa += scale * eps * fa
            pz = self._project(self.z, pz)
        new_h = self.action() + self._kinetic(pz, pa)
        delta = new_h - old_h
        prob = torch.exp(-delta).clamp(max=1.0)
        finite = torch.isfinite(delta)
        accept = finite & (torch.rand(self.chains, generator=self.generator,
                            device=self.device, dtype=self.dtype) < prob)
        reject = ~accept
        self.z[reject], self.a[reject], self.s[reject] = z_old[reject], a_old[reject], s_old[reject]
        self.attempted += 1
        self.accepted += accept
        self.last_accept_prob = torch.where(finite, prob, torch.zeros_like(prob))
        self._wrap_and_shift_s()
        return accept, delta

    def _wrap_and_shift_s(self) -> None:
        wrapped = torch.remainder(self.a + math.pi, 2 * math.pi) - math.pi
        if self.kind == "full":
            delta = torch.round((self.a - wrapped) / (2 * math.pi)).to(torch.int64)
            if self.periodic:
                self.s += (delta[..., 0] + _roll(delta, -1, 1)[..., 1]
                           - _roll(delta, -1, 2)[..., 0] - delta[..., 1])
            else:
                self.s[:, :-1, :-1] += (delta[:, :-1, :-1, 0] + delta[:, 1:, :-1, 1]
                                        - delta[:, :-1, 1:, 0] - delta[:, :-1, :-1, 1])
        self.a = wrapped

    def metropolis_s(self, step: float = 0.5, updates: int = 1) -> None:
        if self.kind != "full":
            return
        valid = torch.ones_like(self.s, dtype=torch.bool)
        if not self.periodic:
            valid[:, -1, :] = False
            valid[:, :, -1] = False
        for _ in range(updates):
            jump = (torch.abs(step * torch.randn(self.s.shape, generator=self.generator,
                    device=self.device, dtype=self.dtype)).to(torch.int64) + 1)
            sign = torch.where(torch.rand(self.s.shape, generator=self.generator,
                               device=self.device) < 0.5, -1, 1)
            proposal = self.s + jump * sign
            da = self.plaquette_da()
            old_s = self.s if self.periodic else self.s[:, :-1, :-1]
            new_s = proposal if self.periodic else proposal[:, :-1, :-1]
            old_f, new_f = da + 2 * math.pi * old_s, da + 2 * math.pi * new_s
            delta = 0.5 * self.c.alpha * (new_f.square() - old_f.square())
            accept_core = torch.log(torch.rand(delta.shape, generator=self.generator,
                                    device=self.device, dtype=self.dtype)) < -delta
            accept = accept_core if self.periodic else torch.nn.functional.pad(accept_core, (0, 1, 0, 1))
            accept &= valid
            self.s = torch.where(accept, proposal, self.s)
            self.metro_attempted += valid.flatten(1).sum(1)
            self.metro_accepted += accept.flatten(1).sum(1)

    def sweep(self, *, s_step: float = 0.5, s_updates: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
        result = self.hmc_step()
        self.metropolis_s(s_step, s_updates)
        return result

    def adapt(self, steps: int, target: float = 0.75, *, s_step: float = 0.5,
              s_updates: int = 1,
              progress: Callable[[float, float], None] | None = None) -> float:
        """Dual-average a single shared epsilon, then freeze it."""
        if steps <= 0:
            return self.epsilon
        mu, log_eps = math.log(10 * self.epsilon), math.log(self.epsilon)
        log_bar, hbar, gamma, t0, kappa = log_eps, 0.0, 0.05, 10.0, 0.75
        for t in range(1, steps + 1):
            self.set_epsilon(math.exp(log_eps))
            self.hmc_step()
            self.metropolis_s(s_step, s_updates)
            eta = 1.0 / (t + t0)
            hbar = (1 - eta) * hbar + eta * (target - float(self.last_accept_prob.mean()))
            log_eps = mu - math.sqrt(t) * hbar / gamma
            weight = t ** -kappa
            log_bar = weight * log_eps + (1 - weight) * log_bar
            if progress is not None:
                progress(self.epsilon, float(self.last_accept_prob.mean()))
        self.set_epsilon(math.exp(log_bar))
        return self.epsilon

    def state_dict(self) -> dict:
        return {"z": self.z.detach().cpu(), "a": self.a.detach().cpu(),
                "s": self.s.detach().cpu(), "epsilon": self.epsilon,
                "trajectory_length": self.trajectory_length,
                "n_leapfrog": self.n_leapfrog,
                "generator_state": self.generator.get_state().cpu(),
                "accepted": self.accepted.cpu(), "attempted": self.attempted.cpu(),
                "metro_accepted": self.metro_accepted.cpu(),
                "metro_attempted": self.metro_attempted.cpu()}

    def load_state_dict(self, state: dict) -> None:
        self.z = state["z"].to(self.device, self.cdtype)
        self.a = state["a"].to(self.device, self.dtype)
        self.s = state["s"].to(self.device, torch.int64)
        saved_length = float(state.get("trajectory_length",
                                       float(state["epsilon"]) * int(state.get("n_leapfrog", 1))))
        if not math.isclose(saved_length, self.trajectory_length, rel_tol=1e-12,
                            abs_tol=1e-12):
            raise ValueError("checkpoint trajectory_length does not match sampler")
        self.set_epsilon(float(state["epsilon"]))
        self.generator.set_state(state["generator_state"].cpu())
        for name in ("accepted", "attempted", "metro_accepted", "metro_attempted"):
            setattr(self, name, state[name].to(self.device))
