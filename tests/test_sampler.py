import math
import unittest

import torch

from cpn_renorm.sampler import BatchedHMCSampler, Couplings
from cpn_renorm.observables import _z_link_phase, rectangular_loops, topo_charge


def make(kind="half", alpha=0.4, alpha1=0.0):
    return BatchedHMCSampler(chains=2, Lx=3, Ly=3, N=2,
        couplings=Couplings(0.7, 0.2, alpha, alpha1, 0), kind=kind,
        periodic=True, dtype=torch.float64, seed=3, epsilon=0.005, n_leapfrog=2)


class SamplerTests(unittest.TestCase):
    def test_z_link_phase_uses_forward_bra_ket_orientation(self):
        left = torch.tensor([[1 + 0j, 0j]], dtype=torch.complex128)
        right = torch.tensor([[1j, 0j]], dtype=torch.complex128)
        self.assertAlmostEqual(float(_z_link_phase(left, right)), math.pi / 2)

    def test_rectangular_z_loop_matches_forward_legacy_formula(self):
        sampler = make()
        measured = rectangular_loops(sampler, [(1, 1)])["1x1"]["z"]
        z = sampler.z
        vdot = lambda left, right: torch.sum(torch.conj(left) * right, dim=-1)
        zx, zy = torch.roll(z, -1, 1), torch.roll(z, -1, 2)
        zxy = torch.roll(z, (-1, -1), (1, 2))
        loop = vdot(z, zx) * vdot(zx, zxy) * vdot(zxy, zy) * vdot(zy, z)
        expected = torch.cos(torch.angle(loop)).mean((1, 2))
        self.assertTrue(torch.allclose(measured, expected, atol=1e-12, rtol=1e-12))

    def test_trajectory_length_survives_step_changes_and_checkpoint(self):
        sampler = BatchedHMCSampler(chains=1, Lx=3, Ly=3, N=2,
            couplings=Couplings(0.7, 0.0, 0.4), dtype=torch.float64,
            seed=9, epsilon=0.31, trajectory_length=1.0)
        for candidate in (0.31, 0.19, 0.07, 2.0):
            sampler.set_epsilon(candidate)
            self.assertAlmostEqual(sampler.epsilon * sampler.n_leapfrog, 1.0)
        state = sampler.state_dict()
        restored = BatchedHMCSampler(chains=1, Lx=3, Ly=3, N=2,
            couplings=Couplings(0.7, 0.0, 0.4), dtype=torch.float64,
            seed=10, epsilon=0.2, trajectory_length=1.0)
        restored.load_state_dict(state)
        self.assertAlmostEqual(restored.epsilon * restored.n_leapfrog, 1.0)

    def test_analytic_forces_match_autograd(self):
        cases = (("half", 0.4, 0.0, 0, 0.2),
                 ("full", 0.3, 0.2, 0, 0.2),
                 ("half", 0.4, 0.0, 1, 0.2),
                 ("half", 0.4, 0.0, 1, -0.2))
        for kind, alpha, alpha1, mod, beta1 in cases:
            with self.subTest(kind=kind, mod=mod, beta1=beta1):
                sampler = make(kind, alpha, alpha1)
                sampler.c = Couplings(0.7, beta1, alpha, alpha1, mod)
                z = sampler.z.detach().requires_grad_(True)
                a = sampler.a.detach().requires_grad_(True)
                sampler.z, sampler.a = z, a
                gz, ga = torch.autograd.grad(sampler.action().sum(), (z, a))
                analytic_z = sampler._project(z, sampler._z_force_pbc())
                analytic_a = sampler._a_force_pbc()
                self.assertTrue(torch.allclose(analytic_z, sampler._project(z, -gz),
                                                atol=2e-9, rtol=2e-8))
                self.assertTrue(torch.allclose(analytic_a, -ga, atol=2e-9, rtol=2e-8))

    def test_full_wrap_preserves_flux_and_action(self):
        sampler = make("full", 0.3, 0.2)
        sampler.a += 7 * math.pi
        flux, action = sampler.villain_flux().clone(), sampler.action().clone()
        sampler._wrap_and_shift_s()
        self.assertTrue(torch.allclose(sampler.villain_flux(), flux))
        self.assertTrue(torch.allclose(sampler.action(), action))

    def test_hmc_preserves_spin_norm(self):
        sampler = make()
        sampler.hmc_step()
        norm = torch.abs(sampler.z).square().sum(-1)
        self.assertTrue(torch.allclose(norm, torch.ones_like(norm), atol=1e-10))

    def test_obc_autograd_force_works_inside_no_grad_sampling(self):
        sampler = BatchedHMCSampler(chains=1, Lx=3, Ly=3, N=2,
            couplings=Couplings(0.2, 0.0, 0.1), periodic=False,
            dtype=torch.float64, seed=5, epsilon=0.01, trajectory_length=0.01)
        with torch.no_grad():
            sampler.sweep()
        self.assertEqual(int(sampler.attempted.item()), 1)

    def test_periodic_topological_charges_are_integer(self):
        sampler = make("full", 0.3, 0.2)
        charge = topo_charge(sampler)
        for tag in ("Q_U", "Q_z", "Q_s"):
            self.assertTrue(torch.allclose(charge[tag], torch.round(charge[tag]), atol=1e-10))
