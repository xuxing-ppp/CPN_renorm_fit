"""
Test that CPN_RefVil_HMCSampler_1plaq (full Villain) and CPNSampler_1plaq
(Original, cos plaquette) produce the SAME distribution of the integer
vortex measured from the gauge field (vortex_U) and from the matter field
(vortex_z), in the beta1 = alpha = 0 limit.

Why they must match
-------------------
With beta1 = 0 the matter deformation vanishes (the F1 piece of _z_force is
proportional to beta1). With alpha = 0 the Villain Gaussian
(alpha/2)*(da + 2*pi*s)^2 vanishes, leaving only

        -alpha1 * (cos(da + 2*pi*s) - 1) = -alpha1 * (cos(da) - 1)

because cos is 2*pi-periodic, so the integer s drops out of the plaquette
action. The (z, a) HMC dynamics therefore target exactly the same Boltzmann
distribution as the Original cos-plaquette heatbath sampler; the integer s is
a free (flat) field that does NOT couple back to z or a. Hence vortex_U (from
a) and vortex_z (from z) have identical distributions in the two samplers.

vortex_s is NOT compared: it is identically 0 in Original and a free random
walk in RefVil at alpha = 0 (the s-Metropolis has dS = 0, accept ~1).

Parallelism
-----------
Each (case, kind, replica) is an independent task run in a ProcessPoolExecutor.
Within a case, all replicas share the SAME frozen boundary (so the pooled
distribution is for one fixed boundary flux); RefVil and Original replicas are
paired replica-by-replica. A tqdm bar tracks the tasks.

Run (conda env myenv):
    python test_RefVil_vs_Original.py
"""
import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent))

import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

from func.func_CPN_RefVil_1plaq_fit import (
    CPN_RefVil_HMCSampler_1plaq, rand_fine_z, rand_fine_U, extract_param)
from func.func_CPN_Original_1plaq_fit import CPNSampler_1plaq


def collect(sampler, kind, n_therm, n_meas, meas_interval):
    """Thermalize, then record vortex_U / vortex_z every meas_interval sweeps."""
    for _ in range(n_therm):
        if kind == 'refvil':
            sampler.sweep(mod=0)
        else:
            sampler.sweep()
    vU, vz = [], []
    for i in range(n_meas):
        if kind == 'refvil':
            sampler.sweep(mod=0)
        else:
            sampler.sweep()
        if i % meas_interval == 0:
            v = sampler.get_vortex()      # [vortex_U, vortex_z, vortex_s]
            vU.append(v[0])
            vz.append(v[1])
    return np.array(vU), np.array(vz)


def run_replica(beta, alpha1, L, N, kind, boundary_seed, sampler_seed,
                n_therm, n_meas, meas_interval):
    """One worker: build a sampler on a fixed boundary, thermalize, measure.

    Must be top-level (not nested) so the spawned worker can import it.
    """
    beta1, alpha = 0.0, 0.0

    # shared boundary (same for every replica & both kinds within a case)
    np.random.seed(boundary_seed)
    z_list = rand_fine_z(L, N)
    U_list = rand_fine_U(L)
    aU, aZ, _ = extract_param(L, z_list, U_list)

    if kind == 'refvil':
        s = CPN_RefVil_HMCSampler_1plaq(
            N=N, L=L, beta=beta, beta1=beta1, alpha=alpha, alpha1=alpha1,
            z_list=z_list.copy(), U_list=U_list.copy(),
            epsilon=0.05, n_leapfrog=20, s_step=0.5, s_update_num=1,
            seed=sampler_seed)
    else:
        s = CPNSampler_1plaq(
            N=N, L=L, beta=beta, alpha1=alpha1,
            z_list=z_list.copy(), U_list=U_list.copy(), seed=sampler_seed)

    rt = {
        'z':  float(np.linalg.norm(z_list - s.get_boundary_z())),
        'U':  float(np.linalg.norm(U_list - s.get_boundary_U())),
        'aU': float(np.linalg.norm(aU - s.get_boundary_coarse_conn())),
        'aZ': float(np.linalg.norm(aZ - s.get_boundary_coarse_conn_z())),
    }
    vU, vz = collect(s, kind, n_therm, n_meas, meas_interval)

    if kind == 'refvil':
        hmc_rate, metro_rate = s.accept_rate
    else:
        hmc_rate = metro_rate = float('nan')
    return dict(kind=kind, vU=vU, vz=vz, rt=rt,
                hmc_rate=hmc_rate, metro_rate=metro_rate,
                da_U=float(np.sum(aU)), da_z=float(np.sum(aZ)))


def pmf(samples, support):
    """Empirical probability mass function over an integer support."""
    bins = np.arange(support[0] - 0.5, support[-1] + 1.5, 1.0)
    counts, _ = np.histogram(samples, bins=bins)
    return counts / counts.sum()


def compare(name, v_ref, v_ori):
    """Print and summarize the agreement of two integer-valued vortex samples."""
    support = np.arange(int(min(v_ref.min(), v_ori.min())),
                        int(max(v_ref.max(), v_ori.max())) + 1)
    p_ref = pmf(v_ref, support)
    p_ori = pmf(v_ori, support)
    tv = 0.5 * np.sum(np.abs(p_ref - p_ori))                     # total variation
    m_ref, s_ref = v_ref.mean(), v_ref.std()
    m_ori, s_ori = v_ori.mean(), v_ori.std()
    se = np.sqrt(s_ref ** 2 / len(v_ref) + s_ori ** 2 / len(v_ori))
    z = abs(m_ref - m_ori) / se if se > 0 else np.inf
    print(f"\n=== {name} ===")
    print(f"  support: {support[0]} .. {support[-1]}    "
          f"(n_ref={len(v_ref)}, n_ori={len(v_ori)})")
    print(f"  {'v':>3} {'p_ref':>9} {'p_ori':>9}")
    for vi, pr, po in zip(support, p_ref, p_ori):
        print(f"  {vi:>3} {pr:>9.4f} {po:>9.4f}")
    print(f"  mean   ref={m_ref:+.4f}  ori={m_ori:+.4f}   |diff|={abs(m_ref - m_ori):.4f}")
    print(f"  std    ref={s_ref:.4f}  ori={s_ori:.4f}")
    print(f"  TV distance = {tv:.4f}    z(naive) = {z:.2f}")
    return tv, abs(m_ref - m_ori)


def main():
    L, N = 10, 2
    n_therm = 300
    n_meas_per_replica = 2000
    meas_interval = 1
    n_replicas = 12                 # independent replicas per (case, kind)
    n_workers = min(os.cpu_count(), 6)

    cases = [
        dict(beta=0.5, alpha1=0.3, base=20240101),   # broad vortex distribution
        dict(beta=1.0, alpha1=0.5, base=20240202),   # narrower vortex distribution
    ]

    # ---- build task list: (case, kind, replica) ----
    tasks, keys = [], []
    for ci, c in enumerate(cases):
        bseed = c['base']                            # one shared boundary per case
        for r in range(n_replicas):
            for kind in ('refvil', 'original'):
                sseed = c['base'] + 1000 * (1 if kind == 'refvil' else 2) + r
                tasks.append((c['beta'], c['alpha1'], L, N, kind, bseed, sseed,
                              n_therm, n_meas_per_replica, meas_interval))
                keys.append((ci, kind, r))

    print(f"Running {len(tasks)} replicas on {n_workers} workers "
          f"({n_replicas} replicas x {len(cases)} cases x 2 samplers)...")

    by_key = {}
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futs = {ex.submit(run_replica, *t): k for t, k in zip(tasks, keys)}
        for fut in tqdm(as_completed(futs), total=len(futs),
                        desc='MC replicas', unit='replica'):
            by_key[futs[fut]] = fut.result()

    # ---- aggregate per (case, kind) and compare ----
    tol_mean, tol_tv = 0.25, 0.08
    all_ok = True
    for ci, c in enumerate(cases):
        beta1, alpha = 0.0, 0.0
        print("\n" + "#" * 60)
        print(f"# beta={c['beta']}, alpha1={c['alpha1']}, beta1={beta1}, "
              f"alpha={alpha}, L={L}, N={N}")
        print("#" * 60)
        r0 = by_key[(ci, 'refvil', 0)]
        print(f"coarse boundary flux  da_U/2pi = {r0['da_U'] / 2 / np.pi:+.4f}   "
              f"da_z/2pi = {r0['da_z'] / 2 / np.pi:+.4f}")

        rt_max = max(by_key[(ci, k, r)]['rt'][m]
                     for k in ('refvil', 'original')
                     for r in range(n_replicas)
                     for m in ('z', 'U', 'aU', 'aZ'))
        print(f"max boundary round-trip residual = {rt_max:.2e} (expect ~1e-15)")

        hmc = np.mean([by_key[(ci, 'refvil', r)]['hmc_rate'] for r in range(n_replicas)])
        metro = np.mean([by_key[(ci, 'refvil', r)]['metro_rate'] for r in range(n_replicas)])
        print(f"RefVil accept rates (mean of {n_replicas}):  "
              f"HMC={hmc:.3f}   s-metro={metro:.3f} "
              f"(s-metro ~1.0 expected: alpha=0 => dS=0)")

        vU_ref = np.concatenate([by_key[(ci, 'refvil', r)]['vU'] for r in range(n_replicas)])
        vU_ori = np.concatenate([by_key[(ci, 'original', r)]['vU'] for r in range(n_replicas)])
        vz_ref = np.concatenate([by_key[(ci, 'refvil', r)]['vz'] for r in range(n_replicas)])
        vz_ori = np.concatenate([by_key[(ci, 'original', r)]['vz'] for r in range(n_replicas)])

        tv_U, dmu_U = compare("vortex_U  (gauge field)", vU_ref, vU_ori)
        tv_z, dmu_z = compare("vortex_z  (matter field)", vz_ref, vz_ori)
        ok = (dmu_U < tol_mean and dmu_z < tol_mean
              and tv_U < tol_tv and tv_z < tol_tv)
        all_ok = all_ok and ok

    print("\n" + "=" * 60)
    print("RESULT:", "PASS  (vortex_U / vortex_z distributions match within tol)"
          if all_ok else "FAIL  (distributions disagree)")
    print("=" * 60)
    return all_ok


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
